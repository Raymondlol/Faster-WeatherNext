"""Fine-tuning of WeatherNext 2 at 0.25° on IFS open data (prototype).

Trains the head (mesh->grid GNN and grid decoder, finetune.HEAD_MODULES; `--modules all` trains
every parameter, through the attention kernel's backward) of one checkpoint with the official
training loss (fair CRPS over 2 members) on one-step examples from IFS open-data analyses
(ifs.training_example: inputs at T-6h and T; targets the analysis at T+6h and the 0-6 h
precipitation of the T run; the cyclone channels have no targets and are masked). One example per
optimizer step; the next example is built on the CPU while the GPU trains on the current one.

Before and after training the loss is evaluated on held-out init times with fixed noise keys, so
both evaluations see the same ensemble noise. This checks that the pipeline learns; it is not a
skill evaluation.

  python scripts/finetune.py --train 2026101912 2026101918 2026102000 --eval 2026103012 \
      --steps 30 --mask sea_surface_temperature --out head.npz --log finetune_run.json
  python scripts/finetune.py --train 2026102200 --compile-only   # memory of one training step
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/finetune.py --modules all ...
      # every parameter: a 25.5 GiB temp buffer + 3.1 GiB of arguments, too much for the 0.9 pool
"""
import argparse
import concurrent.futures
import datetime as dt
import json
import os
import time

# One preallocated pool: the 0.25° training step needs a single ~20 GiB temp buffer. In a pool that
# grows on demand, the regions taken earlier by the evaluation pass are kept but too small, and the
# step then fails to allocate (RESOURCE_EXHAUSTED) although enough memory is free in total.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import numpy as np  # noqa: E402


def parse_init(s):
  return dt.datetime.strptime(s, "%Y%m%d%H")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="WeatherNext2")
  p.add_argument("--checkpoint", type=int, default=1)
  p.add_argument("--train", nargs="+", required=True, help="init times YYYYMMDDHH (inputs T-6h, T; target T+6h)")
  p.add_argument("--eval", nargs="*", default=[], help="held-out init times")
  p.add_argument("--steps", type=int, default=10)
  # 3e-5 without warmup degraded held-out loss by 7-11 % within 60 steps (docs/validation.md §8).
  p.add_argument("--lr", type=float, default=3e-6)
  p.add_argument("--warmup", type=int, default=10, help="steps of linear learning-rate warmup from 0")
  p.add_argument("--weight-decay", type=float, default=0.0)
  p.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip")
  p.add_argument("--eval-keys", type=int, default=2, help="noise keys per eval example")
  p.add_argument("--precision", default="default")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--no-precip", action="store_true", help="mask precipitation instead of fetching its target")
  p.add_argument("--modules", nargs="+", default=None,
                 help='module names to train (default: the head, finetune.HEAD_MODULES); "all" trains every '
                      "parameter, through the attention backward")
  p.add_argument("--sequential-samples", choices=["on", "off"], default="on",
                 help="run the 2 ensemble members one at a time: same loss, about half the activation "
                      "memory (0.25°: head 20.0 -> 12.3 GiB temp, all parameters 45.8 -> 25.5 GiB), ~5 %% slower")
  p.add_argument("--mask", nargs="*", default=[],
                 help="extra target variables to leave out of the loss, e.g. sea_surface_temperature: open data "
                      "has no SST, and its stand-in (skin temperature) has a diurnal cycle real SST lacks")
  p.add_argument("--compile-only", action="store_true")
  p.add_argument("--no-remat", action="store_true", help="diagnostics: compile without block remat")
  p.add_argument("--weights-dir")
  p.add_argument("--out", help="write the fine-tuned checkpoint here (official format)")
  p.add_argument("--log", default="finetune_run.json")
  args = p.parse_args()

  import jax
  import optax
  import faster_weathernext
  from faster_weathernext import finetune, ifs, model

  faster_weathernext.enable(remat=not args.no_remat, sequential_samples=args.sequential_samples == "on")
  task, _ = model.build(args.model)
  load = lambda s: ifs.training_example(parse_init(s), task, precip=not args.no_precip)
  t0 = time.time()
  first = load(args.train[0])
  print(f"example {args.train[0]} built in {time.time() - t0:.1f}s", flush=True)
  masked = sorted(set(finetune.all_nan_vars(first[1])) | set(args.mask))
  _, loss_fn = finetune.build_loss(args.model, masked=masked)
  params = model.load_params(args.model, args.checkpoint, args.weights_dir)
  modules = finetune.HEAD_MODULES if not args.modules else "all" if args.modules == ["all"] else tuple(args.modules)
  head, frozen = finetune.split(params, modules)
  head = jax.device_put(head)
  frozen = jax.device_put(frozen)
  lr = (optax.join_schedules([optax.linear_schedule(0.0, args.lr, args.warmup), optax.constant_schedule(args.lr)],
                             [args.warmup]) if args.warmup else args.lr)
  optimizer = optax.chain(optax.clip_by_global_norm(args.clip),
                          optax.adamw(lr, weight_decay=args.weight_decay))
  opt_state = optimizer.init(head)
  dev = jax.devices()[0]
  log = dict(model=args.model, checkpoint=args.checkpoint, device=dev.device_kind, jax=jax.__version__,
             faster_weathernext=faster_weathernext.__version__, args=vars(args), masked=masked,
             modules=modules, head_params=finetune.head_size(params, modules),
             total_params=int(sum(np.size(v) for v in jax.tree.leaves(params))))
  print(f"{args.model} ckpt {args.checkpoint}: training {log['head_params'] / 1e6:.2f}M of "
        f"{log['total_params'] / 1e6:.1f}M params; masked targets: {len(masked)} "
        f"({', '.join(['cyclone channels'] + [v for v in masked if not v.startswith('cyclone')])})", flush=True)

  key = jax.random.PRNGKey(args.seed)
  step_fn = finetune.train_step(loss_fn, optimizer)
  with jax.default_matmul_precision(args.precision):
    t0 = time.time()
    compiled = step_fn.lower(head, frozen, opt_state, key, *first).compile()
    mem = compiled.memory_analysis()
    log["compile"] = dict(seconds=time.time() - t0, temp_gib=mem.temp_size_in_bytes / 2**30,
                          argument_gib=mem.argument_size_in_bytes / 2**30,
                          output_gib=mem.output_size_in_bytes / 2**30)
    print(f"train step compiled in {log['compile']['seconds']:.0f}s: temp {log['compile']['temp_gib']:.2f} GiB, "
          f"arguments {log['compile']['argument_gib']:.2f} GiB", flush=True)
    if args.compile_only:
      with open(args.log, "w") as f:
        json.dump(log, f, indent=1)
      print("wrote", args.log)
      return

    # frozen is an argument, not a closure: closed-over arrays can become executable constants.
    eval_fn = jax.jit(lambda h, fz, rng, i, t, f: loss_fn(finetune.merge(fz, h), rng, i, t, f))
    eval_examples = {s: load(s) for s in args.eval}
    eval_keys = [jax.random.PRNGKey(1000 + i) for i in range(args.eval_keys)]

    def evaluate(h):
      out = {}
      for s, ex in eval_examples.items():
        vals = [jax.device_get(eval_fn(h, frozen, k, *ex)) for k in eval_keys]
        out[s] = dict(loss=float(np.mean([v[0] for v in vals])),
                      per_var={k: float(np.mean([v[1][k] for v in vals])) for k in vals[0][1]})
      return out

    if eval_examples:
      t0 = time.time()
      log["eval_before"] = evaluate(head)
      print(f"eval before ({time.time() - t0:.0f}s): "
            + "  ".join(f"{s} {v['loss']:.5f}" for s, v in log["eval_before"].items()), flush=True)

    # Epochs over the training inits, each in a fresh random order.
    rng = np.random.default_rng(args.seed)
    names = [args.train[i] for _ in range(-(-args.steps // len(args.train)))
             for i in rng.permutation(len(args.train))][:args.steps]
    pool = concurrent.futures.ThreadPoolExecutor(1)
    ex = first if names[0] == args.train[0] else load(names[0])
    log["steps"] = []
    for step, name in enumerate(names):
      upcoming = pool.submit(load, names[step + 1]) if step + 1 < len(names) else None
      t0 = time.time()
      head, opt_state, metrics = compiled(head, frozen, opt_state, jax.random.fold_in(key, step), *ex)
      metrics = jax.device_get(metrics)
      dt_step = time.time() - t0
      log["steps"].append(dict(step=step, init=name, loss=float(metrics["loss"]),
                               grad_norm=float(metrics["grad_norm"]), seconds=dt_step))
      print(f"step {step:4d} {name}  loss {float(metrics['loss']):.5f}  |g| {float(metrics['grad_norm']):.3e}  "
            f"{dt_step:.2f}s", flush=True)
      if upcoming is not None:
        ex = upcoming.result()
    stats = dev.memory_stats() or {}
    log["peak_gib"] = stats.get("peak_bytes_in_use", 0) / 2**30
    times = [s["seconds"] for s in log["steps"][1:]] or [log["steps"][0]["seconds"]]
    log["median_step_s"] = float(np.median(times))
    print(f"median step {log['median_step_s']:.2f}s, device peak {log['peak_gib']:.2f} GiB", flush=True)

    if eval_examples:
      log["eval_after"] = evaluate(head)
      for s in eval_examples:
        b, a = log["eval_before"][s]["loss"], log["eval_after"][s]["loss"]
        print(f"eval {s}: {b:.5f} -> {a:.5f} ({100 * (a - b) / b:+.2f} %)", flush=True)

  if args.out:
    finetune.save(args.out, finetune.merge(frozen, head), model=args.model, checkpoint=args.checkpoint,
                  weights_dir=args.weights_dir,
                  note=f"Head-only fine-tuned with faster-weathernext {faster_weathernext.__version__}: "
                       f"{args.steps} steps, lr {args.lr}, IFS open data {args.train[0]}..{args.train[-1]}.")
    print("wrote", args.out)
  with open(args.log, "w") as f:
    json.dump(log, f, indent=1)
  print("wrote", args.log)


if __name__ == "__main__":
  main()
