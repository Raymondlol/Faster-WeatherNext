"""Fine-tuning gradients through faster-weathernext vs the official GPU path.

Mini model (1°) on Google's sample data; same checkpoint, inputs, targets and noise (same PRNG key).
The loss is the official training loss (fair CRPS over 2 members, finetune.build_loss) and the
gradients are taken w.r.t. the head (mesh->grid GNN + grid decoder; default) or, with
`--modules all`, every parameter (through the fused attention kernel's backward), at both matmul
precisions. Paths:

  official   patch disabled, attention_type="triblockdiag_mha" (the official GPU path)
  official_mha  patch disabled, attention_type="mha" (the official dense masked attention): a second
             official implementation, so official_mha vs official is the floor for differences that
             come only from rounding. The fair CRPS has sign terms (|pred - target|, |pred_i - pred_j|),
             so rounding-level forward differences flip a few gradient contributions; gradient
             agreement is judged against this floor, not against 0.
  fwn        patch enabled, remat=True, small blocks (grid_chunk 8192, edge_chunk 16384) so the
             65,160-point 1° grid runs through several encoder / decoder / GNN blocks (with the
             default 32,768-point blocks the encoder and decoder would take the stock path)
  fwn_xla      fwn with the XLA attention path, whose backward is JAX autodiff (with --modules all:
             an independent reference for the Pallas backward kernels)
  fwn_noremat  fwn without remat, compiled only: what rematerialisation saves
  fwn_strict   (highest only) fwn with IEEE fp32 attention dots instead of the kernel's 3xTF32

`--loss linear` replaces the CRPS by a smooth one: a fixed random linear functional of all output
fields (one member, no sample dimension). With no sign terms, implementation differences show at
rounding level; use it where no second official implementation fits as a floor (--modules all).

A case that runs out of GPU memory is recorded as such (the official path's full backward may not
fit); the comparisons then use the official path if it ran, else fwn_xla, as the reference.

  python scripts/verify_grads.py --data ".../source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-04.nc"
  python scripts/verify_grads.py --modules all --data ... --out full_grads_mini.json
"""
import argparse
import dataclasses
import gc
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

MODEL = "WeatherNextCyclones_Mini"
BLOCKS = dict(grid_chunk=8192, edge_chunk=16384)


def example(path, task):
  import xarray
  from weathernext.utils import data_utils
  ds = xarray.load_dataset(path).isel(time=slice(0, 3))
  ds = ds.assign_coords(time=ds.time - ds.time[0])
  parts = data_utils.extract_inputs_targets_forcings(ds, target_lead_times=slice("6h", "6h"),
                                                     **dataclasses.asdict(task))
  return tuple(x.drop_vars("datetime", errors="ignore").compute() for x in parts)


def linear_loss(attention_type, weights):
  """loss_fn with the CRPS replaced by sum_v mean(prediction_v * weights_v) (one member)."""
  import jax.numpy as jnp
  import xarray_jax
  from faster_weathernext import model
  _, forward = model.build(MODEL, attention_type=attention_type)

  def loss_fn(params, rng, inputs, targets, forcings):
    preds = forward(params, rng, inputs, targets, forcings)
    terms = {v: jnp.mean(jnp.nan_to_num(xarray_jax.unwrap_data(preds[v])) * w) for v, w in weights.items()}
    return sum(terms.values()), terms

  return loss_fn


def run_case(params, ex, masked, precision, path, modules, compile_only=False, weights=None):
  import jax
  import faster_weathernext
  from faster_weathernext import finetune
  if path.startswith("official"):
    faster_weathernext.disable()
    attention_type = "mha" if path == "official_mha" else "triblockdiag_mha"
  else:
    faster_weathernext.enable(remat=(path != "fwn_noremat"), strict_fp32=(path == "fwn_strict"),
                              attention="xla" if path == "fwn_xla" else "auto", **BLOCKS)
    attention_type = "chunked_mha"
  if weights is None:
    _, loss_fn = finetune.build_loss(MODEL, masked=masked, attention_type=attention_type)
  else:
    loss_fn = linear_loss(attention_type, weights)
  head, frozen = finetune.split(params, modules)
  fn = finetune.loss_and_grads(loss_fn)
  call = (head, frozen, jax.random.PRNGKey(0), *ex)
  out, info = None, {}
  try:
    with jax.default_matmul_precision(precision):
      t0 = time.time()
      compiled = fn.lower(*call).compile()
      info = dict(temp_gib=compiled.memory_analysis().temp_size_in_bytes / 2**30, compile_s=time.time() - t0)
      if not compile_only:
        jax.block_until_ready(compiled(*call))
        t0 = time.time()
        out = jax.device_get(jax.block_until_ready(compiled(*call)))
        info["step_s"] = time.time() - t0
      del compiled
  except jax.errors.JaxRuntimeError as e:
    if "RESOURCE_EXHAUSTED" not in str(e):
      raise
    info["oom"] = str(e).splitlines()[0][:300]
    out = None
  gc.collect()
  jax.clear_caches()
  temp = f"temp {info['temp_gib']:5.2f} GiB" if "temp_gib" in info else "temp   n/a"
  status = ("  OUT OF MEMORY" if "oom" in info else "  (compile only)" if compile_only
            else f"  step {info['step_s']:.2f}s  loss {float(out[0][0]):.6f}")
  print(f"  {path:12s} {precision:8s} {temp}{status}", flush=True)
  return out, info


def compare(got, ref):
  """Head gradients `got` vs `ref` (same tree): global and per-parameter relative L2 error."""
  rows, num, den, dot, n_got = [], 0.0, 0.0, 0.0, 0.0
  for m in sorted(ref):
    for k in sorted(ref[m]):
      a = np.asarray(got[m][k], np.float64).ravel()
      b = np.asarray(ref[m][k], np.float64).ravel()
      nb, nd = float(np.linalg.norm(b)), float(np.linalg.norm(a - b))
      num, den, dot, n_got = num + nd**2, den + nb**2, dot + float(a @ b), n_got + float(a @ a)
      rows.append(dict(param=f"{m}/{k}", size=a.size, ref_norm=nb, got_norm=float(np.linalg.norm(a)),
                       finite=bool(np.isfinite(a).all()),
                       rel_l2=nd / nb if nb > 0 else (0.0 if nd == 0 else float("inf"))))
  # Per-parameter errors are reported for tensors whose gradient is not negligible; tiny ones
  # (e.g. output heads of variables with near-zero loss weight) only enter the global numbers.
  big = [r for r in rows if r["ref_norm"] > 1e-3 * max(r["ref_norm"] for r in rows)]
  return dict(global_rel_l2=np.sqrt(num / den), global_cos=dot / np.sqrt(den * n_got),
              max_rel_l2=max(r["rel_l2"] for r in big), n_params=len(rows), n_compared=len(big),
              n_zero_both=sum(r["ref_norm"] == 0 and r["got_norm"] == 0 for r in rows),
              n_zero_ref_only=sum(r["ref_norm"] == 0 and r["got_norm"] > 0 for r in rows),
              all_finite=all(r["finite"] for r in rows),
              worst=sorted(big, key=lambda r: -r["rel_l2"])[:5])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True, help="local path of Google's 1° steps-04 sample")
  p.add_argument("--precisions", default="highest,default")
  p.add_argument("--modules", nargs="+", default=None, help='modules to differentiate (default: head); "all"')
  p.add_argument("--loss", choices=["crps", "linear"], default="crps")
  p.add_argument("--weights-dir")
  p.add_argument("--out", default="grads_mini.json")
  args = p.parse_args()

  import jax
  import faster_weathernext
  from faster_weathernext import finetune, model

  modules = finetune.HEAD_MODULES if not args.modules else "all" if args.modules == ["all"] else tuple(args.modules)
  task, _ = finetune.build_loss(MODEL)
  params = model.load_params(MODEL, 1, args.weights_dir)
  ex = example(args.data, task)
  masked = finetune.all_nan_vars(ex[1])
  weights = None
  if args.loss == "linear":  # fixed random weights per output field (NaN predictions count as 0)
    rng = np.random.default_rng(0)
    weights = {v: rng.standard_normal(da.shape[:1] + da.shape[2:]).astype(np.float32)[:, None]
               for v, da in ex[1].data_vars.items()}
  result = dict(model=MODEL, device=jax.devices()[0].device_kind, jax=jax.__version__,
                faster_weathernext=faster_weathernext.__version__, blocks=BLOCKS, masked=masked,
                modules=modules, loss=args.loss, head_params=finetune.head_size(params, modules), precisions={})
  print(f"{MODEL}: differentiating {result['head_params'] / 1e6:.2f}M params ({modules}); "
        f"masked targets: {masked}", flush=True)
  for precision in args.precisions.split(","):
    print(f"[{precision}]", flush=True)
    r, outs = {}, {}
    cases = ["official", "official_mha", "fwn", "fwn_xla"] + (["fwn_strict"] if precision == "highest" else [])
    for path in cases:  # fwn_strict: the Pallas kernel's default under `highest` is 3xTF32, not IEEE fp32
      outs[path], r[path] = run_case(params, ex, masked, precision, path, modules, weights=weights)
    _, r["fwn_noremat"] = run_case(params, ex, masked, precision, "fwn_noremat", modules, compile_only=True,
                                   weights=weights)
    ref_name = "official" if outs["official"] is not None else "fwn_xla"
    off_loss, off_grads = outs[ref_name]
    pairs = [(f"{p} vs {ref_name}" + (" (floor)" if p == "official_mha" else ""), outs[p])
             for p in ("fwn", "fwn_strict", "fwn_xla", "official_mha") if p != ref_name and outs.get(p) is not None]
    r["reference"] = ref_name
    r["checks"] = {}
    for label, (loss, grads) in pairs:
      lo, lg = float(off_loss[0]), float(loss[0])
      per_var = {k: abs(float(loss[1][k]) - float(v)) / abs(float(v)) for k, v in off_loss[1].items() if float(v) != 0}
      per_var = per_var or {"-": 0.0}
      g = compare(grads, off_grads)
      r["checks"][label] = dict(loss=dict(official=lo, other=lg, rel_diff=abs(lg - lo) / abs(lo),
                                          max_per_var_rel_diff=max(per_var.values())), grads=g)
      print(f"  {label:34s} loss rel diff {abs(lg - lo) / abs(lo):.2e} (per var max {max(per_var.values()):.2e})  "
            f"grads: global rel L2 {g['global_rel_l2']:.2e}  cos {g['global_cos']:.10f}  "
            f"max per-param rel L2 {g['max_rel_l2']:.2e} ({g['n_compared']}/{g['n_params']} params; "
            f"zero in both {g['n_zero_both']}, zero only in official {g['n_zero_ref_only']})  "
            f"finite {g['all_finite']}", flush=True)
      for w in g["worst"][:3]:
        print(f"      {w['rel_l2']:.2e}  {w['param']}", flush=True)
    result["precisions"][precision] = r
  with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
  print("wrote", args.out)


if __name__ == "__main__":
  main()
