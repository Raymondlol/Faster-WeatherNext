"""Batch size > 1 and the Mini model: per-element equivalence, same process / inputs / noise.

  mini  WeatherNextCyclones_Mini (1°): the official GPU path (`triblockdiag_mha`, patch disabled) vs
        faster-weathernext at batch 1 and 2, all output fields. Fits any GPU.
  wn2   WeatherNext2 at 0.25°: faster-weathernext at batch 2 vs itself at batch 1, element by element
        (the official path at batch 2 needs ~68 GiB). Also records memory and step time per batch size.

The FGN noise is normally drawn inside the model with one call per forward, so element i of a batch
would never see the same noise as a batch-1 run. Here the noise generator is replaced (for every
path alike) by one that inserts fixed rows of a deterministic table, which makes elements comparable.

  python scripts/verify_batch.py mini --data ".../source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-04.nc"
  python scripts/verify_batch.py wn2 --inits 2026092500 2026092400
"""
import argparse
import copy
import dataclasses
import datetime as dt
import functools
import gc
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

OFFICIAL = dict(attention_type="triblockdiag_mha")  # as the demo notebook and earth2studio set it on GPU
_NOISE = {}


def fixed_noise_generator(inputs, forcings, default_dtype, noise_var_dict, *, rows):
  """Drop-in for fgn.gaussian_noise_generator: row `rows[i]` of a fixed N(0,1) table for element i."""
  import jax.numpy as jnp
  import xarray_jax
  inputs = inputs.copy()
  forcings = forcings.copy() if forcings is not None else None
  for var, config in noise_var_dict.items():
    dims = tuple(s[0] for s in config["non_batch_shape"])
    shape = tuple(s[1] for s in config["non_batch_shape"])
    if var not in _NOISE:
      _NOISE[var] = np.random.default_rng(1234).standard_normal((16,) + shape).astype(np.float32)
    source = inputs if config["source"] == "input" else forcings
    assert source.sizes["batch"] == len(rows), (source.sizes["batch"], rows)
    source[var] = xarray_jax.Variable(("batch",) + dims,
                                      jnp.asarray(_NOISE[var][list(rows)], config.get("dtype", default_dtype)))
  return inputs, forcings


def build(model, rows, attention_type="chunked_mha", mask_type=None):
  """model.build(), with the fixed-noise generator and explicit transformer settings."""
  import haiku as hk
  import jax
  from weathernext.utils import fiddle_config_io
  from weathernext.weathernext2 import fgn
  config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{model}")
  pk = copy.deepcopy(config.predictor_kwargs)
  tk = pk["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
  tk["attention_type"] = attention_type
  if mask_type is not None:
    tk["mask_type"] = mask_type
  pk["noise_generator_constructor"] = functools.partial(fixed_noise_generator, rows=tuple(rows))
  cfg = fgn.PredictorConfig(task=config.task, predictor_constructor=config.predictor_constructor,
                            predictor_kwargs=pk, predictor_wrappers=config.predictor_wrappers[:-1])

  @hk.transform
  def run(inputs, targets_template, forcings):
    return fgn.construct_predictor(cfg)(inputs, targets_template=targets_template, forcings=forcings)

  return config.task, jax.jit(lambda params, rng, i, t, f: run.apply(params, rng, i, t, f))


def compare(a, b):
  rows = []
  for v in a.data_vars:
    levels = a[v]["level"].values if "level" in a[v].dims else [None]
    for lev in levels:
      x = np.asarray(a[v].sel(level=lev) if lev is not None else a[v], np.float64).ravel()
      y = np.asarray(b[v].sel(level=lev) if lev is not None else b[v], np.float64).ravel()
      m = np.isfinite(x) & np.isfinite(y)
      sd, d = x[m].std(), x[m] - y[m]
      rows.append(dict(field=f"{v}{'' if lev is None else '@' + str(int(lev))}",
                       corr=float(np.corrcoef(x[m], y[m])[0, 1]) if sd > 0 else 1.0,
                       rel_rms=float(np.sqrt(np.mean(d**2)) / sd) if sd > 0 else 0.0,
                       nan_mask_equal=bool((np.isnan(x) == np.isnan(y)).all())))
  return dict(min_corr=min(r["corr"] for r in rows), max_rel_rms=max(r["rel_rms"] for r in rows),
              median_rel_rms=float(np.median([r["rel_rms"] for r in rows])),
              nan_masks_equal=all(r["nan_mask_equal"] for r in rows), n_fields=len(rows),
              worst=sorted(rows, key=lambda r: -r["rel_rms"])[:3])


def run_case(model, params, example, rows, precision, patched, **transformer):
  """One compiled forward; returns (host predictions, info)."""
  import jax
  import faster_weathernext
  from faster_weathernext import patch
  faster_weathernext.enable() if patched else faster_weathernext.disable()
  _, forward = build(model, rows, **transformer)
  inputs, targets, forcings = example
  call = (params, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
  with jax.default_matmul_precision(precision):
    t0 = time.time()
    compiled = forward.lower(*call).compile()
    t_compile = time.time() - t0
    print(f"  compiled b={len(rows)}: temp {compiled.memory_analysis().temp_size_in_bytes / 2**30:.2f} GiB", flush=True)
    jax.block_until_ready(compiled(*call))
    t0 = time.time()
    out = jax.block_until_ready(compiled(*call))
    t_step = time.time() - t0
  info = dict(temp_gib=compiled.memory_analysis().temp_size_in_bytes / 2**30, step_s=t_step, compile_s=t_compile,
              batch=len(rows), rows=list(rows), patched=patched,
              attention=(patch.attention_impl() if patched else transformer.get("attention_type")))
  preds = jax.device_get(out)
  del compiled, out
  gc.collect()
  jax.clear_caches()
  print(f"  {model} {'fwn' if patched else 'official'} b={len(rows)} {precision:8s} temp {info['temp_gib']:5.2f} GiB "
        f"step {t_step:5.2f}s compile {t_compile:4.0f}s", flush=True)
  return preds, info


def concat(examples):
  import xarray
  return tuple(xarray.concat(parts, dim="batch") for parts in zip(*examples))


def mini_examples(path, task):
  import xarray
  from weathernext.utils import data_utils
  ds = xarray.load_dataset(path)
  out = []
  for start in (0, 1):  # two windows of the 4-step sample: inits 6 h apart
    w = ds.isel(time=slice(start, start + 3))
    w = w.assign_coords(time=w.time - w.time[0])
    out.append(data_utils.extract_inputs_targets_forcings(w, target_lead_times=slice("6h", "6h"),
                                                          **dataclasses.asdict(task)))
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("case", choices=["mini", "wn2"])
  p.add_argument("--data", help="mini: local path of Google's 1° steps-04 sample")
  p.add_argument("--inits", nargs=2, default=["2026092500", "2026092400"], help="wn2: two IFS init times")
  p.add_argument("--precisions", default="highest,default")
  p.add_argument("--weights-dir")
  p.add_argument("--out")
  p.add_argument("--save-preds", metavar="DIR", help="also write each run's predictions as <DIR>/<precision>_<run>.nc")
  args = p.parse_args()

  import jax
  import faster_weathernext
  from faster_weathernext import ifs, model

  faster_weathernext.enable()
  name = "WeatherNextCyclones_Mini" if args.case == "mini" else "WeatherNext2"
  task, _ = model.build(name)
  params = model.load_params(name, 1, args.weights_dir)
  if args.case == "mini":
    ex = mini_examples(args.data, task)
  else:
    ex = [ifs.rollout_inputs(dt.datetime.strptime(s, "%Y%m%d%H"), 1, task) for s in args.inits]
  singles = [(ex[0], (0,)), (ex[1], (1,))]
  pair = (concat(ex), (0, 1))
  result = dict(case=args.case, model=name, device=jax.devices()[0].device_kind, jax=jax.__version__,
                faster_weathernext=faster_weathernext.__version__, precisions={})

  for precision in args.precisions.split(","):
    print(f"[{precision}]", flush=True)
    r = dict(runs={}, checks={})
    fwn = {}
    # Largest case first: a fresh pool has the contiguous room its temp buffer needs.
    fwn["b2"], r["runs"]["fwn/b2"] = run_case(name, params, *pair, precision, True)
    for i, (e, rows) in enumerate(singles):
      fwn[f"b1_{i}"], r["runs"][f"fwn/b1_{i}"] = run_case(name, params, e, rows, precision, True)
    for i in range(2):  # element i of the batch-2 run vs the batch-1 run with the same noise row
      r["checks"][f"fwn_b2[{i}] vs fwn_b1_{i}"] = compare(fwn["b2"].isel(batch=[i]), fwn[f"b1_{i}"])
    if args.case == "mini":
      off = {}
      for i, (e, rows) in enumerate(singles):
        off[f"b1_{i}"], r["runs"][f"official/b1_{i}"] = run_case(name, params, e, rows, precision, False, **OFFICIAL)
      off["b2"], r["runs"]["official/b2"] = run_case(name, params, *pair, precision, False, **OFFICIAL)
      for k in ("b1_0", "b1_1", "b2"):
        r["checks"][f"fwn_{k} vs official_{k}"] = compare(fwn[k], off[k])
      for i in range(2):
        r["checks"][f"official_b2[{i}] vs official_b1_{i}"] = compare(off["b2"].isel(batch=[i]), off[f"b1_{i}"])
    if args.save_preds:
      os.makedirs(args.save_preds, exist_ok=True)
      for k, ds in fwn.items():
        ds.to_netcdf(os.path.join(args.save_preds, f"{precision}_fwn_{k}.nc"))
    for k, v in r["checks"].items():
      print(f"  {k:36s} min corr {v['min_corr']:.10f}  max rel RMS {v['max_rel_rms']:.2e}  "
            f"median {v['median_rel_rms']:.2e}  NaN masks equal {v['nan_masks_equal']}", flush=True)
    result["precisions"][precision] = r

  out = args.out or f"batch_{args.case}.json"
  with open(out, "w") as f:
    json.dump(result, f, indent=1)
  print("wrote", out)


if __name__ == "__main__":
  main()
