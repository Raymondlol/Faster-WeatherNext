"""One 0.25° step: faster-weathernext vs the reference path, same process / inputs / RNG, per output field.

Reference = faster-weathernext configured as the "P0" path: XLA chunked attention (1024-query chunks),
the official GNN / encoder / decoder, unrolled layers, XLA autotuning off. P0 showed that path
matches the official GPU implementation (`triblockdiag_mha`) to fp32 rounding at full
resolution on an A100-80GB (docs/validation.md); the fully official path needs ~34 GiB and
does not fit a 32 GB GPU. The reference needs ~24 GB.

Uses the WeatherNextCyclones checkpoint and Google's 0.25° sample data (the public sample has no
100 m winds, which WeatherNext2 needs; the architectures are identical). Both are downloaded
from Google's bucket and cached (~2 GB).

  python scripts/verify_equivalence.py --out equiv.json
"""
import argparse
import contextlib
import dataclasses
import gc
import json
import os
import time
import urllib.parse
import urllib.request

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

SAMPLE = "source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc"
SAMPLE_URL = "https://storage.googleapis.com/dm_graphcast/weathernext2/dataset/" + urllib.parse.quote(SAMPLE)

REFERENCE = dict(attention="xla", attn_chunk=1024, grid_chunk=0, edge_chunk=0, layer_loop=False,
                 reorder=False)


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


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--precisions", default="highest,default")
  p.add_argument("--data", help=f"local copy of {SAMPLE}")
  p.add_argument("--weights-dir")
  p.add_argument("--out", default="equivalence.json")
  args = p.parse_args()

  import jax
  import xarray
  import faster_weathernext
  from faster_weathernext import ifs, model, patch
  from weathernext.utils import data_utils

  data = args.data or os.path.join(ifs.cache_dir(), SAMPLE)
  if not os.path.exists(data):
    print(f"downloading {SAMPLE_URL}", flush=True)
    urllib.request.urlretrieve(SAMPLE_URL, data + ".part")
    os.replace(data + ".part", data)
  faster_weathernext.enable()
  defaults = dataclasses.asdict(patch.OPTIONS)
  task, _ = model.build("WeatherNextCyclones")
  params = model.load_params("WeatherNextCyclones", 1, args.weights_dir)
  inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
      xarray.load_dataset(data), target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))
  call = (params, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
  dev = jax.devices()[0]
  result = dict(device=dev.device_kind, jax=jax.__version__, faster_weathernext=faster_weathernext.__version__, cases={})
  for precision in args.precisions.split(","):
    preds = {}
    for label, opts, autotune in (("reference", REFERENCE, 0), ("faster_weathernext", {}, None)):
      for k, v in {**defaults, **opts}.items():
        setattr(patch.OPTIONS, k, v)
      _, forward = model.build("WeatherNextCyclones")  # fresh jit: options are read at trace time
      ctx = contextlib.nullcontext() if precision == "default" else jax.default_matmul_precision(precision)
      with ctx:
        t0 = time.time()
        compiled = forward.lower(*call).compile(
            None if autotune is None else {"xla_gpu_autotune_level": autotune})
        t_compile = time.time() - t0
        jax.block_until_ready(compiled(*call))
        t0 = time.time()
        out = jax.block_until_ready(compiled(*call))
        t_step = time.time() - t0
      preds[label] = jax.device_get(out)
      info = dict(temp_gib=compiled.memory_analysis().temp_size_in_bytes / 2**30, step_s=t_step,
                  compile_s=t_compile, attention=patch.attention_impl())
      result["cases"][f"{precision}/{label}"] = info
      print(f"{precision:8s} {label:10s} temp {info['temp_gib']:5.2f} GiB  step {t_step:5.2f}s  "
            f"compile {t_compile:4.0f}s  attention={info['attention']}", flush=True)
      del compiled, out
      gc.collect()
      jax.clear_caches()
    stats = compare(preds["reference"], preds["faster_weathernext"])
    result["cases"][f"{precision}/faster_weathernext"]["vs_reference"] = stats
    print(f"{precision:8s} faster-weathernext vs reference: min corr {stats['min_corr']:.10f}  max rel RMS "
          f"{stats['max_rel_rms']:.2e}  median {stats['median_rel_rms']:.2e}  NaN masks equal "
          f"{stats['nan_masks_equal']}  ({stats['n_fields']} fields)", flush=True)
  for k, v in defaults.items():
    setattr(patch.OPTIONS, k, v)
  with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
  print("wrote", args.out)


if __name__ == "__main__":
  main()
