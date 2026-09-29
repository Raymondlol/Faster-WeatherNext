"""Real-card canary: 0.25° WeatherNextCyclones inference with faster-weathernext on one GPU.

The environment (set before JAX starts) decides the memory pool:
  XLA_PYTHON_CLIENT_PREALLOCATE=false   -> true in-pool peak
  XLA_PYTHON_CLIENT_MEM_FRACTION=f      -> capped pool (e.g. 7.0 GiB = an 8 GB card at 0.9)

  1. one step at precision "highest", saved on every 4th grid point (--npz) for cross-GPU
     comparison (scripts/compare_npz.py);
  2. an N-step rollout at default precision: per-step wall time, finite check, pool peak.

  python cloud/canary_run.py --data SAMPLE.nc --json out.json [--npz out.npz] [--steps 4]
"""
import argparse
import dataclasses
import json
import os
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="Google's 0.25° sample (see scripts/verify_equivalence.py)")
p.add_argument("--weights-dir")
p.add_argument("--json", required=True)
p.add_argument("--npz", default="")
p.add_argument("--steps", type=int, default=4)
p.add_argument("--label", default="")
args = p.parse_args()
res = dict(label=args.label, env={k: v for k, v in os.environ.items() if k.startswith(("XLA_", "JAX_"))}, ok=False)

try:
  import jax
  import numpy as np
  import pandas as pd
  import xarray
  import faster_weathernext
  from faster_weathernext import model, patch
  from weathernext.utils import data_utils

  faster_weathernext.enable()
  G = 2**30
  dev = jax.devices()[0]
  res.update(device=dev.device_kind, jax=jax.__version__, faster_weathernext=faster_weathernext.__version__,
             compute_capability=getattr(dev, "compute_capability", ""), attention=patch.attention_impl(),
             pool_gib=(dev.memory_stats() or {}).get("bytes_limit", 0) / G)
  print(f"{dev.device_kind} cc={res['compute_capability']} attention={res['attention']} "
        f"pool {res['pool_gib']:.2f} GiB", flush=True)
  task, forward = model.build("WeatherNextCyclones")
  params = model.load_params("WeatherNextCyclones", 1, args.weights_dir)
  ds = xarray.load_dataset(args.data)
  inputs, targets1, forcings1 = data_utils.extract_inputs_targets_forcings(
      ds, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))

  if args.npz:
    t0 = time.time()
    with jax.default_matmul_precision("highest"):
      pred = jax.device_get(forward(params, jax.random.PRNGKey(0), inputs, targets1 * np.nan, forcings1))
    res["highest_first_call_s"] = time.time() - t0
    fields = {}
    for v in pred.data_vars:
      a = np.asarray(pred[v].transpose(..., "lat", "lon"))[..., ::4, ::4].astype(np.float32)
      if "level" in pred[v].dims:
        for i, lev in enumerate(pred[v]["level"].values):
          fields[f"{v}@{int(lev)}"] = a.reshape(-1, *a.shape[-3:])[0, i]
      else:
        fields[v] = a.reshape(-1, *a.shape[-2:])[0]
    np.savez_compressed(args.npz, **fields)
    print(f"highest-precision step saved ({len(fields)} fields) in {res['highest_first_call_s']:.0f}s", flush=True)
    del pred

  lead = pd.to_timedelta(np.arange(1, args.steps + 1) * 6, unit="h")
  tvars = {}
  for v, da in targets1.data_vars.items():
    shape = list(da.shape)
    shape[da.dims.index("time")] = args.steps
    tvars[v] = (da.dims, np.broadcast_to(np.float32(np.nan), shape))
  targets = xarray.Dataset(tvars, coords={**{k: c for k, c in targets1.coords.items() if k != "time"}, "time": lead})
  init = pd.Timestamp(ds["datetime"].values[0, 1]).to_pydatetime()
  fds = xarray.Dataset(coords=dict(time=lead, lon=ds.lon, datetime=(("batch", "time"), np.array(
      [[init + td for td in lead.to_pytimedelta()]], dtype="datetime64[ns]"))))
  data_utils.add_derived_vars(fds)
  forcings = fds[list(task.forcing_variables)].drop_vars("datetime")
  forcings = forcings.astype({v: forcings1[v].dtype for v in forcings1.data_vars})
  res["step_s"], res["finite"] = [], []
  t0 = time.time()
  for pred in model.rollout(forward, params, jax.random.PRNGKey(0), inputs, targets, forcings):
    msl = np.asarray(pred["mean_sea_level_pressure"])
    res["step_s"].append(time.time() - t0)
    res["finite"].append(bool(np.isfinite(msl).all()))
    print(f"  step {len(res['step_s'])}: {res['step_s'][-1]:.1f}s  msl min {np.nanmin(msl) / 100:.1f} hPa", flush=True)
    t0 = time.time()
  res["peak_gib"] = (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / G
  res["ok"] = all(res["finite"])
  print(f"peak {res['peak_gib']:.2f} GiB of pool {res['pool_gib']:.2f} GiB; ok={res['ok']}", flush=True)
except Exception as e:  # noqa: BLE001  record failures (e.g. OOM) instead of losing them
  res["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
  res["traceback"] = traceback.format_exc()[-4000:]
  print(res["error"], flush=True)
finally:
  os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
  with open(args.json, "w") as f:
    json.dump(res, f, indent=1)
