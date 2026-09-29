"""faster-weathernext command line (`fwn`).

  fwn info
  fwn forecast --init 2026092800 --steps 40 --checkpoints 1 2 3 4 --samples 2 \\
                   --region 15 60 260 320 --out forecast.nc
"""

import argparse
import datetime as dt
import os
import sys
import time

DEFAULT_VARS = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind",
                "100m_u_component_of_wind", "100m_v_component_of_wind", "total_precipitation_6hr",
                "2m_temperature", "geopotential@500"]


def _configure_xla(args):
  """Must run before JAX initialises its backend."""
  # JAX preallocates 75% of GPU memory by default; a 0.25° step needs ~6.4 GiB, which does not
  # fit in 75% of an 8 GB card. Respect an explicit user setting.
  os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
  flags = [os.environ.get("XLA_FLAGS", "")]
  if getattr(args, "no_autotune", False):
    flags.append("--xla_gpu_autotune_level=0")
  cache = getattr(args, "autotune_cache", None)
  if cache:
    # XLA picks kernels by timing, so picks (and last-bit results) vary between runs; a saved
    # set of picks makes runs reproducible.
    key = "load" if os.path.exists(cache) else "dump"
    flags.append(f"--xla_gpu_{key}_autotune_results_from={cache}" if key == "load"
                 else f"--xla_gpu_dump_autotune_results_to={cache}")
  os.environ["XLA_FLAGS"] = " ".join(f for f in flags if f)


def cmd_info(args):
  _configure_xla(args)
  import jax
  import faster_weathernext
  from faster_weathernext import compat
  dev = jax.devices()[0]
  stats = dev.memory_stats() or {}
  print(f"faster-weathernext {faster_weathernext.__version__} | jax {jax.__version__} | "
        f"weathernext {compat.weathernext_version()} (validated at {compat.TESTED_WEATHERNEXT_COMMIT}) | "
        f"device {dev.device_kind} (compute capability {getattr(dev, 'compute_capability', '?')})")
  print(f"JAX memory pool: {stats.get('bytes_limit', 0) / 2**30:.2f} GiB "
        f"(XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}); "
        "a 0.25° step needs ~6.4 GiB")
  found = compat.problems()
  print("weathernext compatibility:", "OK" if not found else "PROBLEMS:\n  - " + "\n  - ".join(found))
  faster_weathernext.enable(force=bool(found))
  from faster_weathernext import patch
  print(f"attention implementation: {patch.attention_impl()}  (options: {patch.OPTIONS})")


def _subset(pred, variables, region):
  out = {}
  for v in variables:
    name, _, level = v.partition("@")
    da = pred[name]
    if level:
      da = da.sel(level=int(level)).drop_vars("level")
      name = f"{name}_{level}"
    if region:
      lat0, lat1, lon0, lon1 = region
      da = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    out[name] = da
  import xarray
  ds = xarray.Dataset(out)
  return ds.as_numpy() if hasattr(ds, "as_numpy") else ds


def cmd_forecast(args):
  _configure_xla(args)
  import contextlib
  import jax
  import numpy as np
  import xarray
  import faster_weathernext
  from faster_weathernext import ifs, model

  faster_weathernext.enable(attention=args.attention)
  init = dt.datetime.strptime(args.init, "%Y%m%d%H")
  task, forward = model.build(args.model)
  t0 = time.time()
  inputs, targets, forcings = ifs.rollout_inputs(init, args.steps, task)
  print(f"[{init:%Y-%m-%d %HZ}] inputs ready in {time.time() - t0:.0f}s; {args.steps} steps x "
        f"{len(args.checkpoints)} checkpoints x {args.samples} samples", flush=True)
  precision = (jax.default_matmul_precision("highest") if args.precision == "highest"
               else contextlib.nullcontext())
  members = []
  with precision:
    for k in args.checkpoints:
      params = model.load_params(args.model, k, args.weights_dir)
      for s in range(args.samples):
        t1 = time.time()
        rng = jax.random.PRNGKey(args.seed * 1000 + k * 10 + s)
        steps = [_subset(p, args.vars, args.region)
                 for p in model.rollout(forward, params, rng, inputs, targets, forcings)]
        member = xarray.concat(steps, dim="time").expand_dims(member=[f"m{k}s{s}"])
        members.append(member)
        print(f"   checkpoint {k} sample {s}: {time.time() - t1:.0f}s "
              "(first member includes compilation)", flush=True)
      del params
  ds = xarray.concat(members, dim="member")
  ds = ds.assign_coords(valid_time=("time", [np.datetime64(init) + t for t in ds.time.values]))
  ds.attrs.update(init_time=str(init), model=args.model, precision=args.precision,
                  source=f"{args.model} checkpoints {args.checkpoints} (Google DeepMind, CC BY 4.0), "
                         "ECMWF IFS open-data initial conditions (CC BY 4.0), run with faster-weathernext "
                         f"{faster_weathernext.__version__} (unofficial)")
  ds.to_netcdf(args.out + ".tmp", engine="h5netcdf")
  os.replace(args.out + ".tmp", args.out)
  print(f"saved {args.out} ({time.time() - t0:.0f}s total)")


def main(argv=None):
  p = argparse.ArgumentParser(prog="fwn", description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--no-autotune", action="store_true",
                      help="XLA autotune off (deterministic kernel picks; recommended for strict fp32 on A100)")
  common.add_argument("--autotune-cache", metavar="PATH",
                      help="save XLA's kernel picks here on the first run and reuse them afterwards")
  sub.add_parser("info", parents=[common], help="device, memory and compatibility check").set_defaults(fn=cmd_info)
  f = sub.add_parser("forecast", parents=[common], help="run a forecast from ECMWF IFS open data")
  f.add_argument("--init", required=True, help="initialisation time YYYYMMDDHH (00/06/12/18 UTC)")
  f.add_argument("--steps", type=int, default=40, help="number of 6 h steps (default 40 = 10 days)")
  f.add_argument("--model", default="WeatherNext2", choices=["WeatherNext2", "WeatherNextCyclones"])
  f.add_argument("--checkpoints", nargs="+", type=int, default=[1], help="1-4 (independently trained runs)")
  f.add_argument("--samples", type=int, default=1, help="noise samples per checkpoint")
  f.add_argument("--seed", type=int, default=0)
  f.add_argument("--vars", nargs="+", default=DEFAULT_VARS, help="output variables (name or name@level)")
  f.add_argument("--region", nargs=4, type=float, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                 help="crop the output (degrees north, degrees east 0-360)")
  f.add_argument("--precision", choices=["default", "highest"], default="default",
                 help="default = TF32 matmuls (fast); highest = fp32-level")
  f.add_argument("--attention", choices=["auto", "pallas", "xla"], default="auto")
  f.add_argument("--weights-dir", help="directory with (or for) the .npz checkpoints")
  f.add_argument("--out", required=True)
  f.set_defaults(fn=cmd_forecast)
  args = p.parse_args(argv)
  args.fn(args)


if __name__ == "__main__":
  sys.exit(main())
