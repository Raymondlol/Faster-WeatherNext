"""One WeatherNext 2 step through NVIDIA earth2studio's wrapper, with or without faster-weathernext.

Run twice in separate processes (the wrapper builds its jitted forward at load time), then compare:

  python scripts/earth2studio_check.py stock mini out/stock_mini.npz
  python scripts/earth2studio_check.py fwn   mini out/fwn_mini.npz --batch 2
  python scripts/earth2studio_check.py fwn   full out/fwn_full.npz      # 0.25°: does not fit 32 GB unpatched
  python scripts/earth2studio_check.py compare out/stock_mini.npz out/fwn_mini.npz

Inputs are Google's sample for the model (shipped with the earth2studio package), converted to the
wrapper's tensor layout; `--batch n` repeats it along the batch (ensemble) dimension.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402


def sample_tensor(model, package, cls, batch):
  import torch
  import xarray as xr
  from earth2studio.lexicon.wb2 import WB2Lexicon
  ds = xr.load_dataset(package.resolve(cls.SAMPLE_PATH))
  coords = model.input_coords()
  fields = []
  for name in coords["variable"]:
    wn2, level = WB2Lexicon.VOCAB[name].split("::")
    da = ds[wn2].isel(batch=0, time=slice(0, 2))  # the two input times
    if level:
      da = da.sel(level=int(level))
    fields.append(da.reindex(lat=coords["lat"], lon=coords["lon"]).values)  # lat descending, as the wrapper wants
  x = np.stack(fields, axis=1)[None, None]  # (batch, time, lead_time, variable, lat, lon)
  x = np.repeat(x, batch, axis=0)
  coords = coords.copy()
  coords["batch"] = np.arange(batch)
  coords["time"] = np.array([ds.datetime.values[0, 1]])  # valid time of the 0 h input
  return torch.as_tensor(x), coords


def run(mode, which, out, batch):
  if mode == "fwn":
    import faster_weathernext
    faster_weathernext.enable()
  import jax
  import torch
  from earth2studio.models.px import WeatherNext2Cyclones, WeatherNext2CyclonesMini
  cls = WeatherNext2CyclonesMini if which == "mini" else WeatherNext2Cyclones
  package = cls.load_default_package()
  t0 = time.time()
  model = cls.load_model(package, seed=0).to("cuda:0")
  x, coords = sample_tensor(model, package, cls, batch)
  x = x.to("cuda:0")
  print(f"{mode} {which} batch {batch}: input {tuple(x.shape)}, model loaded in {time.time() - t0:.0f}s", flush=True)
  t0 = time.time()
  y, ocoords = model(x, coords)  # compiles
  torch.cuda.synchronize()
  t_first = time.time() - t0
  model.set_rng(0)  # same noise again for the timed call
  t0 = time.time()
  y, ocoords = model(x, coords)
  torch.cuda.synchronize()
  t_step = time.time() - t0
  stats = jax.devices()[0].memory_stats() or {}
  info = dict(mode=mode, model=cls.__name__, batch=batch, first_call_s=t_first, step_s=t_step,
              jax_peak_gib=stats.get("peak_bytes_in_use", 0) / 2**30, out_shape=list(y.shape),
              variables=[str(v) for v in ocoords["variable"]], nan_fraction=float(torch.isnan(y).float().mean()))
  print(json.dumps(info), flush=True)
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  np.savez_compressed(out, y=y.cpu().numpy(), info=json.dumps(info))


def compare(a_path, b_path):
  a, b = np.load(a_path, allow_pickle=False), np.load(b_path, allow_pickle=False)
  ia, ib = json.loads(str(a["info"])), json.loads(str(b["info"]))
  ya, yb = a["y"].astype(np.float64), b["y"].astype(np.float64)
  assert ya.shape == yb.shape and ia["variables"] == ib["variables"], (ya.shape, yb.shape)
  rows = []
  for bi in range(ya.shape[0]):
    for vi, v in enumerate(ia["variables"]):
      x, y = ya[bi, ..., vi, :, :].ravel(), yb[bi, ..., vi, :, :].ravel()
      m = np.isfinite(x) & np.isfinite(y)
      sd = x[m].std()
      rows.append(dict(batch=bi, field=v, corr=float(np.corrcoef(x[m], y[m])[0, 1]) if sd > 0 else 1.0,
                       rel_rms=float(np.sqrt(np.mean((x[m] - y[m])**2)) / sd) if sd > 0 else 0.0,
                       nan_equal=bool((np.isnan(x) == np.isnan(y)).all())))
  summary = dict(a=ia, b=ib, min_corr=min(r["corr"] for r in rows), max_rel_rms=max(r["rel_rms"] for r in rows),
                 median_rel_rms=float(np.median([r["rel_rms"] for r in rows])),
                 nan_masks_equal=all(r["nan_equal"] for r in rows), n=len(rows),
                 worst=sorted(rows, key=lambda r: -r["rel_rms"])[:3])
  print(json.dumps(summary, indent=1))
  return summary


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("mode", choices=["stock", "fwn", "compare"])
  p.add_argument("a")
  p.add_argument("b")
  p.add_argument("--batch", type=int, default=1)
  a = p.parse_args()
  if a.mode == "compare":
    compare(a.a, a.b)
  else:
    run(a.mode, a.a, a.b, a.batch)
