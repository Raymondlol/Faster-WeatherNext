"""Compare one-step outputs between two runs (e.g. two GPUs), per field.

Inputs are `.npz` files (as written by `cloud/canary_run.py --npz` or by `--save` below) or
`fwn forecast` NetCDF files. NetCDF fields are split per level and subsampled on a fixed
lat/lon stride so that a reference fits in a GitHub attachment; the same stride is applied
to both sides, so an `.npz` reference and an `.nc` candidate are comparable.

  python scripts/compare_npz.py reference.npz other.nc
  python scripts/compare_npz.py --save forecast.nc forecast.npz     # subsample to a small .npz
"""
import sys

import numpy as np

def stride(n_lon):
  """7 at 0.25° (721 x 1440 -> 103 x 206 points), 2 at 1°: ~100 fields stay under a 10 MB attachment."""
  return 7 if n_lon >= 1440 else 2 if n_lon >= 360 else 1


def load(path):
  if path.endswith(".npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}
  import xarray
  ds = xarray.open_dataset(path)
  out = {}
  for v in ds.data_vars:
    da = ds[v]
    if not {"lat", "lon"} <= set(da.dims):
      continue
    k = stride(da.sizes["lon"])
    da = da.isel(lat=slice(None, None, k), lon=slice(None, None, k))
    if "level" in da.dims:
      for lev in da["level"].values:
        out[f"{v}@{int(lev)}"] = np.asarray(da.sel(level=lev), np.float32)
    else:
      out[v] = np.asarray(da, np.float32)
  return out


def main(argv):
  if argv[:1] == ["--save"]:
    np.savez_compressed(argv[2], **load(argv[1]))
    print("wrote", argv[2])
    return
  ref, other = load(argv[0]), load(argv[1])
  keys = [k for k in ref if k in other]
  missing = sorted(set(ref) ^ set(other))
  rows = []
  for k in keys:
    x, y = ref[k].astype(np.float64).ravel(), other[k].astype(np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    sd = x[m].std()
    d = x[m] - y[m]
    rows.append((k, np.corrcoef(x[m], y[m])[0, 1] if sd > 0 else 1.0, np.sqrt(np.mean(d**2)) / sd if sd > 0 else 0.0,
                 bool((np.isnan(x) == np.isnan(y)).all())))
  print(f"{len(rows)} fields  min corr {min(r[1] for r in rows):.12f}  max rel RMS {max(r[2] for r in rows):.2e}  "
        f"median rel RMS {np.median([r[2] for r in rows]):.2e}  NaN masks equal: {all(r[3] for r in rows)}")
  for k, c, r, _ in sorted(rows, key=lambda r: -r[2])[:3]:
    print(f"  {k:32s} corr {c:.10f}  rel RMS {r:.2e}")
  if missing:
    print(f"  (fields only on one side, ignored: {len(missing)})")


if __name__ == "__main__":
  main(sys.argv[1:])
