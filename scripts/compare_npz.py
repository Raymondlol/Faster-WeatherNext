"""Compare subsampled one-step outputs (cloud/canary_run.py --npz) between GPUs, per field.

  python scripts/compare_npz.py reference.npz other.npz
"""
import sys

import numpy as np

ref, other = np.load(sys.argv[1]), np.load(sys.argv[2])
rows = []
for k in ref.files:
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
