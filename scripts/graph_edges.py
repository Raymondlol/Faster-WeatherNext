"""Count the edges of the grid<->mesh graphs that the official code builds on the host CPU.

WN2 builds its grid->mesh edges with a ball query (scipy cKDTree, radius = a fraction of the
longest mesh edge) on unit-sphere coordinates computed from float32 lat/lon with NumPy (mesh
lat/lon via np.arctan2 / np.arccos, then np.sin / np.cos). Grid points close to the query radius
are kept or dropped depending on the last bits of those coordinates, and NumPy's results depend on
the code path it picks for the host CPU (AVX-512, AVX2, baseline). This script traces one forward
step on the CPU (no compilation) and prints, per graph, the edge count, a hash of the edge indices
and how many (grid point, mesh node) pairs lie within a relative 1e-12 ... 1e-5 of the radius.

  JAX_PLATFORMS=cpu python scripts/graph_edges.py --data SAMPLE.nc [--weights-dir DIR] [--json out.json]
      [--save-edges edges.npz]        # reuse elsewhere with cloud/canary_run.py --edges edges.npz
  # NumPy's code path for a CPU without AVX-512 (NumPy 2.5 names; "X86_V3 ..." too: no AVX2):
  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" JAX_PLATFORMS=cpu python scripts/graph_edges.py ...
"""
import argparse
import dataclasses
import hashlib
import json
import os
import platform

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="Google's 0.25° sample (see scripts/verify_equivalence.py)")
p.add_argument("--model", default="WeatherNextCyclones")
p.add_argument("--weights-dir")
p.add_argument("--json", default="")
p.add_argument("--save-edges", default="", help=".npz with the edge indices (for cloud/canary_run.py --edges)")
args = p.parse_args()

import jax  # noqa: E402
import xarray  # noqa: E402
from scipy import spatial  # noqa: E402
from weathernext.utils import data_utils  # noqa: E402
from weathernext.utils import model_utils  # noqa: E402
from weathernext.utils import points_mesh_gnn as pmg  # noqa: E402

import faster_weathernext  # noqa: E402
from faster_weathernext import model  # noqa: E402

from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__  # noqa: E402

cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
res = dict(cpu=cpu, numpy=np.__version__, avx512_skx=bool(__cpu_features__.get("AVX512_SKX")),
           dispatch=[f for f in __cpu_dispatch__ if __cpu_features__.get(f)],
           npy_disable_cpu_features=os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), graphs=[])

orig = pmg.PointsMeshTypedGraphGNN._get_edge_indices
edges = {}


def hooked(self, plat, plon, mlat, mlon, faces):
  pts, mesh = orig(self, plat, plon, mlat, mlon, faces)
  g = dict(connectivity=self._connectivity_type.name, n_edges=int(len(pts)),
           sha1=hashlib.sha1(np.stack([pts, mesh]).astype(np.int64).tobytes()).hexdigest()[:16])
  if self._connectivity_type == pmg.ConnectivityType.BALL_QUERY:
    pc = np.stack(model_utils.lat_lon_to_cartesian(plat, plon), -1)
    mc = np.stack(model_utils.lat_lon_to_cartesian(mlat, mlon), -1)
    r = self._ball_query_radius_fraction * pmg._max_edge_distance(mc, faces)
    near = spatial.cKDTree(mc).query_ball_point(pc, r=r * (1 + 1e-5))
    d = np.concatenate([np.linalg.norm(mc[nb] - pc[i], axis=-1) for i, nb in enumerate(near) if nb])
    g.update(dtypes=[str(np.asarray(a).dtype) for a in (plat, plon, mlat, mlon)], radius=float(r),
             pairs_near_radius={f"{t:.0e}": int((np.abs(d - r) <= t * r).sum()) for t in (1e-12, 1e-9, 1e-7, 1e-6, 1e-5)})
  res["graphs"].append(g)
  edges[f"{g['connectivity']}_points"], edges[f"{g['connectivity']}_mesh"] = pts.astype(np.int32), mesh.astype(np.int32)
  print(g, flush=True)
  return pts, mesh


pmg.PointsMeshTypedGraphGNN._get_edge_indices = hooked
faster_weathernext.enable()
task, forward = model.build(args.model)
params = model.load_params(args.model, 1, args.weights_dir)
ds = xarray.load_dataset(args.data)
inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    ds, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))
# Tracing builds the graphs (host numpy/scipy); nothing is compiled or run on a device.
jax.eval_shape(forward, params, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
print(json.dumps(res, indent=1))
if args.json:
  with open(args.json, "w") as f:
    json.dump(res, f, indent=1)
if args.save_edges:
  np.savez_compressed(args.save_edges, **edges)
