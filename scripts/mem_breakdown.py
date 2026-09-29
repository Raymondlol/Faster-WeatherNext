"""Compile one 0.25° step (no execution) and report XLA's memory use and the largest temp slots.

  python scripts/mem_breakdown.py --dump /tmp/xla_dump [--reference] [--data SAMPLE.nc]

Reads XLA's `*memory-usage-report.txt` from the dump directory. Note: HloLiveRange's "peak"
double-counts aliased buffers, and hlo_rematerialization's warnings are not actual usage.
Executable constants are not part of temp; they are reported separately from the optimized HLO.
"""
import argparse
import dataclasses
import glob
import os
import re
import sys

p = argparse.ArgumentParser()
p.add_argument("--dump", required=True)
p.add_argument("--data", help="Google's 0.25° sample (see scripts/verify_equivalence.py)")
p.add_argument("--weights-dir")
p.add_argument("--reference", action="store_true", help="P0 reference options instead of defaults")
p.add_argument("--top", type=int, default=12)
args = p.parse_args()
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={args.dump}"
                           " --xla_dump_hlo_module_re=.*jit.*lambda.* --xla_gpu_autotune_level=0").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import xarray  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import faster_weathernext  # noqa: E402
from faster_weathernext import ifs, model  # noqa: E402
from verify_equivalence import REFERENCE, SAMPLE  # noqa: E402
from weathernext.utils import data_utils  # noqa: E402

faster_weathernext.enable(**(REFERENCE if args.reference else {}))
task, forward = model.build("WeatherNextCyclones")
params = model.load_params("WeatherNextCyclones", 1, args.weights_dir)
data = args.data or os.path.join(ifs.cache_dir(), SAMPLE)
inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    xarray.load_dataset(data), target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))
compiled = forward.lower(params, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings).compile()
m = compiled.memory_analysis()
G = 2**30
print(f"args {m.argument_size_in_bytes / G:.2f} GiB  outputs {m.output_size_in_bytes / G:.2f} GiB  "
      f"temp {m.temp_size_in_bytes / G:.2f} GiB")

report = sorted(glob.glob(os.path.join(args.dump, "*memory-usage-report.txt")), key=os.path.getmtime)
if report:
  print("\nlargest temp slots:")
  print("\n".join(l[:200] for l in open(report[-1]).read().splitlines()[:args.top + 4]))
hlo = sorted(glob.glob(os.path.join(args.dump, "*after_optimizations.txt")), key=os.path.getmtime)
if hlo:
  total = 0
  for mt in re.finditer(r"= [a-z]+(\d+)\[([\d,]*)\]\{[^}]*\} constant\(", open(hlo[-1]).read()):
    total += int(np.prod([int(d) for d in mt.group(2).split(",") if d] or [1])) * int(mt.group(1)) // 8
  print(f"\nexecutable constants (outside JAX's pool): {total / 2**20:.0f} MiB")
