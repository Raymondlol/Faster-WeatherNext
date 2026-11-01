"""Whole-model gradients vs the official path, on a GPU where the official backward fits (Modal H200).

On a 32 GB GPU the official path's end-to-end backward does not fit even for the Mini model at 1°
(compiled temp 82-116 GiB with the fair CRPS, 33-57 GiB with a one-member loss; docs/validation.md
§9). On an H200 (141 GB) it does, so scripts/verify_head_grads.py --modules all can compare
faster-weathernext (Pallas backward kernels) with the official GPU path directly, with a second
official implementation as the floor. Each loss (crps, linear) runs in its own container; the first
also runs the kernel tests (the Pallas kernel's first run on Hopper, sm_90).

  modal run --detach cloud/modal_grads.py --tag grads01
  modal volume get fwn-grads results/grads01 ./grads01
  modal app stop -y <app id>      # detached apps can linger after finishing

Cost (Modal, 2026-11): H200 $4.54/h + 8 cores and 64 GiB ~ $0.9/h; ~40 min per container.
"""
import os
import subprocess
import time
import urllib.parse
import urllib.request

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "https://storage.googleapis.com/dm_graphcast/weathernext2"
WEIGHTS = "params/WeatherNextCyclones_Mini_<2024.npz"
DATA = "dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-04.nc"

app = modal.App("fwn-grads")
vol = modal.Volume.from_name("fwn-grads", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(REPO, "/opt/faster-weathernext", copy=True,
                   ignore=[".git", "**/__pycache__", "docs/results", "*.nc", "*.npz", ".pytest_cache"])
    .run_commands("pip install '/opt/faster-weathernext[cuda12,weathernext,test]'")
)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2)
def fetch():
  for rel in (WEIGHTS, DATA):
    dst = os.path.join("/data", os.path.basename(rel))
    if not os.path.exists(dst):
      urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(rel)}", dst + ".part")
      os.replace(dst + ".part", dst)
  vol.commit()


def _run(cmd, log, env=None):
  t0 = time.time()
  proc = subprocess.run(cmd, cwd="/opt/faster-weathernext", capture_output=True, text=True,
                        env={**os.environ, **(env or {})})
  with open(log, "w") as f:
    f.write(proc.stdout + proc.stderr + f"\nexit {proc.returncode}  wall {time.time() - t0:.0f}s\n")
  vol.commit()
  return proc.returncode


@app.function(image=image, gpu="H200", volumes={"/data": vol}, timeout=9000, cpu=8, memory=65536)
def grads(loss: str, tag: str, run_tests: bool, precisions: str) -> str:
  out = f"/data/results/{tag}"
  os.makedirs(out, exist_ok=True)
  smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
  lines = [f"[{loss}] {smi}"]
  if run_tests:
    rc = _run(["python", "-m", "pytest", "-q", "-rs", "tests/test_pallas_kernel.py"], f"{out}/pytest_h200.txt")
    lines.append(f"kernel tests exit {rc}: " + open(f"{out}/pytest_h200.txt").read().strip().splitlines()[-2])
  # The official mha path needs ~116 GiB with the CRPS: above JAX's default 75 % pool.
  rc = _run(["python", "scripts/verify_head_grads.py", "--modules", "all", "--loss", loss,
             "--data", f"/data/{os.path.basename(DATA)}", "--weights-dir", "/data",
             "--precisions", precisions, "--out", f"{out}/full_grads_mini_{loss}_h200.json"],
            f"{out}/full_grads_mini_{loss}_h200.log", env={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
  log = open(f"{out}/full_grads_mini_{loss}_h200.log").read().splitlines()
  lines += [f"verify exit {rc}"] + [l for l in log if l.startswith(("[", "  ")) and "WARNING" not in l][-40:]
  return "\n".join(lines)


@app.local_entrypoint()
def main(tag: str = "grads01", losses: str = "crps,linear", precisions: str = "highest,default",
         tests: bool = True):
  fetch.remote()
  calls = [grads.spawn(loss, tag, tests and i == 0, precisions) for i, loss in enumerate(losses.split(","))]
  for call in calls:
    print(call.get(), flush=True)
