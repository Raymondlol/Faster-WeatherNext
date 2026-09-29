"""Multi-GPU canary on Modal: install this repo in a clean image and run cloud/canary_run.py.

Per card, two fresh processes: uncapped (true peak, speed, fp32 step for cross-GPU comparison)
and "cap8" (JAX pool capped at 7.0 GiB, i.e. an 8 GB card at MEM_FRACTION=0.9).

  modal run --detach cloud/modal_canary.py --cards a100,a10g,l4,t4 --tag canary01
  modal volume get fwn-canary results/canary01 ./canary01
  modal app stop -y <app id>      # detached apps can linger after finishing

Cost (2026 prices): roughly $0.2-0.5 per card.
"""
import os
import subprocess
import time
import urllib.parse
import urllib.request

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "https://storage.googleapis.com/dm_graphcast/weathernext2"
WEIGHTS = "params/WeatherNextCyclones_<2025_model1.npz"
DATA = "dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc"
CAP_GIB = 7.0

app = modal.App("fwn-canary")
vol = modal.Volume.from_name("fwn-canary", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(REPO, "/opt/faster-weathernext", copy=True,
                   ignore=[".git", "**/__pycache__", "docs/results", "*.nc", "*.npz"])
    .run_commands("pip install '/opt/faster-weathernext[cuda12]'")
)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4)
def fetch():
  for rel in (WEIGHTS, DATA):
    dst = os.path.join("/data", os.path.basename(rel))
    if not os.path.exists(dst):
      urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(rel)}", dst + ".part")
      os.replace(dst + ".part", dst)
  vol.commit()


def _card(card: str, tag: str, steps: int) -> dict:
  import json
  total_mib = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0])
  out = f"/data/results/{tag}"
  os.makedirs(out, exist_ok=True)
  summary = {"card": card, "total_mib": total_mib}
  runs = {"uncapped": dict(XLA_PYTHON_CLIENT_PREALLOCATE="false"),
          "cap8": dict(XLA_PYTHON_CLIENT_MEM_FRACTION=f"{CAP_GIB * 1024 / total_mib:.4f}")}
  for name, env in runs.items():
    cmd = ["python", "/opt/faster-weathernext/cloud/canary_run.py", "--weights-dir", "/data",
           "--data", f"/data/{os.path.basename(DATA)}", "--json", f"{out}/{card}_{name}.json",
           "--steps", str(steps), "--label", f"{card}_{name}"]
    if name == "uncapped":
      cmd += ["--npz", f"{out}/{card}.npz"]
    t0 = time.time()
    proc = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    print(f"[{card}/{name}] exit {proc.returncode}\n" + "\n".join(
        l for l in (proc.stdout + proc.stderr).splitlines() if "step" in l or "peak" in l or "Error" in l)[-2000:],
          flush=True)
    path = f"{out}/{card}_{name}.json"
    summary[name] = json.load(open(path)) if os.path.exists(path) else {"error": f"exit {proc.returncode}"}
    summary[name].pop("traceback", None)
    summary[name]["wall_s"] = time.time() - t0
    vol.commit()
  return summary


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def card_t4(tag: str, steps: int):
  return _card("t4", tag, steps)


@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def card_l4(tag: str, steps: int):
  return _card("l4", tag, steps)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def card_a10g(tag: str, steps: int):
  return _card("a10g", tag, steps)


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def card_a100(tag: str, steps: int):
  return _card("a100", tag, steps)


CARDS = {"t4": card_t4, "l4": card_l4, "a10g": card_a10g, "a100": card_a100}


@app.local_entrypoint()
def main(cards: str = "l4", tag: str = "canary01", steps: int = 4):
  import json
  fetch.remote()
  calls = {c: CARDS[c].spawn(tag, steps) for c in cards.split(",")}
  for c, call in calls.items():
    r = call.get()
    print(f"== {c}: " + json.dumps({k: (v if not isinstance(v, dict) else {
        kk: vv for kk, vv in v.items() if kk in ("device", "attention", "pool_gib", "peak_gib", "step_s", "ok", "error")})
        for k, v in r.items()}), flush=True)
