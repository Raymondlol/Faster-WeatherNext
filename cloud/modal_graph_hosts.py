"""Which grid->mesh graph does each Modal GPU host build? (host CPU, NumPy SIMD dispatch, edges)

Runs scripts/graph_edges.py (CPU only: traces one step, compiles nothing) on the host of each
GPU type and records the CPU model, the CPU flags the container sees and NumPy's dispatch
targets next to the edge count and hash. A host whose graph differs gives different fp32
outputs on any GPU (docs/validation.md §4, A100 note).

  modal run --detach cloud/modal_graph_hosts.py --cards a100,a100_80,a10g,l4,t4 --tag graphhosts01
  modal volume get fwn-canary results/graphhosts01 ./graphhosts01
  modal app stop -y <app id>

Cost (2026 prices): about $0.05-0.15 per card (~3 min each).
"""
import os
import subprocess

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc"

app = modal.App("fwn-graph-hosts")
vol = modal.Volume.from_name("fwn-canary", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(REPO, "/opt/faster-weathernext", copy=True,
                   ignore=[".git", "**/__pycache__", "docs/results", "*.nc", "*.npz"])
    .run_commands("pip install '/opt/faster-weathernext[cuda12,weathernext]'")
)


def _graph(card: str, tag: str) -> dict:
  import json
  out = f"/data/results/{tag}"
  os.makedirs(out, exist_ok=True)
  gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
  flags = next((l.split(":", 1)[1].split() for l in open("/proc/cpuinfo") if l.startswith("flags")), [])
  path = f"{out}/{card}_graph.json"
  proc = subprocess.run(["python", "/opt/faster-weathernext/scripts/graph_edges.py", "--weights-dir", "/data",
                         "--data", f"/data/{DATA}", "--json", path],
                        env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True)
  res = json.load(open(path)) if os.path.exists(path) else {"error": (proc.stdout + proc.stderr)[-3000:]}
  res.update(card=card, gpu=gpu, nproc=os.cpu_count(),
             cpu_flags={f: f in flags for f in ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512bw", "avx512vl")})
  with open(path, "w") as f:
    json.dump(res, f, indent=1)
  vol.commit()
  print(f"[{card}] {gpu} | {res.get('cpu')} | flags {res['cpu_flags']} | numpy dispatch {res.get('dispatch')} | "
        + "; ".join(f"{g['connectivity']} {g['n_edges']} {g['sha1']}" for g in res.get("graphs", [])), flush=True)
  return res


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol}, timeout=1200, cpu=8, memory=32768)
def g_a100(tag: str):
  return _graph("a100", tag)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": vol}, timeout=1200, cpu=8, memory=32768)
def g_a100_80(tag: str):
  return _graph("a100_80", tag)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1200, cpu=8, memory=32768)
def g_a10g(tag: str):
  return _graph("a10g", tag)


@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=1200, cpu=8, memory=32768)
def g_l4(tag: str):
  return _graph("l4", tag)


@app.function(image=image, gpu="T4", volumes={"/data": vol}, timeout=1200, cpu=8, memory=32768)
def g_t4(tag: str):
  return _graph("t4", tag)


CARDS = {"a100": g_a100, "a100_80": g_a100_80, "a10g": g_a10g, "l4": g_l4, "t4": g_t4}


@app.local_entrypoint()
def main(cards: str = "a100,a100_80,a10g,l4,t4", tag: str = "graphhosts01"):
  calls = {c: CARDS[c].spawn(tag) for c in cards.split(",")}
  for c, call in calls.items():
    call.get()
