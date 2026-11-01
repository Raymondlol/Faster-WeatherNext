"""A100 fp32 probe: why does the canary's "highest" step land at TF32 level on A100-SXM4-40GB?

Runs the canary's fp32 step (cloud/canary_run.py --steps 0) in fresh processes with different
XLA settings on an A100-40GB and an A100-80GB. The base run on each card also dumps the autotune
picks, the HLO, the thunk sequence and XLA's device description. Compare each .npz with an
RTX 5090 run of the same code (scripts/compare_npz.py). The "graph" step saves the host's
grid<->mesh graphs and "edges5090" runs the step on the RTX 5090's graphs instead (the cause:
see docs/validation.md §4, A100 note).

  modal run --detach cloud/modal_a100_tf32.py --tag tf32probe01
  modal run --detach cloud/modal_a100_tf32.py --tag tf32probe02 --v40 graph,edges5090 --v80 ""
  modal run --detach cloud/modal_a100_tf32.py --tag tf32probe03 --v40 "" --v80 numpy_avx2
  modal volume get fwn-canary results/tf32probe01 ./tf32probe01
  modal app stop -y <app id>      # detached apps can linger after finishing

Cost (2026 prices): about $1 (A100-40GB ~18 min at ~$2.10/h, A100-80GB ~6 min at ~$2.50/h).
"""
import os
import subprocess
import time

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc"

app = modal.App("fwn-a100-tf32")
vol = modal.Volume.from_name("fwn-canary", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(REPO, "/opt/faster-weathernext", copy=True,
                   ignore=[".git", "**/__pycache__", "docs/results", "*.nc", "*.npz"])
    .run_commands("pip install '/opt/faster-weathernext[cuda12,weathernext]'")
)

# name -> (extra environment, dump autotune picks + HLO, extra canary_run.py arguments)
VARIANTS = {
    "base": ({}, True, []),
    "at0": ({"XLA_FLAGS": "--xla_gpu_autotune_level=0"}, False, []),
    "tf32off": ({"NVIDIA_TF32_OVERRIDE": "0"}, False, []),
    "notritongemm": ({"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}, False, []),
    # The RTX 5090's grid<->mesh graphs (scripts/graph_edges.py --save-edges, uploaded with
    # `modal volume put fwn-canary edges_rtx5090.npz edges/edges_rtx5090.npz`).
    "edges5090": ({}, False, ["--edges", "/data/edges/edges_rtx5090.npz"]),
    # NumPy's AVX2 code path on an AVX-512 host: same host-side graph and features as the
    # A100-40GB hosts (AVX2 only), so the GPUs are compared on identical inputs.
    "numpy_avx2": ({"NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}, False, []),
}


def _run(card: str, tag: str, names: list[str]) -> dict:
  import json
  out = f"/data/results/{tag}"
  os.makedirs(out, exist_ok=True)
  smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,ecc.mode.current",
                        "--format=csv"], capture_output=True, text=True).stdout
  summary = {"card": card, "nvidia_smi": smi,
             "env": {k: v for k, v in os.environ.items() if any(s in k for s in ("TF32", "XLA", "JAX", "CUBLAS", "CUDNN"))}}
  print(f"[{card}] {smi}", flush=True)
  for name in names:
    if name == "graph":  # this host's graphs, saved for reuse elsewhere (CPU only)
      subprocess.run(["python", "/opt/faster-weathernext/scripts/graph_edges.py", "--weights-dir", "/data",
                      "--data", f"/data/{DATA}", "--json", f"{out}/{card}_graph.json",
                      "--save-edges", f"{out}/{card}_edges.npz"], env={**os.environ, "JAX_PLATFORMS": "cpu"})
      summary[name] = json.load(open(f"{out}/{card}_graph.json"))
      print(f"[{card}/{name}] {summary[name]['cpu']} {summary[name]['dispatch']} "
            + "; ".join(f"{g['connectivity']} {g['n_edges']} {g['sha1']}" for g in summary[name]["graphs"]), flush=True)
      vol.commit()
      continue
    extra, dump, extra_args = VARIANTS[name]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", **extra}
    if dump:
      d = f"/tmp/dump_{card}_{name}"
      env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + f" --xla_dump_to={d} --xla_dump_hlo_as_text"
                          " --xla_dump_hlo_module_re=.*lambda.*"
                          f" --xla_gpu_dump_autotune_results_to={out}/{card}_{name}_autotune.txt").strip()
    cmd = ["python", "/opt/faster-weathernext/cloud/canary_run.py", "--weights-dir", "/data",
           "--data", f"/data/{DATA}", "--json", f"{out}/{card}_{name}.json", "--npz", f"{out}/{card}_{name}.npz",
           "--steps", "0", "--label", f"{card}_{name}", *extra_args]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(f"{out}/{card}_{name}.log", "w") as f:
      f.write(proc.stdout + proc.stderr)
    if dump:
      # HLO before/after optimization, thunk sequence (which op runs as which kernel or library
      # call), device description and autotune picks; not the large buffer-assignment files.
      keep = r"optimizations.txt$|thunk_sequence.txt$|gpu_target_config.pbtxt$|autotune_results.pbtxt$|memory-usage-report.txt$"
      subprocess.run(f"cd {d} && tar czf {out}/{card}_{name}_dump.tgz $(ls | grep -E '{keep}')", shell=True, check=False)
    path = f"{out}/{card}_{name}.json"
    summary[name] = json.load(open(path)) if os.path.exists(path) else {"error": f"exit {proc.returncode}"}
    summary[name].pop("traceback", None)
    summary[name]["wall_s"] = time.time() - t0
    print(f"[{card}/{name}] exit {proc.returncode} device {summary[name].get('device')} "
          f"{summary[name]['wall_s']:.0f}s {summary[name].get('error', '')}", flush=True)
    vol.commit()
  with open(f"{out}/{card}_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
  vol.commit()
  return summary


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def a100_40(tag: str, names: list[str]):
  return _run("a100_40", tag, names)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": vol}, timeout=3600, cpu=8, memory=32768)
def a100_80(tag: str, names: list[str]):
  return _run("a100_80", tag, names)


@app.local_entrypoint()
def main(tag: str = "tf32probe01", v40: str = "base,at0,tf32off,notritongemm", v80: str = "base"):
  calls = []
  if v40:
    calls.append(a100_40.spawn(tag, v40.split(",")))
  if v80:
    calls.append(a100_80.spawn(tag, v80.split(",")))
  for call in calls:
    r = call.get()
    print(f"== {r['card']}: " + ", ".join(f"{k}: {v.get('device')} ok={v.get('ok')} {v.get('error', '')}"
                                         for k, v in r.items() if isinstance(v, dict) and k != "env"), flush=True)
