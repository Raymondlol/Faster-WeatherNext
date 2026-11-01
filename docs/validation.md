# Validation

All numbers below come from logged runs; the raw outputs are in [`results/`](results/).
"Relative RMS" is RMS(difference) / std(field), per output field (variable × pressure level,
101 fields for WeatherNextCyclones), reported as the worst (max) and median over fields.
"Reference" is defined per section.

Precisions: `default` = JAX's default matmul precision on GPU (TF32 tensor cores, 10-bit
mantissa); `highest` = `jax.default_matmul_precision("highest")` (fp32-level; the Pallas kernel
uses 3xTF32).

## 1. Chunked attention vs the official GPU path (P0, A100-80GB)

The official GPU attention (`triblockdiag_mha`) needs ~34 GiB for one 0.25° step, so this
comparison ran on an A100-80GB (Modal), with both implementations in the same process, same
weights (WeatherNextCyclones `<2025` model 1), inputs (Google's 2024-10-07 sample) and RNG.
Only the attention differs; everything else is the official code.

| precision | min corr | max rel RMS | NaN masks | temp memory (official → chunked) |
|---|---|---|---|---|
| highest | 0.99999999986 | 1.7e-5 | equal | 33.0 → 20.8 GiB |
| default | 0.99998885 | 4.7e-3 | equal | |

Rollout (default precision, 4 steps): implementation difference (median rel RMS) 5.2e-4 at
+6 h → 2.7e-3 at +24 h; difference between two ensemble members 0.160 → 0.240, i.e. the
implementation difference is **311× → 88× smaller** than the member spread.
Files: [`results/p0/`](results/p0/).

This chunked-attention path is the **reference** for everything below (it fits a 32 GB GPU).

## 2. faster-weathernext vs reference, same GPU (RTX 5090)

One 0.25° step, WeatherNextCyclones model 1, Google sample data; reference = §1 path
(XLA chunked attention, official GNN / encoder / decoder, autotune off).

| stage | precision | min corr | max rel RMS | temp memory | s/step |
|---|---|---|---|---|---|
| reference | default | — | — | 20.8 GiB | 5.47 |
| reference | highest | — | — | 20.8 GiB | 8.94 |
| + blocked GNN / encoder / decoder, layer loop (autotune on) | default | 0.999986 | 5.3e-3 | 4.9 GiB | 2.80 |
| | highest | 0.9999999998 | 1.9e-5 | 4.8 GiB | 3.97 |
| + Pallas attention | default | 0.999987 | 5.0e-3 | 4.0 GiB | 1.00 |
| | highest | 0.9999999998 | 2.2e-5 | 4.1 GiB | 2.34 |
| **+ Hilbert reordering (current)** | **default** | **0.999988** | **4.9e-3** | **3.8 GiB** | **0.78** |
| | **highest** | **0.9999999997** | **2.5e-5** | **4.1 GiB** | **1.64** |
| **faster-weathernext 0.1.0 (release candidate, `scripts/verify_equivalence.py`)** | default | 0.999989 | 4.8e-3 | 3.8 GiB | 0.78 |
| | highest | 0.9999999998 | 2.0e-5 | 4.1 GiB | 1.50 |

All NaN masks equal. The worst field is always `vertical_velocity@50` (tiny variance).
Step times at `highest` vary by ~10% between runs (XLA autotuning picks differ).
The TF32 differences are the same size as the difference between the official GPU path and the
reference at TF32 (§1: 4.7e-3). Files: [`results/same_card/`](results/same_card/), and the packaged code:
[`results/equivalence_rtx5090.json`](results/equivalence_rtx5090.json)
(`scripts/verify_equivalence.py`).

## 3. Nor'easter hindcasts: rollouts with the real WeatherNext 2 checkpoints

A September 2026 US nor'easter, initialised from ECMWF IFS open-data analyses at 2026-09-22,
23, 24 and 25 00Z, run to 09-28 06Z (up to 150 h), with all four WeatherNext 2 checkpoints ×
2 noise samples. Same inputs and RNG seeds for both runs; old = §1 reference code (P0),
new = current faster_weathernext. Verification against IFS analyses.

**Implementation difference vs member spread** (median over inits; member spread = RMS
difference between two members):

| lead | MSLP | 10 m wind | 2 m temperature | Z500 |
|---|---|---|---|---|
| 6 h | 181× smaller | 142× | 179× | 189× |
| 24 h | 59× | 32× | 39× | 57× |
| 72 h | 12× | 4.8× | 6.3× | 17× |
| 150 h | 7.3× | 2.4× | 2.7× | 7.8× |

The difference grows with lead time because the atmosphere is chaotic (any rounding difference
grows), but it stays below the spread between ensemble members.

**Skill vs IFS analyses** (ensemble-mean RMSE, mean over leads):

| init | MSLP old / new (hPa) | 10 m wind old / new (m/s) |
|---|---|---|
| 09-22 00Z | 1.181 / 1.191 | 1.328 / 1.332 |
| 09-23 00Z | 1.074 / 1.075 | 1.269 / 1.272 |
| 09-24 00Z | 0.803 / 0.805 | 1.153 / 1.155 |
| 09-25 00Z | 0.573 / 0.573 | 1.031 / 1.030 |

Old vs new differ by −0.1 … +0.8 %. For scale, two 4-member halves of the same ensemble
(different noise samples) differ by a median of 2.5 % (up to 12 %).
Storm track (ensemble-mean low vs analysed low, median over leads): 188 / 193, 118 / 126,
90 / 90, 56 / 56 km (old / new).
Files: [`results/noreaster/`](results/noreaster/), [`figures/noreaster_p2_vs_p0.png`](figures/noreaster_p2_vs_p0.png).

## 4. Other GPUs (Modal, vast.ai)

Current code, WeatherNextCyclones model 1, Google sample data. "s/step" is the steady-state
rollout step including host overhead (steps 2–4 of a 4-step rollout, default precision).
"8 GB pool" = JAX memory pool capped at 7.0 GiB, i.e. what an 8 GB card gets at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.

| GPU | arch | attention | s/step | peak in pool | fits 8 GB pool | fp32 vs RTX 5090 (max rel RMS) |
|---|---|---|---|---|---|---|
| RTX 5090 (local) | sm_120 | pallas | 0.7 | 6.40 GiB | yes | — |
| RTX 4060 (vast.ai) | sm_89 | pallas | 5.2 | 6.21 GiB | yes (real 8 GB card) | not measured (sm_89 covered by L4) |
| A100-SXM4-40GB | sm_80 | pallas | 1.1 | 6.18 GiB | yes (6.21) | 1.32e-2: its host builds another graph; 1.7e-5 vs an A100-80GB given the same host inputs (see note) |
| A10G | sm_86 | pallas | 2.5 | 6.53 GiB | yes (6.34) | 1.6e-5 |
| L4 | sm_89 | pallas | 3.5 | 6.53 GiB | yes (6.34) | 1.75e-5 |
| T4 | sm_75 | xla (automatic; no TF32) | 44 | 6.49 GiB | yes (6.30) | 9.7e-5 |

Files: [`results/canary_modal/`](results/canary_modal/) (incl. `cross_card_fp32.txt`).

**Real 8 GB card (RTX 4060, vast.ai), faster-weathernext 0.1.0** (Pallas attention): 5.1–5.3 s/step
with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (in-pool peak 6.21 of 6.86 GiB; whole-card peak 7320 of
8188 MiB). With JAX's default 0.75 (5.72 GiB pool) the first step completes and the second runs
out of memory. Installed from the repo on a fresh machine. Files:
[`results/rtx4060_v0.1.0/`](results/rtx4060_v0.1.0/).
An earlier build without the fused attention kernel ran at 18.5 s/step on the same card
([`results/canary_vast_rtx4060/`](results/canary_vast_rtx4060/)).

**A100 note: the host CPU, not the GPU (cause found 2026-11-01).** On Modal's A100-SXM4-40GB the
fp32 step differs from the RTX 5090 by max rel RMS 1.32e-2 (median 6.21e-4; worst fields
`cyclone_usa_rmw_disc`, `vertical_velocity@50`) in every run (canary02 on v0.1.0, canary03 and
tf32probe01 on v0.2; the three agree with each other at 1.5–2.1e-5). It looks like TF32 but is not.
Earlier it was read as "1 of 6 runs, an autotune pick that did not honour fp32"; that was wrong:
Modal served the five "A100-40GB" probe and repeat requests on A100-SXM4-80GB machines (`device`
in [`results/a100_repeats/`](results/a100_repeats/)), so every 40GB run deviated and every 80GB run
matched (1.6–1.9e-5).

The official code builds the grid→mesh and mesh→grid graphs on the host CPU, with NumPy, from
float32 latitudes and longitudes: the mesh nodes' lat/lon come from Cartesian positions through
`np.arctan2`/`np.arccos`, then grid and mesh go back to unit-sphere coordinates through
`np.sin`/`np.cos`, and a `cKDTree` ball query (radius 0.6 × the longest mesh edge, 0.0124) picks the
grid→mesh edges; mesh→grid uses the mesh triangle that contains each grid point. One float32 ULP
of a coordinate moves a 0.0124 distance by ~1e-5 (relative), and about 70 (grid point, mesh node)
pairs lie that close to the radius. NumPy picks its math code at run time from the host CPU, and
the results differ in the last bit: between its AVX-512 and AVX2 code, 12,454 of the 40,962 mesh
latitudes (and 9,964 longitudes) come out different; its baseline code (no AVX2) also hands
`sin`/`cos` to glibc ([`numpy_code_paths.txt`](results/a100_host_graph/local_rtx5090/numpy_code_paths.txt)).
Modal's A100-40GB hosts offer AVX2 but not AVX-512; the RTX 5090 desktop (Zen 5) and Modal's
A100-80GB and T4 hosts offer AVX-512. The two build different graphs: 1,618,832 vs 1,618,841
grid→mesh edges (13 differ), and 517 of 1,038,240 grid points in a different mesh triangle (points
on mesh edges along the meridians 0°, 60°, …, 300°); the edge-feature and node-position constants
also differ in the last bits.
faster-weathernext does not touch this (the patched `_build_typed_graph` calls the stock one,
which calls the official `PointsMeshTypedGraphGNN._get_edge_indices`), so the official code depends
on the host in the same way. Each graph is what the official code builds on that CPU.

Evidence (one 0.25° step at `highest`; files in [`results/a100_host_graph/`](results/a100_host_graph/),
comparisons in its `cross_card_fp32.txt`):
- The HLO that JAX hands to XLA already differs between a 40GB and an 80GB run, in one place: the
  grid→mesh edge-feature constant is `f32[1618832,1,4]` vs `f32[1618841,1,4]`
  (`tf32probe01/hlo_before_optimizations_40gb_vs_80gb.diff`).
- XLA and TF32 settings change nothing on the 40GB card: autotuning off, `NVIDIA_TF32_OVERRIDE=0`
  and Triton GEMMs off all give 1.32e-2 / 6.21e-4 (`cloud/modal_a100_tf32.py`).
- The difference is local. It sits on the 13 changed edges and the 517 re-assigned grid points
  ([map](results/a100_host_graph/errmap_a100_40gb_vs_rtx5090.png),
  [`differing_edges.txt`](results/a100_host_graph/differing_edges.txt)); elsewhere the two cards
  agree at ~1e-6.
- Graph swap (`scripts/graph_edges.py --save-edges`, `cloud/canary_run.py --edges`): the RTX 5090
  on the 40GB host's graph differs from the RTX 5090 on its own graph by 1.32e-2 / 6.21e-4 (same
  worst fields), and so does the A100-40GB on the RTX 5090's graph from the A100-40GB on its own.
  With the graph swapped, the two cards are 1.3e-4 / 1e-5 apart. The remainder is the host-side
  features: on the RTX 5090, with the graph held fixed, switching only NumPy's code path from
  AVX-512 to AVX2 moves the output by the same 1.33e-4 / 1.0e-5.
- One card, one environment variable: on an A100 80GB PCIe (AVX-512 host), forcing NumPy's AVX2
  code (`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`) builds exactly the 40GB host's
  graph (same hash) and moves the output by 1.32e-2 / 6.21e-4 from the same card's default run.
  That output then matches the A100-40GB at 1.69e-5 / 1.47e-6, the usual cross-GPU level
  (tf32probe03).
- Host survey (`cloud/modal_graph_hosts.py`): the T4 and A100-80GB hosts build the RTX 5090's graph
  (same hash); an A10G host with AVX2 only builds the 40GB host's graph. The canary's A10G
  requests ran on `NVIDIA A10` cards (another host type; its CPU was not recorded) and matched.

So the fp32 agreement across GPUs holds for the same host-built graph. On machines whose CPUs take
different NumPy code paths the official model runs on slightly different graphs, and one fp32 step
then differs by up to ~1e-2 max rel RMS (median 6e-4), around the changed edges. `cloud/canary_run.py`
now records the graph hashes and NumPy's SIMD targets: compare fp32 outputs across machines only
when the hashes match, or pass `--edges`. `--no-autotune` is not needed for fp32 (it changed nothing
here). XLA's autotune picks still vary between runs (21 of 149 ops across three A100 repeats), so
use `--autotune-cache` for bitwise-reproducible runs.

**Canary after the attention backward (2026-11-01, branch `head-only-finetune`).** Same runner and
cards as the table above (`modal run cloud/modal_canary.py --cards a100,a10g,l4,t4`, ~$1–2): every
card ran and fits the 8 GB pool; steady-state s/step and peaks as in the table (A100 1.10 s / 6.18
GiB, A10G 2.43 / 6.53, L4 3.57 / 6.53, T4 43 / 6.49 on the XLA path). fp32 outputs vs a fresh
RTX 5090 run of the same code: A10G 1.67e-5, L4 1.85e-5, T4 9.67e-5 max rel RMS; A100 1.32e-2 (a
different host-built graph, not the GPU: note above). Files: [`results/canary_modal/v0.2/`](results/canary_modal/v0.2/).

## 5. Mini model and batch size > 1 (v0.1.1)

`scripts/verify_batch.py`, RTX 5090. The FGN noise is drawn inside the model once per forward,
so element *i* of a batch never sees the same noise as a batch-1 run; the script swaps the noise
generator (for every path alike) for one that inserts fixed rows of a deterministic table, which
makes elements comparable. Two different initialisations per batch.

**WeatherNextCyclones_Mini (1°)**: the official GPU path (`triblockdiag_mha`, patch disabled) vs
faster-weathernext, Google's 1° sample (inits 2024-10-07 00Z and 06Z), all 84 output fields
× levels. The official path fits, so this is a direct comparison at both batch sizes:

| precision | comparison | min corr | max rel RMS | median rel RMS |
|---|---|---|---|---|
| highest | fwn vs official, batch 1 (two inits) | 0.9999999984 | 5.7e-5, 2.0e-5 | 3.0e-6 |
| highest | fwn vs official, batch 2 | 0.9999999998 | 2.2e-5 | 2.9e-6 |
| highest | official batch 2 element *i* vs official batch 1 | 0.9999999998 | 2.1e-5, 1.8e-5 | 2.3e-6 |
| highest | fwn batch 2 element *i* vs fwn batch 1 | 0.9999999993 | 3.8e-5, 1.0e-5 | 1.6e-6 |
| default | all of the above | 0.99989–0.99998 | 0.6–1.5e-2 | 0.9–1.5e-3 |

At fp32 the patch is within the official path's own batch-1-vs-batch-2 rounding; at TF32 every
pair, including official-vs-official, differs at the 1e-2 level (Mini's vertical velocity is the
worst field throughout). Temp memory / step, `highest`: official 2.65 GiB / 0.16 s (batch 1),
3.21 GiB / 0.31 s (batch 2); faster-weathernext 0.61 GiB / 0.10 s and 1.28 GiB / 0.18 s.
Files: [`results/batch/mini_rtx5090.json`](results/batch/mini_rtx5090.json).

**WeatherNext2 (0.25°)**: the official path at batch 2 needs ~68 GiB, so faster-weathernext is
compared with itself, element by element (IFS inits 2026-09-25 00Z and 09-24 00Z):

| precision | comparison | min corr | max rel RMS | median rel RMS |
|---|---|---|---|---|
| highest | batch 2 element 0 vs batch 1 (init 09-25) | 0.9999999999 | 1.3e-5 | 1.9e-6 |
| highest | batch 2 element 1 vs batch 1 (init 09-24) | 0.9999999999 | 1.5e-5 | 2.3e-6 |
| default | batch 2 element 0 vs batch 1 | 0.99976 | 2.2e-2 | 3.0e-3 |
| default | batch 2 element 1 vs batch 1 | 0.99988 | 1.5e-2 | 2.3e-3 |

The TF32 rows are batch-size sensitivity of TF32 kernels, not of the patch: on Mini the official
path differs from its own batch-2 run by 1.3–1.5e-2 (table above); the worst fields are again
vertical velocity. Temp memory / step: batch 1 4.14 GiB / 1.50 s (`highest`), 3.79 GiB / 0.78 s
(default); batch 2 8.28 GiB / 3.40 s and 8.28 GiB / 2.22 s. So on an RTX 5090 batch 2 buys no
throughput (2.22 s for two members vs 2 × 0.78 s); it is a convenience for callers that batch
ensemble members. In 0.1.0, batch 2 needed 13.24 GiB temp: the blocked encoder's output was
stacked in grid layout and reordered afterwards, free at batch 1 but a second whole-grid copy at
batch 2 ([`results/batch/wn2_batch2_memory_fix.txt`](results/batch/wn2_batch2_memory_fix.txt)).
Files: [`results/batch/wn2_rtx5090.json`](results/batch/wn2_rtx5090.json). After that change,
the batch-1 equivalence of §2 was re-run: `highest` 2.0e-5, default 4.9e-3 max rel RMS
([`results/equivalence_rtx5090_v0.1.1.json`](results/equivalence_rtx5090_v0.1.1.json)).

## 6. Inside earth2studio

`scripts/earth2studio_check.py`: NVIDIA earth2studio's `WeatherNext2CyclonesMini` /
`WeatherNext2Cyclones` wrappers (earth2studio main at 89be5bc, its own `weathernext` pin
9c034db, JAX 0.11.2), fed Google's sample in the wrapper's tensor layout, stock vs
`faster_weathernext.enable()` called before `load_model` (separate processes, same seed).
faster-weathernext 0.1.1 installs into that environment without changing JAX or `weathernext`
(0.1.0 could not: its `weathernext` git URL conflicted with earth2studio's pin).

| wrapper | batch | precision | min corr | max rel RMS | median rel RMS | JAX peak stock → patched | step stock → patched |
|---|---|---|---|---|---|---|---|
| WeatherNext2CyclonesMini | 1 | highest | 0.9999999997 | 2.4e-5 | 2.0e-6 | 1.60 → 0.83 GiB | 0.26 → 0.19 s |
| WeatherNext2CyclonesMini | 1 | default (TF32) | 0.99996 | 9.3e-3 | 9.4e-4 | 1.57 → 0.75 GiB | 0.23 → 0.16 s |
| WeatherNext2CyclonesMini | 2 | default (TF32) | 0.99996 | 9.3e-3 | 9.0e-4 | 3.30 → 1.44 GiB | 0.39 → 0.25 s |
| WeatherNext2Cyclones (0.25°) | 1 | default (TF32) | stock needs ~34 GiB: not runnable on this 32 GB GPU | — → 5.22 GiB | — → 1.52 s |

The 0.25° wrapper run with the patch produced the full (84 variables, 721 × 1440) output with the
same NaN pattern as the Mini runs (SST over land, 0.40 %); its numerics are covered by §1–§2 and
§5 (same modules, same code path as the wrapper: `fgn.construct_predictor` with the ensemble
wrapper dropped).

"step" is the wrapper's whole `model(x, coords)` call (xarray conversion and TISR included), so
Mini times are dominated by host work. `highest` was set with `JAX_DEFAULT_MATMUL_PRECISION=highest`
for both processes. Files: [`results/earth2studio/`](results/earth2studio/).

## 7. Intel XPU / Arc GPUs (OpenXLA via oneAPI)

Intel GPUs (such as Arc B-series / Battlemage) run via Intel's OpenXLA PJRT plugin (`jax-oneapi-plugin`). Because Intel GPUs do not use NVIDIA CUDA / Triton Pallas kernels, `faster-weathernext` automatically dispatches to the pure XLA chunked attention implementation (`attention_impl() == "xla"`). A full 0.25° rollout step on Intel is currently a TODO.

### Installation

Requires the Intel compute runtime / Level-Zero and the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (source `oneapi-vars.sh` or `setvars.sh` to initialize the runtime environment).

Install `faster-weathernext` with the `[oneapi]` extra:
```bash
pip install -e ".[oneapi,weathernext]"
```
(installs `jax-oneapi-plugin==0.11.2` and `jax-oneapi-pjrt==0.11.2`).

### Device Check & Microbenchmark

Running `fwn info` on an Intel Arc B580 (12 GB GDDR6):
```
Platform 'oneapi' is experimental and not all JAX functionality may be correctly supported!
faster-weathernext 0.1.1 | jax 0.11.2 | weathernext 0.3.1.dev0 (f2f2c51) (validated at f2f2c51) | device Intel(R) Arc(TM) B580 Graphics (compute capability unknown)
JAX memory pool: 10.20 GiB (XLA_PYTHON_CLIENT_MEM_FRACTION=0.9); a 0.25° step needs ~6.4 GiB
weathernext compatibility: OK
attention implementation: xla  (options: Options(attention='auto', replace_attention=True, reorder=True, strict_fp32=False, attn_chunk=512, grid_chunk=32768, edge_chunk=65536, layer_loop=True))
```

One WN2 layer's masked mesh attention (40,962 nodes, 6 heads × 128 dim, 32-hop mask, `scripts/attn_bench.py`):

| GPU | arch | memory | attention | precision | ms/layer | attention only, ×24 |
|---|---|---|---|---|---|---|
| Intel Arc B580 (contributed) | Xe2 (Battlemage) | 12 GB (10.2 GiB pool) | xla | default | 684.2 ms | 16.42 s |

Files: [`results/intel_b580/`](results/intel_b580/) (`fwn_info.txt`, `attn_bench.txt`). All unit tests pass (`pytest -q`).

## 8. Head-only fine-tuning (prototype)

`faster_weathernext.finetune` trains the head (mesh→grid GNN + grid decoder) with the official
training loss: the config's full predictor stack and FGN's fair CRPS over 2 members, i.e. an
effective batch of 2. Gradients are taken w.r.t. the head only, so the frozen encoder, grid→mesh
GNN and transformer run as in inference. The new option `remat` rematerialises each blocked
GNN / encoder / decoder step in the backward pass. Results here: RTX 5090, JAX 0.11.2.

**Inference is unchanged by the patch edit** (`remat` defaults to off; `scripts/verify_equivalence.py`
re-run as in §2): `highest` max rel RMS 1.98e-5, default 4.87e-3 (v0.1.1: 2.00e-5, 4.93e-3); temp
memory and step time unchanged (4.14 / 4.02 GiB, 1.49 / 0.77 s).
Files: [`results/finetune/equivalence_rtx5090_remat_option.json`](results/finetune/equivalence_rtx5090_remat_option.json).

**Head gradients vs the official GPU path** (`scripts/verify_grads.py`; the runs in §8 and §9 were made
when it and `scripts/finetune.py` were called `verify_head_grads.py` / `finetune_head.py`; WeatherNextCyclones_Mini
at 1°, Google's sample, same inputs / targets / noise). faster-weathernext runs with remat and
small blocks (8,192 grid points, 16,384 edges), so the 1° grid passes through 8 blocks in every
blocked module. The fair CRPS has sign terms (|pred − target|, |pred_i − pred_j|): rounding-level
forward differences flip individual gradient contributions, so gradient agreement is judged
against a floor, a second official implementation (dense `mha` instead of `triblockdiag_mha`),
not against zero. Head: 2.54M parameters in 86 tensors; per-tensor errors are over the 58 tensors
whose gradient norm exceeds 1e-3 of the largest (the others enter the global numbers).

| precision | comparison | loss rel diff | grad global rel L2 | cosine | max per-tensor rel L2 |
|---|---|---|---|---|---|
| highest | fwn vs official | 9.9e-8 | 1.00e-3 | 0.99999950 | 2.3e-3 |
| highest | fwn (`strict_fp32` attention) vs official | 0 | 8.9e-4 | 0.99999962 | 1.4e-3 |
| highest | official `mha` vs official (floor) | 0 | 5.7e-4 | 0.99999984 | 2.4e-3 |
| default | fwn vs official | 1.1e-5 | 1.62e-2 | 0.99987 | 0.21 |
| default | official `mha` vs official (floor) | 1.4e-5 | 1.58e-2 | 0.99988 | 0.21 |

At fp32 the global error is within 2× of the floor (which itself moved from 4.3e-4 to 5.7e-4
between two runs: XLA autotuning) and the worst tensor is no worse than the floor's; at TF32 the
two rows are the same. The worst tensors are output layers of individual variables (10 m wind,
upper-level u; at TF32 the cyclone channels, whose loss terms are tiny). All gradients finite.
Temp memory per step (Mini, effective batch 2): official 7.36 GiB (`highest`), faster-weathernext
1.48 GiB with remat and 9.91 GiB without; 0.22 s vs 0.42 s per step.
Files: [`results/finetune/head_grads_mini_rtx5090.json`](results/finetune/head_grads_mini_rtx5090.json) (+ `.log`).

**WeatherNext 2 at 0.25°** (`scripts/finetune.py`, IFS open data). Compiled training step:
20.03 GiB temp + 1.79 GiB arguments (default precision). Rematerialisation is what makes this
fit: compiled for the CPU backend (whose buffer assignment is larger; the loss forward alone is
21.1 GiB there vs 8.3 GiB on the GPU), the step needs 34.3 GiB temp with remat and 249.6 GiB
without ([`results/finetune/cpu_compile_*.json`](results/finetune/)). The step needs one
contiguous 20 GiB buffer: in a JAX pool that grows on demand, an evaluation pass run first left
regions too small for it and the step failed with RESOURCE_EXHAUSTED
([`results/finetune/finetune_smoke_rtx5090_oom_growing_pool.log`](results/finetune/finetune_smoke_rtx5090_oom_growing_pool.log));
`finetune.py` therefore preallocates 90 % of the card.

**Training runs** (checkpoint 1; 60 steps of one example each, 3.84 s per step, device peak
21.8 GiB; Adam, gradient-norm clip 1). Training examples: 60 of 68 IFS open-data init times
2026-10-04 12Z – 10-24 06Z, each used once; held out: 10-28 00Z/12Z and 10-30 00Z/12Z, a later
week. The held-out
loss is the training loss, evaluated before and after with the same two noise keys per example,
so both evaluations see the same ensemble noise. Targets: the IFS analysis at +6 h, and the 0–6 h
precipitation of the IFS run; the cyclone channels have no targets and are masked.

| run | held-out loss change (4 inits) | per variable (mean of the 4) |
|---|---|---|
| lr 3e-5, no warmup | +6.6 … +7.1 % | 2 m T −8.9 %, precip −11.2 %, SST −15.4 %; pressure-level T/u/v/q/w +12 … +17 % |
| lr 3e-5, no warmup, SST masked | +10.7 … +11.7 % | 2 m T −8.9 %, precip −11.4 %; pressure-level T/u/v/q/w +13 … +18 % |
| **lr 3e-6, 10-step warmup, SST masked** | **−1.5 … −1.6 %** | 2 m T −8.4 %, precip −4.5 %, T −2.6 %, z −2.1 %, MSLP −1.7 %; 13 of 14 variables lower, v wind ±0.0 % |

At 3e-5 the gradient norm grew from ~5 to ~13 within 50 steps and every pressure-level variable got
worse, while the variables where IFS open data differs most from the model's training data
(2 m temperature, precipitation, and SST, which open data replaces with skin temperature) improved
on held-out days. At 3e-6 with warmup the gradient norm fell (3.7 → 2.9) and the held-out loss
dropped on all four days. SST is masked in the last two runs because its open-data stand-in has a
diurnal cycle real SST does not; masking it is not what fixed the first run (the learning rate was).
The saved checkpoint (first run) loads with the stock `weathernext` loader and differs from the
base checkpoint only in head modules: 15 of the 18. The other 3 are the mesh→grid GNN's mesh-node
update (1.2M of the 5.6M head parameters), whose output the model discards, so they get no gradient.

What this shows: the pipeline runs at 0.25° on a 32 GB GPU and lowers a held-out one-step loss.
What it does not show: forecast skill. This is one checkpoint, 60 steps, one week of held-out
6-hour steps, and the loss of a 2-member sample; rollouts, ensemble spread and lead times beyond
6 h were not evaluated.
Files: [`results/finetune/finetune_smoke_rtx5090.json`](results/finetune/finetune_smoke_rtx5090.json),
[`finetune_masksst_rtx5090.json`](results/finetune/finetune_masksst_rtx5090.json),
[`finetune_lr3e-6_rtx5090.json`](results/finetune/finetune_lr3e-6_rtx5090.json) (+ `.log`; per-variable
losses before and after for each held-out init).

## 9. Attention backward and full fine-tuning

The fused attention kernel is differentiable (`pallas_attn.py`, `jax.custom_vjp`): the forward also
stores each row's log-sum-exp, and two tiled kernels compute dQ (per 64-query block, over the
forward's tile list) and dK/dV (per 64-key block, over the transposed mask's tile list, which for
the symmetric k-hop mask is the forward's). With `remat`, each transformer layer is
rematerialised, so the backward pass keeps one [nodes, 768] input per layer instead of q/k/v/output.
Without differentiation the forward kernel is the same as before. Inference re-checked as in §2 on
the final code: `highest` max rel RMS 1.97e-5; default 6.54e-3 in that run (worst field the sparse
`cyclone_exists_gaussian_unit_mode`; vertical velocity 4.9e-3 as usual) and 4.74e-3 on a repeat,
i.e. TF32 run-to-run variation from timing-based kernel selection (§4 note); temp memory and speed
unchanged ([`results/finetune/equivalence_rtx5090_attention_bwd.json`](results/finetune/equivalence_rtx5090_attention_bwd.json),
[`equivalence_rtx5090_attention_bwd_default_repeat.json`](results/finetune/equivalence_rtx5090_attention_bwd_default_repeat.json)).

**Kernel vs references.** `tests/test_pallas_kernel.py`: dQ, dK, dV against an analytic float64
masked-softmax gradient (n = 300, so padding rows are exercised; symmetric and non-symmetric masks;
batch 1 and 2; both `bwd_split` settings): the kernel test file passes on the RTX 5090 (21 tests,
forward and backward; 11 of them also run on CPUs in Pallas interpret mode). One layer of
WeatherNext 2's exact 32-hop mask (40,962 nodes, 6 heads × 128, Hilbert order;
`scripts/attn_bench.py --grad`) against JAX autodiff of the XLA attention path:

| precision | dQ rel RMS | dK rel RMS | dV rel RMS | output rel RMS | Pallas fwd / fwd+bwd | XLA fwd / fwd+bwd |
|---|---|---|---|---|---|---|
| highest | 1.1e-6 | 1.1e-6 | 8.8e-7 | 8.7e-7 | 31.0 / 115.4 ms | 128.4 / 258.5 ms |
| default (TF32) | 3.0e-4 | 2.9e-4 | 1.8e-5 | 2.8e-4 | 9.3 / 42.2 ms | 98.0 / 228.8 ms |

Gradients agree to the same level as the outputs. All attention kernels use one dot precision (the
backward recomputes P from the forward's log-sum-exp). Under `highest` that is 3xTF32, which keeps
hi/lo copies of each operand in shared memory: the backward kernels (3–4 dots per tile) asked for
147 KB of sm_120's 99 KB in 64-row programs, so they run in 32-row programs (`bwd_split=2`). TF32
keeps 64-row programs (43.7 ms with 32 rows); IEEE fp32 dots (`strict_fp32=True`) take 797 ms.
Files: [`results/attn_bwd/`](results/attn_bwd/). The kernel tests also pass on an H200 (sm_90; the
kernel's first run on Hopper): [`results/finetune/h200/grads01/pytest_h200.txt`](results/finetune/h200/grads01/pytest_h200.txt).

**Whole-model gradients vs the official path (Mini, 1°, all 56.7M parameters;
`verify_grads.py --modules all`).** The official path cannot be differentiated end to end on
a 32 GB GPU (compiled backward 82–116 GiB with the fair CRPS on the RTX 5090). On an H200 (141 GB;
`cloud/modal_grads.py`, Modal, ~$4 for the three runs below) it needs 102 GiB (`triblockdiag_mha`)
and 123 GiB (dense `mha`) with the CRPS, 51 and 62 GiB with a one-member loss; faster-weathernext
needs 3.97 and 3.30 GiB (39.1 / 19.7 GiB without remat). Reference: the official GPU path
(`triblockdiag_mha`); floor: the official dense `mha` path against it. Two losses: the fair CRPS,
whose sign terms let rounding flip individual contributions (§8), and `--loss linear`, a fixed
random linear functional of all outputs, which is smooth. Global relative L2 error of all gradients:

| loss | precision | fwn | fwn, `strict_fp32` attention | fwn, XLA attention (autodiff) | floor: official `mha` |
|---|---|---|---|---|---|
| linear | highest | 1.4e-4 | 9.0e-5 | 7.6e-5 | 8.1e-5 |
| linear | default | 4.4e-2 | — | 2.0e-2 | 4.5e-2 |
| CRPS | highest | 4.7e-3 | 1.2e-3 | 1.2e-3 | 9.5e-4 |
| CRPS | default | 5.2e-2 | — | 4.9e-2 | 5.5e-2 |

At the training default (TF32), faster-weathernext's whole-model gradients differ from the official
path's as much as the official path's two attention implementations differ from each other. At
`highest`, IEEE fp32 attention (`strict_fp32=True`) is at the floor; the default 3xTF32 attention,
whose forward differs slightly from fp32, is at 1.7× the floor with the smooth loss and 5× with the
CRPS (cosine 0.99998880), whose sign terms amplify forward rounding. The highest rows are from the
final code (run `grads03`); the default rows from `grads01` (that path has not changed since).
Comparisons fluctuate between runs (XLA autotuning): over the three H200 runs the floor ranged
7.5e-4–9.5e-4 (CRPS) and 7.6e-5–1.3e-4 (linear), and the XLA-attention row once reached 4.3e-3 with
the CRPS. In `grads01` the backward used IEEE dots after a 3xTF32 forward (fwn 2.4e-4 linear,
4.8e-3 CRPS); in `grads02` the training forward used IEEE dots too (6.8e-5, 1.4e-3, at the floor
but 797 instead of 115 ms per layer); the final version uses 3xTF32 throughout. What sets the
distance to the official path at `highest` is the attention forward's precision, not whether
forward and backward match. On the RTX 5090 the same comparisons against the XLA-attention path,
before those changes, gave the same picture at default precision (4.4e-2 CRPS, 2.9e-2 linear).
Files: [`results/finetune/h200/`](results/finetune/h200/) (`grads01`–`grads03`; JSON + log),
[`results/finetune/full_grads_mini_*_rtx5090.json`](results/finetune/).

**Ensemble members one at a time.** At 0.25° the full-model step compiled to 45.8 GiB temp with
FGN's two members as one batch ([`finetune_full_batched_oom_rtx5090.log`](results/finetune/finetune_full_batched_oom_rtx5090.log)).
One rematerialised call per member did not help (46.1 GiB,
[`finetune_full_unrolled_oom_rtx5090.log`](results/finetune/finetune_full_unrolled_oom_rtx5090.log)):
the calls are independent, so XLA ran both backward passes side by side. Running the members in an
`hk.scan` with a rematerialised body forces one at a time: 25.5 GiB. The noise is drawn for the
stacked batch as before, so the loss is unchanged; checked on the CPU (Mini, all parameters, fp32)
with the CRPS's absolute differences replaced by squares to make the loss smooth: loss rel diff
8.4e-8, gradients 1.6e-4 global rel L2 vs the batched call, temp 4.14 → 2.55 GiB.
For the head alone at 0.25° it brings the step from 20.03 to 12.27 GiB temp (+1.79 GiB arguments;
4.03 instead of 3.84 s per step). With the JAX pool on the RTX 5090 capped at what a 24 GB card
gets at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (fraction 0.678), 10 head-only training steps ran with
a device peak of 14.06 GiB; capped at a 16 GB card's pool (14.39 GiB) they also ran, with 0.2 GiB
to spare (14.16 GiB). No real 24 GB or 16 GB card has been tried. `scripts/finetune.py` runs the
members one at a time by default (`--sequential-samples on`).
Files: [`results/finetune/head_sequential_*`](results/finetune/).

**WeatherNext 2 at 0.25°, every parameter** (`finetune.py --modules all`; same data and
held-out days as §8; SST masked; lr 1e-6 with a 5-step warmup; 20 steps). Compiled step: 25.46 GiB
temp + 3.12 GiB arguments (parameters, Adam state, one example), device peak 28.6 GiB, so it needs
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` on a 32 GB card (at 0.9 the step failed to allocate:
[`finetune_full_scan_pool90_oom_rtx5090.log`](results/finetune/finetune_full_scan_pool90_oom_rtx5090.log)).
8.51 s per step. Held-out loss −1.8 … −2.4 % on the four days (same noise before and after); every
variable lower: 2 m T −9.3 %, z −6.1 %, precipitation −5.3 %, MSLP −4.6 %, T −3.9 %, the rest
−0.4 … −1.5 % (SST, masked, −0.1 %). The same caveats as §8 apply: one checkpoint, 20 steps, one
week of held-out one-step losses, no rollout or spread evaluation.
Files: [`results/finetune/finetune_full_smoke_rtx5090.json`](results/finetune/finetune_full_smoke_rtx5090.json) (+ `.log`).

## Limitations

- Single-step equivalence (§1, §2) used one initialisation (2024-10-07) and the
  WeatherNextCyclones checkpoint (same architecture as WeatherNext 2; the public sample data
  lacks the 100 m winds WeatherNext 2 needs). WeatherNext 2 itself was validated through the
  rollouts in §3 and the batch checks in §5.
- Batch sizes > 1 were checked at 2 (§5, §6); larger batches only change memory.
- Without a fixed autotune cache, results are not bitwise reproducible between runs (§4 note).
- fp32 agreement across machines assumes the same host-built graphs. The official code builds the
  grid↔mesh graphs with NumPy on the host CPU, and they follow NumPy's SIMD code path: between an
  AVX2 and an AVX-512 host one fp32 step differs by up to 1.3e-2 max rel RMS (median 6e-4), with or
  without faster-weathernext (§4, A100 note). Only x86 hosts were checked.
- TPU (`splash_mha`) numerics were not compared; TPU matmuls use bf16 passes by default and
  differ from any of the GPU paths above.
- Whole-model gradients were compared with the official path on the Mini model at 1° (§9, on an
  H200: the official backward does not fit a 32 GB GPU). At 0.25° the official backward does not
  fit any single GPU, so the 0.25° training runs rest on the Mini comparison, the kernel checks
  against a float64 reference and the real-mask comparison with JAX autodiff.

## README figures

- `figures/noreaster_comparison.gif`: member `m1s0` (WeatherNext 2 checkpoint 1, noise sample 0)
  of the 2026-09-25 00Z hindcast in §3 — old (reference code) vs new (faster-weathernext), same
  inputs and seed, next to the ECMWF IFS analysis; 10 m wind speed and mean sea-level pressure,
  +6 h to +78 h. RMS sea-level pressure difference between the two runs: 0.002 hPa at +6 h,
  0.12 hPa at +78 h.
- `figures/perf_*.svg`: memory from §1 (official path, A100-80GB) and §2/§4 (rollout peak),
  speed from §2 (RTX 5090, default precision).
- `figures/chaos_*.svg`: MSLP rows of [`results/noreaster/noreaster_p2_vs_p0.csv`](results/noreaster/noreaster_p2_vs_p0.csv)
  (median over the four initialisations).
- `figures/social_preview.png`: the repository's link-preview card (1280×640); the +12 h frame of
  `noreaster_comparison.gif` under the title.
