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
| A100-SXM4-40GB | sm_80 | pallas | 1.1 | 6.18 GiB | yes (6.21) | 1.6–1.9e-5 in 5 of 6 runs; see note |
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

**A100 note.** One of six A100 fp32 runs deviated from the RTX 5090 at TF32 level (max rel RMS
1.3e-2, median 6.2e-4) and could not be reproduced (autotune on / off and three fresh repeats:
1.6–1.9e-5). XLA's autotuner picks kernels by timing and its picks vary between runs (21 of
149 autotuned ops differed across three repeats); the most likely explanation is one pick that
did not honour fp32 in that run. For strict fp32 use `--no-autotune` (5/5 matching A100 runs)
and for reproducible runs `--autotune-cache`. Files: [`results/a100_repeats/`](results/a100_repeats/).

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

## Limitations

- Single-step equivalence (§1, §2) used one initialisation (2024-10-07) and the
  WeatherNextCyclones checkpoint (same architecture as WeatherNext 2; the public sample data
  lacks the 100 m winds WeatherNext 2 needs). WeatherNext 2 itself was validated through the
  rollouts in §3 and the batch checks in §5.
- Batch sizes > 1 were checked at 2 (§5, §6); larger batches only change memory.
- Without a fixed autotune cache, results are not bitwise reproducible between runs (§4 note).
- TPU (`splash_mha`) numerics were not compared; TPU matmuls use bf16 passes by default and
  differ from any of the GPU paths above.

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
