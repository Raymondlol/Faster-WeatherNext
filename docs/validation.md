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
The TF32 differences are the same size as the reference path's own TF32-vs-fp32 difference
(§1: 4.7e-3). Files: [`results/same_card/`](results/same_card/), and the packaged code:
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

## Limitations

- Single-step equivalence (§1, §2) used one initialisation (2024-10-07) and the
  WeatherNextCyclones checkpoint (same architecture as WeatherNext 2; the public sample data
  lacks the 100 m winds WeatherNext 2 needs). WeatherNext 2 itself was validated through the
  rollouts in §3.
- WeatherNextCyclones_Mini and batch sizes > 1 (several members per call) are untested.
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
