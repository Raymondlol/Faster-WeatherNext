# Changelog

## Unreleased

- The A100 fp32 deviation in the canaries (1.3e-2 max rel RMS vs the RTX 5090) is explained, and
  it is not GPU precision. The official code builds its grid↔mesh graphs with NumPy on the host
  CPU, and Modal's A100-40GB hosts (AVX2 only) build a slightly different graph than AVX-512
  hosts. `cloud/canary_run.py` records the graph hashes and NumPy's SIMD targets, runs on
  another machine's graph with `--edges`, and skips the rollout with `--steps 0`. New
  `scripts/graph_edges.py` counts and saves the graphs on the CPU; new `cloud/modal_a100_tf32.py`
  and `cloud/modal_graph_hosts.py` are the probes. Validation: docs/validation.md §4, A100 note.

## 0.2.0 — 2026-11-01

Fine-tuning (experimental), IFS open-data mirror fallback, Intel Arc. Inference is unchanged:
re-validated against the reference at both precisions and on A100 / A10G / L4 / T4.

- Head-only fine-tuning (prototype): `faster_weathernext.finetune` trains the mesh→grid GNN and
  grid decoder (5.6M of WeatherNext 2's 184M parameters) with the official training loss (fair
  CRPS, 2 members) while the encoder, grid→mesh GNN and transformer stay frozen. Gradients never
  enter the frozen part, so no attention backward pass is needed. New option `remat` (off by
  default; inference unchanged) rematerialises each GNN / encoder / decoder block in the backward
  pass. `ifs.training_example` builds one-step examples from IFS open data, and
  `scripts/finetune.py` / `scripts/verify_grads.py` train and validate. New `[train]`
  extra (optax). Validation and memory: docs/validation.md §8.
- The fused attention kernel is differentiable (FlashAttention-2 backward as two more tiled
  Pallas kernels, dQ and dK/dV, over the same tile lists), so every parameter can be trained:
  `finetune.split(params, "all")`, `scripts/finetune.py --modules all`. `remat` now also
  rematerialises each transformer layer and each XLA attention chunk, and the new option
  `sequential_samples` runs the 2 ensemble members one at a time in training (same noise; half
  the activation memory). Forward and inference unchanged. Whole-model gradients checked against
  the official path on an H200 (`cloud/modal_grads.py`; the official backward needs 102 GiB even
  for the 1° Mini model). Validation: docs/validation.md §9.
- IFS open-data inputs fall back across mirrors: AWS, then Google Cloud
  (`storage.googleapis.com/ecmwf-open-data`), then ECMWF (`data.ecmwf.int`). The AWS bucket
  throttles a freshly published run with 503 SlowDown on every request, sometimes for hours;
  `fwn forecast` used to sit in its retry backoff. A 503 burst is now ridden out for ~10 s
  before moving on, the mirror that worked is tried first for the next file, and
  `FWN_IFS_MIRRORS` (comma-separated) overrides the order. Checked: the three mirrors return
  identical fields for the 2026-09-29 12Z run.
- `fwn forecast --vars all` writes every output field (levels kept as a dimension).
  `scripts/compare_npz.py` accepts `fwn forecast` NetCDF files on either side (levels split,
  fixed lat/lon stride) and `--save` writes a subsampled `.npz` small enough to attach to an issue.
- Intel Arc B580 via Intel's OpenXLA plugin (`[oneapi]` extra): runs the XLA attention path and
  the unit tests; a full 0.25° step is still open (docs/validation.md §7; contributed by @guidov,
  #1). The Pallas test no longer crashes on GPUs whose compute capability is not a number.

## 0.1.1 — 2026-09-29

Packaging, one memory fix for batch > 1, and validation.

- Batch > 1 at 0.25°: the blocked grid encoder now emits each block points-major, so the whole
  grid is no longer transposed after stacking. That transpose is free at batch 1 (size-1 axis)
  but was a second whole-grid copy at batch 2: temp memory 13.2 → 8.3 GiB (= 2 × batch 1).
  Batch 1 is unchanged (4.14 GiB temp; equivalence re-verified).
- `weathernext` is no longer a hard dependency (it is not on PyPI, and the git URL conflicted
  with environments that pin it, e.g. earth2studio). Install it with the new `[weathernext]`
  extra, or bring your own; `compat.check()` verifies the internals at `enable()`.
- JAX requirement relaxed from `==0.11.2` to `>=0.11.2,<0.12`.
- `fwn info` reports the installed `weathernext` version and commit.
- `model.build()` takes `attention_type` / `mask_type` (official transformer settings).
- Validated: WeatherNextCyclones_Mini and batch size 2 against the official GPU path
  (fp32: max rel RMS 5.7e-5); 0.25° batch 2 element-wise against batch 1; stock vs patched
  inside NVIDIA earth2studio's wrappers (docs/validation.md §5–6; `scripts/verify_batch.py`,
  `scripts/earth2studio_check.py`).
- Tests: Pallas kernel at batch 2, `_merge_point_blocks`.

## 0.1.0 — 2026-09-29

First public version.

- `faster_weathernext.enable()` / `disable()`: run-time patch for the official
  `weathernext` code (commit f2f2c51, JAX 0.11.2); official checkpoints load unchanged.
- Fused Pallas (Triton) tiled flash attention for the 32-hop mesh mask, with Hilbert-ordered
  mesh nodes; XLA chunked attention fallback for pre-Ampere and non-NVIDIA devices.
- Blocked grid↔mesh GNNs (in-place updates, per-block edge encoder), blocked grid
  encoder/decoder, transformer layers as a loop, one shared copy of the attention mask.
- `fwn info` and `fwn forecast` (ECMWF IFS open-data initial conditions, NetCDF output).
- Validation: docs/validation.md (RTX 5090, A100, A10G, L4, T4, RTX 4060; nor'easter hindcasts).
