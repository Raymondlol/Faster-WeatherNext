# Changelog

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
