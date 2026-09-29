# Changelog

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
