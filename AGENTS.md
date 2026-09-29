# AGENTS.md — guidance for coding agents working on faster-weathernext

faster-weathernext makes Google DeepMind's WeatherNext 2 (WN2) run at full 0.25° resolution on consumer
GPUs by patching the official `weathernext` code at run time. The value of this project rests on
one claim: **it changes how the model executes, never what it computes.** Every rule below
protects that claim.

## Invariants (do not break)

1. **Numerics.** Any change to `src/faster_weathernext/patch.py` or `src/faster_weathernext/pallas_attn.py` must be
   re-validated on a GPU with `scripts/verify_equivalence.py` at **both** precisions
   (`highest` and `default`) against the reference path, and the numbers reported. Acceptable
   differences are rounding-level (see "Equivalence thresholds"). "It runs" is not validation.
2. **Parameter names.** Official checkpoints must load unchanged. Haiku derives parameter names
   from module names and the module scope active at construction:
   - Patch behaviour by **subclassing** (the Haiku metaclass wraps subclass methods in the
     module's name scope). Monkeypatching a method that *creates* sub-modules puts their
     parameters in the wrong scope.
   - Always pass explicit `name=` (e.g. `ChunkedTransformer` keeps Haiku's default
     `"transformer"`, blocks are `block_XX`).
   - Haiku raises at `apply` time on a missing parameter; a successful apply with the official
     checkpoint is necessary (not sufficient) evidence that names are intact.
3. **No large constants inside loop bodies.** A concrete numpy/jax array closed over by
   `hk.switch`/`hk.scan` branches is inlined as a literal **per branch** (this once put
   24 × 113 MB of masks into the executable, outside JAX's memory pool). Create such arrays
   once and pass them through `jax.lax.optimization_barrier` so they become one traced operand.
4. **Validated versions.** JAX 0.11.2 and weathernext commit `f2f2c51` (`compat.TESTED_*`;
   `compat.check()` warns on other JAX versions and verifies the weathernext internals it
   patches). `pyproject.toml` allows `jax>=0.11.2,<0.12` and does **not** depend on
   `weathernext` (not on PyPI; a git URL there conflicts with environments that pin it, such
   as earth2studio, which pins `9c034db`, docs-only away from f2f2c51). The `[weathernext]`
   extra installs the validated commit. Raising `TESTED_*` requires the full validation matrix
   (docs/validation.md): equivalence at both precisions, a multi-step rollout comparison, and
   the multi-GPU canary (sm_80/86/89/120 + a Turing card for the XLA fallback).
5. **No claims without evidence.** Numbers in README/docs must come from a logged run
   (`docs/validation.md` links each to its script). Say "tested on X" rather than "works on X".

## Layout

```
src/faster_weathernext/
  __init__.py      enable() / disable() / options()
  patch.py         the patch: attention dispatch, layer loop, blocked GNN/encoder/decoder
  pallas_attn.py   Pallas (Triton) tiled flash-attention kernel, tile-mask builder, Hilbert order
  compat.py        startup self-check of the weathernext internals the patch relies on
  model.py         weights download (Google's public bucket), predictor build, rollout
  ifs.py           WN2 inputs from ECMWF IFS open data
  cli.py           `fwn info|forecast`
tests/             CPU tests (masks, tiling, kernel in interpret mode, install/uninstall)
scripts/           GPU validation, memory and kernel benchmarks
cloud/             Modal scripts for multi-GPU canaries
docs/              validation.md (all evidence), how-it-works.md
```

## Commands

```bash
# CPU unit tests (no GPU needed; the Pallas kernel runs in interpret mode)
JAX_PLATFORMS=cpu pytest -q

# GPU: reference vs faster-weathernext, one 0.25° step, both precisions (needs ~24 GB for the reference path)
python scripts/verify_equivalence.py --out equiv.json

# Memory breakdown of one compiled step (compile only; dumps XLA buffer assignment)
python scripts/mem_breakdown.py --dump /tmp/xla_dump

# Attention kernel microbenchmark (one layer, exact model mask)
python scripts/attn_bench.py --variants xla,pallas --precision default
```

## Equivalence thresholds (one 0.25° step, 101 output fields, vs the reference)

| precision | min correlation | max relative RMS | notes |
|---|---|---|---|
| `highest` | ≥ 0.9999999995 | ≲ 3e-5 | worst field is `vertical_velocity@50` (tiny variance) |
| `default` (TF32) | ≥ 0.99998 | ≲ 6e-3 | TF32 rounding; same level as reference-vs-reference with different kernels |

NaN masks must be identical. Over a rollout, differences grow chaotically; they must stay far
below the ensemble member-to-member spread (docs/validation.md).

## Hard-won gotchas

- **Memory accounting.** Use `compiled.memory_analysis()` (temp) and XLA dumps
  (`--xla_dump_to`, `*memory-usage-report.txt`). HloLiveRange's "peak" double-counts aliased
  buffers; the `hlo_rematerialization` warning's numbers are meaningless here. Executable
  constants live outside JAX's pool — check the HLO for `constant(` sizes.
- **JAX's pool** defaults to 75% of GPU memory, also with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  A 0.25° step needs ~6.4 GiB in-pool, so 8 GB cards need `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
  (the CLI sets it unless the user did).
- **Consumer Blackwell (sm_120) has ~99 KB shared memory per block**: Pallas tiles are
  64 queries × 32 keys, 1 stage. Larger tiles/stages fail with "Shared memory size limit exceeded".
- **Triton IEEE fp32 dots do not use tensor cores** (~10× slower on sm_120); under `highest`
  the kernel uses 3xTF32 (`strict_fp32=True` for IEEE). Turing (sm_75) has no TF32 → the
  XLA attention path is selected automatically.
- **XLA autotuning is timing-based**: kernel picks (and last bits) differ between runs; once, an
  A100 fp32 run deviated at TF32 level (1 of 6, not reproduced). Use `--autotune-cache` for
  reproducible runs and `--no-autotune` for strict fp32.
- **In-place block updates** need the new block wrapped in `optimization_barrier` before the
  `dynamic_update_slice`, or XLA fuses the read into the update and copies the whole buffer.
- **Transformer layers as a loop** (`hk.scan` + `hk.switch`) avoid XLA heap fragmentation
  (~2 GiB with 24 unrolled layers).
- **Nsight Compute**: JAX 0.11's CUDA 13.4 runtime crashes ncu 2025.3; use ncu ≥ 2026.3.
- **eccodes** must be imported after pyproj (heap corruption at exit otherwise): keep it lazy.
- **Compile time**: ~1 min on a fast desktop CPU, 2–3 min on cloud vCPUs; building the tile
  mask (`adj**32` + tiling) takes ~10 s of CPU per process.

## Style

2-space indentation and Google-style docstrings (matching upstream weathernext). Comments say
*why* (constraints, measured effects), not what. Keep patches minimal and local; prefer calling
the official module over re-implementing it.

## Things that need a human decision

Cloud spend (Modal/vast.ai), pushing or publishing anything, changing pinned versions, and any
wording in README/docs that describes accuracy or hardware support.
