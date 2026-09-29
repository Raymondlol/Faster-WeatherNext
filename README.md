# faster-weathernext

**Run Google DeepMind's WeatherNext 2 at full 0.25° resolution on consumer GPUs — 8 GB and up.**

> Unofficial. Not affiliated with or endorsed by Google / Google DeepMind. WeatherNext is an
> experimental research model, not an operational forecast; it does not replace official
> warnings from meteorological agencies.

The official WeatherNext 2 code targets TPUs; its GPU path needs about **34 GiB** for one
0.25° step (an H100/A100-80GB). `faster-weathernext` patches the official code at run time so that the
same model, with the same checkpoints, runs in about **6.4 GiB**, and **7× faster** on the same
GPU, while computing the same thing:

- **Exact attention, no approximation**; outputs match the official GPU implementation to fp32
  rounding (minimum correlation 0.9999999997 over 101 output fields, max relative RMS 2.5e-5).
- Under the default TF32 precision, the difference is about 100× smaller than the spread
  between ensemble members after 24 h, and forecast skill on a real case is indistinguishable
  (see [docs/validation.md](docs/validation.md)).

| GPU | VRAM | s / 6 h step | notes |
|---|---|---|---|
| RTX 5090 | 32 GB | 0.78 | a 10-day forecast in ~31 s per ensemble member |
| A100 | 40 GB | 1.1 | |
| A10G | 24 GB | 2.5 | |
| L4 | 24 GB | 3.5 | |
| RTX 4060 | 8 GB | 18.5 * | needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (the CLI sets it) |
| T4 | 16 GB | 44 | no TF32 on Turing; uses the XLA attention path automatically |

\* measured before the fused attention kernel was added; expected to be several times faster
now (the same architecture, L4, runs at 3.5 s). Default precision (TF32); see
[docs/validation.md](docs/validation.md) for how each number was measured.

Tested on NVIDIA GPUs only. The fused attention kernel is used on NVIDIA compute capability
8.0+ (Ampere and newer); other devices (older NVIDIA GPUs, AMD ROCm, CPU) automatically use
the slower XLA attention path — untested on AMD so far.

## Quick start

```bash
# Python 3.12; pick cuda12 or cuda13 to match your driver
pip install "faster-weathernext[cuda12] @ git+https://github.com/Raymondlol/Faster-WeatherNext.git"

fwn info            # GPU, memory pool, attention path, compatibility check

# 10-day forecast from the latest ECMWF open-data analysis, 4 checkpoints x 2 noise samples,
# cropped to North America
fwn forecast --init 2026092800 --steps 40 --checkpoints 1 2 3 4 --samples 2 \
    --region 15 60 230 310 --out forecast.nc
```

Weights (~735 MB per checkpoint) are downloaded from Google's public bucket on first use,
initial conditions from ECMWF's open data on AWS; both are cached in `~/.cache/faster-weathernext`
(override with `FWN_CACHE`). The first step includes compilation (~1 min; 2–3 min on
slower CPUs).

### From Python, with the official API

```python
import faster_weathernext
faster_weathernext.enable()   # before the model is first traced

# ...then use google-deepmind/weathernext exactly as in its demo notebook. Whatever attention
# type the config asks for is routed to faster-weathernext's implementation, and the official
# checkpoints load unchanged.
```

or with the helpers:

```python
import jax, faster_weathernext
from faster_weathernext import ifs, model
import datetime as dt

faster_weathernext.enable()
task, forward = model.build("WeatherNext2")
params = model.load_params("WeatherNext2", checkpoint=1)
inputs, targets, forcings = ifs.rollout_inputs(dt.datetime(2026, 9, 28, 0), n_steps=40, task=task)
for step, pred in enumerate(model.rollout(forward, params, jax.random.PRNGKey(0), inputs, targets, forcings)):
    ...  # pred: xarray.Dataset for lead time 6 h * (step + 1)
```

## Settings that matter

| | |
|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX reserves 75% of GPU memory by default; a step needs ~6.4 GiB, so 8 GB cards need 0.9 (the CLI sets 0.9 unless you set it). `XLA_PYTHON_CLIENT_PREALLOCATE=false` keeps the 75% cap. |
| `--precision highest` | fp32-level matmuls (the attention kernel uses 3xTF32); ~2× slower than the default TF32. |
| `--autotune-cache PATH` | XLA picks kernels by timing, so results differ in the last bits between runs; this saves the picks on the first run and reuses them, making runs reproducible. |
| `--no-autotune` | deterministic default kernels; recommended with `--precision highest` on A100 (see the note in docs/validation.md). |

## How it works

`faster_weathernext.enable()` swaps a few classes in the official `weathernext` modules for subclasses that
execute the same math differently (details: [docs/how-it-works.md](docs/how-it-works.md)):

1. **Fused masked attention** — a Pallas (Triton) flash-attention kernel for the model's
   32-hop mesh mask: it walks only 64×32 query/key tiles that contain a valid key and never
   writes the attention matrix to memory. Mesh nodes are processed in Hilbert-curve order,
   which halves the tiles (exact: the mask is permuted identically, the order restored after).
2. **Blocked grid↔mesh GNNs** — the official code materialises per-edge tensors for 3.1 M
   edges (8.9 GiB each); here edges and grid points are processed in blocks, with in-place
   updates.
3. **Blocked encoder / decoder** over grid points, and the 24 transformer layers run as a loop
   (one layer's buffers instead of a fragmented 24-layer heap).

Memory for one step: 33 GiB (official GPU path) → 4 GiB of temporaries + 1.7 GiB of weights,
inputs and outputs.

## Related work

- [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) — the official
  code and weights; on GPU the full model needs an H100-class card.
- [NVIDIA earth2studio](https://github.com/NVIDIA/earth2studio) wraps WeatherNext 2 with the
  official GPU attention (80 GB GPUs).
- [kashif/weathernext2](https://huggingface.co/kashif/weathernext2) — a PyTorch / `transformers`
  port with the weights as safetensors; its model card states ~50 GB per ensemble member at 0.25°.

## Citation and licenses

faster-weathernext is Apache-2.0 (see [LICENSE](LICENSE), [NOTICE](NOTICE)). It downloads, but does not
redistribute, the WeatherNext weights (© Google, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
and ECMWF open data (© ECMWF, CC BY 4.0). If you use WeatherNext 2, cite:

```bibtex
@article{alet2025skillful,
  title   = {Skillful joint probabilistic weather forecasting from marginals},
  author  = {Alet, Ferran and Price, Ilan and El-Kadi, Andrew and Masters, Dominic and Markou, Stratis and
             Andersson, Tom R and Stott, Jacklynn and Lam, Remi and Willson, Matthew and
             Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  journal = {arXiv preprint arXiv:2506.10772},
  year    = {2025}
}
```
