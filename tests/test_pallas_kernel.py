"""Pallas tiled flash attention vs a float64 dense masked-softmax reference.

Runs on a GPU with compute capability >= 8.0, or in Pallas interpret mode elsewhere (if the
installed JAX supports interpreting the Triton primitives; otherwise skipped).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from faster_weathernext import pallas_attn


def _reference(q, k, v, mask, scale):
  s = np.einsum("nhd,mhd->hnm", q[0].astype(np.float64), k[0].astype(np.float64)) * scale
  s = np.where(mask[None], s, -np.inf)
  p = np.exp(s - s.max(-1, keepdims=True))
  p /= p.sum(-1, keepdims=True)
  return np.einsum("hnm,mhd->nhd", p, v[0].astype(np.float64))


def _on_gpu():
  dev = jax.devices()[0]
  return dev.platform == "gpu" and float(getattr(dev, "compute_capability", "0") or 0) >= 8.0


@pytest.mark.parametrize("precision,tol", [("highest", 1e-5), ("default", 2e-2)])
def test_matches_dense_reference(precision, tol):
  if precision == "default" and not _on_gpu():
    pytest.skip("TF32 only exists on GPUs")
  rng = np.random.default_rng(0)
  n, h, d = 300, 2, 64
  q, k, v = (rng.standard_normal((1, n, h, d)).astype(np.float32) for _ in range(3))
  m = (sparse.random(n, n, density=0.1, random_state=1, format="csr").astype(bool)
       + sparse.identity(n, dtype=bool)).tocsr()
  tiles = pallas_attn.build_tile_mask(m)
  try:
    with jax.default_matmul_precision(precision):
      out = pallas_attn.tiled_flash_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), *tiles,
                                              scale=d**-0.5, interpret=not _on_gpu())
  except NotImplementedError as e:  # interpret mode lacks a Triton primitive in this JAX
    pytest.skip(f"Pallas interpret mode unsupported: {e}")
  ref = _reference(q, k, v, m.toarray(), d**-0.5)
  err = np.abs(np.asarray(out[0], np.float64) - ref).max() / np.abs(ref).max()
  assert err < tol, err
