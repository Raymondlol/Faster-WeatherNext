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
  """One batch element: q, k, v are (n, heads, d)."""
  s = np.einsum("nhd,mhd->hnm", q.astype(np.float64), k.astype(np.float64)) * scale
  s = np.where(mask[None], s, -np.inf)
  p = np.exp(s - s.max(-1, keepdims=True))
  p /= p.sum(-1, keepdims=True)
  return np.einsum("hnm,mhd->nhd", p, v.astype(np.float64))


def _on_gpu():
  dev = jax.devices()[0]
  if dev.platform != "gpu" or "cuda" not in str(getattr(dev.client, "platform_version", "")).lower():
    return False
  try:
    cc = float(getattr(dev, "compute_capability", "0") or 0)
  except ValueError:
    return False
  return cc >= 8.0


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("precision,tol", [("highest", 1e-5), ("default", 2e-2)])
def test_matches_dense_reference(precision, tol, batch):
  if precision == "default" and not _on_gpu():
    pytest.skip("TF32 only exists on GPUs")
  rng = np.random.default_rng(0)
  n, h, d = 300, 2, 64
  q, k, v = (rng.standard_normal((batch, n, h, d)).astype(np.float32) for _ in range(3))
  m = (sparse.random(n, n, density=0.1, random_state=1, format="csr").astype(bool)
       + sparse.identity(n, dtype=bool)).tocsr()
  tiles = pallas_attn.build_tile_mask(m)
  try:
    with jax.default_matmul_precision(precision):
      out = pallas_attn.tiled_flash_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), *tiles,
                                              scale=d**-0.5, interpret=not _on_gpu())
  except NotImplementedError as e:  # interpret mode lacks a Triton primitive in this JAX
    pytest.skip(f"Pallas interpret mode unsupported: {e}")
  for b in range(batch):  # each element attends only to itself
    ref = _reference(q[b], k[b], v[b], m.toarray(), d**-0.5)
    err = np.abs(np.asarray(out[b], np.float64) - ref).max() / np.abs(ref).max()
    assert err < tol, (b, err)


def _reference_grads(q, k, v, do, mask, scale):
  """float64 dQ, dK, dV of the masked softmax attention for one batch element ((n, heads, d))."""
  q, k, v, do = (x.astype(np.float64) for x in (q, k, v, do))
  s = np.where(mask[None], np.einsum("nhd,mhd->hnm", q, k) * scale, -np.inf)
  p = np.exp(s - s.max(-1, keepdims=True))
  p /= p.sum(-1, keepdims=True)                                # [h, n, m]
  dp = np.einsum("nhd,mhd->hnm", do, v)
  ds = p * (dp - (dp * p).sum(-1, keepdims=True))
  dq = np.einsum("hnm,mhd->nhd", ds, k) * scale
  dk = np.einsum("hnm,nhd->mhd", ds, q) * scale
  dv = np.einsum("hnm,nhd->mhd", p, do)
  return dq, dk, dv


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("precision,tol", [("highest", 1e-5), ("default", 3e-2)])
def test_grads_match_dense_reference(precision, tol, batch, symmetric, split):
  """Backward kernels (dQ per query block, dK/dV per key block over the transposed tile list)."""
  if precision == "default" and not _on_gpu():
    pytest.skip("TF32 only exists on GPUs")
  rng = np.random.default_rng(2)
  n, h, d = 300, 2, 64  # n is not a multiple of the 64-row blocks: padding rows are exercised
  q, k, v, do = (rng.standard_normal((batch, n, h, d)).astype(np.float32) for _ in range(4))
  m = sparse.random(n, n, density=0.08, random_state=3, format="csr").astype(bool)
  m = ((m + m.T) if symmetric else m) + sparse.identity(n, dtype=bool)
  m = m.tocsr().astype(bool)
  tiles = pallas_attn.build_tile_mask(m)
  tiles_t = None if symmetric else pallas_attn.build_tile_mask(m.T.tocsr())
  fn = lambda q, k, v: pallas_attn.tiled_flash_attention(q, k, v, *tiles, tiles_t=tiles_t, scale=d**-0.5,
                                                         bwd_split=split, interpret=not _on_gpu())
  try:
    with jax.default_matmul_precision(precision):
      _, vjp = jax.vjp(fn, *(jnp.asarray(x) for x in (q, k, v)))
      grads = vjp(jnp.asarray(do))
  except NotImplementedError as e:
    pytest.skip(f"Pallas interpret mode unsupported: {e}")
  for b in range(batch):
    for name, got, ref in zip("qkv", grads, _reference_grads(q[b], k[b], v[b], do[b], m.toarray(), d**-0.5)):
      err = np.abs(np.asarray(got[b], np.float64) - ref).max() / np.abs(ref).max()
      assert err < tol, (name, b, err)


def test_nonsymmetric_mask_needs_transposed_tiles():
  """Reusing the forward tiles for a non-symmetric mask gives wrong dK/dV (guards the default)."""
  rng = np.random.default_rng(4)
  n, h, d = 130, 1, 32
  q, k, v, do = (rng.standard_normal((1, n, h, d)).astype(np.float32) for _ in range(4))
  m = (sparse.random(n, n, density=0.1, random_state=5, format="csr").astype(bool)
       + sparse.identity(n, dtype=bool)).tocsr()
  tiles = pallas_attn.build_tile_mask(m)
  try:
    _, vjp = jax.vjp(lambda k: pallas_attn.tiled_flash_attention(jnp.asarray(q), k, jnp.asarray(v), *tiles,
                                                                 scale=d**-0.5, interpret=not _on_gpu()),
                     jnp.asarray(k))
    (dk,) = vjp(jnp.asarray(do))
  except NotImplementedError as e:
    pytest.skip(f"Pallas interpret mode unsupported: {e}")
  ref = _reference_grads(q[0], k[0], v[0], do[0], m.toarray(), d**-0.5)[1]
  assert np.abs(np.asarray(dk[0], np.float64) - ref).max() / np.abs(ref).max() > 1e-2
