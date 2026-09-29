"""Fused masked attention for WeatherNext 2's k-hop mesh mask, as a Pallas (Triton) GPU kernel.

The mask (each of 40,962 mesh nodes attends to its ~3,100-node 32-hop neighbourhood) is
stored per block of BQ queries as a CSR list of the BK-key tiles that contain at least one
valid key, plus one uint32 bitmask word per query row per tile (BK = 32). Each program
handles one (query block, head): it walks only its active tiles with an online softmax
(FlashAttention-2), so logits never reach DRAM and fully masked tiles are skipped.

Numerics match the masked softmax of the stock paths up to summation order: masked logits
are set to -1e30 before the softmax (exp underflows to exactly 0), matmuls are TF32 under
JAX's default matmul precision and full fp32 under jax.default_matmul_precision("highest").
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from scipy import sparse

BLOCK_Q = 64
BLOCK_K = 32  # one uint32 mask word per query row per tile
NUM_WARPS = 4
NUM_STAGES = 1
MASK_VALUE = -1e30
# Under jax.default_matmul_precision("highest"): Triton's strict-fp32 ("ieee") dot does not use
# tensor cores and is ~10x slower on consumer Blackwell (sm_120), so by default use 3xTF32
# (fp32-level accuracy on tensor cores); strict_fp32=True selects IEEE fp32 dots.


def _hilbert3(ints, bits):
  """3D Hilbert index of non-negative ints [n, 3] < 2**bits (Skilling's transpose algorithm)."""
  x = ints.T.copy()
  m = 1 << (bits - 1)
  q = m
  while q > 1:
    p = q - 1
    for i in range(3):
      hi = (x[i] & q) != 0
      x[0] = np.where(hi, x[0] ^ p, x[0])
      t = np.where(hi, 0, (x[0] ^ x[i]) & p)
      x[0] ^= t
      x[i] ^= t
    q >>= 1
  for i in range(1, 3):
    x[i] ^= x[i - 1]
  t = np.zeros_like(x[0])
  q = m
  while q > 1:
    t = np.where((x[2] & q) != 0, t ^ (q - 1), t)
    q >>= 1
  for i in range(3):
    x[i] ^= t
  code = np.zeros(x.shape[1], np.int64)
  for bit in range(bits - 1, -1, -1):
    for i in range(3):
      code = (code << 1) | ((x[i] >> bit) & 1)
  return code


def hilbert_order(lat_deg, lon_deg, bits: int = 10) -> np.ndarray:
  """Permutation sorting points on the sphere along a 3D Hilbert curve (non-finite points last).

  For the 32-hop mesh mask this makes each block of 64 consecutive queries a compact patch,
  so the tiles it must scan drop from ~12.2k keys (model's banded order) to ~5.5k.
  """
  lat, lon = np.deg2rad(np.asarray(lat_deg, np.float64).ravel()), np.deg2rad(np.asarray(lon_deg, np.float64).ravel())
  xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
  ok = np.isfinite(xyz).all(-1)
  ints = np.clip(((np.nan_to_num(xyz) + 1) / 2 * (2**bits - 1)).round(), 0, 2**bits - 1).astype(np.int64)
  code = np.where(ok, _hilbert3(ints, bits), np.iinfo(np.int64).max)
  return np.argsort(code, kind="stable")


def build_tile_mask(mask: sparse.spmatrix, block_q: int = BLOCK_Q, block_k: int = BLOCK_K):
  """CSR tile list for a boolean [n, n] mask.

  Returns (tile_ptr int32[n_qblocks + 1], tile_kb int32[n_tiles], tile_bits uint32[n_tiles, block_q]):
  query block i owns tiles tile_ptr[i]:tile_ptr[i+1]; tile t covers keys
  tile_kb[t]*block_k + [0, block_k) and bit j of tile_bits[t, r] says whether query row r
  of the block may attend to key tile_kb[t]*block_k + j.
  """
  assert block_k == 32
  mask = sparse.coo_matrix(mask.astype(bool))
  rows, cols = mask.row.astype(np.int64), mask.col.astype(np.int64)
  n = mask.shape[0]
  n_qblocks = -(-n // block_q)
  qb, kb = rows // block_q, cols // block_k
  key = qb * (-(-mask.shape[1] // block_k)) + kb
  uniq, tile_of_entry = np.unique(key, return_inverse=True)
  n_kblocks = -(-mask.shape[1] // block_k)
  tile_qb, tile_kb = uniq // n_kblocks, uniq % n_kblocks
  bits = np.zeros((len(uniq), block_q), np.uint32)
  np.bitwise_or.at(bits, (tile_of_entry, rows % block_q),
                   (np.uint32(1) << (cols % block_k).astype(np.uint32)))
  tile_ptr = np.zeros(n_qblocks + 1, np.int64)
  np.add.at(tile_ptr, tile_qb + 1, 1)
  tile_ptr = np.cumsum(tile_ptr)
  return tile_ptr.astype(np.int32), tile_kb.astype(np.int32), bits


def _kernel(q_ref, k_ref, v_ref, ptr_ref, kb_ref, bits_ref, o_ref, *, block_k, scale, precision):
  qb = pl.program_id(0)
  q = plgpu.load(q_ref)                                    # [BQ, D]
  block_q, head_dim = q.shape
  cols = lax.broadcasted_iota(jnp.uint32, (1, block_k), 1)

  def body(t, carry):
    acc, m_prev, l_prev = carry
    keys = pl.ds(kb_ref[t] * block_k, block_k)
    k = plgpu.load(k_ref.at[keys, :])                      # [BK, D]
    s = plgpu.dot(q, k, trans_b=True, precision=precision) * scale
    words = plgpu.load(bits_ref.at[t, :])                  # [BQ] uint32
    valid = ((words[:, None] >> cols) & jnp.uint32(1)) != 0
    s = jnp.where(valid, s, MASK_VALUE)
    m_next = jnp.maximum(m_prev, jnp.max(s, axis=1))
    corr = jnp.exp(m_prev - m_next)
    p = jnp.exp(s - m_next[:, None])
    l_next = corr * l_prev + jnp.sum(p, axis=1)
    v = plgpu.load(v_ref.at[keys, :])
    acc = corr[:, None] * acc + plgpu.dot(p, v, precision=precision)
    return acc, m_next, l_next

  init = (jnp.zeros((block_q, head_dim), jnp.float32),
          jnp.full((block_q,), -jnp.inf, jnp.float32),
          jnp.zeros((block_q,), jnp.float32))
  acc, _, l = lax.fori_loop(ptr_ref[qb], ptr_ref[qb + 1], body, init)
  # Every real query attends at least to itself; the guard only affects padding rows.
  o_ref[...] = (acc / jnp.maximum(l, 1e-30)[:, None]).astype(o_ref.dtype)


def tiled_flash_attention(q, k, v, tile_ptr, tile_kb, tile_bits, *, scale, strict_fp32=False,
                          block_q=BLOCK_Q, block_k=BLOCK_K, num_warps=NUM_WARPS,
                          num_stages=NUM_STAGES, interpret=False):
  """q, k, v: [batch, n, heads, head_dim] float32 -> [batch, n, heads, head_dim]."""
  batch, n, heads, head_dim = q.shape
  n_qblocks = tile_ptr.shape[0] - 1
  n_q = n_qblocks * block_q
  n_k = -(-n // block_k) * block_k  # mask keys are < n
  # Head-major layout so each program reads contiguous [rows, head_dim] tiles.
  to_hm = lambda x, rows: jnp.pad(x.transpose(0, 2, 1, 3), ((0, 0), (0, 0), (0, rows - n), (0, 0)))
  qh, kh, vh = to_hm(q, n_q), to_hm(k, n_k), to_hm(v, n_k)
  if interpret:
    precision = None  # CPU interpreter: plain fp32 dots (no tensor-core presets)
  elif jax.config.jax_default_matmul_precision in ("highest", "float32", lax.Precision.HIGHEST):
    precision = (lax.DotAlgorithmPreset.F32_F32_F32 if strict_fp32
                 else lax.DotAlgorithmPreset.TF32_TF32_F32_X3)
  else:
    precision = lax.DotAlgorithmPreset.TF32_TF32_F32
  kernel = functools.partial(_kernel, block_k=block_k, scale=scale, precision=precision)
  full = lambda a: pl.BlockSpec(a.shape, lambda i, h, b: (0,) * a.ndim)
  out = pl.pallas_call(
      kernel,
      grid=(n_qblocks, heads, batch),
      in_specs=[
          pl.BlockSpec((None, None, block_q, head_dim), lambda i, h, b: (b, h, i, 0)),
          pl.BlockSpec((None, None, n_k, head_dim), lambda i, h, b: (b, h, 0, 0)),
          pl.BlockSpec((None, None, n_k, head_dim), lambda i, h, b: (b, h, 0, 0)),
          full(tile_ptr), full(tile_kb), full(tile_bits),
      ],
      out_specs=pl.BlockSpec((None, None, block_q, head_dim), lambda i, h, b: (b, h, i, 0)),
      out_shape=jax.ShapeDtypeStruct((batch, heads, n_q, head_dim), q.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="wn2_tiled_flash_attention",
      interpret=interpret,
  )(qh, kh, vh, jnp.asarray(tile_ptr), jnp.asarray(tile_kb), jnp.asarray(tile_bits))
  return out[:, :, :n].transpose(0, 2, 1, 3)
