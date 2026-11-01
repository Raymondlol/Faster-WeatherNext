"""Fused masked attention for WeatherNext 2's k-hop mesh mask, as a Pallas (Triton) GPU kernel.

The mask (each of 40,962 mesh nodes attends to its ~3,100-node 32-hop neighbourhood) is
stored per block of BQ queries as a CSR list of the BK-key tiles that contain at least one
valid key, plus one uint32 bitmask word per query row per tile (BK = 32). Each program
handles one (query block, head): it walks only its active tiles with an online softmax
(FlashAttention-2), so logits never reach DRAM and fully masked tiles are skipped.

Numerics match the masked softmax of the stock paths up to summation order: masked logits
are set to -1e30 before the softmax (exp underflows to exactly 0), matmuls are TF32 under
JAX's default matmul precision and full fp32 under jax.default_matmul_precision("highest").

The kernel is differentiable (jax.custom_vjp, FlashAttention-2 backward): the forward also stores
each row's log-sum-exp, and two more tiled kernels compute dQ (per query block, over the same tile
list) and dK/dV (per key block, over the transposed mask's tile list, which for the symmetric k-hop
mask is the same list). Without differentiation the forward is unchanged.
"""

import dataclasses
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
BWD_NUM_WARPS = 4  # tuned on the GPU (docs/validation.md §9)
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


def _kernel(q_ref, k_ref, v_ref, ptr_ref, kb_ref, bits_ref, o_ref, *lse_ref, block_k, scale, precision):
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
  acc, m, l = lax.fori_loop(ptr_ref[qb], ptr_ref[qb + 1], body, init)
  # Every real query attends at least to itself; the guard only affects padding rows.
  o_ref[...] = (acc / jnp.maximum(l, 1e-30)[:, None]).astype(o_ref.dtype)
  if lse_ref:  # training: per-row log-sum-exp for the backward pass (padding rows: finite garbage)
    lse_ref[0][...] = m + jnp.log(jnp.maximum(l, 1e-30))


# Backward (FlashAttention-2): dS = P * (dP - delta), delta = rowsum(dO * O). Both kernels recompute
# P = exp(S - lse) tile by tile from the forward's log-sum-exp, so no attention matrix is stored.
# Masked entries are zeroed with `where` rather than through exp underflow, so the finite garbage
# that padding rows carry (padded dO is 0) cannot leak into real rows.
# With split = 2 each program takes half of a 64-row tile-list block (32 rows): it walks the
# block's tiles and reads its half of each tile's mask words, which halves the operands a program
# keeps in shared memory without a second tile list (tiles empty for that half are just masked).


def _row_words(bits_ref, t, split, part):
  """This program's rows of tile t's mask words (`part`: which of the block's `split` parts)."""
  if split == 1:
    return plgpu.load(bits_ref.at[t, :])
  rows = bits_ref.shape[1] // split
  return plgpu.load(bits_ref.at[t, pl.ds(part * rows, rows)])


def _bwd_dq_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, ptr_ref, kb_ref, bits_ref, dq_ref,
                   *, block_k, split, scale, precision):
  """dQ for one (query block, head): walks the query block's key tiles, as the forward does."""
  qb, part = pl.program_id(0) // split, pl.program_id(0) % split
  q = plgpu.load(q_ref)                                    # [BQ, D]
  do = plgpu.load(do_ref)                                  # [BQ, D]
  lse = plgpu.load(lse_ref)                                # [BQ]
  delta = plgpu.load(delta_ref)                            # [BQ]
  cols = lax.broadcasted_iota(jnp.uint32, (1, block_k), 1)

  def body(t, dq):
    keys = pl.ds(kb_ref[t] * block_k, block_k)
    k = plgpu.load(k_ref.at[keys, :])                      # [BK, D]
    v = plgpu.load(v_ref.at[keys, :])
    s = plgpu.dot(q, k, trans_b=True, precision=precision) * scale
    words = _row_words(bits_ref, t, split, part)
    valid = ((words[:, None] >> cols) & jnp.uint32(1)) != 0
    p = jnp.where(valid, jnp.exp(s - lse[:, None]), 0.0)
    dp = plgpu.dot(do, v, trans_b=True, precision=precision)
    ds = p * (dp - delta[:, None])
    return dq + plgpu.dot(ds, k, precision=precision) * scale

  dq = lax.fori_loop(ptr_ref[qb], ptr_ref[qb + 1], body, jnp.zeros(q.shape, jnp.float32))
  dq_ref[...] = dq.astype(dq_ref.dtype)


def _bwd_dkv_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, ptr_ref, kb_ref, bits_ref,
                    dk_ref, dv_ref, *, block_q, split, scale, precision):
  """dK, dV for one (key block, head): walks the key block's query tiles (the transposed tile list)."""
  kb, part = pl.program_id(0) // split, pl.program_id(0) % split
  k = plgpu.load(k_ref)                                    # [BKV, D]
  v = plgpu.load(v_ref)
  cols = lax.broadcasted_iota(jnp.uint32, (1, block_q), 1)

  def body(t, carry):
    dk, dv = carry
    queries = pl.ds(kb_ref[t] * block_q, block_q)
    q = plgpu.load(q_ref.at[queries, :])                   # [BQT, D]
    do = plgpu.load(do_ref.at[queries, :])
    lse = plgpu.load(lse_ref.at[queries])                  # [BQT]
    delta = plgpu.load(delta_ref.at[queries])
    st = plgpu.dot(k, q, trans_b=True, precision=precision) * scale  # [BKV, BQT] = S^T
    words = _row_words(bits_ref, t, split, part)                 # bit j of row r: mask[query j, key r]
    valid = ((words[:, None] >> cols) & jnp.uint32(1)) != 0
    pt = jnp.where(valid, jnp.exp(st - lse[None, :]), 0.0)
    dv = dv + plgpu.dot(pt, do, precision=precision)
    dpt = plgpu.dot(v, do, trans_b=True, precision=precision)
    dst = pt * (dpt - delta[None, :])
    dk = dk + plgpu.dot(dst, q, precision=precision) * scale
    return dk, dv

  zeros = jnp.zeros(k.shape, jnp.float32)
  dk, dv = lax.fori_loop(ptr_ref[kb], ptr_ref[kb + 1], body, (zeros, zeros))
  dk_ref[...] = dk.astype(dk_ref.dtype)
  dv_ref[...] = dv.astype(dv_ref.dtype)


@dataclasses.dataclass(frozen=True)
class _Config:
  scale: float
  precision: object
  block_q: int
  block_k: int
  num_warps: int
  num_stages: int
  bwd_num_warps: int
  bwd_split: int
  interpret: bool


def _full(a):
  return pl.BlockSpec(a.shape, lambda i, h, b: (0,) * a.ndim)


def _forward(qh, kh, vh, tiles, cfg, with_lse):
  batch, heads, n_q, head_dim = qh.shape
  n_k = kh.shape[2]
  out_shape = jax.ShapeDtypeStruct((batch, heads, n_q, head_dim), qh.dtype)
  out_specs = pl.BlockSpec((None, None, cfg.block_q, head_dim), lambda i, h, b: (b, h, i, 0))
  if with_lse:
    out_shape = (out_shape, jax.ShapeDtypeStruct((batch, heads, n_q), jnp.float32))
    out_specs = (out_specs, pl.BlockSpec((None, None, cfg.block_q), lambda i, h, b: (b, h, i)))
  return pl.pallas_call(
      functools.partial(_kernel, block_k=cfg.block_k, scale=cfg.scale, precision=cfg.precision),
      grid=(n_q // cfg.block_q, heads, batch),
      in_specs=[
          pl.BlockSpec((None, None, cfg.block_q, head_dim), lambda i, h, b: (b, h, i, 0)),
          pl.BlockSpec((None, None, n_k, head_dim), lambda i, h, b: (b, h, 0, 0)),
          pl.BlockSpec((None, None, n_k, head_dim), lambda i, h, b: (b, h, 0, 0)),
          *(_full(t) for t in tiles),
      ],
      out_specs=out_specs,
      out_shape=out_shape,
      compiler_params=plgpu.CompilerParams(num_warps=cfg.num_warps, num_stages=cfg.num_stages),
      name="wn2_tiled_flash_attention",
      interpret=cfg.interpret,
  )(qh, kh, vh, *tiles)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _attention(qh, kh, vh, tiles, tiles_t, cfg):
  """Head-major padded attention: qh [B, H, n_q, D], kh/vh [B, H, n_k, D] -> [B, H, n_q, D]."""
  del tiles_t
  return _forward(qh, kh, vh, tiles, cfg, with_lse=False)


def _attention_fwd(qh, kh, vh, tiles, tiles_t, cfg):
  # The backward recomputes the logits and normalises them with this pass's log-sum-exp; all
  # kernels use one dot precision, so the recomputed P is the forward's.
  out, lse = _forward(qh, kh, vh, tiles, cfg, with_lse=True)
  return out, (qh, kh, vh, out, lse, tiles, tiles_t)


def _attention_bwd(cfg, res, dout):
  qh, kh, vh, out, lse, tiles, tiles_t = res
  batch, heads, n_q, head_dim = qh.shape
  n_k = kh.shape[2]
  delta = jnp.sum(out * dout, axis=-1)                     # [B, H, n_q]
  split = cfg.bwd_split
  rows_per = cfg.block_q // split
  params = plgpu.CompilerParams(num_warps=cfg.bwd_num_warps, num_stages=cfg.num_stages)
  rows = lambda size: pl.BlockSpec((None, None, size, head_dim), lambda i, h, b: (b, h, i, 0))
  vec = lambda size: pl.BlockSpec((None, None, size), lambda i, h, b: (b, h, i))
  whole = lambda x: pl.BlockSpec((None, None, *x.shape[2:]), lambda i, h, b: (b, h) + (0,) * (x.ndim - 2))

  dq = pl.pallas_call(
      functools.partial(_bwd_dq_kernel, block_k=cfg.block_k, split=split, scale=cfg.scale,
                        precision=cfg.precision),
      grid=(n_q // rows_per, heads, batch),
      in_specs=[rows(rows_per), whole(kh), whole(vh), rows(rows_per), vec(rows_per), vec(rows_per),
                *(_full(t) for t in tiles)],
      out_specs=rows(rows_per),
      out_shape=jax.ShapeDtypeStruct(qh.shape, qh.dtype),
      compiler_params=params, name="wn2_tiled_flash_attention_dq", interpret=cfg.interpret,
  )(qh, kh, vh, dout, lse, delta, *tiles)

  # Key blocks of block_q rows walk query tiles of block_k: the transposed mask in the forward's
  # tile layout. Keys are padded to whole key blocks; queries already span whole query tiles.
  n_kb = tiles_t[0].shape[0] - 1
  pad = lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, n_kb * cfg.block_q - n_k), (0, 0)))
  dk, dv = pl.pallas_call(
      functools.partial(_bwd_dkv_kernel, block_q=cfg.block_k, split=split, scale=cfg.scale,
                        precision=cfg.precision),
      grid=(n_kb * split, heads, batch),
      in_specs=[whole(qh), rows(rows_per), rows(rows_per), whole(dout), whole(lse), whole(delta),
                *(_full(t) for t in tiles_t)],
      out_specs=(rows(rows_per), rows(rows_per)),
      out_shape=(jax.ShapeDtypeStruct((batch, heads, n_kb * cfg.block_q, head_dim), kh.dtype),) * 2,
      compiler_params=params, name="wn2_tiled_flash_attention_dkv", interpret=cfg.interpret,
  )(qh, pad(kh), pad(vh), dout, lse, delta, *tiles_t)
  return dq, dk[:, :, :n_k], dv[:, :, :n_k], None, None


_attention.defvjp(_attention_fwd, _attention_bwd)


def tiled_flash_attention(q, k, v, tile_ptr, tile_kb, tile_bits, *, scale, strict_fp32=False,
                          tiles_t=None, block_q=BLOCK_Q, block_k=BLOCK_K, num_warps=NUM_WARPS,
                          num_stages=NUM_STAGES, bwd_num_warps=BWD_NUM_WARPS, bwd_split=None,
                          interpret=False):
  """q, k, v: [batch, n, heads, head_dim] float32 -> [batch, n, heads, head_dim].

  Differentiable: the backward pass runs two more tiled kernels (dQ per query block; dK and dV per
  key block). The dK/dV kernel needs the tile list of the transposed mask (`tiles_t`, from
  build_tile_mask(mask.T)); it defaults to the forward's, which is correct for symmetric masks
  such as the k-hop mesh neighbourhoods. bwd_split (1 or 2; default: 2 under `highest`, else 1) is
  the number of backward programs per 64-row block.
  """
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
  # 3xTF32 (and IEEE fp32) dots keep extra operand copies in shared memory: the forward (2 dots
  # per tile) fits sm_120's 99 KB in 64-row programs, the backward kernels (3-4 dots) asked for
  # 147 KB, so under `highest` they run in 32-row programs (bwd_split=2).
  if bwd_split is None:
    bwd_split = 1 if precision in (None, lax.DotAlgorithmPreset.TF32_TF32_F32) else 2
  cfg = _Config(scale=scale, precision=precision, block_q=block_q, block_k=block_k,
                num_warps=num_warps, num_stages=num_stages, bwd_num_warps=bwd_num_warps, bwd_split=bwd_split,
                interpret=interpret)
  tiles = tuple(jnp.asarray(t) for t in (tile_ptr, tile_kb, tile_bits))
  tiles_t = tiles if tiles_t is None else tuple(jnp.asarray(t) for t in tiles_t)
  out = _attention(qh, kh, vh, tiles, tiles_t, cfg)
  return out[:, :, :n].transpose(0, 2, 1, 3)
