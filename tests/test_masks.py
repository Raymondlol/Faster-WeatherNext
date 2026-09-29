"""CPU tests: mask layouts, Hilbert ordering, block helpers, patch install/uninstall."""
import numpy as np
import pytest
from scipy import sparse

from faster_weathernext import pallas_attn


def _random_mask(n, density, seed=0):
  m = sparse.random(n, n, density=density, random_state=seed, format="csr").astype(bool)
  return (m + sparse.identity(n, dtype=bool, format="csr")).tocsr()


def _tiles_to_dense(ptr, kb, bits, n, bq, bk):
  dense = np.zeros((len(ptr) - 1) * bq, dtype=bool)[:, None].repeat(max(kb.max() + 1, 1) * bk, 1)
  for qb in range(len(ptr) - 1):
    for t in range(ptr[qb], ptr[qb + 1]):
      cols = kb[t] * bk + np.arange(bk)
      rows = qb * bq + np.arange(bq)
      word = bits[t]
      dense[np.ix_(rows, cols)] = ((word[:, None] >> np.arange(bk, dtype=np.uint32)) & 1).astype(bool)
  return dense[:n, :n]


@pytest.mark.parametrize("n", [100, 257, 1000])
def test_tile_mask_roundtrip(n):
  m = _random_mask(n, 0.05)
  ptr, kb, bits = pallas_attn.build_tile_mask(m)
  assert ptr[0] == 0 and ptr[-1] == len(kb) == len(bits)
  np.testing.assert_array_equal(_tiles_to_dense(ptr, kb, bits, n, pallas_attn.BLOCK_Q, pallas_attn.BLOCK_K),
                                m.toarray())
  # Within a query block, tiles are distinct and no tile is empty.
  for qb in range(len(ptr) - 1):
    seg = kb[ptr[qb]:ptr[qb + 1]]
    assert len(np.unique(seg)) == len(seg)
    assert (bits[ptr[qb]:ptr[qb + 1]].any(axis=1)).all()


def test_hilbert_order_is_permutation_and_local():
  rng = np.random.default_rng(0)
  lat, lon = rng.uniform(-90, 90, 5000), rng.uniform(0, 360, 5000)
  lat[:3] = np.nan
  perm = pallas_attn.hilbert_order(lat, lon)
  assert sorted(perm.tolist()) == list(range(5000))
  assert set(perm[-3:].tolist()) == {0, 1, 2}  # non-finite points last
  # Consecutive points along the curve are much closer than random pairs.
  p = perm[:-3]
  xyz = np.stack([np.cos(np.deg2rad(lat[p])) * np.cos(np.deg2rad(lon[p])),
                  np.cos(np.deg2rad(lat[p])) * np.sin(np.deg2rad(lon[p])), np.sin(np.deg2rad(lat[p]))], -1)
  step = np.linalg.norm(np.diff(xyz, axis=0), axis=1).mean()
  rand = np.linalg.norm(xyz[rng.permutation(len(p))] - xyz, axis=1).mean()
  assert step < 0.2 * rand


def test_chunked_mask_roundtrip():
  from faster_weathernext import patch
  n, chunk = 700, 64
  # Banded mask (as for the model's mesh ordering).
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
  m = sparse.csr_matrix(np.abs(i - j) <= 40)
  starts, bits, window, n_kv = patch.build_chunked_mask(m, chunk)
  dense = np.zeros((len(starts) * chunk, n_kv), bool)
  unpacked = np.unpackbits(bits, axis=-1, count=window, bitorder="little").astype(bool)
  for c, s in enumerate(starts):
    dense[c * chunk:(c + 1) * chunk, s:s + window] = unpacked[c]
  np.testing.assert_array_equal(dense[:n, :n], m.toarray())


def test_num_blocks_divides_when_possible():
  from faster_weathernext.patch import _num_blocks
  assert _num_blocks(1038240, 32768) == 32 and 1038240 % 32 == 0
  n = 1_000_003  # prime: falls back to ceil
  assert _num_blocks(n, 32768) == -(-n // 32768)


def test_enable_disable_restores_modules():
  import faster_weathernext
  from weathernext.utils import deep_gnn, sparse_transformer as st, xarray_dense
  from faster_weathernext import patch
  orig = (st.Transformer, st.Block, deep_gnn.DeepGNN, xarray_dense.DataArrayDictDenseEncoder)
  faster_weathernext.enable()
  assert patch.is_enabled() and st.Block is patch.ChunkedBlock and deep_gnn.DeepGNN is patch.ChunkedDeepGNN
  faster_weathernext.disable()
  assert (st.Transformer, st.Block, deep_gnn.DeepGNN, xarray_dense.DataArrayDictDenseEncoder) == orig
  with pytest.raises(TypeError):
    faster_weathernext.enable(not_an_option=1)
  faster_weathernext.disable()


def test_compat_no_problems():
  from faster_weathernext import compat
  assert compat.problems() == []
