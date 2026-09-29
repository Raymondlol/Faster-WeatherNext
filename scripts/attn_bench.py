"""Microbenchmark: one WN2 layer's masked mesh attention (40,962 nodes, 6 heads x 128).

Builds the model's exact 32-hop mask from the icosahedral mesh and compares:
  xla      chunked banded-window attention (the path used on pre-Ampere GPUs)
  pallas   fused tiled kernel, official node order
  hilbert  fused tiled kernel, Hilbert-ordered nodes (the default on Ampere+)

  python scripts/attn_bench.py --variants xla,pallas,hilbert --precision default
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from faster_weathernext import pallas_attn
from faster_weathernext.patch import build_chunked_mask

H, D, K_HOP = 6, 128, 32


def mesh_mask():
  from weathernext.utils import data_modalities, icosahedral_mesh
  m = data_modalities.TriangularMeshData.with_icosahedral_mesh(data=None, splits_list=[2] * 6)
  n = m.point_dims_shape[0]
  s, r = icosahedral_mesh.faces_to_edges(m.finest_faces)
  adj = sparse.csr_matrix((np.ones(len(s), bool), (s, r)), shape=(n, n))
  adj = (adj + sparse.identity(n, dtype=bool, format="csr")).astype(bool)
  return (adj**K_HOP).astype(bool).tocsr(), np.asarray(m.lat).ravel(), np.asarray(m.lon).ravel()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--variants", default="xla,pallas,hilbert")
  p.add_argument("--precision", default="default")
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--chunk", type=int, default=512)
  args = p.parse_args()

  t0 = time.time()
  mask, lat, lon = mesh_mask()
  n = mask.shape[0]
  print(f"mask: {n} nodes, {mask.nnz / n:.0f} keys/query, built in {time.time() - t0:.0f}s", flush=True)
  q, k, v = (jax.random.normal(kk, (1, n, H, D), jnp.float32) for kk in jax.random.split(jax.random.PRNGKey(0), 3))
  scale = D**-0.5
  fns = {}
  if "xla" in args.variants:
    starts, bits, window, n_kv = build_chunked_mask(mask, args.chunk)
    starts, bits = jnp.asarray(starts), jnp.asarray(bits)
    c = args.chunk

    def xla(q, k, v):
      nc = starts.shape[0]
      qc = jnp.pad(q, ((0, 0), (0, nc * c - n), (0, 0), (0, 0))).reshape(1, nc, c, H, D).transpose(1, 0, 2, 3, 4)
      kp = jnp.pad(k, ((0, 0), (0, n_kv - n), (0, 0), (0, 0)))
      vp = jnp.pad(v, ((0, 0), (0, n_kv - n), (0, 0), (0, 0)))

      def one(a):
        qq, s0, packed = a
        kc = jax.lax.dynamic_slice_in_dim(kp, s0, window, axis=1)
        vc = jax.lax.dynamic_slice_in_dim(vp, s0, window, axis=1)
        logits = jnp.einsum("bqhd,bkhd->bhqk", qq, kc) * scale
        m = jnp.unpackbits(packed, axis=-1, count=window, bitorder="little").astype(bool)
        w = jax.nn.softmax(jnp.where(m[None, None], logits, -1e30), axis=-1)
        return jnp.einsum("bhqk,bkhd->bqhd", w, vc)

      out = jax.lax.map(one, (qc, starts, bits))
      return out.transpose(1, 0, 2, 3, 4).reshape(1, nc * c, H, D)[:, :n]
    fns["xla"] = xla
  if "pallas" in args.variants:
    tiles = pallas_attn.build_tile_mask(mask)
    fns["pallas"] = lambda q, k, v: pallas_attn.tiled_flash_attention(q, k, v, *tiles, scale=scale)
    print(f"pallas tiles: {len(tiles[1]) * pallas_attn.BLOCK_K / (len(tiles[0]) - 1):.0f} keys/query-block")
  if "hilbert" in args.variants:
    perm = pallas_attn.hilbert_order(lat, lon)
    inv = np.argsort(perm)
    htiles = pallas_attn.build_tile_mask(mask[perm][:, perm])
    print(f"hilbert tiles: {len(htiles[1]) * pallas_attn.BLOCK_K / (len(htiles[0]) - 1):.0f} keys/query-block")

    def hilbert(q, k, v):
      pq, pk, pv = (x[:, perm] for x in (q, k, v))
      return pallas_attn.tiled_flash_attention(pq, pk, pv, *htiles, scale=scale)[:, inv]
    fns["hilbert"] = hilbert

  outs = {}
  with jax.default_matmul_precision(args.precision):
    for name in args.variants.split(","):
      fn = jax.jit(fns[name])
      out = jax.block_until_ready(fn(q, k, v))
      ts = []
      for _ in range(args.iters):
        t0 = time.time()
        jax.block_until_ready(fn(q, k, v))
        ts.append(time.time() - t0)
      outs[name] = np.asarray(out, np.float64)
      print(f"{name:8s} {np.median(ts) * 1e3:7.1f} ms/layer  (x24 = {24 * np.median(ts):.2f} s/step)", flush=True)
  ref_name = args.variants.split(",")[0]
  for name, o in outs.items():
    if name != ref_name:
      r = outs[ref_name]
      print(f"{name} vs {ref_name}: rel RMS {np.sqrt(np.mean((o - r) ** 2)) / r.std():.2e}")


if __name__ == "__main__":
  main()
