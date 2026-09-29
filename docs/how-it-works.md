# How it works

WeatherNext 2 (FGN, arXiv:2506.10772) maps two 0.25° atmospheric states (721 × 1440 grid,
13 pressure levels) to the state 6 h later:

```
grid (1,038,240 points) --encoder--> latent 768
  --grid→mesh GNN (1.6 M edges)--> icosahedral mesh (40,962 nodes)
  --24-layer transformer, attention restricted to 32-hop mesh neighbourhoods (~3,100 keys/query)-->
  --mesh→grid GNN (3.1 M edges)--> grid --decoder--> 101 output fields
```

A 32-dimensional noise vector conditions every LayerNorm; one draw = one ensemble member.

## Where the official GPU path spends memory

`compiled.memory_analysis()` and XLA's buffer-assignment dumps for one step (official code,
`triblockdiag_mha`): ~33 GiB of temporaries. The largest live buffers are per-edge tensors of the
mesh→grid GNN (3.1 M × 768 × fp32 = 8.9 GiB each, two alive at once), the attention's dense
banded blocks, and full-grid latents (1,038,240 × 768 × fp32 = 3 GiB each).

## The changes (none alters the math)

`faster_weathernext.enable()` replaces classes in the official modules with subclasses; parameter names and
module structure are unchanged, so the official checkpoints load as-is.

**1. Grid↔mesh GNNs in blocks (`ChunkedDeepGNN`).** The official interaction network computes
the edge MLP for all edges, then segment-sums messages into nodes. For mesh→grid every grid
point has exactly three incoming edges, sorted by receiver, so a block of grid points owns a
contiguous block of edges: edge MLP, aggregation and node update run per block (32,768 points)
inside a scan, writing results in place. For grid→mesh, fixed-size edge blocks (65,536) are
segment-summed into a small fp32 mesh accumulator. The edge encoder is applied per block too.
The same Haiku sub-modules are called; only the order of evaluation changes (summation order
within segment sums may differ by rounding).

**2. Encoder / decoder in blocks.** Both are pointwise over grid points, so they run on blocks
of 32,768 lat-major points; inputs that only vary along one grid axis (e.g. time-of-day over
longitude) are broadcast first. The encoder emits each block points-major (`[block, batch,
latent]`), so the stacked result already has the layout the grid→mesh GNN consumes. Stacking in
grid layout and reordering afterwards costs nothing at batch 1 (only size-1 axes move) but
materialised a second whole-grid copy at batch 2 (13.2 instead of 8.3 GiB temp; v0.1.1).

**3. Transformer as a loop.** The 24 blocks run as `hk.scan` over the layer index with
`hk.switch` selecting `block_XX`. Unrolled, XLA's heap allocator fragments across layers
(~2 GiB); as a loop it reuses one layer's buffers. All layers share one device copy of the
attention mask: a constant closed over by 24 switch branches would otherwise be inlined 24
times into the executable (outside JAX's memory pool).

**4. Fused masked attention (Pallas).** The mask is stored per block of 64 queries as a list of
the 32-key tiles that contain at least one valid key, with one 32-bit word per query row per
tile. Each GPU program (one query block × one head) walks its tiles with an online softmax
(FlashAttention-2): QKᵀ on tensor cores, masked entries set to −1e30 exactly as the official
code does (exp underflows to 0), PV accumulated in fp32. The attention matrix never reaches
GPU memory, and fully masked tiles are skipped. Precision follows JAX: TF32 by default,
3xTF32 under `highest`.

On GPUs without TF32 (before Ampere) an XLA version is used instead: queries in chunks of 512,
each against the contiguous window of keys its chunk needs (the official mesh ordering is
banded), with the mask bit-packed per chunk.

**5. Hilbert ordering.** The official mesh ordering is banded but not compact: a block of 64
consecutive nodes has neighbours spread over ~12,000 keys. Sorting the mesh nodes along a 3D
Hilbert curve makes each block a compact patch (~5,500 keys, 57 % of scanned entries useful
instead of 26 %). Attention, LayerNorm and the feed-forward layers are per-node or
permutation-equivariant with a consistently permuted mask, so the transformer runs on the
permuted nodes and the order is restored afterwards — exact up to summation order.

## Result (RTX 5090, one step)

| | temporaries | step (TF32) |
|---|---|---|
| official GPU path | 33 GiB (does not fit 32 GB) | — |
| chunked attention (P0 reference) | 20.8 GiB | 5.47 s |
| + blocked GNNs / encoder / decoder / loop | 4.0–4.9 GiB | 2.80 s |
| + Pallas attention | 4.0 GiB | 1.00 s |
| + Hilbert ordering | 3.8 GiB | 0.78 s |

After these changes the attention kernel runs at ~72 % of the tensor-core pipeline (Nsight
Compute) and is no longer memory-bound; roughly 45 % of the step is attention and 30 %
other matrix multiplies.
