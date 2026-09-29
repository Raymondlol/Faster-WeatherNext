"""Memory-lean, fast GPU execution of WeatherNext 2, installed as a patch on the official code.

`enable()` changes how the official `weathernext` modules execute, never what they compute:
parameter names, checkpoints and the model's math are unchanged (see docs/validation.md).

1. Attention (sparse_transformer): the k-hop masked attention runs either as a fused Pallas
   (Triton) tiled flash-attention kernel (`pallas_attn.py`; GPUs with compute capability >= 8.0)
   or as an XLA chunked banded-window attention (older GPUs). Same masked softmax as the
   official paths (masked logits = -1e30).
2. Transformer: the 24 blocks run as hk.scan over the layer index with hk.switch selecting
   block_XX (one layer's temporaries instead of a fragmented heap), and all blocks share one
   device copy of the mask. On the Pallas path the mesh nodes are processed in Hilbert-curve
   order (the mask is permuted identically and the order restored afterwards).
3. Grid<->mesh GNNs (deep_gnn): run over blocks of grid points / edges so no
   [num_edges, latent] tensor is materialised; grid-point updates are in place; the edge
   encoder is applied per block.
4. Grid encoder / decoder (xarray_dense): the pointwise encoder and decoder run over the same
   blocks of grid points.

Options are read when the model is traced (i.e. at jit time), so change them before the first
call of a jitted function.
"""

import dataclasses
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import xarray as xr
import xarray_jax

from weathernext.utils import deep_gnn
from weathernext.utils import dense
from weathernext.utils import gather_scatter_ops
from weathernext.utils import mesh_transformer
from weathernext.utils import points_mesh_gnn
from weathernext.utils import sparse_transformer as st
from weathernext.utils import sparse_transformer_utils as st_utils
from weathernext.utils import typed_graph_net
from weathernext.utils import xarray_dense

from faster_weathernext import pallas_attn


@dataclasses.dataclass
class Options:
  """Execution options, read at trace time."""
  # Attention kernel: "pallas", "xla", or "auto" (pallas on GPUs with compute capability >= 8.0).
  attention: str = "auto"
  # Route every attention_type the official configs use (e.g. "triblockdiag_mha", "splash_mha")
  # to this implementation. If False, only attention_type="chunked_mha" is handled.
  replace_attention: bool = True
  # Pallas path: process mesh nodes in Hilbert-curve order (exact; ~2x fewer key tiles).
  reorder: bool = True
  # Under jax.default_matmul_precision("highest") the Pallas kernel uses 3xTF32 (fp32-level
  # accuracy on tensor cores); True forces IEEE fp32 dots (much slower on consumer GPUs).
  strict_fp32: bool = False
  # XLA attention path: queries per chunk.
  attn_chunk: int = 512
  # Grid points per block (GNNs, encoder, decoder) and edges per grid->mesh block; 0 = stock.
  grid_chunk: int = 32768
  edge_chunk: int = 65536
  # Run the transformer layers as a loop (hk.scan + hk.switch) instead of unrolled.
  layer_loop: bool = True


OPTIONS = Options()
_CHUNKED = "chunked_mha"


def attention_impl():
  """The attention implementation `enable()` would use on the default JAX device."""
  if OPTIONS.attention != "auto":
    return OPTIONS.attention
  dev = jax.devices()[0]
  # The Pallas kernel is validated on NVIDIA GPUs with TF32 tensor cores (sm_80+). Everything
  # else (older NVIDIA GPUs, AMD ROCm which reports e.g. "gfx1100", CPU) uses the XLA path.
  if dev.platform != "gpu" or "cuda" not in str(getattr(dev.client, "platform_version", "")).lower():
    return "xla"
  try:
    cc = float(getattr(dev, "compute_capability", "0") or "0")
  except ValueError:
    return "xla"
  return "pallas" if cc >= 8.0 else "xla"


# ----------------------------------------------------------------------------------------------
# Attention


def build_chunked_mask(mask: sparse.spmatrix, chunk: int):
  """Returns (starts int32[n_chunks], packed_bits uint8[n_chunks, chunk, W/8], W, n_kv)."""
  mask = mask.tocsr().astype(bool)
  n = mask.shape[0]
  n_chunks = -(-n // chunk)
  row_lo = np.array([mask.indices[mask.indptr[i]:mask.indptr[i + 1]].min() for i in range(n)])
  row_hi = np.array([mask.indices[mask.indptr[i]:mask.indptr[i + 1]].max() for i in range(n)])
  span = max(row_hi[i * chunk:(i + 1) * chunk].max() - row_lo[i * chunk:(i + 1) * chunk].min() + 1
             for i in range(n_chunks))
  window = int(-(-span // 128) * 128)
  n_kv = max(n, window)
  mask = sparse.hstack([mask, sparse.csr_matrix((n, n_kv - n), dtype=bool)]).tocsr()
  starts = np.zeros(n_chunks, np.int32)
  bits = np.zeros((n_chunks, chunk, window // 8), np.uint8)
  for i in range(n_chunks):
    r0, r1 = i * chunk, min((i + 1) * chunk, n)
    starts[i] = min(row_lo[r0:r1].min(), n_kv - window)
    block = mask[r0:r1, starts[i]:starts[i] + window].toarray()
    # Every real row must keep all of its keys inside the window.
    assert block.sum() == mask[r0:r1].sum(), f"chunk {i} window too narrow"
    bits[i, :r1 - r0] = np.packbits(block, axis=-1, bitorder="little")
  return starts, bits, window, n_kv


def chunked_mha(x, starts, bits, window, n_kv, chunk, cfg):
  """XLA path: masked MHA over banded-window key blocks. x: (batch, nodes, d_model)."""
  batch, n, _ = x.shape
  q = st.multihead_linear(x, "q", cfg)
  k = st.multihead_linear(x, "k", cfg)
  v = st.multihead_linear(x, "v", cfg)
  n_chunks = starts.shape[0]
  _, _, heads, head_dim = q.shape
  q = jnp.pad(q, ((0, 0), (0, n_chunks * chunk - n), (0, 0), (0, 0)))
  q = q.reshape(batch, n_chunks, chunk, heads, head_dim).transpose(1, 0, 2, 3, 4)
  k = jnp.pad(k, ((0, 0), (0, n_kv - n), (0, 0), (0, 0)))
  v = jnp.pad(v, ((0, 0), (0, n_kv - n), (0, 0), (0, 0)))
  scale = cfg.key_size**-0.5

  def one_chunk(args):
    qc, start, packed = args
    kc = jax.lax.dynamic_slice_in_dim(k, start, window, axis=1)
    vc = jax.lax.dynamic_slice_in_dim(v, start, window, axis=1)
    logits = jnp.einsum("bqhd,bkhd->bhqk", qc, kc) * scale
    m = jnp.unpackbits(packed, axis=-1, count=window, bitorder="little").astype(bool)
    logits = jnp.where(m[None, None], logits, -1e30)
    weights = st_utils.wrap_fn_for_upcast_downcast(logits, jax.nn.softmax)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, vc)

  out = jax.lax.map(one_chunk, (q, starts, bits))
  out = out.transpose(1, 0, 2, 3, 4).reshape(batch, n_chunks * chunk, heads * head_dim)[:, :n]
  attn_winit_final = hk.initializers.VarianceScaling(cfg.attn_winit_final_mult / cfg.num_layers)
  return hk.Linear(cfg.d_model, name="mha_final", w_init=attn_winit_final)(out)


def tiled_mha(x, tiles, cfg):
  """Pallas path: same projections as chunked_mha, fused tiled attention. x: (batch, nodes, d_model)."""
  batch, n, _ = x.shape
  q = st.multihead_linear(x, "q", cfg)
  k = st.multihead_linear(x, "k", cfg)
  v = st.multihead_linear(x, "v", cfg)
  _, _, heads, head_dim = q.shape
  out = pallas_attn.tiled_flash_attention(q, k, v, *tiles, scale=cfg.key_size**-0.5,
                                          strict_fp32=OPTIONS.strict_fp32)
  attn_winit_final = hk.initializers.VarianceScaling(cfg.attn_winit_final_mult / cfg.num_layers)
  return hk.Linear(cfg.d_model, name="mha_final", w_init=attn_winit_final)(
      out.reshape(batch, n, heads * head_dim))


_ORIG = dict(
    transformer_init=st.Transformer.__init__,
    Transformer=st.Transformer,
    Block=st.Block,
    DeepGNN=deep_gnn.DeepGNN,
    Encoder=xarray_dense.DataArrayDictDenseEncoder,
    Decoder=xarray_dense.DataArrayDictDenseDecoder,
    build_typed_graph=points_mesh_gnn.PointsMeshTypedGraphGNN._build_typed_graph,
    get_adjacency_matrix=mesh_transformer._get_adjacency_matrix,
)


def _transformer_init(self, adj_mat, attention_k_hop, attention_type, mask_type,
                      num_heads=1, name=None, block_q=None, block_kv=None,
                      block_kv_compute=None, block_q_dkv=None, block_kv_dkv=None,
                      block_kv_dkv_compute=None, **kwargs):
  # Keep Haiku's default module name ("transformer") for the ChunkedTransformer subclass.
  name = name or "transformer"
  if OPTIONS.replace_attention:
    attention_type = _CHUNKED
  if attention_type != _CHUNKED:
    return _ORIG["transformer_init"](
        self, adj_mat, attention_k_hop, attention_type, mask_type, num_heads=num_heads,
        name=name, block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv, block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute, **kwargs)
  hk.Module.__init__(self, name=name)
  if attention_impl() == "pallas":
    mask = adj_mat**attention_k_hop
    latlon = getattr(adj_mat, "_fwn_latlon", None)
    perm = pallas_attn.hilbert_order(*latlon) if OPTIONS.reorder and latlon is not None else None
    if perm is not None:
      mask = mask.tocsr()[perm][:, perm]
    self.mask = pallas_attn.build_tile_mask(mask)
    self.num_padding_nodes = ("pallas", perm)
  else:
    starts, bits, window, n_kv = build_chunked_mask(adj_mat**attention_k_hop, OPTIONS.attn_chunk)
    self.mask = (starts, bits)
    # Reuse the num_padding_nodes slot to carry static chunking metadata to Block.
    self.num_padding_nodes = (window, n_kv, OPTIONS.attn_chunk)
  self._cfg = st._ModelConfig(mask_block_size=0, attention_type=attention_type,
                              mask_type=mask_type, num_heads=num_heads, **kwargs)


class ChunkedBlock(_ORIG["Block"]):
  """st.Block plus the chunked/Pallas attention branch. Subclassed (not method-patched) so
  Haiku's metaclass wraps __call__ in the module's name scope."""

  def __call__(self, x, global_norm_conditioning=jax.Array, mask_arrays=None):
    if self._cfg.attention_type != _CHUNKED:
      return super().__call__(x, global_norm_conditioning=global_norm_conditioning)
    # ChunkedTransformer passes one shared device copy of the mask; converting
    # self.mask here would embed a large constant per layer in the executable.
    mask = tuple(mask_arrays) if mask_arrays is not None else tuple(map(jnp.asarray, self.mask))

    def norm_conditioning_layer(y):
      return dense.LinearNormConditioning(name=self.name + "_norm_conditioning")(
          y, norm_conditioning=jnp.expand_dims(global_norm_conditioning, 1))

    h = norm_conditioning_layer(st.layernorm(x, create_scale=False, create_offset=False))
    if self.num_padding_nodes[0] == "pallas":
      x = x + tiled_mha(h, mask, self._cfg)
    else:
      window, n_kv, chunk = self.num_padding_nodes
      x = x + chunked_mha(h, *mask, window, n_kv, chunk, self._cfg)
    x = x + st.ffw(
        norm_conditioning_layer(st.layernorm(x, create_scale=False, create_offset=False)),
        self._cfg)
    return x


class ChunkedTransformer(_ORIG["Transformer"]):
  """All blocks share one copy of the mask; with layer_loop the blocks run as hk.scan over
  the layer index with hk.switch picking block_XX. Parameter names are unchanged."""

  def __call__(self, node_features, global_norm_conditioning):
    if self._cfg.attention_type != _CHUNKED:
      return super().__call__(node_features, global_norm_conditioning)
    # A concrete array closed over by the switch branches is inlined as a literal in every
    # branch (24 copies of the mask in the executable); the barrier makes the mask a traced
    # value, so it is one constant passed to the loop as an operand.
    mask_arrays = jax.lax.optimization_barrier(tuple(jnp.asarray(m) for m in self.mask))
    blocks = [ChunkedBlock(cfg=self._cfg, mask=self.mask, num_nodes=node_features.shape[1],
                           num_padding_nodes=self.num_padding_nodes, name="block_%02d" % i)
              for i in range(self._cfg.num_layers)]

    def apply(block, x):
      return block(x, global_norm_conditioning=global_norm_conditioning, mask_arrays=mask_arrays)

    perm = self.num_padding_nodes[1] if self.num_padding_nodes[0] == "pallas" else None
    if perm is not None:
      node_features = jnp.take(node_features, jnp.asarray(perm, jnp.int32), axis=1)

    if OPTIONS.layer_loop:
      branches = [functools.partial(apply, b) for b in blocks]
      x, _ = hk.scan(lambda x, i: (hk.switch(i, branches, x), None), node_features,
                     jnp.arange(len(blocks)))
    else:
      x = node_features
      for b in blocks:
        x = apply(b, x)
    x = st.layernorm(x, create_scale=False, create_offset=False)
    x = dense.LinearNormConditioning(name=self.name + "_final_norm_conditioning")(
        x, norm_conditioning=jnp.expand_dims(global_norm_conditioning, 1))
    if perm is not None:
      x = jnp.take(x, jnp.asarray(np.argsort(perm), jnp.int32), axis=1)
    return x


def _get_adjacency_matrix(triangular_mesh_data, add_self_edges):
  """Stock adjacency, tagged with the node coordinates for the Hilbert reordering."""
  adj = _ORIG["get_adjacency_matrix"](triangular_mesh_data, add_self_edges=add_self_edges)
  lat, lon = triangular_mesh_data.lat, triangular_mesh_data.lon
  if isinstance(lat, np.ndarray) and isinstance(lon, np.ndarray) and lat.size == lon.size == adj.shape[0]:
    adj._fwn_latlon = (lat.ravel(), lon.ravel())
  return adj


# ----------------------------------------------------------------------------------------------
# Grid <-> mesh GNNs

MESH = points_mesh_gnn.MESH_NODES_NAME
POINTS = points_mesh_gnn.POINT_NODES_NAME


def _num_blocks(n, block):
  """Smallest block count >= ceil(n / block) that divides n (so no padding copy), else ceil."""
  lo = -(-n // block)
  for nb in range(lo, 2 * lo + 1):
    if n % nb == 0:
      return nb
  return lo


def _pad_rows(x, rows):
  return x if x.shape[0] == rows else jnp.pad(x, [(0, rows - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def _update_blocks_in_place(fn, blocks, xs=()):
  """blocks[i] <- fn(blocks[i], *xs[i]), as a scan whose carry XLA updates in place."""

  def body(buf, args):
    i, *rest = args
    new = fn(jax.lax.dynamic_index_in_dim(buf, i, keepdims=False), *rest)
    # The barrier keeps the block computation (which reads `buf`) out of the
    # dynamic-update-slice fusion, so XLA can alias the update with `buf`.
    new = jax.lax.optimization_barrier(new.astype(buf.dtype))
    return jax.lax.dynamic_update_index_in_dim(buf, new, i, 0), None

  out, _ = hk.scan(body, blocks, (jnp.arange(blocks.shape[0]), *xs))
  return out


class ChunkedDeepGNN(_ORIG["DeepGNN"]):
  """deep_gnn.DeepGNN with a blocked path for the one-step grid<->mesh GNNs.

  The official InteractionNetwork materialises the edge MLP input, hidden layer and messages
  for every edge (3.1M mesh->grid edges x 768 x f32 = 8.9 GiB each). Here the same Haiku
  sub-modules (captured from `_networks_builder`, so parameter names are unchanged) are
  applied block by block inside hk.scan:

  * mesh->points: every point has exactly k in-edges and edges are sorted by receiver, so a
    block of points owns a contiguous block of edges; edge update, aggregation and the point
    update all happen inside the block.
  * points->mesh: fixed-size edge blocks, messages are segment-summed into a
    [num_mesh, B, F] accumulator carried through a scan; the (pointwise) point update is then
    run in point blocks.

  Updated edge features are not returned (the input edge features are passed through);
  PointsMeshTypedGraphGNN only reads node features.
  """

  # Set by the patched PointsMeshTypedGraphGNN._build_typed_graph when the graph carries raw
  # (unencoded) spatial edge features: maps [edges, batch, n_features] -> [edges, batch, latent].
  deferred_edge_encoder = None

  def __call__(self, input_graph, global_norm_conditioning=None, is_training=None):
    mode = self._chunk_mode(input_graph)
    if mode is None:
      if self.deferred_edge_encoder is not None:  # raw edge features: encode them the stock way
        input_graph = input_graph._replace(edges={
            k: es._replace(features=self.deferred_edge_encoder(es.features))
            for k, es in input_graph.edges.items()})
      return super().__call__(input_graph, global_norm_conditioning, is_training)
    captured = {}
    orig = typed_graph_net.InteractionNetwork
    typed_graph_net.InteractionNetwork = lambda **kw: captured.update(kw)
    try:
      self._networks_builder(input_graph, global_norm_conditioning)
    finally:
      typed_graph_net.InteractionNetwork = orig
    edge_key = next(iter(input_graph.edges))
    if mode == "mesh_to_points":
      nodes = self._mesh_to_points(input_graph, edge_key, captured)
    else:
      nodes = self._points_to_mesh(input_graph, edge_key, captured)
    return input_graph._replace(nodes={
        k: ns._replace(features=nodes[k]) for k, ns in input_graph.nodes.items()})

  def _encode_edges(self, e):
    return e if self.deferred_edge_encoder is None else self.deferred_edge_encoder(e)

  def _chunk_mode(self, graph):
    if (OPTIONS.grid_chunk <= 0 or self._num_message_passing_steps != 1
        or self._num_processor_repetitions != 1 or not self._pre_gather_matmul
        or set(graph.nodes) != {MESH, POINTS} or len(graph.edges) != 1):
      return None
    (key, es), = graph.edges.items()
    if any(ns.features.ndim != 3 for ns in graph.nodes.values()) or graph.context.features:
      return None
    snd, rcv = es.indices.senders, es.indices.receivers
    if not (isinstance(snd, np.ndarray) and isinstance(rcv, np.ndarray)):
      return None
    if key.node_sets == (MESH, POINTS):
      n = graph.nodes[POINTS].features.shape[0]
      k = len(rcv) // n
      if k * n == len(rcv) and np.array_equal(rcv, np.repeat(np.arange(n), k)):
        return "mesh_to_points"
    elif key.node_sets == (POINTS, MESH) and OPTIONS.edge_chunk > 0 and np.all(np.diff(rcv) >= 0):
      return "points_to_mesh"
    return None

  def _aggregate(self, messages, segment_ids, num_segments):
    # Same as DeepGNN's maybe_upcast_and_normalize_wrapper around a sorted segment_sum.
    dtype = messages.dtype
    if self._f32_aggregation:
      messages = messages.astype(jnp.float32)
    out = jax.ops.segment_sum(messages, segment_ids, num_segments, indices_are_sorted=True)
    if self._aggregate_normalization:
      out = out / self._aggregate_normalization
    return out.astype(dtype) if self._f32_aggregation else out

  def _mesh_to_points(self, graph, edge_key, f):
    name = edge_key.name
    mesh = graph.nodes[MESH].features
    pts = graph.nodes[POINTS].features
    es = graph.edges[edge_key]
    n, batch, feat = pts.shape
    k = len(es.indices.receivers) // n
    nb = _num_blocks(n, OPTIONS.grid_chunk)
    block = -(-n // nb)
    senders = np.pad(np.asarray(es.indices.senders, np.int32), (0, (nb * block - n) * k))
    local_rcv = np.repeat(np.arange(block, dtype=np.int32), k)
    sent_proj = f["pre_gather_senders_for_edges_fn"][name](mesh)
    gather_rcv = f["gather_nodes_for_edges_fn"].receivers is not None

    def one_block(p, e, s):
      sent = gather_scatter_ops.gather_with_fill(sent_proj, s)
      recv = (gather_scatter_ops.gather_with_fill(
          f["pre_gather_receivers_for_edges_fn"][name](p), local_rcv, indices_are_sorted=True)
              if gather_rcv else None)
      msg = f["update_edge_fn"][name](
          f["pre_gather_edges_for_edges_fn"][name](self._encode_edges(e)), sent, recv,
          graph.context.features)
      agg = self._aggregate(msg, local_rcv, block)
      return p + f["update_node_fn"][POINTS](p, {}, {name: agg}, graph.context.features)

    new_pts = _update_blocks_in_place(one_block, _pad_rows(pts, nb * block).reshape(nb, block, batch, feat), (
        _pad_rows(es.features, nb * block * k).reshape(nb, block * k, *es.features.shape[1:]),
        jnp.asarray(senders.reshape(nb, block * k))))
    new_pts = new_pts.reshape(nb * block, batch, feat)[:n]
    new_mesh = mesh + f["update_node_fn"][MESH](mesh, {}, {}, graph.context.features)
    return {MESH: new_mesh, POINTS: new_pts}

  def _points_to_mesh(self, graph, edge_key, f):
    name = edge_key.name
    mesh = graph.nodes[MESH].features
    pts = graph.nodes[POINTS].features
    es = graph.edges[edge_key]
    num_mesh, batch, feat = mesh.shape
    n_edges = len(es.indices.senders)
    edge_chunk = OPTIONS.edge_chunk
    nb = -(-n_edges // edge_chunk)
    pad = nb * edge_chunk - n_edges
    senders = np.pad(np.asarray(es.indices.senders, np.int32), (0, pad))
    # Padded edges point at segment `num_mesh`, which segment_sum drops.
    receivers = np.pad(np.asarray(es.indices.receivers, np.int32), (0, pad),
                       constant_values=num_mesh)
    gather_rcv = f["gather_nodes_for_edges_fn"].receivers is not None
    mesh_rcv_proj = f["pre_gather_receivers_for_edges_fn"][name](mesh) if gather_rcv else None
    acc_dtype = jnp.float32 if self._f32_aggregation else mesh.dtype

    def one_block(acc, xs):
      e, s, r = xs
      # Row-wise matmul commutes with the gather; gathering first avoids a [points, F] buffer.
      sent = f["pre_gather_senders_for_edges_fn"][name](gather_scatter_ops.gather_with_fill(pts, s))
      recv = gather_scatter_ops.gather_with_fill(mesh_rcv_proj, r) if gather_rcv else None
      msg = f["update_edge_fn"][name](
          f["pre_gather_edges_for_edges_fn"][name](self._encode_edges(e)), sent, recv,
          graph.context.features)
      acc = acc + jax.ops.segment_sum(msg.astype(acc_dtype), r, num_mesh, indices_are_sorted=True)
      return acc, None

    acc, _ = hk.scan(one_block, jnp.zeros((num_mesh, batch, feat), acc_dtype), (
        _pad_rows(es.features, nb * edge_chunk).reshape(nb, edge_chunk, *es.features.shape[1:]),
        jnp.asarray(senders.reshape(nb, edge_chunk)),
        jnp.asarray(receivers.reshape(nb, edge_chunk))))
    if self._aggregate_normalization:
      acc = acc / self._aggregate_normalization
    agg = acc.astype(mesh.dtype)
    new_mesh = mesh + f["update_node_fn"][MESH](mesh, {}, {name: agg}, graph.context.features)

    n = pts.shape[0]
    pb = _num_blocks(n, OPTIONS.grid_chunk)
    block = -(-n // pb)
    new_pts = _update_blocks_in_place(
        lambda p: p + f["update_node_fn"][POINTS](p, {}, {}, graph.context.features),
        _pad_rows(pts, pb * block).reshape(pb, block, batch, feat))
    return {MESH: new_mesh, POINTS: new_pts.reshape(pb * block, batch, feat)[:n]}


def _build_typed_graph(self, triangular_mesh_data, lat_lon_points_data, global_inputs_conditioning):
  """Stock graph construction, except that the edge encoder is deferred to ChunkedDeepGNN.

  The official code encodes all edges up front ([3.1M mesh->grid edges, 32] plus the encoder's
  intermediates). Here the encoder is swapped for a pass-through while the graph is built, so
  the graph carries the raw [edges, batch, 4] spatial features, and ChunkedDeepGNN applies the
  same encoder module (same parameters) block by block.
  """
  gnn = self._typed_graph_gnn
  if OPTIONS.grid_chunk <= 0 or not isinstance(gnn, ChunkedDeepGNN):
    return _ORIG["build_typed_graph"](self, triangular_mesh_data, lat_lon_points_data,
                                      global_inputs_conditioning)
  encoder = self._edge_encoder_dense
  seen = {}

  def passthrough(mapping, norm_conditioning):
    (name, da), = mapping.items()
    da = da.transpose("points", "batch", ...)
    seen.update(name=name, dims=da.dims, coords=_drop_grid_coords(da.coords),
                norm_conditioning=norm_conditioning)
    return jnp.asarray(xarray_jax.unwrap_data(da))

  self._edge_encoder_dense = passthrough
  try:
    graph = _ORIG["build_typed_graph"](self, triangular_mesh_data, lat_lon_points_data,
                                       global_inputs_conditioning)
  finally:
    self._edge_encoder_dense = encoder
  coords = {k: c for k, c in seen["coords"].items() if not set(c.dims) & {"points", "batch"}}

  def encode(raw):
    da = xarray_jax.DataArray(raw, dims=seen["dims"], coords=coords)
    return encoder({seen["name"]: da}, seen["norm_conditioning"])

  gnn.deferred_edge_encoder = encode
  return graph


# ----------------------------------------------------------------------------------------------
# Grid encoder / decoder

GRID_DIMS = ("batch", "lat", "lon")


def _grid_blocks(n_lat, n_lon):
  n = n_lat * n_lon
  if OPTIONS.grid_chunk <= 0 or n <= 2 * OPTIONS.grid_chunk:
    return None
  nb = _num_blocks(n, OPTIONS.grid_chunk)
  return n, nb, -(-n // nb)


def _merge_point_blocks(x, lat_axis, n_lat, n_lon, n):
  """[nb, ..., 1 (lat), block (lon), ...] -> [..., n_lat, n_lon, ...] (blocks are lat-major point ranges)."""
  x = jnp.moveaxis(x, 0, lat_axis + 1)
  shape = x.shape
  x = x.reshape(shape[:lat_axis] + (-1,) + shape[lat_axis + 3:])
  x = jax.lax.slice_in_dim(x, 0, n, axis=lat_axis)
  return x.reshape(shape[:lat_axis] + (n_lat, n_lon) + shape[lat_axis + 3:])


def _drop_grid_coords(coords):
  return {k: c for k, c in coords.items() if not set(c.dims) & set(GRID_DIMS[1:])}


class ChunkedDenseEncoder(_ORIG["Encoder"]):
  """The (pointwise) grid encoder applied to blocks of lat-major grid points, so its
  [points, hidden] intermediates are never materialised for the whole grid."""

  def __call__(self, data_array_mapping, norm_conditioning=None):
    lat, lon = GRID_DIMS[1:]
    full = [da for da in data_array_mapping.values() if lat in da.dims and lon in da.dims]
    geom = None
    if tuple(self._preserved_dims) == GRID_DIMS and full:
      n_lat, n_lon = full[0].sizes[lat], full[0].sizes[lon]
      geom = _grid_blocks(n_lat, n_lon)
    if geom is None:
      return super().__call__(data_array_mapping, norm_conditioning)
    n, nb, block = geom
    flat = {}
    for name, da in data_array_mapping.items():
      if lat not in da.dims and lon not in da.dims:
        continue  # e.g. year_progress: broadcast by the stock encoder
      rest = [d for d in da.dims if d not in (lat, lon)]
      x = jnp.asarray(xarray_jax.unwrap_data(da.transpose(*rest, *[d for d in (lat, lon) if d in da.dims])))
      # Inputs with only one grid dim (e.g. day_progress over lon) are broadcast to the full grid.
      if lat not in da.dims:
        x = x[..., None, :]
      elif lon not in da.dims:
        x = x[..., None]
      x = jnp.broadcast_to(x, x.shape[:-2] + (n_lat, n_lon)).reshape(x.shape[:-2] + (n,))
      x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, nb * block - n)]) if nb * block != n else x
      flat[name] = (x, (*rest, lat, lon), _drop_grid_coords(da.coords))
    stock_call = super().__call__

    def one_block(i):
      mapping = dict(data_array_mapping)
      for name, (x, dims, coords) in flat.items():
        b = jax.lax.dynamic_slice_in_dim(x, i * block, block, axis=-1)
        mapping[name] = xarray_jax.DataArray(b.reshape(b.shape[:-1] + (1, block)), dims=dims, coords=coords)
      return stock_call(mapping, norm_conditioning)

    return _merge_point_blocks(hk.map(one_block, jnp.arange(nb)), 1, n_lat, n_lon, n)


class ChunkedDenseDecoder(_ORIG["Decoder"]):
  """The (pointwise) grid decoder applied to blocks of lat-major grid points."""

  def __call__(self, inputs, output_template):
    lat, lon = GRID_DIMS[1:]
    geom = None
    if (tuple(self._preserved_dims) == GRID_DIMS and inputs.ndim == 4
        and all(lat in da.dims and lon in da.dims and da.dims.index(lon) == da.dims.index(lat) + 1
                for da in output_template.values())):
      geom = _grid_blocks(inputs.shape[1], inputs.shape[2])
    if geom is None:
      return super().__call__(inputs, output_template)
    n, nb, block = geom
    batch, n_lat, n_lon, feat = inputs.shape
    x = _pad_rows(jnp.moveaxis(inputs.reshape(batch, n, feat), 1, 0), nb * block)
    names = list(output_template.keys())
    block_template = {}
    for name in names:
      da = output_template[name]
      shape = tuple(1 if d == lat else block if d == lon else da.sizes[d] for d in da.dims)
      block_template[name] = xr.DataArray(np.zeros(shape, da.dtype), dims=da.dims,
                                          coords=_drop_grid_coords(da.coords))
    if isinstance(output_template, xr.Dataset):
      block_template = xr.Dataset(block_template)
    stock_call = super().__call__

    def one_block(i):
      b = jax.lax.dynamic_slice_in_dim(x, i * block, block, axis=0)
      out = stock_call(jnp.moveaxis(b, 0, 1).reshape(batch, 1, block, feat), block_template)
      return tuple(xarray_jax.unwrap_data(out[name]) for name in names)

    outs = hk.map(one_block, jnp.arange(nb))
    result = {}
    for name, o in zip(names, outs):
      da = output_template[name]
      result[name] = xarray_jax.DataArray(
          _merge_point_blocks(o, da.dims.index(lat), n_lat, n_lon, n), dims=da.dims, coords=da.coords)
    return xr.Dataset(result) if isinstance(output_template, xr.Dataset) else result


# ----------------------------------------------------------------------------------------------
# Install / uninstall

_enabled = False


def install():
  """Swap the patched classes/functions into the weathernext modules (idempotent)."""
  global _enabled
  mesh_transformer._get_adjacency_matrix = _get_adjacency_matrix
  _ORIG["Transformer"].__init__ = _transformer_init
  # Transformer.__call__ looks up `Block` from module globals at call time; mesh_transformer
  # constructs `sparse_transformer.Transformer` at call time.
  st.Block = ChunkedBlock
  st.Transformer = ChunkedTransformer
  # points_mesh_gnn constructs `deep_gnn.DeepGNN(...)` at module-init time; architecture.py
  # constructs the xarray_dense encoder/decoder at call time.
  deep_gnn.DeepGNN = ChunkedDeepGNN
  points_mesh_gnn.PointsMeshTypedGraphGNN._build_typed_graph = _build_typed_graph
  xarray_dense.DataArrayDictDenseEncoder = ChunkedDenseEncoder
  xarray_dense.DataArrayDictDenseDecoder = ChunkedDenseDecoder
  _enabled = True


def uninstall():
  """Restore the original weathernext classes/functions (modules already built keep theirs)."""
  global _enabled
  mesh_transformer._get_adjacency_matrix = _ORIG["get_adjacency_matrix"]
  _ORIG["Transformer"].__init__ = _ORIG["transformer_init"]
  st.Block = _ORIG["Block"]
  st.Transformer = _ORIG["Transformer"]
  deep_gnn.DeepGNN = _ORIG["DeepGNN"]
  points_mesh_gnn.PointsMeshTypedGraphGNN._build_typed_graph = _ORIG["build_typed_graph"]
  xarray_dense.DataArrayDictDenseEncoder = _ORIG["Encoder"]
  xarray_dense.DataArrayDictDenseDecoder = _ORIG["Decoder"]
  _enabled = False


def is_enabled() -> bool:
  return _enabled
