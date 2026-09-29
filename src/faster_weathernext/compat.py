"""Startup checks: the patch relies on internals of the official `weathernext` package."""

import importlib
import inspect
import warnings

# Versions the equivalence results in docs/validation.md were produced with.
TESTED_JAX = "0.11.2"
TESTED_WEATHERNEXT_COMMIT = "f2f2c51"  # github.com/google-deepmind/weathernext, 2026-09-04

# (module, attribute[, required parameters of the callable]) the patch reads or replaces.
_REQUIRED = [
    ("weathernext.utils.sparse_transformer", "Transformer",
     ("adj_mat", "attention_k_hop", "attention_type", "mask_type")),
    ("weathernext.utils.sparse_transformer", "Block", ()),
    ("weathernext.utils.sparse_transformer", "_ModelConfig", ()),
    ("weathernext.utils.sparse_transformer", "multihead_linear", ()),
    ("weathernext.utils.sparse_transformer", "layernorm", ()),
    ("weathernext.utils.sparse_transformer", "ffw", ()),
    ("weathernext.utils.sparse_transformer_utils", "wrap_fn_for_upcast_downcast", ()),
    ("weathernext.utils.deep_gnn", "DeepGNN", ()),
    ("weathernext.utils.typed_graph_net", "InteractionNetwork", ()),
    ("weathernext.utils.points_mesh_gnn", "PointsMeshTypedGraphGNN", ()),
    ("weathernext.utils.points_mesh_gnn", "MESH_NODES_NAME", ()),
    ("weathernext.utils.points_mesh_gnn", "POINT_NODES_NAME", ()),
    ("weathernext.utils.xarray_dense", "DataArrayDictDenseEncoder", ()),
    ("weathernext.utils.xarray_dense", "DataArrayDictDenseDecoder", ()),
    ("weathernext.utils.mesh_transformer", "_get_adjacency_matrix",
     ("triangular_mesh_data", "add_self_edges")),
    ("weathernext.utils.gather_scatter_ops", "gather_with_fill", ()),
    ("weathernext.utils.dense", "LinearNormConditioning", ()),
    ("haiku", "map", ()),
    ("haiku", "scan", ()),
    ("haiku", "switch", ()),
]
_REQUIRED_METHODS = [
    ("weathernext.utils.deep_gnn", "DeepGNN", "_networks_builder"),
    ("weathernext.utils.points_mesh_gnn", "PointsMeshTypedGraphGNN", "_build_typed_graph"),
]


def problems() -> list[str]:
  """Everything that looks incompatible with this version of faster-weathernext (empty list = fine)."""
  out = []
  for mod_name, attr, params in _REQUIRED:
    try:
      mod = importlib.import_module(mod_name)
    except ImportError as e:
      out.append(f"cannot import {mod_name}: {e}")
      continue
    obj = getattr(mod, attr, None)
    if obj is None:
      out.append(f"{mod_name}.{attr} is missing")
      continue
    if params:
      target = obj.__init__ if inspect.isclass(obj) else obj
      try:
        names = set(inspect.signature(target).parameters)
      except (TypeError, ValueError):
        continue
      missing = [p for p in params if p not in names]
      if missing:
        out.append(f"{mod_name}.{attr} no longer takes {missing}")
  for mod_name, cls, meth in _REQUIRED_METHODS:
    try:
      if not hasattr(getattr(importlib.import_module(mod_name), cls), meth):
        out.append(f"{mod_name}.{cls}.{meth} is missing")
    except ImportError:
      pass
  return out


def check(force: bool = False) -> None:
  """Raise (or warn, with force=True) if the installed weathernext/JAX look incompatible."""
  import jax
  if jax.__version__ != TESTED_JAX:
    warnings.warn(f"faster-weathernext was validated with JAX {TESTED_JAX}; found {jax.__version__}. "
                  "Re-run scripts/verify_equivalence.py before relying on results.", stacklevel=3)
  found = problems()
  if found:
    msg = ("faster-weathernext patches internals of the official `weathernext` package (tested at commit "
           f"{TESTED_WEATHERNEXT_COMMIT}); the installed version looks incompatible:\n  - "
           + "\n  - ".join(found))
    if not force:
      raise RuntimeError(msg + "\nPass force=True to enable() anyway.")
    warnings.warn(msg, stacklevel=3)
