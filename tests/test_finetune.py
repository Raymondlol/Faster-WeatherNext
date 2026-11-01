"""CPU tests: head/frozen parameter split, remat of the blocked scans, checkpoint save/load."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from faster_weathernext import finetune
from faster_weathernext import patch

PARAMS = {
    "multimodality_forward/grid_encoder/shared_dense/mlp/~/linear_0": {"w": np.ones((3, 4), np.float32)},
    "multimodality_forward/mesh_transformer/transformer/block_00/mha_final": {"w": np.ones((4, 4), np.float32)},
    "multimodality_forward/mesh_to_grid_gnn/deep_gnn/processor_nodes_0_point_nodes": {"b": np.ones(4, np.float32)},
    "multimodality_forward/grid_decoder/split_output_linear": {"w_2m_temperature": np.ones((4, 1), np.float32)},
    "multimodality_forward/grid_to_mesh_gnn/deep_gnn/processor_nodes_0_mesh_nodes": {"b": np.ones(4, np.float32)},
}


def test_split_selects_mesh_to_grid_and_decoder_only():
  head, frozen = finetune.split(PARAMS)
  assert sorted(m.split("/")[1] for m in head) == ["grid_decoder", "mesh_to_grid_gnn"]
  assert not any(finetune.is_head(m) for m in frozen)
  assert finetune.merge(frozen, head).keys() == PARAMS.keys()
  assert finetune.head_size(PARAMS) == 4 + 4
  trained, frozen = finetune.split(PARAMS, "all")
  assert trained.keys() == PARAMS.keys() and not frozen
  trained, _ = finetune.split(PARAMS, ("mesh_transformer", *finetune.HEAD_MODULES))
  assert len(trained) == 3


def test_masked_variables_get_zero_weight_and_nan_masking():
  pytest.importorskip("weathernext")
  cfg = finetune.predictor_config("WeatherNext2", masked=["sea_surface_temperature", "cyclone_usa_wind_disc"])
  kw = cfg.predictor_kwargs["fgn_loss_function_kwargs"]
  # A zero weight is what removes a variable that has data (SST); the NaN list alone would not.
  assert kw["per_variable_weights"]["sea_surface_temperature"] == 0.0
  assert kw["per_variable_weights"]["2m_temperature"] == 1.0  # others keep the config's weights
  assert {"sea_surface_temperature", "cyclone_usa_wind_disc"} <= set(kw["vars_with_nan_targets"])
  with pytest.raises(ValueError, match="not targets"):
    finetune.predictor_config("WeatherNext2", masked=["not_a_variable"])


@pytest.fixture
def remat_option():
  saved = patch.OPTIONS.remat
  yield
  patch.OPTIONS.remat = saved


def test_remat_leaves_values_and_grads_unchanged(remat_option):
  def f(blocks, xs):
    lin = hk.Linear(5, name="lin")  # built outside the scan, as ChunkedDeepGNN captures its modules

    def fn(b, x):
      return jnp.tanh(lin(b) + x)
    return jnp.sum(patch._update_blocks_in_place(fn, blocks, (xs,)) ** 2)

  model = hk.transform(f)
  blocks = jnp.asarray(np.random.default_rng(0).standard_normal((4, 3, 5)), jnp.float32)
  xs = jnp.asarray(np.random.default_rng(1).standard_normal((4, 3, 5)), jnp.float32)
  params = model.init(jax.random.PRNGKey(0), blocks, xs)
  results = {}
  for remat in (False, True):
    patch.OPTIONS.remat = remat
    results[remat] = jax.value_and_grad(lambda p, b: model.apply(p, None, b, xs), argnums=(0, 1))(params, blocks)
  (v0, (gp0, gb0)), (v1, (gp1, gb1)) = results[False], results[True]
  np.testing.assert_allclose(v0, v1, rtol=1e-6)
  np.testing.assert_allclose(gb0, gb1, rtol=1e-5, atol=1e-6)
  for a, b in zip(jax.tree.leaves(gp0), jax.tree.leaves(gp1)):
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_remat_is_in_the_jaxpr_only_when_enabled(remat_option):
  def f(blocks):
    lin = hk.Linear(5, name="lin")
    return patch._update_blocks_in_place(lin, blocks)

  model = hk.transform(f)
  blocks = jnp.ones((2, 3, 5), jnp.float32)
  params = model.init(jax.random.PRNGKey(0), blocks)
  for remat in (False, True):
    patch.OPTIONS.remat = remat
    jaxpr = str(jax.make_jaxpr(lambda p: jax.grad(lambda q: model.apply(q, None, blocks).sum())(p))(params))
    assert ("checkpoint" in jaxpr or "remat" in jaxpr) == remat


def test_save_roundtrip_and_shape_check(tmp_path, monkeypatch):
  from weathernext.utils import checkpoint as ckpt_lib
  from weathernext.weathernext2 import fgn
  from faster_weathernext import model

  base = tmp_path / "base.npz"
  with open(base, "wb") as f:
    ckpt_lib.dump(f, fgn.CheckPoint(params=PARAMS, description="base", license="CC BY 4.0"))
  monkeypatch.setattr(model, "weights_path", lambda *a, **k: str(base))

  head, frozen = finetune.split(PARAMS)
  head = jax.tree.map(lambda v: v * 2, head)
  out = tmp_path / "tuned.npz"
  finetune.save(str(out), finetune.merge(frozen, head), note="head-only test")
  loaded = finetune.load(str(out))
  for m, p in finetune.merge(frozen, head).items():
    for k, v in p.items():
      np.testing.assert_array_equal(loaded[m][k], v)
  with open(out, "rb") as f:
    ck = ckpt_lib.load(f, fgn.CheckPoint)
  assert ck.description == "base\n\nhead-only test" and ck.license == "CC BY 4.0"

  bad = finetune.merge(frozen, jax.tree.map(lambda v: v[..., :1], head))
  with pytest.raises(ValueError, match="do not match"):
    finetune.save(str(tmp_path / "bad.npz"), bad)


def test_take_slices_one_stacked_member_and_keeps_static_fields():
  import xarray
  import xarray_jax
  a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)  # batch, sample, lat, lon
  ds = xarray.Dataset({"x": (("batch", "sample", "lat", "lon"), a),
                       "static": (("lat", "lon"), np.ones((3, 4), np.float32))},
                      coords={"lat": np.arange(3.0), "lon": np.arange(4.0)})
  st = ds.rename({"batch": "_batch"}).stack(batch=("_batch", "sample")).transpose("batch", ...)

  def f(i):
    one = patch._take(st, i)
    assert dict(one["x"].sizes) == {"batch": 1, "lat": 3, "lon": 4} and "batch" not in one["static"].dims
    return xarray_jax.unwrap_data(one["x"])

  np.testing.assert_array_equal(jax.jit(f)(1)[0], a[0, 1])
  np.testing.assert_array_equal(jax.jit(f)(0)[0], a[0, 0])
