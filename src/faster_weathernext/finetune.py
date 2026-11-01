"""Fine-tuning of WeatherNext 2 (prototype).

By default only the "head" is trained: everything after the mesh transformer, i.e. the mesh->grid
GNN (`mesh_to_grid_gnn`) and the grid decoder (`grid_decoder`), ~5.6M of WeatherNext 2's 184M
parameters. Gradients are taken with respect to the trained modules only, so JAX never
differentiates a frozen prefix of the model: with the head alone, the encoder, grid->mesh GNN and
transformer run as in inference. `split(params, "all")` (or any set of module names, e.g.
("mesh_transformer", *HEAD_MODULES)) trains more; the backward pass then runs through the fused
attention kernel's backward and needs `enable(remat=True)` to fit.

The loss is the official training loss: the checkpoint config's full predictor stack (residual
normalisation, NaN cleaning, autoregressive wrapper, two-member sample dimension) and FGN's fair
CRPS with the config's per-variable weights. Target variables the data cannot provide (or should
not be trained on) are masked out of the loss (`predictor_config`).

    import faster_weathernext, optax
    from faster_weathernext import finetune, ifs, model

    faster_weathernext.enable(remat=True)   # remat: blocked GNN/decoder rematerialised in backward
    task, _ = model.build("WeatherNext2")
    inputs, targets, forcings = ifs.training_example(init, task)
    _, loss_fn = finetune.build_loss("WeatherNext2", masked=finetune.all_nan_vars(targets))
    head, frozen = finetune.split(model.load_params("WeatherNext2", 1))
    optimizer = optax.adamw(1e-5)
    opt_state = optimizer.init(head)
    step = finetune.train_step(loss_fn, optimizer)
    head, opt_state, metrics = step(head, frozen, opt_state, rng, inputs, targets, forcings)
    finetune.save("head.npz", finetune.merge(frozen, head), model="WeatherNext2", checkpoint=1)

Fine-tuned weights are a different model: the equivalence claims in docs/validation.md cover how
faster-weathernext executes a checkpoint, not the skill of a fine-tuned one.
"""

import copy
import os

import haiku as hk
import jax
import numpy as np

HEAD_MODULES = ("mesh_to_grid_gnn", "grid_decoder")


def is_head(module_name: str, modules=HEAD_MODULES) -> bool:
  """Whether a Haiku module path (e.g. 'multimodality_forward/grid_decoder/...') is trained:
  one of its path components is in `modules`, or modules == "all"."""
  return modules == "all" or bool(set(module_name.split("/")) & set(modules))


def split(params, modules=HEAD_MODULES):
  """(trained, frozen) partition of a parameter tree by module path (see is_head)."""
  return hk.data_structures.partition(lambda m, n, v: is_head(m, modules), params)


def merge(*parts):
  return hk.data_structures.merge(*parts)


def all_nan_vars(targets) -> list:
  """Target variables that are NaN everywhere (no data): pass them as `masked` to build_loss."""
  return [v for v, da in targets.data_vars.items() if np.isnan(np.asarray(da.values)).all()]


def predictor_config(model: str = "WeatherNext2", *, masked=(), num_samples: int = 2,
                     attention_type: str = "chunked_mha", mask_type: str | None = None):
  """The checkpoint's training PredictorConfig, with `masked` target variables out of the loss.

  A masked variable gets loss weight 0 and is listed in `vars_with_nan_targets`. Both are needed:
  the official NaN mechanism only zeroes the loss where a target is NaN (it handles the all-NaN
  cyclone channels, but not a variable with data, such as SST, whose land NaNs NaNCleaner fills
  anyway), and a zero weight alone would give 0 * NaN = NaN for NaN targets.
  """
  from weathernext.utils import ensemble
  from weathernext.utils import fiddle_config_io
  from weathernext.weathernext2 import fgn

  config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{model}")
  pk = copy.deepcopy(config.predictor_kwargs)
  tk = pk["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
  tk["attention_type"] = attention_type
  if mask_type is not None:
    tk["mask_type"] = mask_type
  unknown = set(masked) - set(config.task.target_variables)
  if unknown:
    raise ValueError(f"masked variables are not targets of {model}: {sorted(unknown)}")
  loss_kwargs = pk["fgn_loss_function_kwargs"]
  loss_kwargs["vars_with_nan_targets"] = sorted(set(loss_kwargs["vars_with_nan_targets"]) | set(masked))
  loss_kwargs["loss_for_nan_targets"] = 0.0
  loss_kwargs["per_variable_weights"] = {**loss_kwargs.get("per_variable_weights", {}),
                                         **{v: 0.0 for v in masked}}
  wrappers = []
  for w in config.predictor_wrappers:
    if w["constructor"] is ensemble.WithSampleDim:
      w = dict(w, kwargs=dict(w.get("kwargs", {}), num_samples=num_samples))
    wrappers.append(w)
  return fgn.PredictorConfig(task=config.task, predictor_constructor=config.predictor_constructor,
                             predictor_kwargs=pk, predictor_wrappers=wrappers)


def build_loss(model: str = "WeatherNext2", *, masked=(), num_samples: int = 2,
               attention_type: str = "chunked_mha", mask_type: str | None = None):
  """(task, loss_fn) with loss_fn(params, rng, inputs, targets, forcings) -> (loss, per-variable losses).

  `loss` is the official training loss averaged over the batch, without the `masked` target
  variables (see predictor_config; their targets may be NaN). The per-variable losses are
  unweighted and include masked variables. `attention_type`/`mask_type` are the official
  transformer settings, as in model.build (with the patch enabled every attention type runs through
  faster-weathernext's kernel; the official GPU path is attention_type="triblockdiag_mha" with the
  patch disabled).
  """
  import xarray_jax
  from weathernext.weathernext2 import fgn

  cfg = predictor_config(model, masked=masked, num_samples=num_samples,
                         attention_type=attention_type, mask_type=mask_type)

  @hk.transform
  def run(inputs, targets, forcings):
    loss, diagnostics = fgn.construct_predictor(cfg).loss(inputs, targets, forcings)
    per_var = {k: xarray_jax.unwrap_data(v).mean() for k, v in diagnostics.items()}
    return xarray_jax.unwrap_data(loss).mean(), per_var

  return cfg.task, run.apply


def loss_and_grads(loss_fn):
  """jit(head, frozen, rng, inputs, targets, forcings) -> ((loss, per-variable), head grads)."""

  def fn(head, frozen, rng, inputs, targets, forcings):
    return jax.value_and_grad(
        lambda h: loss_fn(merge(frozen, h), rng, inputs, targets, forcings), has_aux=True)(head)

  return jax.jit(fn)


def train_step(loss_fn, optimizer):
  """jit(head, frozen, opt_state, rng, inputs, targets, forcings) -> (head, opt_state, metrics).

  `head` and `opt_state` are donated. metrics: loss, grad_norm and the per-variable losses.
  """
  import optax

  def step(head, frozen, opt_state, rng, inputs, targets, forcings):
    (loss, per_var), grads = jax.value_and_grad(
        lambda h: loss_fn(merge(frozen, h), rng, inputs, targets, forcings), has_aux=True)(head)
    updates, opt_state = optimizer.update(grads, opt_state, head)
    head = optax.apply_updates(head, updates)
    return head, opt_state, dict(loss=loss, grad_norm=optax.global_norm(grads), per_var=per_var)

  return jax.jit(step, donate_argnums=(0, 2))


def save(path: str, params, *, model: str = "WeatherNext2", checkpoint: int = 1,
         weights_dir: str | None = None, note: str = ""):
  """Write params as an official-format checkpoint (loads with the stock weathernext code).

  Description and license are taken from the base checkpoint; `note` is appended to the
  description. Parameter names and shapes must match the base checkpoint.
  """
  from weathernext.utils import checkpoint as ckpt_lib
  from weathernext.weathernext2 import fgn
  from faster_weathernext.model import weights_path

  with open(weights_path(model, checkpoint, weights_dir), "rb") as f:
    base = ckpt_lib.load(f, fgn.CheckPoint)
  host = jax.device_get(params)
  out = {m: {k: np.asarray(v) for k, v in host[m].items()} for m in host}
  ref = {(m, k): v.shape for m, p in base.params.items() for k, v in p.items()}
  got = {(m, k): v.shape for m, p in out.items() for k, v in p.items()}
  if ref != got:
    diff = sorted(set(ref.items()) ^ set(got.items()))[:5]
    raise ValueError(f"params do not match {model} checkpoint {checkpoint}: {diff}")
  description = base.description + (f"\n\n{note}" if note else "")
  with open(path + ".tmp", "wb") as f:
    ckpt_lib.dump(f, fgn.CheckPoint(params=out, description=description, license=base.license))
  os.replace(path + ".tmp", path)


def load(path: str):
  """Params of a checkpoint written by `save` (or any official-format checkpoint file)."""
  from weathernext.utils import checkpoint as ckpt_lib
  from weathernext.weathernext2 import fgn
  with open(path, "rb") as f:
    return ckpt_lib.load(f, fgn.CheckPoint).params


def head_size(params, modules=HEAD_MODULES) -> int:
  head, _ = split(params, modules)
  return int(sum(np.size(v) for v in jax.tree.leaves(head)))
