"""Load WeatherNext 2 checkpoints (from Google's public bucket) and build a jitted forward step."""

import functools
import os
import urllib.parse
import urllib.request

WEIGHTS_URL = "https://storage.googleapis.com/dm_graphcast/weathernext2/params/"
# model name -> weight file for checkpoint k (the <2025 releases; 4 independently trained runs).
MODELS = {
    "WeatherNext2": "WeatherNext2_<2025_model{k}.npz",
    "WeatherNextCyclones": "WeatherNextCyclones_<2025_model{k}.npz",
    "WeatherNextCyclones_Mini": "WeatherNextCyclones_Mini_<2024.npz",
}


def weights_path(model: str = "WeatherNext2", checkpoint: int = 1, weights_dir: str | None = None) -> str:
  """Local path of a checkpoint, downloading it from Google's bucket if needed (~735 MB each)."""
  from faster_weathernext.ifs import cache_dir
  name = MODELS[model].format(k=checkpoint)
  path = os.path.join(weights_dir or cache_dir("weights"), name)
  if not os.path.exists(path):
    url = WEIGHTS_URL + urllib.parse.quote(name)
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)
  return path


def load_params(model: str = "WeatherNext2", checkpoint: int = 1, weights_dir: str | None = None):
  from weathernext.utils import checkpoint as ckpt_lib
  from weathernext.weathernext2 import fgn
  with open(weights_path(model, checkpoint, weights_dir), "rb") as f:
    return ckpt_lib.load(f, fgn.CheckPoint).params


def build(model: str = "WeatherNext2", *, attention_type: str = "chunked_mha", mask_type: str | None = None):
  """(task, forward) with forward(params, rng, inputs, targets_template, forcings) -> predictions.

  Call faster_weathernext.enable() first. As in the official demo notebook, the training-time ensemble
  wrapper (WithSampleDim) is dropped: one call = one ensemble member (the batch dimension is free).
  `attention_type` / `mask_type` are the official transformer settings; with the patch enabled any
  attention type runs through faster-weathernext's kernel. The official GPU path is
  `attention_type="triblockdiag_mha"` with the patch disabled (used by the validation scripts).
  """
  import haiku as hk
  import jax
  from weathernext.utils import fiddle_config_io
  from weathernext.weathernext2 import fgn

  config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{model}")
  transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords[
      "transformer_kwargs"]
  transformer_kwargs["attention_type"] = attention_type
  if mask_type is not None:
    transformer_kwargs["mask_type"] = mask_type
  cfg = fgn.PredictorConfig(task=config.task, predictor_constructor=config.predictor_constructor,
                            predictor_kwargs=config.predictor_kwargs,
                            predictor_wrappers=config.predictor_wrappers[:-1])

  @hk.transform
  def run(inputs, targets_template, forcings):
    return fgn.construct_predictor(cfg)(inputs, targets_template=targets_template, forcings=forcings)

  # Params are an argument, so all checkpoints of one model share one compilation.
  forward = jax.jit(lambda params, rng, inputs, targets_template, forcings:
                    run.apply(params, rng, inputs, targets_template, forcings))
  return config.task, forward


def rollout(forward, params, rng, inputs, targets_template, forcings):
  """Yields one prediction (xarray.Dataset, on host) per 6 h step, autoregressively."""
  from weathernext.utils import rollout as rollout_lib
  yield from rollout_lib.chunked_prediction_generator(
      functools.partial(forward, params), rng, inputs, targets_template,
      num_steps_per_chunk=1, forcings=forcings)
