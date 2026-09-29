"""faster-weathernext: run WeatherNext 2 at full 0.25° resolution on consumer GPUs (unofficial).

    import faster_weathernext
    faster_weathernext.enable()          # patch the official `weathernext` code, then use it as usual

`enable()` changes how the official modules execute (memory-blocked GNNs, fused Pallas
attention, ...), not what they compute; checkpoints and parameter names are unchanged.
Not an official Google product.
"""

__version__ = "0.1.1"


def enable(force: bool = False, **options):
  """Patch the official weathernext modules for low-memory, fast GPU execution.

  Call before the model is first traced (e.g. before the first call of a jitted forward).
  Keyword options update `faster_weathernext.patch.Options` (attention, reorder, strict_fp32,
  grid_chunk, edge_chunk, attn_chunk, layer_loop, replace_attention).
  Returns the active Options.
  """
  from faster_weathernext import compat
  compat.check(force=force)
  from faster_weathernext import patch
  for key, value in options.items():
    if not hasattr(patch.OPTIONS, key):
      raise TypeError(f"unknown option {key!r}; see faster_weathernext.patch.Options")
    setattr(patch.OPTIONS, key, value)
  patch.install()
  return patch.OPTIONS


def disable():
  """Restore the original weathernext modules (already-built models keep the patched classes)."""
  from faster_weathernext import patch
  patch.uninstall()


def options():
  from faster_weathernext import patch
  return patch.OPTIONS
