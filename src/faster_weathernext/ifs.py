"""WeatherNext 2 inputs from ECMWF IFS open data (0.25°, 13 pressure levels).

Only the needed GRIB messages are downloaded (index-driven HTTP range requests) and cached.
Files come from the first mirror in `MIRRORS` (or `FWN_IFS_MIRRORS`, comma-separated) that serves them.
Known approximation: IFS open data has no SST, so skin temperature over the ocean
(land-sea mask < 0.5) is used, clamped at the sea-ice freezing point (271.46 K).
Static fields (orography, land-sea mask) are taken from Google's 0.25° sample file, exactly
as the model saw them in training (read by HTTP range request, ~8 MB, then cached).

ECMWF open data is CC BY 4.0: "Copyright ECMWF"; ECMWF does not accept any liability
for errors or omissions in the data.
"""

import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import xarray

# ECMWF open-data mirrors with identical paths and files, tried in this order. The AWS bucket
# throttles a freshly published run (503 SlowDown on the whole prefix while demand peaks; ECMWF
# points users to the mirrors), and a mirror can lag in publishing a run (404).
MIRRORS = ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
           "https://storage.googleapis.com/ecmwf-open-data",
           "https://data.ecmwf.int/forecasts")
_last_good = None  # the mirror that served the previous file; tried first for the next one
STATIC_URL = ("https://storage.googleapis.com/dm_graphcast/weathernext2/dataset/"
              + urllib.parse.quote("source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-01.nc"))
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
G = 9.80665
# WN2 name -> IFS shortName
PL_VARS = {"temperature": "t", "geopotential": "gh", "u_component_of_wind": "u",
           "v_component_of_wind": "v", "vertical_velocity": "w", "specific_humidity": "q"}
SFC_VARS = {"2m_temperature": "2t", "mean_sea_level_pressure": "msl",
            "10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v",
            "100m_u_component_of_wind": "100u", "100m_v_component_of_wind": "100v",
            "sea_surface_temperature": "skt"}
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")
SEA_ICE_SST = 271.46  # K; HRES/ERA5 SST under sea ice sits at the freezing point.


def cache_dir(*parts) -> str:
  path = os.path.join(os.environ.get("FWN_CACHE", os.path.expanduser("~/.cache/faster-weathernext")), *parts)
  os.makedirs(path, exist_ok=True)
  return path


def _url(base: str, run: dt.datetime, step: int, ext: str) -> str:
  d, h = run.strftime("%Y%m%d"), run.strftime("%H")
  return f"{base}/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-{step}h-oper-fc.{ext}"


def _get(url: str, byte_range=None, retries: int = 4) -> bytes:
  req = urllib.request.Request(url)
  if byte_range:
    req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
  for attempt in range(retries):
    try:
      with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
      # Ride out a short 503 burst (~10 s), then let the caller fail over to the next mirror:
      # a throttled prefix can stay throttled for hours.
      if attempt == retries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code not in (500, 503)):
        raise
      time.sleep(min(2 ** attempt, 8) * (0.5 + random.random()))


def _mirrors() -> list:
  env = os.environ.get("FWN_IFS_MIRRORS")
  mirrors = [m.strip().rstrip("/") for m in env.split(",") if m.strip()] if env else list(MIRRORS)
  if _last_good in mirrors:
    mirrors.remove(_last_good)
    mirrors.insert(0, _last_good)
  return mirrors


def _download_from(base: str, run: dt.datetime, step: int, wanted: set) -> bytes:
  index = [json.loads(line) for line in _get(_url(base, run, step, "index")).decode().splitlines()]
  recs = [r for r in index if (r["param"], r["levtype"], int(r.get("levelist", 0))) in wanted]
  missing = wanted - {(r["param"], r["levtype"], int(r.get("levelist", 0))) for r in recs}
  if missing:
    raise KeyError(f"IFS open data {run:%Y-%m-%d %HZ} +{step}h is missing {sorted(missing)}")
  recs.sort(key=lambda r: r["_offset"])
  url = _url(base, run, step, "grib2")
  with concurrent.futures.ThreadPoolExecutor(6) as ex:
    futures = [ex.submit(_get, url, (r["_offset"], r["_offset"] + r["_length"] - 1)) for r in recs]
    try:
      parts = [f.result() for f in futures]
    except BaseException:
      for f in futures:  # fail over now instead of requesting the rest from this mirror
        f.cancel()
      raise
  for r, part in zip(recs, parts):
    if len(part) != r["_length"]:  # e.g. a server that ignores Range and sends the whole file
      raise OSError(f"{url}: got {len(part)} bytes for a {r['_length']}-byte range")
  return b"".join(parts)


def _download(run: dt.datetime, step: int, wanted: set) -> bytes:
  """The wanted GRIB messages of one IFS file, from the first mirror that serves all of them."""
  global _last_good
  errors = []
  mirrors = _mirrors()
  for i, base in enumerate(mirrors):
    try:
      data = _download_from(base, run, step, wanted)
    except (OSError, ValueError, http.client.HTTPException) as e:
      errors.append(f"{base}: {e}")
      if i + 1 < len(mirrors):
        print(f"IFS open data {run:%Y-%m-%d %HZ} +{step}h: {base} failed ({e}); trying {mirrors[i + 1]}",
              flush=True)
      continue
    _last_good = base
    return data
  raise RuntimeError(f"IFS open data {run:%Y-%m-%d %HZ} +{step}h: no mirror served it (a run is published "
                     "~7-8 h after its init time)\n  " + "\n  ".join(errors))


def fetch_fields(run: dt.datetime, step: int, wanted: set) -> dict:
  """{(shortName, level_or_0): 2D array (lat ascending, lon from 0)} for wanted (param, levtype, level)."""
  import eccodes  # imported lazily: importing eccodes before pyproj corrupts the heap at exit
  tag = f"{run:%Y%m%d%H}-{step}h-" + hashlib.md5(repr(sorted(wanted)).encode()).hexdigest()[:8]
  path = os.path.join(cache_dir("ifs"), tag + ".grib2")
  if not os.path.exists(path):
    data = _download(run, step, wanted)
    with open(path + ".tmp", "wb") as f:
      f.write(data)
    os.replace(path + ".tmp", path)
  out = {}
  with open(path, "rb") as f:
    while (h := eccodes.codes_grib_new_from_file(f)) is not None:
      try:
        name = eccodes.codes_get(h, "shortName")
        level = eccodes.codes_get(h, "level") if eccodes.codes_get(h, "typeOfLevel") == "isobaricInhPa" else 0
        nj, ni = eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        vals = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
      finally:
        eccodes.codes_release(h)
      assert (nj, ni) == (721, 1440), (nj, ni)
      if lat0 > 0:  # north-to-south -> south-to-north
        vals = vals[::-1]
      shift = int(round((lon0 % 360) / 0.25))  # first column sits at lon0; roll so col 0 is lon=0
      out[(name, level)] = np.roll(vals, shift, axis=1)
  return out


def static_fields() -> dict:
  """Orography and land-sea mask exactly as in Google's 0.25° sample data (cached)."""
  path = os.path.join(cache_dir(), "static_0p25.nc")
  if not os.path.exists(path):
    import fsspec
    with fsspec.open(STATIC_URL, "rb", block_size=2**22) as f, \
        xarray.open_dataset(f, engine="h5netcdf") as ref:
      ref[list(STATIC_VARS)].load().to_netcdf(path + ".tmp")
    os.replace(path + ".tmp", path)
  with xarray.open_dataset(path) as ds:
    return {name: ds[name].values.astype(np.float32) for name in STATIC_VARS}


def analysis_state(t: dt.datetime) -> dict:
  """IFS step-0 fields at time t, mapped to WN2 variable names (no batch/time dims)."""
  wanted = {(p, "pl", lev) for p in PL_VARS.values() for lev in LEVELS}
  wanted |= {(p, "sfc", 0) for p in SFC_VARS.values()} | {("lsm", "sfc", 0)}
  f = fetch_fields(t, 0, wanted)
  state = {}
  for name, p in PL_VARS.items():
    arr = np.stack([f[(p, lev)] for lev in LEVELS])
    state[name] = arr * G if p == "gh" else arr
  for name, p in SFC_VARS.items():
    state[name] = f[(p, 0)]
  sst = np.maximum(state["sea_surface_temperature"], SEA_ICE_SST)
  state["sea_surface_temperature"] = np.where(f[("lsm", 0)] < 0.5, sst, np.nan).astype(np.float32)
  state.update(static_fields())
  return state


def build_example(init: dt.datetime, n_steps: int, target_vars) -> xarray.Dataset:
  """WN2-format dataset: 2 input times (init-6h, init) + n_steps NaN target times."""
  times = [init - dt.timedelta(hours=6)] + [init + dt.timedelta(hours=6 * i) for i in range(n_steps + 1)]
  lat = np.linspace(-90, 90, 721, dtype=np.float32)
  lon = np.arange(1440, dtype=np.float32) * 0.25
  states = [analysis_state(t) for t in times[:2]]
  nt = len(times)
  data = {}
  for name in list(PL_VARS) + list(SFC_VARS):
    shape = states[0][name].shape
    arr = np.full((1, nt) + shape, np.nan, np.float32)
    arr[0, 0], arr[0, 1] = states[0][name], states[1][name]
    dims = ("batch", "time", "level", "lat", "lon") if len(shape) == 3 else ("batch", "time", "lat", "lon")
    data[name] = (dims, arr)
  for name in STATIC_VARS:
    data[name] = (("lat", "lon"), states[1][name])
  for name in target_vars:
    if name not in data:  # target-only fields (precip, cyclone channels) are NaN templates
      data[name] = (("batch", "time", "lat", "lon"), np.full((1, nt, 721, 1440), np.nan, np.float32))
  return xarray.Dataset(
      data,
      coords=dict(time=pd.to_timedelta(np.arange(nt) * 6, unit="h"), lat=lat, lon=lon,
                  level=np.array(LEVELS, np.int32),
                  datetime=(("batch", "time"), np.array([times], dtype="datetime64[ns]"))))


def rollout_inputs(init: dt.datetime, n_steps: int, task):
  """(inputs, targets_template, forcings) for an n_steps rollout from IFS analyses at init-6h, init.

  The target template is a broadcast NaN array and the forcings are derived from the valid times,
  so an N-step rollout does not materialise N steps of input fields.
  """
  import dataclasses
  from weathernext.utils import data_utils
  ds3 = build_example(init, 1, task.target_variables)
  inputs, targets1, forcings1 = data_utils.extract_inputs_targets_forcings(
      ds3, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))
  lead = pd.to_timedelta(np.arange(1, n_steps + 1) * 6, unit="h")
  tvars = {}
  for v, da in targets1.data_vars.items():
    shape = list(da.shape)
    shape[da.dims.index("time")] = n_steps
    tvars[v] = (da.dims, np.broadcast_to(np.float32(np.nan), shape))
  coords = {k: c for k, c in targets1.coords.items() if k != "time"}
  targets = xarray.Dataset(tvars, coords={**coords, "time": lead})
  fds = xarray.Dataset(coords=dict(
      time=lead, lon=ds3.lon,
      datetime=(("batch", "time"), np.array([[init + td for td in lead.to_pytimedelta()]], dtype="datetime64[ns]"))))
  data_utils.add_derived_vars(fds)
  forcings = fds[list(task.forcing_variables)].drop_vars("datetime")
  forcings = forcings.astype({v: forcings1[v].dtype for v in forcings1.data_vars})
  # The hand-built forcings must equal what the library derives for the first step.
  for v in forcings1.data_vars:
    np.testing.assert_allclose(forcings[v].isel(time=0).values, forcings1[v].isel(time=0).values, atol=1e-6)
  return inputs.compute(), targets, forcings


def truth_fields(t: dt.datetime) -> dict:
  """Verification at valid time t: IFS analysis msl/10u/10v and the 0-6h IFS precip ending at t."""
  f = fetch_fields(t, 0, {("msl", "sfc", 0), ("10u", "sfc", 0), ("10v", "sfc", 0)})
  tp = fetch_fields(t - dt.timedelta(hours=6), 6, {("tp", "sfc", 0)})
  return {"mean_sea_level_pressure": f[("msl", 0)], "10m_u_component_of_wind": f[("10u", 0)],
          "10m_v_component_of_wind": f[("10v", 0)], "total_precipitation_6hr": tp[("tp", 0)]}
