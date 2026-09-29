"""CPU tests: IFS open-data mirror fallback (network replaced by a fake `_get`)."""
import datetime as dt
import io
import json
import urllib.error

import pytest

from faster_weathernext import ifs

RUN = dt.datetime(2026, 9, 29, 12)
WANTED = {("msl", "sfc", 0), ("t", "pl", 500)}
GRIB = b"AAAAmslBBBBBBt500CCC"  # stand-in file: 'msl' at bytes 4-6, 't500' at 13-16
INDEX = "\n".join(json.dumps(r) for r in [
    {"param": "msl", "levtype": "sfc", "_offset": 4, "_length": 3},
    {"param": "t", "levtype": "pl", "levelist": "500", "_offset": 13, "_length": 4},
    {"param": "u", "levtype": "pl", "levelist": "500", "_offset": 17, "_length": 3},
]).encode()
A, B, C = "https://a.example", "https://b.example", "https://c.example"


def _http_error(url, code):
  return urllib.error.HTTPError(url, code, "fake", {}, io.BytesIO())


@pytest.fixture
def fake(monkeypatch):
  """behaviour[base] = 'ok' | 'short' | an HTTP status code to fail with; returns the request log."""
  behaviour, calls = {}, []

  def get(url, byte_range=None, retries=4):
    base = next(b for b in behaviour if url.startswith(b + "/"))
    calls.append(base)
    mode = behaviour[base]
    if isinstance(mode, int):
      raise _http_error(url, mode)
    if url.endswith(".index"):
      return INDEX
    data = GRIB[byte_range[0]:byte_range[1] + 1]
    return data[:-1] if mode == "short" else data

  monkeypatch.setattr(ifs, "_get", get)
  monkeypatch.setattr(ifs, "MIRRORS", (A, B, C))
  monkeypatch.setattr(ifs, "_last_good", None)
  monkeypatch.delenv("FWN_IFS_MIRRORS", raising=False)
  return behaviour, calls


def test_first_mirror_serves(fake):
  behaviour, calls = fake
  behaviour.update({A: "ok", B: "ok", C: "ok"})
  assert ifs._download(RUN, 0, WANTED) == b"mslt500"
  assert set(calls) == {A}


def test_throttled_mirror_fails_over_and_next_file_starts_at_the_good_one(fake):
  behaviour, calls = fake
  behaviour.update({A: 503, B: "ok", C: "ok"})
  assert ifs._download(RUN, 0, WANTED) == b"mslt500"
  assert calls[0] == A and set(calls[1:]) == {B}
  calls.clear()
  ifs._download(RUN, 6, WANTED)
  assert set(calls) == {B}


def test_short_range_fails_over(fake):
  behaviour, calls = fake
  behaviour.update({A: "short", B: "ok", C: "ok"})
  assert ifs._download(RUN, 0, WANTED) == b"mslt500"
  assert ifs._last_good == B


def test_all_mirrors_fail_names_each(fake):
  behaviour, _ = fake
  behaviour.update({A: 503, B: 404, C: 404})
  with pytest.raises(RuntimeError) as e:
    ifs._download(RUN, 0, WANTED)
  assert all(m in str(e.value) for m in (A, B, C))


def test_missing_field_is_not_a_mirror_failure(fake):
  behaviour, calls = fake
  behaviour.update({A: "ok", B: "ok", C: "ok"})
  with pytest.raises(KeyError):
    ifs._download(RUN, 0, WANTED | {("q", "pl", 500)})
  assert set(calls) == {A}


def test_env_overrides_mirror_order(fake, monkeypatch):
  behaviour, calls = fake
  behaviour.update({A: "ok", B: "ok", C: "ok"})
  monkeypatch.setenv("FWN_IFS_MIRRORS", f"{C}/, {A}")
  assert ifs._mirrors() == [C, A]
  ifs._download(RUN, 0, WANTED)
  assert set(calls) == {C}
