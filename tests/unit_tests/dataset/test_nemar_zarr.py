"""Tests for streaming NEMAR's Zarr serving copies (eegdash.dataset.nemar_zarr).

Offline tests build a small store in NEMAR's exact layout (sharded int16
level 0, biosigio's per-channel quantisation, an events group) and serve the
HTTP side (index, GitHub sidecars and annex pointers, Range reads of
events.parquet) from memory. Tests marked ``network`` check the live service.
"""

import csv
import io
import json
import multiprocessing
import pickle
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import zarr
from torch.utils.data import DataLoader
from zarr.abc.store import SuffixByteRequest
from zarr.storage import FsspecStore, WrapperStore

import eegdash.dataset.dataset as dataset_mod
import eegdash.dataset.nemar_zarr as nz
from braindecode.preprocessing import create_fixed_length_windows
from eegdash.dataset.dataset import EEGDashDataset
from eegdash.dataset.exceptions import DataIntegrityError
from eegdash.dataset.nemar_zarr import (
    EEGDashZarrRaw,
    EventsTooLargeError,
    MissingChunkError,
    NEMARDataWarning,
    RawNEMARZarr,
)
from eegdash.schemas import create_record

DS = "nm999999"
COMMIT = "c0ffee" * 6 + "abcd"
BIDS_PATH = "sub-01/eeg/sub-01_task-oddball_eeg.set"
STORE_PATH = "sub-01/eeg/sub-01_task-oddball_eeg.zarr"
ANNEX_KEY = "SHA256E-s100--" + "ab" * 32 + ".set"
GROUP = "eeg_250hz"
N_SAMPLES, CHUNK, SHARD = 12_000, 1000, 5000
CHANNELS = [  # label, biosigio type, unit
    ("Fz", "EEG", "uV"),
    ("Cz", "EEG", "uV"),
    ("Pz", "EEG", "µV"),
    ("Oz", "EEG", "V"),
    ("VEOG", "VEOG", "uV"),
    ("Marker", "TRIG", "uV"),  # codes 0-7; the unit label must not be applied
    ("Aux", "MISC", "%"),
]
CHANNELS_TSV = (
    "name\ttype\tunits\tstatus\tstatus_description\n"
    "Fz\tEEG\tuV\tgood\tn/a\nCz\tEEG\tuV\tbad\tflat\nPz\tEEG\tuV\tgood\tn/a\n"
    "Oz\tEEG\tV\tgood\tn/a\nVEOG\tVEOG\tuV\tgood\tn/a\nMarker\tTRIG\tuV\tgood\tn/a\n"
    "Aux\tMISC\t%\tgood\tn/a\n"
)


# ---------------------------------------------------------------------------
# Fixture store in NEMAR's layout
# ---------------------------------------------------------------------------


def quantise(x):
    """Quantise as biosigio does: int16 over each channel's range, min -> -32768."""
    lo, hi = float(x.min()), float(x.max())
    s = (hi - lo) / 65535.0
    o = lo + 32768.0 * s
    d = np.clip(np.round((x - o) / s), -32768, 32767).astype(np.int16)
    return d, s, o


def physical_signals(rng):
    t = np.arange(N_SAMPLES) / 250.0
    eeg = [
        20 * np.sin(2 * np.pi * f * t) + rng.normal(0, 5, N_SAMPLES)
        for f in (6, 10, 11)
    ]
    oz = 2e-5 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 5e-6, N_SAMPLES)  # volts
    veog = 80 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 10, N_SAMPLES)
    marker = np.zeros(N_SAMPLES)
    onsets = np.arange(250, N_SAMPLES - 250, 400)
    codes = (np.arange(len(onsets)) % 7) + 1
    for o, c in zip(onsets, codes):
        marker[o : o + 25] = c
    aux = 50 + 10 * np.sin(2 * np.pi * 0.05 * t)
    return np.vstack([*eeg, oz, veog, marker, aux]), onsets, codes


def build_store(base, *, channels=CHANNELS, nonfinite=None, meg=False):
    """Write a store under ``base/<DS>/zarr/<STORE_PATH>``; return (url, physical, events)."""
    rng = np.random.default_rng(7)
    x, onsets, codes = physical_signals(rng)
    if meg:
        channels = [*channels[:-1], ("MEG0111", "MEGMAG", "T")]
    digital, scales, offsets = zip(*(quantise(row) for row in x))
    digital = np.vstack(digital)
    url = f"{base}/{DS}/zarr/{STORE_PATH}"
    root = zarr.open_group(url, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "format": "biosigio-zarr",
            "format_version": 2,
            "biosigio_version": "1.2.7",
            "dtype": "int16",
            "channel_groups": [GROUP],
            "anti_alias_filter": "scipy.signal.resample_poly (polyphase FIR)",
            "source_format": "eeglab",
            "power_line_frequency": 50.0,
            "electrode_positions": {"Fz": [60.7, 0.0, 59.5], "Cz": [0.0, 0.0, 85.0]},
            "electrode_coordinate_system": "EEGLAB",
            "electrode_coordinate_units": "mm",
            "recording_metadata": {"source_file": BIDS_PATH},
            "note": "Derived serving copy.",
            "nemar": {"dataset_id": DS, "doi": "10.0/x", "source_commit": COMMIT},
        }
    )
    g = root.create_group(GROUP)
    chans = []
    for i, (label, ctype, unit) in enumerate(channels):
        c = {
            "label": label,
            "channel_type": ctype,
            "unit": unit,
            "scale": scales[i],
            "offset": offsets[i],
            "row_index": i,
            "original_rate": 1000.0,
            "target_rate": 250.0,
            "anti_aliased": ctype != "TRIG",
            "usable_for_inference": ctype != "TRIG",
        }
        if nonfinite and label in nonfinite:
            c["nonfinite_samples"] = nonfinite[label]
            c["usable_for_inference"] = False
        chans.append(c)
    g.attrs.update(
        {
            "rate": 250.0,
            "original_rate": 1000.0,
            "n_channels": len(chans),
            "n_samples": N_SAMPLES,
            "channels": chans,
        }
    )
    arr = g.create_array(
        "0",
        shape=digital.shape,
        chunks=(len(chans), CHUNK),
        shards=(len(chans), SHARD),
        dtype="int16",
        fill_value=0,
        attributes={
            "scale": list(scales),
            "offset": list(offsets),
            "physical_formula": "physical = digital * scale + offset",
        },
    )
    arr[:] = digital
    ev = root.create_group("events")
    ev_onset = onsets / 250.0
    ev.create_array("onset", data=ev_onset)
    ev.create_array("duration", data=np.zeros(len(onsets)))  # all fill: never written
    ev.create_array("code", data=np.ones(len(onsets), dtype=np.int32))
    ev.attrs.update(
        {"label_map": {"0": "STATUS", "1": "stimulus"}, "n_events": len(onsets)}
    )
    physical = np.vstack(
        [digital[i] * scales[i] + offsets[i] for i in range(len(chans))]
    )
    return url, physical, (ev_onset, codes)


def fake_index(base):
    return {
        "format_version": 3,
        "dataset_id": DS,
        "source_commit": COMMIT,
        "biosigio_version": "1.2.7",
        "data_base": f"{base}/{DS}/zarr/",
        "stores": [
            {
                "path": BIDS_PATH,
                "zarr": STORE_PATH,
                "groups": [
                    {
                        "name": GROUP,
                        "rate": 250.0,
                        "n_channels": 7,
                        "n_samples": N_SAMPLES,
                    }
                ],
                "units_report": {
                    "sidecar": "sub-01/eeg/sub-01_task-oddball_channels.tsv"
                },
            }
        ],
    }


class FakeHTTP:
    """Serves URLs from memory with HEAD and Range support; records every call."""

    def __init__(self, files=None, status=None):
        self.files = dict(files or {})
        self.status = dict(status or {})  # url substring -> forced status
        self.calls = []

    def __call__(self, url, timeout, *, method="GET", headers=None):
        self.calls.append((method, url, (headers or {}).get("Range")))
        r = requests.Response()
        r.url = url
        for part, code in self.status.items():
            if part in url:
                r.status_code, r._content, r.reason = code, b"", "forced"
                return r
        data = self.files.get(url)
        if data is None:
            r.status_code, r._content, r.reason = 404, b"", "Not Found"
            return r
        if method == "HEAD":
            r.status_code, r._content = 200, b""
            r.headers.update({"Content-Length": str(len(data)), "ETag": '"e1"'})
            return r
        rng = (headers or {}).get("Range")
        if rng:
            start, end = (int(v) for v in rng.removeprefix("bytes=").split("-"))
            r.status_code, r._content = 206, data[start : end + 1]
        else:
            r.status_code, r._content = 200, data
        return r


def github(path):
    return nz.GITHUB_RAW.format(ds=DS, commit=COMMIT, path=path)


@pytest.fixture(autouse=True)
def _cold_caches():
    nz._open_root.cache_clear()
    nz._open_level0.cache_clear()
    nz._CHUNK_CACHE.clear()
    nz._events_from_parquet.cache_clear()
    nz._fetch_index.cache_clear()
    yield
    nz._open_root.cache_clear()
    nz._open_level0.cache_clear()


@pytest.fixture
def store(tmp_path):
    return build_store(str(tmp_path))


@pytest.fixture
def http(monkeypatch, tmp_path):
    """Fake HTTP with the channels.tsv and the raw file's annex pointer at COMMIT."""
    fake = FakeHTTP(
        {
            github(
                "sub-01/eeg/sub-01_task-oddball_channels.tsv"
            ): CHANNELS_TSV.encode(),
            github(
                BIDS_PATH
            ): f"../../.git/annex/objects/xx/yy/{ANNEX_KEY}/{ANNEX_KEY}".encode(),
        }
    )
    monkeypatch.setattr(nz, "_http_get", fake)
    monkeypatch.setattr(nz, "_fetch_index", lambda ds: fake_index(str(tmp_path)))
    return fake


def open_quiet(url, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        return RawNEMARZarr(url, **kw)


class Instrumented(WrapperStore):
    """Counts requests that reach the underlying store, below the reader's caches."""

    def __init__(self, store):
        super().__init__(store)
        self.reset()

    def _with_store(self, store):
        return type(self)(store)

    @property
    def _supports_sync_io(self):
        return False

    def reset(self):
        self.meta = self.index = self.chunks = 0

    async def get(self, key, prototype, byte_range=None):
        if isinstance(byte_range, SuffixByteRequest):
            self.index += 1
        elif "/c/" in key:
            self.chunks += 1
        else:
            self.meta += 1
        return await self._store.get(key, prototype, byte_range)

    async def get_ranges(self, key, byte_ranges, *, prototype, **kw):
        self.chunks += 1
        async for g in self._store.get_ranges(
            key, byte_ranges, prototype=prototype, **kw
        ):
            yield g


class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


class RangeHandler(BaseHTTPRequestHandler):
    """Serves ``server.root`` with HEAD and single Range requests; 404 if absent."""

    def _body(self):
        path = Path(self.server.root) / self.path.lstrip("/").split("?")[0]
        if not path.is_file():
            self.send_error(404)
            return None
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            return data
        start, end = rng.removeprefix("bytes=").split("-")
        if not start:
            start, end = len(data) - int(end), len(data) - 1
        start, end = int(start), int(end) if end else len(data) - 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        return data[start : end + 1]

    def do_HEAD(self):
        data = self._body()
        if data is not None:
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()

    def do_GET(self):
        data = self._body()
        if data is not None:
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server(tmp_path):
    """A real HTTP server for ``tmp_path``, for behaviour that fakes would hide."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    server.root = str(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def shard_path(url, i):
    return Path(url) / GROUP / "0" / "c" / "0" / str(i)


class Flaky(WrapperStore):
    """Raises ``exc`` on the first ``n`` chunk reads."""

    def __init__(self, store, n=0, exc=None):
        super().__init__(store)
        self.n, self.exc, self.calls = n, exc, 0

    def _with_store(self, store):
        return type(self)(store, self.n, self.exc)

    @property
    def _supports_sync_io(self):
        return False

    def _maybe_fail(self, key):
        if "/c/" in key:
            self.calls += 1
            if self.calls <= self.n:
                raise self.exc

    async def get(self, key, prototype, byte_range=None):
        self._maybe_fail(key)
        return await self._store.get(key, prototype, byte_range)

    async def get_ranges(self, key, byte_ranges, *, prototype, **kw):
        self._maybe_fail(key)
        async for g in self._store.get_ranges(
            key, byte_ranges, prototype=prototype, **kw
        ):
            yield g


def wrap_base(monkeypatch, cls, **kw):
    holder = {}

    def make(url, so):
        holder["st"] = cls(
            FsspecStore.from_url(url, storage_options=so, read_only=True), **kw
        )
        return holder["st"]

    monkeypatch.setattr(nz, "_make_base_store", make)
    return holder


# ---------------------------------------------------------------------------
# Lazy reads
# ---------------------------------------------------------------------------


def test_open_reads_metadata_only(store, monkeypatch):
    holder = wrap_base(monkeypatch, Instrumented)
    open_quiet(store[0], events=None, bids_status=False)
    st = holder["st"]
    assert (st.index, st.chunks) == (0, 0) and st.meta <= 4


def test_short_read_fetches_one_chunk_and_caches_it(store, monkeypatch):
    holder = wrap_base(monkeypatch, Instrumented)
    raw = open_quiet(store[0], events=None, bids_status=False)
    st = holder["st"]
    st.reset()
    raw.get_data(start=6100, stop=6600)  # inside chunk 6
    assert (st.index, st.chunks, st.meta) == (1, 1, 0)
    st.reset()
    raw.get_data(start=6200, stop=6700)
    assert (st.index, st.chunks, st.meta) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Numerical correctness and calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cache_bytes", [nz._DEFAULT_CHUNK_CACHE_BYTES, 0, 30_000])
def test_intervals_and_channels_match_the_array(store, cache_bytes):
    url, physical, _ = store
    nz.set_chunk_cache_size(cache_bytes)
    try:
        raw = open_quiet(url, events=None, bids_status=False)
        si = np.array(
            [1e-6, 1e-6, 1e-6, 1.0, 1e-6, 1.0, 1.0]
        )  # unit -> SI (codes unscaled)
        rng = np.random.default_rng(0)
        cases = [
            (0, 1),
            (N_SAMPLES - 1, N_SAMPLES),
            (999, 1001),
            (4999, 5001),
            (0, N_SAMPLES),
        ]
        cases += [
            (a, a + n)
            for n in (1, 7, 999, 1001, 6000)
            for a in rng.integers(0, N_SAMPLES - n, 4)
        ]
        eeg_idx = [0, 1, 2, 3, 4, 6]  # not the rounded trigger
        for a, b in cases:
            idx = list(rng.permutation(eeg_idx)[: rng.integers(1, 7)])
            got = raw.get_data(
                picks=[raw.ch_names[i] for i in idx], start=int(a), stop=int(b)
            )
            exp = physical[idx, a:b] * si[idx, None]
            np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-18)
    finally:
        nz.set_chunk_cache_size(nz._DEFAULT_CHUNK_CACHE_BYTES)


def test_crop_concatenate_pickle_and_projection(store):
    url, physical, _ = store
    raw = open_quiet(url, events=None, bids_status=False)
    c = raw.copy().crop(10.004, 20.0)  # sample 2501
    np.testing.assert_allclose(
        c.get_data(picks="Fz") * 1e6, physical[[0], 2501:5001], rtol=1e-12
    )
    cat = mne.concatenate_raws([raw.copy().crop(0, 2), raw.copy().crop(30, 32)])
    exp = np.hstack([physical[[1], :501], physical[[1], 7500:8001]])
    np.testing.assert_allclose(cat.get_data(picks="Cz") * 1e6, exp, rtol=1e-12)
    clone = pickle.loads(pickle.dumps(raw))
    np.testing.assert_array_equal(
        clone.get_data(start=100, stop=600), raw.get_data(start=100, stop=600)
    )
    rp = (
        raw.copy()
        .pick(["Fz", "Cz", "Pz"])
        .set_eeg_reference("average", projection=True)
    )
    rp.apply_proj()
    seg = physical[:3, 3000:3500]
    np.testing.assert_allclose(
        rp.get_data(start=3000, stop=3500) * 1e6, seg - seg.mean(0), atol=1e-9
    )


def test_units_and_trigger_codes(store):
    raw = open_quiet(store[0], events=None, bids_status=False)
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    assert types == {
        "Fz": "eeg", "Cz": "eeg", "Pz": "eeg", "Oz": "eeg",
        "VEOG": "eog", "Marker": "stim", "Aux": "misc",
    }  # fmt: skip
    oz = raw.get_data(picks="Oz")
    assert 1e-6 < np.abs(oz).max() < 1e-4  # stored in V, read in V
    codes = raw.get_data(picks="Marker")[0]
    assert set(np.unique(codes)) == set(range(8))  # exact integers, 'uV' not applied
    events = mne.find_events(raw)
    _, _, (ev_onset, ev_codes) = store
    np.testing.assert_array_equal(events[:, 0], np.round(ev_onset * 250).astype(int))
    np.testing.assert_array_equal(events[:, 2], ev_codes)


def test_unknown_voltage_unit_warns(tmp_path):
    chans = [("Fz", "EEG", "n/a"), *CHANNELS[1:]]
    url, *_ = build_store(str(tmp_path), channels=chans)
    with pytest.warns(NEMARDataWarning, match=r"Fz \('n/a'\)"):
        RawNEMARZarr(url, events=None, bids_status=False)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_bids_bad_status_and_montage(store, http):
    raw = open_quiet(store[0], events=None)
    assert raw.info["bads"] == ["Cz"]
    assert (
        sum(
            u == github("sub-01/eeg/sub-01_task-oddball_channels.tsv")
            for _, u, _ in http.calls
        )
        == 1
    )
    assert raw.info["line_freq"] == 50.0
    fz = raw.info["chs"][raw.ch_names.index("Fz")]["loc"][:3]
    np.testing.assert_allclose(
        fz, [0.0, 0.0607, 0.0595], atol=1e-9
    )  # EEGLAB x=nose -> MNE y


def test_bids_status_absent_or_unreadable(store, http):
    http.files.pop(github("sub-01/eeg/sub-01_task-oddball_channels.tsv"))
    assert open_quiet(store[0], events=None).info["bads"] == []
    nz._open_root.cache_clear()
    http.status["raw.githubusercontent.com"] = 500
    with pytest.warns(NEMARDataWarning, match="could not read BIDS channel status"):
        raw = RawNEMARZarr(store[0], events=None)
    assert (
        raw.info["bads"] == [] and "500" in raw.nemar_provenance["bids_status"]["error"]
    )


def test_nonfinite_channels_warn_and_optionally_go_bad(tmp_path):
    url, *_ = build_store(str(tmp_path), nonfinite={"Pz": 120})
    with pytest.warns(NEMARDataWarning, match=r"1 of 7 channels had NaN/inf samples"):
        raw = RawNEMARZarr(url, events=None, bids_status=False)
    assert raw.info["bads"] == [] and raw.nonfinite_samples == {"Pz": 120}
    with pytest.warns(NEMARDataWarning, match="marked bad"):
        raw = RawNEMARZarr(url, events=None, bids_status=False, nonfinite="bad")
    assert raw.info["bads"] == ["Pz"]


def test_meg_channels_warn(tmp_path):
    url, *_ = build_store(str(tmp_path), meg=True)
    with pytest.warns(NEMARDataWarning, match="MEG channels are read as 'misc'"):
        RawNEMARZarr(url, events=None, bids_status=False)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def events_parquet(n_stores=6, rows_per_store=4000):
    """A parquet sorted by store_path with one row group per recording, as NEMAR writes.

    Large enough (> 64 KiB, pyarrow's footer read) that a ranged read must
    fetch only part of it.
    """
    rng = np.random.default_rng(1)
    frames = []
    for i in range(n_stores):
        sp = STORE_PATH if i == 3 else f"sub-0{i}/eeg/sub-0{i}_task-other_eeg.zarr"
        frames.append(
            pd.DataFrame(
                {
                    "store_path": sp,
                    "group_name": GROUP,
                    "onset_s": np.sort(rng.uniform(0, 600, rows_per_store)),
                    "duration_s": np.full(rows_per_store, 0.1, dtype=np.float32),
                    "value": [str(70 + (j % 2) * 10) for j in range(rows_per_store)],
                    "trial_type": "stimulus",
                }
            )
        )
    df = pd.concat(frames).sort_values("store_path", kind="stable")
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        buf,
        row_group_size=rows_per_store,
    )
    return buf.getvalue()


def test_events_parquet_is_range_read(monkeypatch):
    data = events_parquet()
    url = f"https://example.org/{DS}/zarr/{STORE_PATH}"
    fake = FakeHTTP({f"https://example.org/{DS}/zarr/events.parquet": data})
    monkeypatch.setattr(nz, "_http_get", fake)
    onset, duration, desc, prov = nz._events_from_parquet(
        url, GROUP, nz.EVENTS_MAX_BYTES
    )
    assert len(onset) == 4000 and set(desc) == {"70", "80"}
    assert set(duration.tolist()) == {0.1}  # float32 0.1 restored exactly
    # sorted by store_path, the oddball recording is the second row group
    assert prov["row_groups_read"] == [1] and prov["etag"] == '"e1"'
    assert 0 < prov["bytes_fetched"] < 0.5 * len(data), (
        prov["bytes_fetched"],
        len(data),
    )
    ranges = [rng for method, _, rng in fake.calls if method == "GET"]
    assert len(ranges) == len(set(ranges))  # no range fetched twice
    assert all(
        rng for method, _, rng in fake.calls if method == "GET"
    )  # never a full GET


def test_events_limit_policy(store, monkeypatch):
    url = f"https://example.org/{DS}/zarr/{STORE_PATH}"
    monkeypatch.setattr(
        nz,
        "_http_get",
        FakeHTTP({f"https://example.org/{DS}/zarr/events.parquet": events_parquet()}),
    )
    root = nz._open_root(store[0], "{}")
    with pytest.warns(NEMARDataWarning, match="above the .* limit.*store's own events"):
        ann, prov = nz._load_annotations("auto", url, GROUP, root, 10)
    assert prov["source"] == "store events group" and set(ann.description) == {
        "stimulus"
    }
    with pytest.raises(EventsTooLargeError, match="events_max_bytes=None"):
        nz._load_annotations("parquet", url, GROUP, root, 10)


def test_events_failure_falls_back_or_raises(store, monkeypatch):
    url = f"https://example.org/{DS}/zarr/{STORE_PATH}"
    monkeypatch.setattr(nz, "_http_get", FakeHTTP(status={"events.parquet": 503}))
    root = nz._open_root(store[0], "{}")
    with pytest.warns(NEMARDataWarning, match="could not be read.*503"):
        _, prov = nz._load_annotations("auto", url, GROUP, root, None)
    assert prov["source"] == "store events group"
    with pytest.raises(requests.HTTPError, match="503"):
        nz._load_annotations("parquet", url, GROUP, root, None)


def test_local_store_uses_its_events_group(store):
    _, _, (ev_onset, _) = store
    with pytest.warns(NEMARDataWarning, match="only fetched over HTTP"):
        raw = RawNEMARZarr(store[0], bids_status=False)
    np.testing.assert_allclose(raw.annotations.onset, ev_onset)
    assert set(raw.annotations.duration) == {0.0}  # duration chunk was never written


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_transient_chunk_errors_are_retried(store, monkeypatch):
    holder = wrap_base(monkeypatch, Flaky, n=2, exc=FakeHTTPError(503))
    raw = open_quiet(store[0], events=None, bids_status=False)
    assert raw.get_data(start=0, stop=10).shape == (7, 10)
    assert holder["st"].calls == 4  # 2 failed index reads, then index + chunk


@pytest.mark.parametrize(
    "status,calls,exc",
    [(503, nz._ATTEMPTS, FakeHTTPError), (403, 1, MissingChunkError)],
)
def test_persistent_or_denied_chunk_errors_are_raised(
    store, monkeypatch, status, calls, exc
):
    holder = wrap_base(monkeypatch, Flaky, n=99, exc=FakeHTTPError(status))
    raw = open_quiet(store[0], events=None, bids_status=False)
    with pytest.raises(exc, match=str(status)):
        raw.get_data(start=0, stop=10)  # a 403 on signal is never read as fill
    assert holder["st"].calls == calls


def test_only_event_chunks_may_be_unwritten():
    assert nz._may_be_unwritten("events/duration/c/0")
    assert not nz._may_be_unwritten("eeg_250hz/0/c/0/1")
    assert not nz._may_be_unwritten("events/zarr.json")


def test_missing_local_shard_is_an_error_not_flat_data(store):
    url = store[0]
    shard_path(url, 1).unlink()  # samples 5000-9999
    raw = open_quiet(url, events=None, bids_status=False)
    assert raw.get_data(start=0, stop=10).shape == (7, 10)
    with pytest.raises(MissingChunkError, match="flat data"):
        raw.get_data(start=5000, stop=5005)


def test_missing_shard_over_http_404_is_an_error(store, http_server):
    shard_path(store[0], 1).unlink()
    url = f"{http_server}/{DS}/zarr/{STORE_PATH}"
    raw = open_quiet(url, events=None, bids_status=False)
    np.testing.assert_allclose(
        raw.get_data(start=0, stop=10, picks="Fz")[0] * 1e6, store[1][0, :10]
    )
    with pytest.raises(MissingChunkError, match="flat data"):
        raw.get_data(start=5000, stop=5005)


def test_constant_channel_with_zero_scale(store):
    url = store[0]
    arr = zarr.open_array(f"{url}/{GROUP}/0", mode="r+")
    scale, offset = list(arr.attrs["scale"]), list(arr.attrs["offset"])
    scale[0], offset[0] = 0.0, 12.5  # Fz flat at 12.5 uV
    scale[5], offset[5] = 0.0, 3.0  # Marker stuck at code 3
    arr.attrs.update({"scale": scale, "offset": offset})
    raw = open_quiet(url, events=None, bids_status=False)
    data = raw.get_data(start=0, stop=600)
    np.testing.assert_allclose(data[0], 12.5e-6)
    np.testing.assert_array_equal(data[5], 3.0)
    np.testing.assert_allclose(data[1] * 1e6, store[1][1, :600], rtol=1e-6)


def test_events_fallback_on_malformed_parquet(store, monkeypatch):
    url = f"https://example.org/{DS}/zarr/{STORE_PATH}"
    root = nz._open_root(store[0], "{}")
    for exc in (KeyError("Content-Length"), pa.ArrowInvalid("bad footer")):

        def broken(*args, exc=exc):
            raise exc

        monkeypatch.setattr(nz, "_events_from_parquet", broken)
        with pytest.warns(NEMARDataWarning, match="could not be read"):
            _, prov = nz._load_annotations("auto", url, GROUP, root, None)
        assert prov["source"] == "store events group"
        with pytest.raises(type(exc)):
            nz._load_annotations("parquet", url, GROUP, root, None)


class _Root:
    def __init__(self, exc):
        self.exc = exc

    def __getitem__(self, key):
        raise self.exc


def test_store_events_absent_vs_failing():
    with pytest.warns(NEMARDataWarning, match="no readable events group"):
        ann, _ = nz._events_from_store(_Root(FakeHTTPError(403)))
    assert len(ann) == 0
    with pytest.raises(FakeHTTPError, match="503"):
        nz._events_from_store(_Root(FakeHTTPError(503)))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance(store, http):
    raw = open_quiet(store[0], events=None)
    p = raw.nemar_provenance
    assert p["source_commit"] == COMMIT and COMMIT in raw.info["description"]
    t = p["nemar_transformations"]
    assert (t["resampling"]["original_rate_hz"], t["resampling"]["served_rate_hz"]) == (
        1000.0,
        250.0,
    )
    assert "resample_poly" in t["resampling"]["continuous_channels"]
    assert t["quantization"]["dtype"] == "int16"
    a = p["reader_adjustments"]
    assert a["unit_factor_to_SI"] == {"uV": 1e-6, "µV": 1e-6, "V": 1.0, "%": 1.0}
    assert a["integer_code_rounding"] == ["Marker"]
    assert p["bids_status"]["commit"] == COMMIT and p["bids_status"]["bads"] == ["Cz"]
    assert pickle.loads(pickle.dumps(raw)).nemar_provenance == p


# ---------------------------------------------------------------------------
# EEGDash integration
# ---------------------------------------------------------------------------


def nemar_record(annex_key=ANNEX_KEY, backend="nemar", dataset=DS):
    rec = create_record(
        dataset=dataset,
        storage_base=f"s3://nemar/{DS}"
        if backend == "nemar"
        else f"s3://openneuro.org/{dataset}",
        storage_backend=backend,
        bids_relpath=BIDS_PATH,
        subject="01",
        task="oddball",
        sampling_frequency=1000.0,  # the BIDS original, not the store
        ntimes=N_SAMPLES * 4,
        ch_names=[c[0] for c in CHANNELS],
        nchans=len(CHANNELS),
    )
    if annex_key:
        rec["storage"]["annex_keys"] = {BIDS_PATH: annex_key}
    return rec


def zarr_raw(tmp_path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        return EEGDashZarrRaw(
            nemar_record(**kw), str(tmp_path / "cache"), zarr_options={"events": None}
        )


def test_length_rate_and_repr_come_from_the_store(store, http, tmp_path):
    ds = zarr_raw(tmp_path)
    assert len(ds) == N_SAMPLES  # the record's ntimes describes the 1000 Hz original
    assert ds._make_mne_info()["sfreq"] == 250.0
    assert "250.0 Hz" in repr(ds) and ds._raw is None


def test_raw_streams_and_writes_nothing(store, http, tmp_path):
    ds = zarr_raw(tmp_path)
    ds.download_all = None  # unused; downloading is a no-op for this class
    ds._download_required_files()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        raw = ds.raw
    assert isinstance(raw, RawNEMARZarr) and len(ds) == N_SAMPLES
    assert raw.nemar_provenance["eegdash_snapshot_check"]["status"] == "verified"
    assert not (tmp_path / "cache" / DS).exists() or not any(
        (tmp_path / "cache" / DS).rglob("*.*")
    )
    X, _ = ds[100]
    assert X.shape == (7, 1)


def test_snapshot_mismatch_is_a_data_integrity_error(store, http, tmp_path):
    ds = zarr_raw(tmp_path, annex_key="SHA256E-s1--different.set")
    with pytest.raises(DataIntegrityError, match="different version"):
        _ = ds.raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        lenient = EEGDashZarrRaw(
            nemar_record(annex_key="SHA256E-s1--different.set"),
            str(tmp_path / "c2"),
            zarr_options={"events": None},
            on_error="skip",
        )
    assert lenient.raw is None and lenient._skipped


def test_snapshot_unverifiable_warns(store, http, tmp_path):
    ds = zarr_raw(tmp_path, annex_key=None)
    with pytest.warns(NEMARDataWarning, match="cannot verify"):
        _ = ds.raw
    assert ds.raw.nemar_provenance["eegdash_snapshot_check"]["status"].startswith(
        "unverified"
    )


def test_rejects_non_nemar_records_and_unknown_options(tmp_path):
    with pytest.raises(ValueError, match="needs NEMAR records"):
        # an OpenNeuro id: EEGDash self-heals nm* ids back to the nemar backend
        EEGDashZarrRaw(nemar_record(backend="s3", dataset="ds000117"), str(tmp_path))
    with pytest.raises(ValueError, match="unknown zarr_options"):
        EEGDashZarrRaw(
            nemar_record(), str(tmp_path), zarr_options={"not_an_option": None}
        )


def test_dataset_signal_source(store, http, tmp_path):
    dataset = EEGDashDataset(
        cache_dir=tmp_path / "cache",
        records=[nemar_record()],
        signal_source="zarr",
        zarr_options={"events": None},
        _suppress_comp_warning=True,
    )
    assert isinstance(dataset.datasets[0], EEGDashZarrRaw) and len(dataset) == N_SAMPLES
    with pytest.raises(ValueError, match="signal_source must be"):
        EEGDashDataset(
            cache_dir=tmp_path, records=[nemar_record()], signal_source="hdf5"
        )
    with pytest.raises(ValueError, match="zarr_options only applies"):
        EEGDashDataset(cache_dir=tmp_path, records=[nemar_record()], zarr_options={})
    with pytest.raises(ValueError, match="download=False"):
        EEGDashDataset(
            cache_dir=tmp_path, dataset=DS, download=False, signal_source="zarr"
        )


def test_braindecode_windows_and_multiprocess_dataloader(store, http, tmp_path):
    url, physical, _ = store
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        dataset = EEGDashDataset(
            cache_dir=tmp_path / "cache",
            records=[nemar_record()],
            signal_source="zarr",
            zarr_options={"events": None, "bids_status": False},
            _suppress_comp_warning=True,
        )
        windows = create_fixed_length_windows(
            dataset,
            window_size_samples=500,
            window_stride_samples=500,
            drop_last_window=True,
            picks=["Fz", "Cz"],
        )
    assert len(windows) == N_SAMPLES // 500
    X0, *_ = windows[3]
    np.testing.assert_allclose(
        X0 * 1e6, physical[:2, 1500:2000], rtol=1e-5
    )  # float32 windows
    single = np.stack([windows[i][0] for i in range(len(windows))])
    loader = DataLoader(
        windows, batch_size=4, num_workers=2, multiprocessing_context="spawn"
    )
    multi = np.concatenate([batch[0].numpy() for batch in loader])
    np.testing.assert_array_equal(multi, single)


def test_dataloader_forked_workers_over_http(store, http, http_server, tmp_path):
    """Linux's default ``fork`` start method must not reuse the parent's sessions."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    index = fake_index(str(tmp_path))
    index["data_base"] = f"{http_server}/{DS}/zarr/"
    with (
        patch.object(nz, "_fetch_index", lambda ds: index),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", NEMARDataWarning)
        dataset = EEGDashDataset(
            cache_dir=tmp_path / "cache",
            records=[nemar_record()],
            signal_source="zarr",
            zarr_options={"events": None, "bids_status": False},
            _suppress_comp_warning=True,
        )
        windows = create_fixed_length_windows(
            dataset,
            window_size_samples=1000,
            window_stride_samples=1000,
            drop_last_window=True,
        )
        nz._CHUNK_CACHE.clear()  # make the workers fetch over HTTP themselves
        loader = DataLoader(
            windows, batch_size=3, num_workers=2, multiprocessing_context="fork"
        )
        multi = np.concatenate([batch[0].numpy() for batch in loader])
    single = np.stack([windows[i][0] for i in range(len(windows))])
    np.testing.assert_array_equal(multi, single)


def test_download_all_downloads_nothing(store, http, tmp_path):
    dataset = EEGDashDataset(
        cache_dir=tmp_path / "cache",
        records=[nemar_record()],
        signal_source="zarr",
        _suppress_comp_warning=True,
    )
    with (
        patch.object(
            dataset_mod.EEGDashRaw, "_download_required_files", autospec=True
        ) as per_file,
        patch.object(dataset, "_download_dataset_files") as globals_mock,
    ):
        dataset.download_all(n_jobs=2)
        dataset.download_all(n_jobs=1)
    per_file.assert_not_called()
    globals_mock.assert_not_called()


def two_group_store(url):
    root = zarr.open_group(url, mode="r+")
    root.attrs["channel_groups"] = [GROUP, "emg_1000hz"]


def two_group_index(base):
    index = fake_index(base)
    entry = index["stores"][0]
    entry["groups"][0]["modality"] = "eeg"
    entry["groups"].append(
        {
            "name": "emg_1000hz",
            "modality": "emg",
            "rate": 1000.0,
            "n_channels": 2,
            "n_samples": N_SAMPLES * 4,
        }
    )
    return index


def test_multi_group_store_uses_matching_or_requested_group(
    store, http, monkeypatch, tmp_path
):
    two_group_store(store[0])
    monkeypatch.setattr(nz, "_fetch_index", lambda ds: two_group_index(str(tmp_path)))
    rec = nemar_record()
    rec["datatype"] = "eeg"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        ds = EEGDashZarrRaw(rec, str(tmp_path), zarr_options={"events": None})
        assert len(ds) == N_SAMPLES and ds.raw.info["sfreq"] == 250.0
        assert ds.raw.nemar_provenance["group"] == GROUP
        rec_unknown = dict(rec, datatype="ieeg")
        ambiguous = EEGDashZarrRaw(rec_unknown, str(tmp_path), on_error="skip")
        assert len(ambiguous) == 0
        assert ambiguous.raw is None and ambiguous._skipped
        wrong = EEGDashZarrRaw(rec, str(tmp_path), zarr_options={"group": "nope"})
        with pytest.raises(DataIntegrityError, match="zarr_options"):
            _ = wrong.raw


def test_snapshot_uses_the_store_commit(store, http, monkeypatch, tmp_path):
    index = fake_index(str(tmp_path))
    index["source_commit"] = "f" * 40  # re-exported index; this store is older
    monkeypatch.setattr(nz, "_fetch_index", lambda ds: index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NEMARDataWarning)
        check = zarr_raw(tmp_path).raw.nemar_provenance["eegdash_snapshot_check"]
    assert check["status"] == "verified"
    assert check["source_commit"] == COMMIT and check["commit_from"] == "store"
    assert check["index_source_commit"] == "f" * 40


def test_raw_file_absent_at_commit_is_a_mismatch(store, http, tmp_path):
    del http.files[github(BIDS_PATH)]
    http.files[github("dataset_description.json")] = b"{}"
    with pytest.raises(DataIntegrityError, match="which has no"):
        _ = zarr_raw(tmp_path).raw


def test_open_failures_go_through_on_error(store, http, monkeypatch, tmp_path):
    index = fake_index(str(tmp_path))
    index["stores"][0]["zarr"] = "sub-01/eeg/missing.zarr"
    monkeypatch.setattr(nz, "_fetch_index", lambda ds: index)
    with pytest.raises(DataIntegrityError, match="could not open"):
        _ = zarr_raw(tmp_path).raw
    lenient = EEGDashZarrRaw(nemar_record(), str(tmp_path), on_error="skip")
    assert lenient.raw is None and lenient._skipped


def test_len_raises_on_transport_errors_but_not_missing_copies(
    store, http, monkeypatch, tmp_path
):
    def down(ds):
        raise requests.ConnectionError("index unreachable")

    monkeypatch.setattr(nz, "_fetch_index", down)
    with pytest.raises(DataIntegrityError, match="index unreachable"):
        len(zarr_raw(tmp_path))
    index = fake_index(str(tmp_path))
    index["stores"] = []
    monkeypatch.setattr(nz, "_fetch_index", lambda ds: index)
    assert len(zarr_raw(tmp_path)) == 0


@pytest.mark.parametrize(
    "options",
    [
        {"events": "parqeut"},
        {"nonfinite": "drop"},
        {"events_max_bytes": -1},
        {"events_max_bytes": True},
        {"bids_status": "yes"},
        {"group": 3},
    ],
)
def test_option_values_are_validated_up_front(options, tmp_path):
    with pytest.raises(ValueError, match="must be"):
        EEGDashZarrRaw(nemar_record(), str(tmp_path), zarr_options=options)


def test_missing_extra_names_the_install_command(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "eegdash.dataset.nemar_zarr", None)
    with pytest.raises(ImportError, match=r"eegdash\[zarr\]"):
        EEGDashDataset(
            cache_dir=tmp_path, records=[nemar_record()], signal_source="zarr"
        )


# ---------------------------------------------------------------------------
# Live NEMAR (deselected by default; run with -m network)
# ---------------------------------------------------------------------------

LIVE_DS, LIVE_PATH = "nm000132", "sub-001/eeg/sub-001_task-MMN_eeg.set"


@pytest.mark.network
def test_live_reader_and_bids_events():
    raw = nz.read_raw_nemar_zarr(LIVE_DS, LIVE_PATH)
    assert raw.info["sfreq"] == 250.0 and raw.n_times == 152_000
    tsv_url = f"https://data.nemar.org/{LIVE_DS}/latest/sub-001/eeg/sub-001_task-MMN_events.tsv"
    rows = list(
        csv.DictReader(
            io.StringIO(requests.get(tsv_url, timeout=60).text), delimiter="\t"
        )
    )
    assert list(raw.annotations.description) == [r["value"] for r in rows]
    assert (
        raw.nemar_provenance["events"]["bytes_fetched"]
        < raw.nemar_provenance["events"]["file_bytes"]
    )


@pytest.mark.network
def test_live_dataset_signal_source(tmp_path):
    dataset = EEGDashDataset(
        cache_dir=tmp_path,
        dataset=LIVE_DS,
        subject="001",
        task="MMN",
        signal_source="zarr",
        _suppress_comp_warning=True,
    )
    ds = dataset.datasets[0]
    assert len(ds) == 152_000  # the record says 622592 samples at 1024 Hz
    assert ds.raw.nemar_provenance["eegdash_snapshot_check"]["status"] == "verified"
    assert not any(tmp_path.rglob("*.set"))


@pytest.mark.network
def test_live_index_lookup_matches_record():
    index = nz._fetch_index(LIVE_DS)
    entry = next(s for s in index["stores"] if s["path"] == LIVE_PATH)
    assert json.loads(json.dumps(entry))["groups"][0]["rate"] == 250.0
