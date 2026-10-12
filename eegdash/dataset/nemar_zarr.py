# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.
"""Lazy reading of NEMAR's Zarr serving copies.

NEMAR publishes each BIDS recording as a Zarr v3 store (``biosigio-zarr``)::

    <recording>.zarr/
      zarr.json            root attrs: channel_groups, electrode_positions, nemar{...}
      <group>/zarr.json    group attrs: rate, original_rate, channels[...]
      <group>/0            level 0: int16 (n_channels, n_samples), the inference signal
      <group>/view/<L>     min/max render envelopes (never read here)
      events/              onset, duration, code (trial_type only)

with ``physical = digital * scale + offset`` per channel. The dataset-wide
``<data_base>events.parquet`` holds every event with its BIDS ``value``.

:class:`RawNEMARZarr` is an :class:`mne.io.BaseRaw` that reads only metadata
when opened and fetches signal chunks on demand. :class:`EEGDashZarrRaw` puts
it behind the usual :class:`~eegdash.dataset.base.EEGDashRaw` interface; select
it with ``EEGDashDataset(..., signal_source="zarr")``.

What NEMAR has already done to the signal
-----------------------------------------
The store is a derived serving copy; the BIDS archive stays authoritative.
NEMAR (biosigio) has:

* **resampled** each continuous channel to the group rate (250 Hz for EEG/MEG,
  1000 Hz for iEEG/EMG by default) with ``scipy.signal.resample_poly``, never
  upsampling; trigger and clock channels by nearest sample instead;
* **quantised** each channel to int16 over its own range:
  ``scale = (max - min) / 65535`` with ``min`` mapped to -32768;
* **zero-filled** NaN/inf samples (``nonfinite_samples``; positions not kept);
* applied the BIDS ``channels.tsv`` units to the signal.

``RawNEMARZarr.nemar_provenance`` records all of it per recording, with the
source commit, tool versions, where events and bad-channel status came from,
and this reader's own adjustments (unit conversion to SI, rounding of integer
trigger codes).

Installation
------------
Reading needs ``pip install "eegdash[zarr]"`` (zarr >= 3.4, fsspec, aiohttp,
pyarrow). S3 is reached over plain HTTPS, so s3fs/aiobotocore are never used
(see eegdash/EEGDash#397).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
import threading
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import Any

import mne
import numpy as np
import requests
import zarr
from mne.io import BaseRaw
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.storage import FsspecStore, WrapperStore

from braindecode.datasets.base import _ReprBuilder

from ..logging import logger
from .base import EEGDashRaw
from .exceptions import DataIntegrityError

try:  # needed for HTTP(S) stores, which is how NEMAR serves them
    import aiohttp
except ImportError:  # pragma: no cover - exercised only without the extra
    aiohttp = None

try:  # needed for events.parquet
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only without the extra
    pq = None

__all__ = [
    "EEGDashZarrRaw",
    "EventsTooLargeError",
    "NEMARDataWarning",
    "RawNEMARZarr",
    "chunk_cache_info",
    "read_raw_nemar_zarr",
    "set_chunk_cache_size",
]

INDEX_URL = "https://zarr.nemar.org/{ds}/zarr/index.json"
CONTRACT_BASE = "https://zarr.nemar.org/{ds}/zarr/"
GITHUB_RAW = "https://raw.githubusercontent.com/nemarDatasets/{ds}/{commit}/{path}"

#: Most bytes of ``events.parquet`` downloaded automatically for one recording.
EVENTS_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_CHUNK_CACHE_BYTES = 64 * 1024 * 1024
_MIN_ZARR = (3, 4)
_INSTALL_HINT = 'install the extra with: pip install "eegdash[zarr]"'

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_ATTEMPTS = 4

# biosigio channel_type -> MNE channel type. Discrete types (biosigio's
# DISCRETE_CHANNEL_TYPES) hold codes, not measurements: stored without
# rescaling or anti-aliasing, and read back as codes.
_TYPE_TO_MNE = {
    "EEG": "eeg",
    "EOG": "eog",
    "HEOG": "eog",
    "VEOG": "eog",
    "ECG": "ecg",
    "EMG": "emg",
    "SEEG": "seeg",
    "ECOG": "ecog",
    "DBS": "dbs",
    "RESP": "resp",
    "GSR": "gsr",
    "TEMP": "temperature",
    "TRIG": "stim",
    "SYSCLOCK": "syst",
    "CTRL": "syst",
    "MISC": "misc",
    "OTHER": "misc",
    "PPG": "misc",
    "ACC": "misc",
    "REF": "misc",
}
_DISCRETE_TYPES = frozenset({"TRIG", "SYSCLOCK", "CTRL"})
_MEG_TYPES = frozenset(
    {"MEGMAG", "MEGGRADPLANAR", "MEGGRADAXIAL", "MEGREFMAG", "MEGREFGRADAXIAL"}
)
_VOLTAGE_MNE_TYPES = frozenset({"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs"})

# Unit spellings seen in NEMAR stores -> factor to SI. "microvol" is EDF's
# 8-character physical-dimension field cut short; µ appears as U+00B5 and U+03BC.
_UNIT_TO_SI = {
    "V": 1.0,
    "mV": 1e-3,
    "uV": 1e-6,
    "µV": 1e-6,
    "μV": 1e-6,
    "microV": 1e-6,
    "microvol": 1e-6,
    "microvolt": 1e-6,
    "nV": 1e-9,
    "T": 1.0,
    "fT": 1e-15,
    "T/m": 1.0,
    "fT/cm": 1e-13,
    "S": 1.0,
    "mS": 1e-3,
    "uS": 1e-6,
    "µS": 1e-6,
    "nS": 1e-9,
    "Celsius": 1.0,
    "DegC": 1.0,
    "°C": 1.0,
}

_EVENT_COLUMNS = [
    "store_path",
    "group_name",
    "onset_s",
    "duration_s",
    "value",
    "trial_type",
]
_READER_OPTIONS = frozenset(
    {"events", "events_max_bytes", "nonfinite", "bids_status", "verify_snapshot"}
)


class NEMARDataWarning(UserWarning):
    """Something about a NEMAR store the caller should know before analysing it."""


class EventsTooLargeError(RuntimeError):
    """The events for a recording exceed the configured download limit."""


# ---------------------------------------------------------------------------
# Dependencies and HTTP
# ---------------------------------------------------------------------------


def _check_dependencies(url: str) -> None:
    found = tuple(int(x) for x in re.findall(r"\d+", zarr.__version__)[:2])
    if found < _MIN_ZARR:
        raise ImportError(
            f"reading NEMAR Zarr stores needs zarr>={'.'.join(map(str, _MIN_ZARR))}, "
            f"found {zarr.__version__}; {_INSTALL_HINT}"
        )
    if url.startswith(("http://", "https://")) and aiohttp is None:
        raise ImportError(
            f"reading NEMAR Zarr stores over HTTPS needs aiohttp; {_INSTALL_HINT}"
        )


_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _session() -> requests.Session:
    """Shared session retrying 429/5xx with backoff, like the EEGDash API client.

    NEMAR's metadata endpoints intermittently answer 500 with an empty body; a
    retry after a short backoff succeeds. Once the retries run out the last
    response is returned so callers surface its status.
    """
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            retry = Retry(
                total=_ATTEMPTS - 1,
                status_forcelist=sorted(_RETRY_STATUS),
                allowed_methods=frozenset({"GET", "HEAD"}),
                backoff_factor=0.5,
                raise_on_status=False,
            )
            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=retry))
            session.mount("http://", HTTPAdapter(max_retries=retry))
            _SESSION = session
        return _SESSION


def _http_get(
    url: str,
    timeout: float,
    *,
    method: str = "GET",
    headers: dict | None = None,
) -> requests.Response:
    """HTTP request through the retrying session (separate so tests can stub it)."""
    return _session().request(
        method, url, timeout=timeout, headers=headers, allow_redirects=True
    )


# ---------------------------------------------------------------------------
# Store access and caching
# ---------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Whether a store-level exception is worth retrying."""
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status in _RETRY_STATUS:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return aiohttp is not None and isinstance(
        exc, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError)
    )


def _is_absent_chunk(exc: BaseException, key: str) -> bool:
    """A chunk of the store's ``events`` arrays that was never written.

    Zarr does not store chunks equal to the fill value (e.g. an all-zero
    duration array), and S3 without list permission answers 403, not 404, for
    absent keys, so the two cannot be told apart by status. That is accepted
    only for the small ``events/*`` arrays, where all-fill chunks really occur.
    For signal chunks a 403/404 is raised: substituting the fill value could
    silently turn an access error into flat data. Missing metadata always fails.
    """
    if not (key.startswith("events/") and "/c/" in key):
        return False
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    return isinstance(exc, FileNotFoundError) or status in (403, 404)


async def _retrying(fn, attempts: int = _ATTEMPTS):
    """Await ``fn()``, retrying transient failures with backoff; re-raise the rest."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise
        await asyncio.sleep(0.25 * 2**attempt)
    raise AssertionError("unreachable")


async def _collect(agen):
    return [group async for group in agen]


class _ByteLRU:
    """Thread-safe LRU of byte buffers, bounded by their total size."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            buf = self._items.get(key)
            if buf is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return buf

    def put(self, key, buf) -> None:
        n = len(buf)
        if n > self.max_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._size -= len(old)
            self._items[key] = buf
            self._size += n
            self._evict()

    def resize(self, max_bytes: int) -> None:
        with self._lock:
            self.max_bytes = max_bytes
            self._evict()

    def _evict(self) -> None:
        while self._items and self._size > self.max_bytes:
            _, evicted = self._items.popitem(last=False)
            self._size -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._size = 0
            self.hits = self.misses = 0

    @property
    def nbytes(self) -> int:
        return self._size


_CHUNK_CACHE = _ByteLRU(_DEFAULT_CHUNK_CACHE_BYTES)


def set_chunk_cache_size(nbytes: int) -> None:
    """Set the per-process limit of the NEMAR chunk cache in bytes (0 disables it).

    The cache is shared by every :class:`RawNEMARZarr` in the process; each
    DataLoader worker process has its own.
    """
    _CHUNK_CACHE.resize(int(nbytes))


def chunk_cache_info() -> dict:
    """Size, limit and hit/miss counts of the per-process chunk cache."""
    return {
        "nbytes": _CHUNK_CACHE.nbytes,
        "max_bytes": _CHUNK_CACHE.max_bytes,
        "hits": _CHUNK_CACHE.hits,
        "misses": _CHUNK_CACHE.misses,
    }


class CachingStore(WrapperStore):
    """Read-only store wrapper: shard-index and chunk caching, retries.

    Shard indexes (~1.2 kB per shard) are kept per store for its lifetime.
    Chunk byte ranges go into the shared, size-bounded LRU keyed by store URL,
    object key and range. Transient failures are retried with backoff and then
    raised. Entries are never invalidated, so only read-only stores may be
    wrapped.
    """

    def __init__(self, store, cache_id: str):
        super().__init__(store)
        self._cache_id = cache_id
        self._index: dict = {}

    def _with_store(self, store):
        return type(self)(store, self._cache_id)

    @property
    def _supports_sync_io(self):
        # Keep zarr on the async paths below so every read is cached and retried.
        return False

    def _range_key(self, key, r):
        if isinstance(r, RangeByteRequest):
            return (self._cache_id, key, r.start, r.end)
        return None

    async def _get(self, key, prototype, byte_range):
        try:
            return await _retrying(lambda: self._store.get(key, prototype, byte_range))
        except Exception as e:
            if _is_absent_chunk(e, key):
                return None  # zarr substitutes the fill value
            raise

    async def _get_ranges(self, key, byte_ranges, prototype, kwargs):
        try:
            return await _retrying(
                lambda: _collect(
                    self._store.get_ranges(
                        key, byte_ranges, prototype=prototype, **kwargs
                    )
                )
            )
        except Exception as e:
            if _is_absent_chunk(e, key):
                return [[(i, None) for i in range(len(byte_ranges))]]
            raise

    async def get(self, key, prototype, byte_range=None):
        if isinstance(byte_range, SuffixByteRequest):
            ik = (key, byte_range.suffix)
            buf = self._index.get(ik)
            if buf is None:
                buf = await self._get(key, prototype, byte_range)
                if buf is not None:
                    self._index[ik] = buf
            return buf
        ck = self._range_key(key, byte_range)
        if ck is not None and _CHUNK_CACHE.max_bytes:
            buf = _CHUNK_CACHE.get(ck)
            if buf is None:
                buf = await self._get(key, prototype, byte_range)
                if buf is not None:
                    _CHUNK_CACHE.put(ck, buf)
            return buf
        return await self._get(key, prototype, byte_range)

    async def get_ranges(self, key, byte_ranges, *, prototype, **kwargs):
        byte_ranges = list(byte_ranges)
        if not _CHUNK_CACHE.max_bytes:
            for group in await self._get_ranges(key, byte_ranges, prototype, kwargs):
                yield group
            return
        hits, miss = [], []
        for i, r in enumerate(byte_ranges):
            ck = self._range_key(key, r)
            buf = _CHUNK_CACHE.get(ck) if ck is not None else None
            if buf is None:
                miss.append(i)
            else:
                hits.append((i, buf))
        if hits:
            yield hits
        if miss:
            fetched = await self._get_ranges(
                key, [byte_ranges[i] for i in miss], prototype, kwargs
            )
            for group in fetched:
                out = []
                for j, buf in group:
                    i = miss[j]
                    ck = self._range_key(key, byte_ranges[i])
                    if ck is not None and buf is not None:
                        _CHUNK_CACHE.put(ck, buf)
                    out.append((i, buf))
                yield out


def _make_base_store(url: str, storage_options: dict | None):
    """Uncached read-only store for ``url`` (separate so tests can instrument it)."""
    return FsspecStore.from_url(url, storage_options=storage_options, read_only=True)


@lru_cache(maxsize=64)
def _open_root(url: str, storage_options_json: str):
    """Open (and cache per process) a store's root group through CachingStore."""
    storage_options = json.loads(storage_options_json) or None
    store = CachingStore(_make_base_store(url, storage_options), cache_id=url)
    # zarr_format=3 skips the v2 metadata probes, which NEMAR's S3 answers with 403.
    return zarr.open_group(store, mode="r", zarr_format=3)


@lru_cache(maxsize=64)
def _open_level0(url: str, group: str, storage_options_json: str):
    """Level-0 array of one channel group, opened once per process.

    Cached apart from the root: ``root[group]["0"]`` re-reads two ``zarr.json``
    documents each time, which would cost two round trips per read.
    """
    return _open_root(url, storage_options_json)[group]["0"]


def _split_store_url(url: str) -> tuple[str, str]:
    """``<data_base>`` and the store path within it, from ``.../<ds>/zarr/<path>``."""
    head, sep, tail = url.partition("/zarr/")
    if not sep or not tail:
        raise ValueError(f"cannot locate the dataset's zarr/ prefix in {url!r}")
    return head + "/zarr/", tail


@lru_cache(maxsize=16)
def _fetch_index(dataset: str) -> dict:
    r = _http_get(INDEX_URL.format(ds=dataset), timeout=60)
    r.raise_for_status()
    return r.json()


def _data_base(index: dict, dataset: str) -> str:
    # data_base only exists from index format_version 3; the contract base
    # always works (it redirects to S3).
    return index.get("data_base") or CONTRACT_BASE.format(ds=dataset)


def _index_entry(dataset: str, recording: str) -> tuple[str, dict, dict]:
    """Store URL, index entry and index for a BIDS recording path."""
    index = _fetch_index(dataset)
    for store in index["stores"]:
        if store["path"] == recording:
            return _data_base(index, dataset) + store["zarr"], store, index
    raise LookupError(
        f"{recording!r} has no Zarr copy in the {dataset} index "
        f"({len(index['stores'])} stores)"
    )


def find_store_url(dataset: str, recording: str) -> str:
    """Resolve a BIDS recording path to its Zarr store URL via NEMAR's index."""
    return _index_entry(dataset, recording)[0]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class _HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file fetched with Range requests.

    Bytes already fetched are kept and reused, so overlapping reads (pyarrow
    reads the last 64 KiB for the footer, then column chunks that may lie in
    it) never download the same range twice.
    """

    def __init__(self, url: str, size: int, timeout: float = 60):
        self.url, self.size, self.timeout = url, size, timeout
        self.pos = self.bytes_fetched = self.requests = 0
        self._blocks: list[tuple[int, bytes]] = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for start, data in self._blocks:
            if start <= self.pos and end < start + len(data):
                n = end - self.pos + 1
                b[:n] = data[self.pos - start : self.pos - start + n]
                self.pos += n
                return n
        r = _http_get(
            self.url, timeout=self.timeout, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        if r.status_code != 206:  # a 200 would be the whole file: refuse it
            r.raise_for_status()
            raise OSError(
                f"server ignored the Range request for {self.url} (HTTP {r.status_code})"
            )
        n = len(r.content)
        b[:n] = r.content
        self._blocks.append((self.pos, r.content))
        self.pos += n
        self.bytes_fetched += n
        self.requests += 1
        return n


@lru_cache(maxsize=32)
def _events_from_parquet(url: str, group: str, max_bytes: int | None):
    """Fetch this recording's rows of ``events.parquet`` with Range requests.

    Only the footer and the row groups whose ``store_path`` statistics can hold
    this recording are downloaded, and only the needed columns. Raises
    :class:`EventsTooLargeError` above ``max_bytes``. Returns
    ``(onset, duration, description, provenance)``.
    """
    data_base, store_path = _split_store_url(url)
    events_url = data_base + "events.parquet"
    head = _http_get(events_url, timeout=30, method="HEAD")
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    raw_file = _HTTPRangeFile(events_url, size)
    pf = pq.ParquetFile(raw_file)
    md = pf.metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    col = {n: names.index(n) for n in _EVENT_COLUMNS}
    selected = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(col["store_path"]).statistics
        if st is None or not st.has_min_max or st.min <= store_path <= st.max:
            selected.append(i)
    planned = sum(
        md.row_group(i).column(col[c]).total_compressed_size
        for i in selected
        for c in _EVENT_COLUMNS
    )
    if max_bytes is not None and planned > max_bytes:
        raise EventsTooLargeError(
            f"this recording's events need {planned / 1e6:.1f} MB of {events_url} "
            f"({len(selected)} of {md.num_row_groups} row groups), above the "
            f"{max_bytes / 1e6:.1f} MB limit"
        )
    df = pf.read_row_groups(selected, columns=_EVENT_COLUMNS).to_pandas()
    df = df[
        (df["store_path"].astype(str) == store_path)
        & (df["group_name"].astype(str) == group)
    ]
    # Label by the BIDS ``value`` (as MNE's EEGLAB reader does); trial_type if absent.
    labels = (
        df["value"]
        .astype("string")
        .fillna(df["trial_type"].astype("string"))
        .fillna("n/a")
    )
    # duration_s is float32 in the parquet (0.1 reads back as 0.10000000149);
    # BIDS durations are decimal text, so recover each float32's shortest form.
    durations = np.array(
        [float(str(d)) for d in df["duration_s"].to_numpy(dtype=np.float32)]
    )
    provenance = {
        "source": "events.parquet",
        "url": events_url,
        "etag": head.headers.get("ETag"),
        "last_modified": head.headers.get("Last-Modified"),
        "file_bytes": size,
        "row_groups_read": selected,
        "bytes_fetched": raw_file.bytes_fetched,
        "range_requests": raw_file.requests,
        "n_events": int(len(df)),
        "commit_pinned": False,
        "note": "dataset-level file, replaced when NEMAR re-exports; "
        "identify the version by etag/last_modified",
    }
    return (
        df["onset_s"].to_numpy(dtype=float),
        np.nan_to_num(durations, nan=0.0),
        labels.astype(str).to_numpy(),
        provenance,
    )


def _events_from_store(root):
    """In-store events: exact onsets and durations, but trial_type labels only."""
    empty = (
        mne.Annotations([], [], []),
        {"source": "store events group", "n_events": 0},
    )
    # Opened by name: NEMAR's S3 refuses listing. A missing events group shows
    # up as KeyError locally and as HTTP 403 remotely.
    try:
        eg = root["events"]
    except Exception as e:
        warnings.warn(
            f"store has no readable events group ({type(e).__name__}); "
            "no annotations added",
            NEMARDataWarning,
            stacklevel=4,
        )
        return empty
    if not dict(eg.attrs).get("n_events"):
        return empty
    label_map = dict(eg.attrs).get("label_map", {})
    codes = eg["code"][:]
    ann = mne.Annotations(
        onset=eg["onset"][:].astype(float),
        duration=eg["duration"][:].astype(float),
        description=[label_map.get(str(int(c)), str(int(c))) for c in codes],
    )
    return ann, {
        "source": "store events group",
        "n_events": len(ann),
        "labels": "trial_type only",
        "commit_pinned": True,
        "note": "written with the signal from source_commit",
    }


def _load_annotations(events, url, group, root, max_bytes):
    """Annotations and their provenance under the chosen events policy."""
    if events is None:
        return None, {"source": None}
    if events not in ("auto", "parquet", "store"):
        raise ValueError(
            f"events must be 'auto', 'parquet', 'store' or None, not {events!r}"
        )
    if events in ("auto", "parquet"):
        reason = None
        if not url.startswith(("http://", "https://")):
            reason = "events.parquet is only fetched over HTTP(S)"
        elif pq is None:
            reason = f"pyarrow is not installed ({_INSTALL_HINT})"
        if reason is None:
            try:
                onset, duration, desc, prov = _events_from_parquet(
                    url, group, max_bytes
                )
                return mne.Annotations(onset, duration, desc), dict(prov)
            except EventsTooLargeError as e:
                if events == "parquet":
                    raise EventsTooLargeError(
                        f"{e}; pass events_max_bytes=None to download it anyway"
                    ) from None
                reason = str(e)
            except (requests.RequestException, OSError) as e:
                if events == "parquet":
                    raise
                reason = f"events.parquet could not be read ({type(e).__name__}: {e})"
        if events == "parquet":
            raise RuntimeError(f"cannot read events.parquet: {reason}")
        warnings.warn(
            f"{reason}; using the store's own events, which carry trial_type "
            "labels only (no per-event BIDS value)",
            NEMARDataWarning,
            stacklevel=4,
        )
    return _events_from_store(root)


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------


def _channel_types(chans) -> tuple[list[str], list[str]]:
    """MNE types for the store's channels, plus notes about lossy mappings."""
    types, meg, unknown = [], [], set()
    for c in chans:
        t = str(c.get("channel_type", "MISC")).upper()
        if t in _TYPE_TO_MNE:
            types.append(_TYPE_TO_MNE[t])
        else:
            types.append("misc")
            if t in _MEG_TYPES:
                meg.append(c["label"])
            else:
                unknown.add(t)
    notes = []
    if meg:
        notes.append(
            f"{len(meg)} MEG channels are read as 'misc': the store carries no "
            "sensor geometry or coil types"
        )
    if unknown:
        notes.append(f"unrecognised channel types {sorted(unknown)} are read as 'misc'")
    return types, notes


def _calibrations(chans, scale, mne_types) -> tuple[np.ndarray, list[str]]:
    """Per-channel factor from stored digital values to MNE's SI units."""
    cal, unclear = np.empty(len(chans)), []
    for i, (c, t) in enumerate(zip(chans, mne_types)):
        if str(c.get("channel_type", "")).upper() in _DISCRETE_TYPES:
            # Codes are dequantised but never unit-converted, whatever their label.
            cal[i] = scale[i]
            continue
        factor = _UNIT_TO_SI.get(str(c.get("unit", "n/a")).strip())
        if factor is None:
            factor = 1.0
            if t in _VOLTAGE_MNE_TYPES:
                unclear.append(f"{c['label']} ({c.get('unit')!r})")
        cal[i] = scale[i] * factor
    return cal, unclear


def _integer_code_rows(chans, scale, offset) -> np.ndarray:
    """Discrete channels whose stored range shows exactly recoverable integer codes.

    biosigio maps each channel's [min, max] onto the full int16 range, so
    ``min = offset - 32768 * scale`` and ``max = min + 65535 * scale``. When
    both ends are integers and one step is below 1, rounding restores every
    code exactly (int16 storage otherwise turns 4 into 3.99994, which MNE's
    ``find_events`` truncates to 3). Analog trigger channels are left alone.
    """
    lo = offset - 32768.0 * scale
    hi = lo + 65535.0 * scale
    discrete = np.array(
        [str(c.get("channel_type", "")).upper() in _DISCRETE_TYPES for c in chans]
    )
    integral = (np.abs(lo - np.round(lo)) < 1e-3) & (np.abs(hi - np.round(hi)) < 1e-3)
    return discrete & integral & (scale < 1.0)


def _read_channels_tsv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _bids_bads(url: str, attrs: dict, ch_names: list[str]) -> dict | None:
    """Bad channels from the BIDS channels.tsv the store was built from.

    Read at the store's own ``source_commit`` from the dataset's GitHub mirror,
    so the status matches the exact BIDS snapshot that was converted. The
    sidecar is first looked for under the recording's own name (right for
    ~99% of NEMAR recordings); only if that is missing is the dataset index
    consulted. Returns None when there is no status; raises on HTTP errors.
    """
    nemar = attrs.get("nemar") or {}
    ds, commit = nemar.get("dataset_id"), nemar.get("source_commit")
    if not ds or not commit:
        return None
    _, store_path = _split_store_url(url)
    sidecar = store_path.rsplit("_", 1)[0] + "_channels.tsv"
    r = _http_get(GITHUB_RAW.format(ds=ds, commit=commit, path=sidecar), timeout=30)
    if r.status_code == 404:
        # e.g. split recordings share an unsplit sidecar, which the index records
        recorded = next(
            (
                (st.get("units_report") or {}).get("sidecar")
                for st in _fetch_index(ds)["stores"]
                if st["zarr"] == store_path
            ),
            None,
        )
        if not recorded or recorded == sidecar:
            return None
        sidecar = recorded
        r = _http_get(GITHUB_RAW.format(ds=ds, commit=commit, path=sidecar), timeout=30)
        if r.status_code == 404:
            return None
    r.raise_for_status()
    rows = _read_channels_tsv(r.content.decode("utf-8-sig", "replace"))
    if not rows or "status" not in rows[0] or "name" not in rows[0]:
        return None
    bads = [
        row["name"]
        for row in rows
        if (row.get("status") or "").strip().lower() == "bad"
    ]
    missing = [b for b in bads if b not in ch_names]
    if missing:
        warnings.warn(
            f"channels.tsv marks {missing} bad but the store has no such channels; "
            "ignoring them",
            NEMARDataWarning,
            stacklevel=5,
        )
    return {
        "bads": [b for b in bads if b in ch_names],
        "sidecar": sidecar,
        "commit": commit,
        "url": GITHUB_RAW.format(ds=ds, commit=commit, path=sidecar),
        "ignored_missing": missing,
    }


def _montage_from_attrs(attrs, ch_names):
    """Head-frame montage from the store's electrode_positions, if any."""
    pos = attrs.get("electrode_positions")
    if not pos or attrs.get("electrode_coordinate_system") != "EEGLAB":
        # Only EEGLAB's convention (x = nose, y = left) is known to map onto
        # MNE's head frame (x = right, y = nose); don't guess for others.
        return None
    to_m = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[
        attrs.get("electrode_coordinate_units", "mm")
    ]
    ch_pos = {
        name: np.array([-y, x, z]) * to_m
        for name, (x, y, z) in pos.items()
        if name in ch_names
    }
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def _apply_calibration(data, block, idx, cals, mult) -> None:
    """Calibrate the rows ``idx`` of ``block`` into ``data`` (MNE's segment contract)."""
    if mult is not None:
        data[:] = mult @ block[idx]
    else:
        np.multiply(block[idx], np.asarray(cals).reshape(-1, 1), out=data)


def _provenance(
    url,
    group,
    attrs,
    gattrs,
    aattrs,
    chans,
    units,
    scale,
    filled,
    unclear,
    rounded,
    bids_prov,
    bids_status,
):
    nemar = attrs.get("nemar") or {}
    rec = attrs.get("recording_metadata") or {}
    discrete = [
        c["label"]
        for c in chans
        if str(c.get("channel_type", "")).upper() in _DISCRETE_TYPES
    ]
    not_aa = [
        c["label"]
        for c in chans
        if c.get("anti_aliased") is False
        and c["label"] not in discrete
        and c.get("original_rate") != c.get("target_rate")
    ]
    steps = {}
    for u in sorted(set(units)):
        su = scale[[i for i, x in enumerate(units) if x == u]]
        steps[u] = {
            "min": float(su.min()),
            "median": float(np.median(su)),
            "max": float(su.max()),
        }
    factors = {
        u: _UNIT_TO_SI.get(u.strip(), 1.0)
        for c, u in zip(chans, units)
        if c["label"] not in discrete
    }
    return {
        "reader": "eegdash.dataset.nemar_zarr",
        "store_url": url,
        "group": group,
        "dataset_id": nemar.get("dataset_id"),
        "doi": nemar.get("doi"),
        "license": nemar.get("license"),
        "citation": nemar.get("citation"),
        "source_commit": nemar.get("source_commit"),
        "source_tree": nemar.get("source_tree"),
        "derived_dataset": nemar.get("derived"),
        "source_file": rec.get("source_file"),
        "source_format": attrs.get("source_format"),
        "store_format": f"{attrs.get('format')} v{attrs.get('format_version')}",
        "biosigio_version": attrs.get("biosigio_version"),
        "engine_version": nemar.get("engine_version"),
        "store_created_utc": attrs.get("created_utc"),
        "nemar_transformations": {
            "resampling": {
                "original_rate_hz": gattrs.get("original_rate"),
                "served_rate_hz": gattrs.get("rate"),
                "continuous_channels": attrs.get("anti_alias_filter"),
                "discrete_channels": "nearest sample, no anti-aliasing"
                if discrete
                else None,
                "channels_resampled_without_anti_aliasing": not_aa,
            },
            "quantization": {
                "dtype": attrs.get("dtype"),
                "formula": aattrs.get(
                    "physical_formula", "physical = digital * scale + offset"
                ),
                "scheme": "per channel over its own range: "
                "scale = (max - min) / 65535, min -> -32768",
                "step_by_unit": steps,
            },
            "nonfinite_zero_filled": filled,
            "channels_tsv_units": attrs.get("channels_tsv_units"),
            "note": attrs.get("note"),
        },
        "reader_adjustments": {
            "unit_factor_to_SI": factors,
            "units_left_as_stored": unclear,
            "discrete_channels_not_unit_converted": discrete,
            "integer_code_rounding": rounded,
            "view_pyramid": "not read (render-only envelopes)",
        },
        "bids_status": bids_prov if bids_status else {"source": None, "disabled": True},
    }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class RawNEMARZarr(BaseRaw):
    """Raw object backed by a NEMAR Zarr store, read lazily.

    Parameters
    ----------
    url : str
        Store URL, e.g. ``https://nemar.s3.us-east-2.amazonaws.com/<ds>/zarr/<rec>.zarr``.
    group : str | None
        Channel group to read (e.g. ``"eeg_250hz"``). Required only if the
        store has more than one.
    events : {"auto", "parquet", "store"} | None
        Where annotations come from. ``"parquet"`` reads this recording's rows
        of the dataset's ``events.parquet`` with HTTP Range requests and labels
        each event by its BIDS ``value``. ``"store"`` reads the store's
        ``events`` group, whose labels are ``trial_type`` only. ``"auto"``
        (default) uses the parquet when it can within ``events_max_bytes`` and
        otherwise falls back to the store with a :class:`NEMARDataWarning`.
        ``None`` adds no annotations.
    events_max_bytes : int | None
        Most bytes of ``events.parquet`` to download for this recording
        (default 16 MB). ``events="auto"`` falls back above it;
        ``events="parquet"`` raises :class:`EventsTooLargeError`. ``None``
        removes the limit.
    nonfinite : {"warn", "bad", "ignore"}
        Channels in which NEMAR replaced NaN/inf samples with 0 (positions are
        not recorded). ``"warn"`` (default) issues a :class:`NEMARDataWarning`;
        ``"bad"`` also adds them to ``info["bads"]``. Often every channel is
        affected because a time segment was missing, in which case marking them
        bad leaves nothing to interpolate from. Counts are in
        :attr:`nonfinite_samples`.
    bids_status : bool
        Mark channels bad as declared by the ``status`` column of the BIDS
        ``channels.tsv`` the store was converted from (fetched at the store's
        ``source_commit``). Default True; skipped silently without a status
        column, with a :class:`NEMARDataWarning` if the sidecar cannot be read.
    storage_options : dict | None
        Passed to fsspec when opening the store.
    preload : bool
        Load all samples into memory now instead of on demand.
    verbose : bool | str | int | None
        MNE verbosity.

    """

    def __init__(
        self,
        url,
        group=None,
        *,
        events="auto",
        events_max_bytes=EVENTS_MAX_BYTES,
        nonfinite="warn",
        bids_status=True,
        storage_options=None,
        preload=False,
        verbose=None,
    ):
        if nonfinite not in ("warn", "bad", "ignore"):
            raise ValueError(
                f"nonfinite must be 'warn', 'bad' or 'ignore', not {nonfinite!r}"
            )
        url = url.rstrip("/")
        _check_dependencies(url)
        so_json = json.dumps(storage_options or {}, sort_keys=True)
        root = _open_root(url, so_json)
        attrs = dict(root.attrs)
        groups = attrs.get("channel_groups")
        if not groups:
            # Listing the store to discover groups isn't possible on NEMAR's S3.
            raise ValueError(
                "store root has no 'channel_groups' attribute; is this a biosigio-zarr store?"
            )
        if group is None:
            if len(groups) != 1:
                raise ValueError(
                    f"store has several channel groups {groups}; pass group="
                )
            group = groups[0]
        elif group not in groups:
            raise ValueError(f"group {group!r} not in store; available: {groups}")

        arr = _open_level0(url, group, so_json)
        gattrs = dict(root[group].attrs)
        chans = gattrs["channels"]
        if [c["row_index"] for c in chans] != list(range(arr.shape[0])):
            raise RuntimeError("channel metadata does not line up with array rows")
        scale = np.asarray(arr.attrs["scale"], dtype=np.float64)
        offset = np.asarray(arr.attrs["offset"], dtype=np.float64)

        ch_names = [c["label"] for c in chans]
        ch_types, type_notes = _channel_types(chans)
        info = mne.create_info(ch_names, float(gattrs["rate"]), ch_types)
        for note in type_notes:
            warnings.warn(note, NEMARDataWarning, stacklevel=2)

        # Calibration turns (digital + offset/scale) into SI units:
        #   (d + o/s) * cal == (d * s + o) * unit_factor  with  cal = s * unit_factor
        units = [str(c.get("unit", "n/a")) for c in chans]
        cals, unclear = _calibrations(chans, scale, ch_types)
        if unclear:
            warnings.warn(
                f"{len(unclear)} voltage channels have units with no known SI factor "
                f"and are left as stored (e.g. {', '.join(unclear[:5])})",
                NEMARDataWarning,
                stacklevel=2,
            )
        for ch, cal in zip(info["chs"], cals):
            ch["cal"] = cal
            ch["range"] = 1.0
        nemar = attrs.get("nemar", {})
        info["line_freq"] = attrs.get("power_line_frequency")
        info["description"] = (
            f"NEMAR {nemar.get('dataset_id')} {url.rsplit('/', 1)[-1]} [{group}] "
            f"doi:{nemar.get('doi')} commit:{nemar.get('source_commit')} "
            "(derived serving copy; see raw.nemar_provenance)"
        )

        montage = _montage_from_attrs(attrs, ch_names)
        if montage is not None:
            info.set_montage(montage, on_missing="ignore")

        bads, bids_prov = [], None
        if bids_status:
            try:
                bids_prov = _bids_bads(url, attrs, ch_names)
                bads = list(bids_prov["bads"]) if bids_prov else []
            except requests.RequestException as e:
                bids_prov = {"error": f"{type(e).__name__}: {e}"}
                warnings.warn(
                    f"could not read BIDS channel status ({e}); no channels marked bad",
                    NEMARDataWarning,
                    stacklevel=2,
                )
        lo = offset - 32768.0 * scale
        lossy = [
            c["label"]
            for c, sc, low in zip(chans, scale, lo)
            if str(c.get("channel_type", "")).upper() in _DISCRETE_TYPES
            and sc > 1.0
            and abs(low - round(low)) < 1e-3
        ]
        if lossy:
            warnings.warn(
                f"trigger/code channels {lossy} span more values than int16 can hold "
                "exactly (step > 1), so their codes are approximate in this store; "
                "use the annotations, or the BIDS original, for events",
                NEMARDataWarning,
                stacklevel=2,
            )
        filled = {
            c["label"]: int(c["nonfinite_samples"])
            for c in chans
            if c.get("nonfinite_samples")
        }
        if filled and nonfinite != "ignore":
            fr = sorted(v / arr.shape[1] for v in filled.values())
            everyone = len(filled) == len(ch_names)
            warnings.warn(
                f"{len(filled)} of {len(ch_names)} channels had NaN/inf samples that "
                f"NEMAR replaced with 0 ({fr[0]:.2%}-{fr[-1]:.2%} of samples; "
                f"positions not recorded): {', '.join(list(filled)[:8])}"
                f"{' ...' if len(filled) > 8 else ''}. "
                + (
                    "Every channel is affected, which points to missing time segments "
                    "rather than bad channels. "
                    if everyone
                    else ""
                )
                + (
                    "They are marked bad (nonfinite='bad')."
                    if nonfinite == "bad"
                    else "Pass nonfinite='bad' to mark them bad; see raw.nonfinite_samples."
                ),
                NEMARDataWarning,
                stacklevel=2,
            )
            if nonfinite == "bad":
                bads += [c for c in filled if c not in bads]
        info["bads"] = bads

        round_rows = _integer_code_rows(chans, scale, offset)
        provenance = _provenance(
            url,
            group,
            attrs,
            gattrs,
            dict(arr.attrs),
            chans,
            units,
            scale,
            filled,
            unclear,
            [ch_names[i] for i in np.flatnonzero(round_rows)],
            bids_prov,
            bids_status,
        )
        raw_extras = {
            "url": url,
            "group": group,
            "storage_options_json": so_json,
            "offset_over_scale": offset / scale,
            "nonfinite_samples": filled,
            "round_rows": round_rows,
            "provenance": provenance,
        }
        super().__init__(
            info,
            preload=preload,
            last_samps=[arr.shape[1] - 1],
            filenames=[None],  # MNE requires local paths; the URL lives in raw_extras
            raw_extras=[raw_extras],
            orig_format="short",
            orig_units=dict(zip(ch_names, units)),
            verbose=verbose,
        )

        annotations, events_prov = _load_annotations(
            events, url, group, root, events_max_bytes
        )
        self._raw_extras[0]["provenance"]["events"] = events_prov
        if annotations is not None:
            self.set_annotations(annotations)

    @property
    def nemar_provenance(self) -> dict | list[dict]:
        """Where this recording came from and every transformation applied to it.

        One dict per source store (a list after ``concatenate_raws``).
        """
        provs = [extra["provenance"] for extra in self._raw_extras]
        return provs[0] if len(provs) == 1 else provs

    @property
    def nonfinite_samples(self) -> dict[str, int]:
        """Per channel, how many samples NEMAR zero-filled because they were NaN/inf."""
        out: dict[str, int] = {}
        for extra in self._raw_extras:
            for ch, n in extra.get("nonfinite_samples", {}).items():
                if ch in self.ch_names:
                    out[ch] = out.get(ch, 0) + n
        return out

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Fetch samples [start, stop) of every channel; calibrate the rows in idx."""
        extra = self._raw_extras[fi]
        arr = _open_level0(extra["url"], extra["group"], extra["storage_options_json"])
        # Each chunk spans all channels, so one request serves every row; repeated
        # per-channel calls (e.g. MNE's non-preloaded resample) hit the chunk cache.
        block = arr[:, start:stop].astype(np.float64)
        block += extra["offset_over_scale"][:, np.newaxis]
        _apply_calibration(data, block, idx, cals, mult)
        round_rows = extra["round_rows"]
        if mult is None and round_rows.any():
            sel = round_rows[np.arange(round_rows.size)[idx]]  # data rows follow idx
            if sel.any():
                data[sel] = np.round(data[sel])


def read_raw_nemar_zarr(
    url_or_dataset,
    recording=None,
    *,
    group=None,
    events="auto",
    events_max_bytes=EVENTS_MAX_BYTES,
    nonfinite="warn",
    bids_status=True,
    storage_options=None,
    preload=False,
    verbose=None,
):
    """Open a NEMAR Zarr recording as a lazy :class:`RawNEMARZarr`.

    Pass either a store URL, or a dataset id plus the BIDS recording path as
    listed in the dataset's Zarr ``index.json``.

    Examples
    --------
    >>> raw = read_raw_nemar_zarr("nm000132", "sub-001/eeg/sub-001_task-MMN_eeg.set")
    >>> raw.get_data(start=0, stop=500).shape
    (33, 500)

    """
    url = (
        url_or_dataset
        if recording is None
        else find_store_url(url_or_dataset, recording)
    )
    return RawNEMARZarr(
        url,
        group,
        events=events,
        events_max_bytes=events_max_bytes,
        nonfinite=nonfinite,
        bids_status=bids_status,
        storage_options=storage_options,
        preload=preload,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# EEGDash integration
# ---------------------------------------------------------------------------


def _annex_key_at_commit(dataset: str, commit: str, path: str) -> str | None:
    """git-annex key of ``path`` at ``commit``, or None if it is not annexed."""
    r = _http_get(GITHUB_RAW.format(ds=dataset, commit=commit, path=path), timeout=30)
    r.raise_for_status()
    try:
        target = r.content.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if "/annex/objects/" not in target:
        return None
    return target.rsplit("/", 1)[-1]


class EEGDashZarrRaw(EEGDashRaw):
    """An EEGDash recording read lazily from NEMAR's Zarr serving copy.

    Behaves like :class:`~eegdash.dataset.base.EEGDashRaw`, but ``.raw`` is a
    :class:`RawNEMARZarr` streamed from NEMAR instead of a downloaded BIDS
    file: nothing is written to ``cache_dir``. The signal is NEMAR's derived
    copy (resampled, int16-quantised; see the module documentation), so its
    sampling rate and length differ from the BIDS record's.

    Parameters
    ----------
    record : dict
        A NEMAR record (``storage.backend == "nemar"``).
    cache_dir : str
        Kept for interface compatibility; nothing is written there.
    zarr_options : dict | None
        Options for the reader: ``events``, ``events_max_bytes``,
        ``nonfinite`` and ``bids_status`` (see :class:`RawNEMARZarr`), and
        ``verify_snapshot`` (default True).
    **kwargs
        Passed to :class:`~eegdash.dataset.base.EEGDashRaw`.

    Notes
    -----
    With ``verify_snapshot`` the reader checks, before streaming, that the
    store was converted from the same raw file the record describes: the
    record's git-annex key for its raw file must equal that file's key at the
    store's ``source_commit``. A mismatch raises
    :class:`~eegdash.dataset.exceptions.DataIntegrityError` (handled by
    ``on_error``), since the signal would not match the record's metadata.
    When either key is unavailable the check is skipped with a
    :class:`NEMARDataWarning` rather than blocking loading.

    """

    def __init__(self, record: dict[str, Any], cache_dir: str, **kwargs):
        options = dict(kwargs.pop("zarr_options", None) or {})
        unknown = set(options) - _READER_OPTIONS
        if unknown:
            raise ValueError(
                f"unknown zarr_options {sorted(unknown)}; allowed: {sorted(_READER_OPTIONS)}"
            )
        super().__init__(record, cache_dir, **kwargs)
        backend = (self.record.get("storage") or {}).get("backend")
        if backend != "nemar":
            raise ValueError(
                f"signal_source='zarr' needs NEMAR records; {self.record['dataset']} "
                f"uses storage backend {backend!r}"
            )
        self._verify_snapshot = options.pop("verify_snapshot", True)
        self._reader_options = options

    # -- location and metadata from the NEMAR index (one fetch per dataset) --

    def _zarr_location(self) -> tuple[str, dict, dict]:
        try:
            return _index_entry(self.record["dataset"], self.record["bids_relpath"])
        except LookupError as e:
            raise DataIntegrityError(
                message=str(e),
                record=self.record,
                issues=["NEMAR has no Zarr copy of this recording"],
            ) from e

    def _zarr_group(self) -> dict:
        return self._zarr_location()[1]["groups"][0]

    def _check_snapshot(self, url: str, index: dict) -> dict:
        commit = index.get("source_commit")
        raw_key = self.record["bids_relpath"]
        expected = ((self.record.get("storage") or {}).get("annex_keys") or {}).get(
            raw_key
        )
        result = {"record_annex_key": expected, "source_commit": commit}
        if not expected or not commit:
            result["status"] = "unverified: record has no annex key or index no commit"
            warnings.warn(
                f"cannot verify that {url} matches the record ({result['status']})",
                NEMARDataWarning,
                stacklevel=4,
            )
            return result
        try:
            found = _annex_key_at_commit(self.record["dataset"], commit, raw_key)
        except requests.RequestException as e:
            result["status"] = f"unverified: {type(e).__name__}: {e}"
            warnings.warn(
                f"cannot verify that {url} matches the record ({e})",
                NEMARDataWarning,
                stacklevel=4,
            )
            return result
        result["store_annex_key"] = found
        if found != expected:
            raise DataIntegrityError(
                message=(
                    f"NEMAR Zarr store {url} was converted from a different version of "
                    f"{raw_key} than the EEGDash record describes"
                ),
                record=self.record,
                issues=[
                    f"record annex key {expected}",
                    f"annex key at source_commit {commit}: {found}",
                ],
            )
        result["status"] = "verified"
        return result

    # -- EEGDashRaw overrides --

    def _download_required_files(self) -> None:
        """Nothing to download: the signal is streamed from the store."""

    def _fetch_nemar_session_metadata(self, filesystem, seen_scans) -> None:
        """No BIDS session files are needed for the Zarr copy."""

    def _ensure_raw(self) -> None:
        if self._raw is not None:
            return
        url, _, index = self._zarr_location()
        snapshot = (
            self._check_snapshot(url, index)
            if self._verify_snapshot
            else {"status": "not checked (verify_snapshot=False)"}
        )
        try:
            raw = RawNEMARZarr(url, **self._reader_options)
        except (OSError, requests.RequestException) as e:
            raise DataIntegrityError(
                message=f"could not open NEMAR Zarr store {url}: {e}",
                record=self.record,
                issues=[str(e)],
            ) from e
        raw._raw_extras[0]["provenance"]["eegdash_snapshot_check"] = snapshot
        self._raw = raw

    def __len__(self) -> int:
        if self._skipped:
            return 0
        if self._raw is not None:
            return len(self._raw)
        try:
            return int(self._zarr_group()["n_samples"])
        except Exception as e:
            logger.warning(
                "Could not read NEMAR Zarr metadata for %s, marking as invalid "
                "(length=0). Error: %s",
                self.record.get("bids_relpath"),
                e,
            )
            return 0

    def _make_mne_info(self):
        """:class:`mne.Info` at the store's rate, without loading the signal."""
        if self._raw is not None:
            return self._raw.info
        group = self._zarr_group()
        ch_names = list(self.record.get("ch_names") or [])
        if len(ch_names) != int(group["n_channels"]):
            ch_names = [f"ch{i}" for i in range(int(group["n_channels"]))]
        ch_type = (self.record.get("datatype") or "misc").lower()
        if ch_type not in mne.io.get_channel_type_constants():
            ch_type = "misc"
        return mne.create_info(ch_names, sfreq=float(group["rate"]), ch_types=ch_type)

    def _build_repr(self):
        if self._raw is not None:
            return super()._build_repr()
        b = _ReprBuilder(type(self).__name__)
        try:
            group = self._zarr_group()
            n_ch, sfreq, n_times = (
                int(group["n_channels"]),
                float(group["rate"]),
                int(group["n_samples"]),
            )
            b.add_header(f"{n_ch} ch", "Channels", str(n_ch))
            b.add_header(f"{sfreq:.1f} Hz", "Sfreq", f"{sfreq:.1f} Hz")
            b.add_header(
                f"{n_times} samples ({n_times / sfreq:.1f} s)",
                "Samples",
                f"{n_times} ({n_times / sfreq:.1f} s)",
            )
        except Exception as e:
            b.add_header("Zarr metadata unavailable", "Status", type(e).__name__)
        if self.description is not None:
            b.add_row(
                "description",
                ", ".join(f"{k}={v}" for k, v in self.description.items()),
            )
        b.add_footnote("NEMAR Zarr serving copy (resampled, int16); not loaded yet.")
        return b
