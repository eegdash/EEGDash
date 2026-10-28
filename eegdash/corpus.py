# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Fetch, resume, verify and hold a corpus of datasets: :class:`EEGDashCorpus`.

A corpus is a list of dataset ids (a named table, a CSV with a ``dataset_id``
column, or the ids themselves). The object plans every expected file from the
catalogue, downloads them resumably into the EEGDash cache, checks them against
their records and answers "is this corpus on disk, whole?" from per-dataset
manifests without touching the network::

    corpus = EEGDashCorpus(["ds004395", "nm000289"], cache_dir="/data/cache")
    plan = corpus.plan()                        # catalogue only
    report = corpus.download(n_jobs=16, n_datasets=4, time_budget=6 * 3600)
    report = corpus.verify(level="hash")        # writes COMPLETE per dataset
    status = corpus.status()                    # offline, from the manifests
    for rec in corpus.records():                # every recording, offline
        ...

Command line::

    python -m eegdash.corpus plan eegdash-corpus-v1 --cache-dir /data/cache
    python -m eegdash.corpus download ds002718,ds004395 --n-jobs 8
    python -m eegdash.corpus verify my-corpus.csv --level hash
    python -m eegdash.corpus status my-corpus.csv --online

Cache layout, per dataset: the files as ``EEGDashDataset`` lays them out, plus
``<dataset>/.eegdash/manifest.jsonl`` (one line per expected file: path, size,
git-annex key, state, bytes seen, when, error class, message) and
``<dataset>/.eegdash/COMPLETE``, written by :meth:`EEGDashCorpus.verify` only.
Methods never raise on a bad file: they return a :class:`CorpusReport`.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
import requests
from botocore.exceptions import BotoCoreError, ClientError

from . import downloader
from .dataset._source_inference import nemar_twin_of
from .logging import logger
from .paths import get_default_cache_dir

_MANIFEST_DIR = ".eegdash"
_DIRECTORY_SUFFIXES = (".ds", ".mefd")
_PRESENT_STATES = ("done", "stale-record")
# The record lists the file without a git-annex key and the archive answers "not found": a companion the
# digest assumed (on002680 lists an .fdt beside every .set that embeds its data; nm000181 a README that is
# README.md), 2026-10-27. Reported, not failed, as EEGDashDataset skips it; retried by the next download.
_ABSENT = "absent"
_RETRYABLE_ERRORS = ("network", "integrity", "unknown")
_RETRY_ROUNDS = 3
_RETRY_BACKOFF = (5.0, 20.0, 60.0)  # seconds before each retry round
_LOCK_STALE_SECONDS = 24 * 3600  # a lock from another host older than this
_MANIFEST_LOCK_WAIT = 60.0  # seconds to wait for the manifest lock
_PART_STALE_SECONDS = 3600  # a .part not written to for this long is dead
_FLUSH_SECONDS = 5.0  # manifest rewrite cadence during a download
_FLUSH_CHANGES = 100
_ERROR_CLASSES = ("catalogue", "network", "integrity", "storage", "unknown")
_NETWORK_ERRNOS = (
    errno.ECONNRESET,
    errno.ECONNREFUSED,
    errno.ECONNABORTED,
    errno.ETIMEDOUT,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.ENETDOWN,
    errno.EPIPE,
)
_DATASET_ID_RE = re.compile(r"^(ds|nm|on)\d+$")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _size_on_disk(path: Path) -> int:
    """Bytes of a file, or of every file under a directory recording."""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _write_staged(dest: Path, payload: bytes) -> None:
    """Write through a unique ``.part`` sibling and rename, like the downloader."""
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.part")
    try:
        tmp.write_bytes(payload)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _drop_stale_parts(dest: Path, stale_after: float = _PART_STALE_SECONDS) -> None:
    """Remove ``<name>.*.part`` siblings a killed job left (not written for an hour).

    A live transfer writes its ``.part`` continuously, so an old modification
    time means nobody owns it; it is removed only under the target's lock.
    """
    cutoff = time.time() - stale_after
    for part in dest.parent.glob(f"{dest.name}.*.part"):
        try:
            if part.stat().st_mtime < cutoff:
                part.unlink()
        except OSError:
            pass


def _short(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}".replace("\n", " ")[:300]


def _classify(exc: BaseException) -> str:
    """The failure class of an exception: catalogue, network, integrity, storage, unknown.

    ``catalogue``: the record names an object the archive does not serve (404,
    403 on an unpublished NEMAR blob); ``network``: a transport fault, retried;
    ``integrity``: the bytes disagree with the record; ``storage``: the local
    disk refused the write.
    """
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for e in chain:
        if isinstance(e, downloader.DownloadIntegrityError):
            return "integrity"
        if isinstance(e, OSError) and e.errno in (
            errno.ENOSPC,
            errno.EDQUOT,
            errno.EROFS,
            errno.EIO,
            errno.EACCES,
            errno.EPERM,
        ):
            return "storage"
        if isinstance(e, (FileNotFoundError, PermissionError)):
            return "catalogue"
        if isinstance(e, urllib.error.HTTPError):
            return "catalogue" if e.code in (403, 404) else "network"
        if isinstance(e, ClientError):
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(e.response.get("Error", {}).get("Code"))
            if status in (403, 404) or code in (
                "NoSuchKey",
                "NoSuchBucket",
                "AccessDenied",
            ):
                return "catalogue"
            return "network"
        if isinstance(
            e,
            (
                BotoCoreError,
                urllib.error.URLError,
                requests.RequestException,
                TimeoutError,
                ConnectionError,
            ),
        ) or (isinstance(e, OSError) and e.errno in _NETWORK_ERRNOS):
            return "network"
        name = e.__class__.__name__
        if name == "StorageAccessError":
            return "catalogue"
        if name == "DataIntegrityError":
            text = str(e).lower()
            if "not found" in text or "denied" in text or "not yet published" in text:
                return "catalogue"
            return "integrity"
    return "unknown"


# ---------------------------------------------------------------- manifest


@dataclass
class FileEntry:
    """One line of a dataset manifest: an expected file and its state."""

    path: str
    kind: str = "recording"  # dataset | recording | sidecar | directory
    size: int | None = None  # from the git-annex key, when the record has one
    key: str | None = None  # the git-annex key (size and hash)
    state: str = "planned"  # planned | done | stale-record | absent | failed | corrupt | missing | locked
    bytes: int | None = None  # size seen on disk
    when: str | None = None
    error: str | None = None  # catalogue | network | integrity | storage | unknown
    message: str | None = None
    record: str | None = None  # bids_relpath of the recording the file belongs to
    subject: str | None = None

    @property
    def present(self) -> bool:
        return self.state in _PRESENT_STATES

    @property
    def expected_bytes(self) -> int | None:
        """The size the file must have: the key's, or the accepted size of a stale record."""
        return self.bytes if self.state == "stale-record" else self.size

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FileEntry:
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


class _Manifest:
    """``<dataset>/.eegdash/manifest.jsonl`` and ``COMPLETE``, rewritten atomically under a lock.

    The first line is a header (``dataset``, ``planned``, ``fingerprint``,
    ``subjects``); every other line is a :class:`FileEntry`. A rewrite merges
    into what is on disk, so two jobs sharing a cache keep each other's states.
    """

    def __init__(self, dataset_dir: Path):
        self.dir = dataset_dir / _MANIFEST_DIR
        self.path = self.dir / "manifest.jsonl"
        self.lock_path = self.dir / "manifest.lock"
        self.complete_path = self.dir / "COMPLETE"

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> tuple[dict[str, Any], dict[str, FileEntry]]:
        header: dict[str, Any] = {}
        entries: dict[str, FileEntry] = {}
        if not self.path.exists():
            return header, entries
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue  # a torn line from a killed writer; the file is rewritten whole
                if "path" in data:
                    entries[data["path"]] = FileEntry.from_dict(data)
                elif "dataset" in data:
                    header = data
        return header, entries

    def _write_unlocked(
        self, header: Mapping[str, Any], entries: Mapping[str, FileEntry]
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"manifest.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(header), separators=(",", ":")) + "\n")
            for entry in entries.values():
                fh.write(entry.to_json() + "\n")
        tmp.replace(self.path)

    def write(
        self, header: Mapping[str, Any], entries: Mapping[str, FileEntry]
    ) -> None:
        """Replace the manifest whole (plan and verify)."""
        with _FileLock(
            self.lock_path, wait=_MANIFEST_LOCK_WAIT, stale_after=_MANIFEST_LOCK_WAIT
        ):
            self._write_unlocked(header, entries)

    def update(
        self, changed: Mapping[str, FileEntry], header: Mapping[str, Any] | None = None
    ) -> None:
        """Merge changed entries into the manifest on disk (download)."""
        if not changed and header is None:
            return
        with _FileLock(
            self.lock_path, wait=_MANIFEST_LOCK_WAIT, stale_after=_MANIFEST_LOCK_WAIT
        ):
            on_disk_header, entries = self.read()
            entries.update(changed)
            self._write_unlocked(header or on_disk_header, entries)

    def is_complete(self) -> bool:
        return self.complete_path.exists()

    def complete_info(self) -> dict[str, Any]:
        try:
            return json.loads(self.complete_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def mark_complete(self, info: Mapping[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_staged(
            self.complete_path, (json.dumps(dict(info), indent=1) + "\n").encode()
        )

    def clear_complete(self) -> None:
        self.complete_path.unlink(missing_ok=True)


class _FileLock:
    """A lock file ``<target>.lock`` naming its holder (host, pid, Slurm job, time).

    Lets several jobs share one cache without fetching a file twice. A lock is
    stale when its holder is a dead process of this host, or when it is older
    than *stale_after* (a job on another host that died with it). ``acquire``
    returns False when a live holder has it; with *wait* it polls that long.
    """

    def __init__(
        self, path: Path, *, wait: float = 0.0, stale_after: float = _LOCK_STALE_SECONDS
    ):
        self.path = (
            path if path.suffix == ".lock" else path.with_name(path.name + ".lock")
        )
        self.wait = wait
        self.stale_after = stale_after
        self.held = False

    def holder(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _is_stale(self) -> bool:
        info = self.holder()
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False  # gone already
        if info is None:
            return age > self.stale_after
        if info.get("host") == socket.gethostname() and isinstance(
            info.get("pid"), int
        ):
            try:
                os.kill(info["pid"], 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False  # alive, another user
            return info["pid"] == os.getpid() and age > self.stale_after
        return age > self.stale_after

    def _try(self) -> bool:
        """Create the lock file exclusively; False when it exists."""
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "job": os.environ.get("SLURM_JOB_ID"),
                        "time": _now(),
                    }
                )
            )
        self.held = True
        return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait
        while True:
            if self._try():
                return True
            if self._is_stale():
                self.path.unlink(missing_ok=True)
                continue  # the next _try races fairly with any peer
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self) -> _FileLock:
        if not self.acquire():
            # The manifest lock is never held longer than a rewrite takes: break it.
            self.path.unlink(missing_ok=True)
            self._try()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ---------------------------------------------------------------- reports


@dataclass
class DatasetReport:
    """What one method found or did for one dataset."""

    dataset_id: str
    state: str = "planned"
    served_as: str | None = (
        None  # the id the catalogue serves it under (ds* -> on* twin)
    )
    entries: dict[str, FileEntry] = field(default_factory=dict)
    size_bytes: int | None = None  # the catalogue's size of the whole dataset
    subjects: int | None = None  # subjects the catalogue serves
    table_subjects: int | None = None  # subjects the table expects
    problems: list[str] = field(default_factory=list)
    drift: dict[str, list[str]] | None = None  # status(online=True): added, removed
    seconds: float = 0.0

    @property
    def counts(self) -> dict[str, int]:
        c = Counter(e.state for e in self.entries.values())
        present = c["done"] + c["stale-record"]
        return {
            "files": len(self.entries),
            "present": present,
            "stale": c["stale-record"],
            "failed": c["failed"],
            "corrupt": c["corrupt"],
            "locked": c["locked"],
            "absent": c[_ABSENT],
            "missing": len(self.entries)
            - present
            - c[_ABSENT]
            - c["failed"]
            - c["corrupt"]
            - c["locked"],
        }

    @property
    def errors(self) -> dict[str, int]:
        """Failed files by error class."""
        return dict(
            Counter(
                e.error or "unknown"
                for e in self.entries.values()
                if e.state == "failed"
            )
        )

    @property
    def bytes_present(self) -> int:
        return sum(e.bytes or 0 for e in self.entries.values() if e.present)

    @property
    def bytes_expected(self) -> int:
        """Bytes the records size (keyless files count 0)."""
        return sum(e.size or 0 for e in self.entries.values())

    @property
    def ok(self) -> bool:
        c = self.counts
        return (
            not self.problems
            and c["files"] > 0
            and c["present"] + c["absent"] == c["files"]
        )

    def row(self) -> dict[str, Any]:
        c = self.counts
        return {
            "dataset_id": self.dataset_id,
            "served_as": self.served_as or "",
            "state": self.state,
            "files": c["files"],
            "present": c["present"],
            "missing": c["missing"],
            "failed": c["failed"],
            "corrupt": c["corrupt"],
            "stale": c["stale"],
            "absent": c["absent"],
            "gb_present": round(self.bytes_present / 1e9, 2),
            "gb_expected": round((self.size_bytes or self.bytes_expected) / 1e9, 2),
            "subjects": self.subjects,
            "errors": ",".join(f"{k}:{v}" for k, v in sorted(self.errors.items())),
            "problems": "; ".join(self.problems),
        }


@dataclass
class CorpusReport:
    """The result of plan, download, verify or status, per dataset and per file."""

    action: str
    datasets: list[DatasetReport] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def problems(self) -> list[str]:
        return [f"{d.dataset_id}: {p}" for d in self.datasets for p in d.problems]

    @property
    def ok(self) -> bool:
        """No problem; and, after a download, verify or status, every expected file present."""
        if self.action == "plan":
            return not self.problems
        return all(d.ok for d in self.datasets)

    @property
    def files(self) -> dict[str, FileEntry]:
        """Every entry, keyed ``<dataset>/<path>``."""
        return {
            f"{d.dataset_id}/{p}": e
            for d in self.datasets
            for p, e in d.entries.items()
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.row() for d in self.datasets])

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "seconds": round(self.seconds, 1),
            "ok": self.ok,
            "datasets": [
                {
                    **d.row(),
                    "drift": d.drift,
                    "files_not_present": [
                        asdict(e) for e in d.entries.values() if not e.present
                    ],
                }
                for d in self.datasets
            ],
        }

    def summary(self) -> str:
        c = Counter()
        for d in self.datasets:
            c.update(d.counts)
        gb = sum(d.bytes_present for d in self.datasets) / 1e9
        states = Counter(d.state for d in self.datasets)
        return (
            f"{self.action}: {len(self.datasets)} datasets "
            f"({', '.join(f'{v} {k}' for k, v in sorted(states.items()))}); "
            f"{c['present']} of {c['files']} files present, {gb:,.1f} GB; "
            f"{c['absent']} listed but not on the archive; "
            f"{c['failed']} failed, {c['corrupt']} corrupt, {c['locked']} locked; "
            f"{len(self.problems)} problem(s); {self.seconds:.0f} s"
        )

    def __str__(self) -> str:
        if not self.datasets:
            return self.summary()
        frame = self.to_frame().drop(columns=["problems"])
        lines = [frame.to_string(index=False), ""]
        lines += [f"PROBLEM {p}" for p in self.problems]
        lines.append(self.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------- planning


@dataclass
class _Transfer:
    """How one expected file is fetched (not persisted; rebuilt from the records)."""

    entry: FileEntry
    dest: Path
    dataset_id: str
    backend: str
    base: str
    relpath: str
    uri: str | None = None
    version_id: str | None = None
    inline: str | None = None
    record: dict[str, Any] | None = None


@dataclass
class _DatasetPlan:
    report: DatasetReport
    transfers: list[_Transfer]
    header: dict[str, Any]
    served: bool


class _Flusher:
    """Collects file state changes of one dataset and rewrites its manifest in batches.

    A rewrite per file would copy a 36,000-line manifest 36,000 times; the
    size check at resume covers whatever a killed job did not flush.
    """

    def __init__(self, manifest: _Manifest, header: Mapping[str, Any]):
        self.manifest = manifest
        self.header = dict(header)
        self.pending: dict[str, FileEntry] = {}
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def note(self, entry: FileEntry) -> None:
        with self.lock:
            self.pending[entry.path] = entry
            due = (
                len(self.pending) >= _FLUSH_CHANGES
                or time.monotonic() - self.last >= _FLUSH_SECONDS
            )
        if due:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            pending, self.pending = self.pending, {}
            self.last = time.monotonic()
        if pending:
            self.manifest.update(pending, self.header)


class EEGDashCorpus:
    """A corpus of datasets: plan, download, verify and status over the EEGDash cache.

    Parameters
    ----------
    source : str | Sequence[str]
        The name of a corpus table shipped with EEGDash (``eegdash/corpora/<name>.csv``),
        the path of a CSV with a ``dataset_id`` column, or a sequence of dataset ids.
    cache_dir : str | Path | None
        Where the datasets live, as for :class:`EEGDashDataset` (``EEGDASH_CACHE_DIR``,
        else ``./.eegdash_cache``).
    eeg_dash_instance : EEGDash | None
        The catalogue client; one is created on first use.
    max_concurrency : int
        Connection pool of the S3 accessor shared by the download workers.
    name : str | None
        A name for the corpus file ``<cache_dir>/.eegdash/corpus-<name>.json``.

    """

    def __init__(
        self,
        source: str | Path | Sequence[str],
        *,
        cache_dir: str | Path | None = None,
        eeg_dash_instance: Any = None,
        max_concurrency: int = 20,
        name: str | None = None,
    ):
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                self.table, self.name = self._read_table(path), path.stem
            else:
                self.table, self.name = self._named_table(str(source)), str(source)
        else:
            ids = [str(s).strip() for s in source if str(s).strip()]
            if not ids:
                raise ValueError("EEGDashCorpus needs at least one dataset id")
            self.table = pd.DataFrame({"dataset_id": ids})
            self.name = "ids"
        if name:
            self.name = name
        dup = self.table["dataset_id"][self.table["dataset_id"].duplicated()].tolist()
        if dup:
            raise ValueError(f"duplicate dataset ids in the corpus: {sorted(set(dup))}")
        self.ids: list[str] = self.table["dataset_id"].tolist()
        self.cache_dir = Path(cache_dir or get_default_cache_dir())
        self.max_concurrency = max_concurrency
        self._client = eeg_dash_instance
        self._datasets: dict[str, Any] | None = None

    # -- construction -------------------------------------------------------

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "dataset_id" not in rows[0]:
            raise ValueError(f"{path}: a corpus table needs a dataset_id column")
        frame = pd.DataFrame(rows)
        frame["dataset_id"] = frame["dataset_id"].astype(str).str.strip()
        return frame[frame["dataset_id"] != ""].reset_index(drop=True)

    @classmethod
    def _named_table(cls, name: str) -> pd.DataFrame:
        """A table shipped in ``eegdash/corpora/``; none ships yet."""
        path = Path(__file__).parent / "corpora" / f"{name}.csv"
        if path.is_file():
            return cls._read_table(path)
        raise FileNotFoundError(
            f"no corpus table named {name!r} ships with EEGDash (looked for {path}); "
            "pass a list of dataset ids or EEGDashCorpus.from_table(<csv>)"
        )

    @classmethod
    def from_table(cls, path: str | Path, **kwargs: Any) -> EEGDashCorpus:
        """A corpus from any CSV with a ``dataset_id`` column (``n_subjects`` is checked when present)."""
        return cls(Path(path), **kwargs)

    @property
    def client(self) -> Any:
        if self._client is None:
            from .api import EEGDash  # noqa: PLC0415 (offline methods never need it)

            self._client = EEGDash()
        return self._client

    def _select(self, datasets: Sequence[str] | str | None) -> list[str]:
        """Dataset ids in corpus order, restricted to *datasets*; an unknown id is refused."""
        if datasets is None:
            return list(self.ids)
        wanted = datasets.split(",") if isinstance(datasets, str) else list(datasets)
        wanted = [w.strip() for w in wanted if w.strip()]
        unknown = [w for w in wanted if w not in self.ids]
        if unknown:
            raise ValueError(f"not in the corpus: {', '.join(unknown)}")
        return [d for d in self.ids if d in set(wanted)]

    def _table_subjects(self, dsid: str) -> int | None:
        if "n_subjects" not in self.table.columns:
            return None
        value = self.table.loc[self.table["dataset_id"] == dsid, "n_subjects"].iloc[0]
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _folder(self, dsid: str) -> str:
        """The cache folder of a dataset, offline: its NEMAR twin's when the plan wrote there.

        The catalogue serves 53 of the 56 OpenNeuro datasets of eegdash-corpus-v1
        under their ``on*`` re-host (2026-10-26); the twin is a pure function of
        the id, so status and verify find the folder without the network.
        """
        twin = nemar_twin_of(dsid)
        if twin and _Manifest(self.cache_dir / twin).exists():
            return twin
        return dsid

    # -- plan ---------------------------------------------------------------

    def plan(
        self,
        *,
        datasets: Sequence[str] | str | None = None,
        subjects: Sequence[str] | None = None,
    ) -> CorpusReport:
        """Query the catalogue and write every dataset's manifest; no download.

        A dataset the catalogue does not serve, or serves with fewer subjects
        than the table's ``n_subjects``, is a problem of the report. Expected
        files new since the last plan reset ``COMPLETE``; the report's ``drift``
        lists what the catalogue added or dropped.
        """
        t0 = time.monotonic()
        report = CorpusReport("plan")
        for dsid in self._select(datasets):
            report.datasets.append(self._plan_dataset(dsid, subjects).report)
        report.seconds = time.monotonic() - t0
        self._write_corpus_file(report)
        return report

    def _plan_dataset(self, dsid: str, subjects: Sequence[str] | None) -> _DatasetPlan:
        t0 = time.monotonic()
        rep = DatasetReport(dsid, table_subjects=self._table_subjects(dsid))
        query = {
            "dataset": dsid,
            **({"subject": [str(s) for s in subjects]} if subjects else {}),
        }
        try:
            doc = self.client.get_dataset(dsid)
            records = list(self.client.find(**query)) if doc is not None else []
        except Exception as exc:  # one unreachable dataset must not hide the others
            rep.problems.append(f"catalogue query failed: {_short(exc)}")
            rep.state = "problem"
            rep.seconds = time.monotonic() - t0
            return _DatasetPlan(rep, [], {}, served=False)
        if subjects:
            wanted = {str(s) for s in subjects}
            records = [r for r in records if str(_subject_of(r)) in wanted]
        rep.size_bytes = (doc or {}).get("size_bytes")
        # A retired OpenNeuro id answers with its NEMAR re-host (ds003810 ->
        # on003810, eegdash #404): EEGDashDataset caches under the served id,
        # so the files, the manifest and every route follow it.
        served = str(
            (doc or {}).get("dataset_id")
            or (records[0].get("dataset") if records else None)
            or dsid
        )
        if served != dsid:
            rep.served_as = served
        manifest = _Manifest(self.cache_dir / served)
        rep.subjects = len(
            {_subject_of(r) for r in records if _subject_of(r) is not None}
        )
        if doc is None:
            rep.problems.append("not served by the catalogue (no dataset document)")
        elif not records:
            rep.problems.append("not served by the catalogue (no recordings)")
        if rep.problems:
            rep.state = "not served"
            rep.seconds = time.monotonic() - t0
            return _DatasetPlan(rep, [], {}, served=False)
        if (
            rep.table_subjects is not None
            and not subjects
            and rep.subjects < rep.table_subjects
        ):
            rep.problems.append(
                f"the catalogue lists {rep.subjects} subjects, the table {rep.table_subjects}"
            )
        transfers = self._transfers_for(served, doc, records)
        header = {
            "dataset": served,
            **({"requested": dsid} if served != dsid else {}),
            "planned": _now(),
            "fingerprint": (doc or {}).get("ingestion_fingerprint"),
            "subjects": sorted(str(s) for s in subjects) if subjects else None,
        }
        _, old = manifest.read()
        new_paths = {t.entry.path for t in transfers}
        added = sorted(new_paths - set(old))
        # A subject-restricted plan sees part of the dataset: it adds to the
        # manifest and drops nothing, so the other subjects keep their states.
        removed = [] if subjects else sorted(set(old) - new_paths)
        for t in transfers:
            prev = old.get(t.entry.path)
            if (
                prev is not None
                and prev.size == t.entry.size
                and prev.key == t.entry.key
            ):
                t.entry.state, t.entry.bytes, t.entry.when = (
                    prev.state,
                    prev.bytes,
                    prev.when,
                )
                t.entry.error, t.entry.message = prev.error, prev.message
        if manifest.exists():
            rep.drift = {"added": added, "removed": removed}
        entries = {p: e for p, e in old.items() if subjects and p not in new_paths}
        entries.update((t.entry.path, t.entry) for t in transfers)
        manifest.write(header, entries)
        if added and manifest.is_complete():
            manifest.clear_complete()
        rep.entries = entries
        rep.state = _dataset_state(manifest, entries)
        rep.seconds = time.monotonic() - t0
        return _DatasetPlan(rep, transfers, header, served=True)

    def _transfers_for(
        self,
        dsid: str,
        doc: Mapping[str, Any] | None,
        records: Sequence[Mapping[str, Any]],
    ) -> list[_Transfer]:
        """The expected files of a dataset: dataset-level files, recordings, sidecars."""
        root = self.cache_dir / dsid
        out: dict[str, _Transfer] = {}
        storage = (doc or {}).get("storage") or {}
        if storage.get("base") and storage.get("backend") in ("s3", "https", "nemar"):
            for rel in [storage.get("raw_key"), *(storage.get("dep_keys") or [])]:
                if rel:
                    self._add_transfer(
                        out, root, dsid, rel, rel, "dataset", storage, None, None
                    )
        for rec in records:
            st = rec.get("storage") or {}
            raw = rec.get("bids_relpath") or st.get("raw_key")
            if not raw or not st.get("base"):
                continue
            raw_key = st.get("raw_key") or raw
            dep_keys = list(st.get("dep_keys") or [])
            subject = _subject_of(rec)
            for rel in [
                *(st.get("sidecar_inline") or {}),
                *(st.get("annex_keys") or {}),
            ]:
                if "/" not in rel and rel not in (raw_key, raw) and rel not in dep_keys:
                    self._add_transfer(
                        out, root, dsid, rel, rel, "dataset", st, None, None
                    )
            kind = (
                "directory"
                if PurePosixPath(raw).suffix.lower() in _DIRECTORY_SUFFIXES
                else "recording"
            )
            self._add_transfer(
                out,
                root,
                dsid,
                raw,
                raw_key,
                kind,
                st,
                dict(rec),
                subject,
                version_id=st.get("version_id"),
            )
            pins = st.get("version_ids") or {}
            for dep in dep_keys:
                self._add_transfer(
                    out,
                    root,
                    dsid,
                    dep,
                    dep,
                    "sidecar",
                    st,
                    dict(rec),
                    subject,
                    version_id=pins.get(dep),
                )
        return list(out.values())

    @staticmethod
    def _add_transfer(
        out: dict[str, _Transfer],
        root: Path,
        dsid: str,
        path: str,
        relpath: str,
        kind: str,
        storage: Mapping[str, Any],
        record: dict[str, Any] | None,
        subject: str | None,
        *,
        version_id: str | None = None,
    ) -> None:
        """One expected file; the first record that lists a shared sidecar owns it."""
        if path in out:
            return
        backend = storage.get("backend") or ""
        base = (storage.get("base") or "").rstrip("/")
        keys = storage.get("annex_keys") or {}
        key = keys.get(relpath) or keys.get(path)
        size = downloader.annex_key_info(key)[0] if key else None
        inline = (storage.get("sidecar_inline") or {}).get(relpath)
        uri = None
        if backend in ("s3", "https"):
            uri = f"{base}/{relpath}"
        elif backend == "nemar" and key:
            uri = f"{base}/objects/{key}"
        entry = FileEntry(
            path=path,
            kind=kind,
            size=size,
            key=key,
            record=(record or {}).get("bids_relpath") if kind != "dataset" else None,
            subject=str(subject) if subject is not None and kind != "dataset" else None,
        )
        out[path] = _Transfer(
            entry=entry,
            dest=root / path,
            dataset_id=dsid,
            backend=backend,
            base=base,
            relpath=relpath,
            uri=uri,
            version_id=version_id,
            inline=inline,
            record=record,
        )

    # -- download -----------------------------------------------------------

    def download(
        self,
        *,
        n_jobs: int = 8,
        n_datasets: int = 2,
        datasets: Sequence[str] | str | None = None,
        subjects: Sequence[str] | None = None,
        time_budget: float | None = None,
    ) -> CorpusReport:
        """Fetch every missing file, resumably; the plan runs first.

        Datasets are taken smallest first, *n_datasets* at a time, each with
        its share of *n_jobs* file workers; within a dataset the dataset-level
        files go first and once, then the files largest first. A file the
        manifest marks present at the size on disk is skipped without network.
        Failures are classed (``catalogue``, ``network``, ``integrity``,
        ``storage``), the transient ones retried in three rounds with backoff,
        then reported. No new dataset starts once *time_budget* seconds have
        passed: the datasets in flight finish, the rest are ``deferred``.
        """
        t0 = time.monotonic()
        report = CorpusReport("download")
        plans = [self._plan_dataset(d, subjects) for d in self._select(datasets)]
        queue = sorted(
            (p for p in plans if p.served),
            key=lambda p: (p.report.size_bytes is None, p.report.size_bytes or 0),
        )
        n_datasets = max(1, n_datasets)
        workers = max(1, n_jobs // n_datasets)
        filesystem = downloader.get_s3_filesystem(max_concurrency=self.max_concurrency)
        deadline = None if time_budget is None else t0 + time_budget
        running: dict[Any, _DatasetPlan] = {}
        with ThreadPoolExecutor(max_workers=n_datasets) as pool:
            while queue or running:
                while queue and len(running) < n_datasets:
                    if deadline is not None and time.monotonic() >= deadline:
                        for p in queue:
                            p.report.state = "deferred"
                            p.report.problems.append("not started: time budget spent")
                        queue = []
                        break
                    p = queue.pop(0)
                    running[
                        pool.submit(self._download_dataset, p, workers, filesystem)
                    ] = p
                if running:
                    done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for fut in done:
                        p = running.pop(fut)
                        exc = fut.exception()
                        if exc is not None:
                            p.report.problems.append(f"download aborted: {_short(exc)}")
                            p.report.state = "problem"
        report.datasets = [p.report for p in plans]
        report.seconds = time.monotonic() - t0
        self._write_corpus_file(report)
        return report

    def _download_dataset(
        self, plan: _DatasetPlan, workers: int, filesystem: Any
    ) -> None:
        t0 = time.monotonic()
        rep = plan.report
        manifest = _Manifest(self.cache_dir / (rep.served_as or rep.dataset_id))
        flusher = _Flusher(manifest, plan.header)
        todo: list[_Transfer] = []
        for t in plan.transfers:
            if self._already_there(t):
                if not t.entry.present:
                    t.entry.state, t.entry.error, t.entry.message = "done", None, None
                    t.entry.bytes, t.entry.when = _size_on_disk(t.dest), _now()
                    flusher.note(t.entry)
            else:
                todo.append(t)
        if todo:
            manifest.clear_complete()
        fetched = 0
        first = [t for t in todo if t.entry.kind == "dataset"]
        rest = sorted(
            (t for t in todo if t.entry.kind != "dataset"),
            key=lambda t: (t.entry.size is None, -(t.entry.size or 0)),
        )
        for t in first:
            fetched += self._fetch_one(t, filesystem, flusher)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched += sum(
                pool.map(lambda t: self._fetch_one(t, filesystem, flusher), rest)
            )
            for rnd in range(_RETRY_ROUNDS):
                again = [
                    t
                    for t in todo
                    if t.entry.state == "locked"
                    or (
                        t.entry.state == "failed" and t.entry.error in _RETRYABLE_ERRORS
                    )
                ]
                if not again:
                    break
                time.sleep(_RETRY_BACKOFF[rnd])
                fetched += sum(
                    pool.map(lambda t: self._fetch_one(t, filesystem, flusher), again)
                )
        flusher.flush()
        rep.state = _dataset_state(manifest, rep.entries)
        rep.seconds = time.monotonic() - t0
        c = rep.counts
        errors = ", ".join(f"{k} {v}" for k, v in sorted(rep.errors.items()))
        logger.info(
            "%s: %d files, %d fetched, %d present, %d failed%s, %d locked, %.1f GB, %.0f s",
            rep.dataset_id,
            c["files"],
            fetched,
            c["present"],
            c["failed"],
            f" ({errors})" if errors else "",
            c["locked"],
            rep.bytes_present / 1e9,
            rep.seconds,
        )

    @staticmethod
    def _already_there(t: _Transfer) -> bool:
        """Present at the expected size (no network); an unsized file only when the manifest says so."""
        try:
            got = _size_on_disk(t.dest)
        except FileNotFoundError:
            return False
        if t.entry.kind == "directory":
            return t.entry.present and got > 0
        expected = t.entry.expected_bytes
        if expected is None:
            return t.entry.present
        return got == expected

    def _fetch_one(self, t: _Transfer, filesystem: Any, flusher: _Flusher) -> int:
        """Fetch one file under its lock; returns 1 when bytes moved. Never raises."""
        e = t.entry
        if t.backend == "local":  # nothing to fetch: it is there or it is not
            e.state, e.error, e.message = (
                "failed",
                "catalogue",
                "local record, not on disk",
            )
            e.when = _now()
            flusher.note(e)
            return 0
        lock = _FileLock(t.dest)
        if not lock.acquire():
            holder = lock.holder() or {}
            e.state = "locked"
            e.message = f"held by {holder.get('host')} pid {holder.get('pid')} job {holder.get('job')}"
            flusher.note(e)
            return 0
        moved = 0
        try:
            _drop_stale_parts(t.dest)
            if self._already_there(t):  # a peer finished it while we waited
                e.state = e.state if e.present else "done"
                e.bytes, e.error, e.message = _size_on_disk(t.dest), None, None
            else:
                t.dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    moved = int(self._transfer(t, filesystem))
                except Exception as exc:
                    klass = _classify(exc)
                    if klass == "integrity" and self._accept_stale(t, filesystem):
                        moved = 1
                    else:
                        absent = (
                            klass == "catalogue" and not e.key and e.kind != "recording"
                        )
                        e.state = _ABSENT if absent else "failed"
                        e.error, e.message = klass, _short(exc)
                        e.bytes = None
                        logger.debug(
                            "%s/%s: %s %s", t.dataset_id, e.path, klass, e.message
                        )
                else:
                    e.state, e.error, e.message = "done", None, None
                    e.bytes = _size_on_disk(t.dest)
                if not e.present:
                    moved = 0
            e.when = _now()
        except Exception as exc:  # a lock or disk fault must not take the dataset down
            e.state, e.error, e.message = "failed", _classify(exc), _short(exc)
        finally:
            lock.release()
        flusher.note(e)
        return moved

    def _transfer(self, t: _Transfer, filesystem: Any) -> bool:
        """Put one file in place; True when bytes were fetched. Raises on any failure."""
        if t.entry.kind == "directory":
            self._fetch_directory(t)
            return True
        if t.inline is not None and not t.entry.key:
            _write_staged(t.dest, t.inline.encode("utf-8"))
            return True
        if t.backend == "nemar" and t.uri is None:
            from .dataset.base import _resolve_one_nemar_entry  # noqa: PLC0415

            uri = _resolve_one_nemar_entry(
                dataset_id=t.dataset_id,
                relpath=t.relpath,
                base=t.base,
                dest=t.dest,
                stored_key=t.entry.key,
                stored_sidecar=t.inline,
                is_required=t.entry.kind != "sidecar",
            )
            if uri is None:
                if not t.dest.exists():
                    raise FileNotFoundError(f"{t.base}/{t.relpath}")
                return True
            t.uri = uri
        if t.uri is None:
            raise RuntimeError(f"no route for {t.backend} record {t.relpath}")
        pin = {"version_id": t.version_id} if t.version_id else {}
        if t.entry.size is None and t.dest.exists():
            # Keyless: the archive's own size is the only check there is. A peer
            # job may have published the file since this job planned it.
            try:
                served = filesystem.info(t.uri, **pin).get("size")
            except Exception:
                served = None
            if served is not None and _size_on_disk(t.dest) == served:
                return False
        marker = t.dest.with_name(t.dest.name + ".s3-version.json")
        marker.unlink(missing_ok=True)
        filesystem.get_file(
            t.uri,
            str(t.dest),
            callback=None,
            **pin,
            **({"annex_key": t.entry.key} if t.entry.key else {}),
        )
        if t.version_id:  # what EEGDashRaw expects beside a pinned object
            marker.write_text(json.dumps({"uri": t.uri, "version_id": t.version_id}))
        return True

    def _fetch_directory(self, t: _Transfer) -> None:
        """A CTF ``.ds`` or MEF3 ``.mefd`` recording is a prefix: EEGDashRaw knows its files."""
        from .dataset.base import EEGDashRaw  # noqa: PLC0415 (heavy import)

        EEGDashRaw(
            dict(t.record or {}), cache_dir=str(self.cache_dir)
        )._download_required_files()
        if not t.dest.is_dir() or _size_on_disk(t.dest) == 0:
            raise FileNotFoundError(f"{t.base}/{t.relpath}")

    def _accept_stale(self, t: _Transfer, filesystem: Any) -> bool:
        """The stale-record policy: a cross-source key that disagrees, but the archive's own size matches.

        When the URI ends in the key (NEMAR ``objects/<key>``) the key names the
        object and a mismatch is refused. Otherwise the record describes a copy
        elsewhere: if the archive reports another size for the object, the file
        is fetched against that size and recorded ``stale-record``.
        """
        e = t.entry
        if (
            not e.key
            or not t.uri
            or downloader.annex_key_info(PurePosixPath(t.uri).name)[0]
        ):
            return False
        try:
            served = filesystem.info(
                t.uri, **({"version_id": t.version_id} if t.version_id else {})
            ).get("size")
        except Exception:
            return False
        if served is None or served == e.size:
            return False
        try:
            filesystem.get_file(
                t.uri,
                str(t.dest),
                callback=None,
                **({"version_id": t.version_id} if t.version_id else {}),
            )
        except Exception:
            return False
        got = _size_on_disk(t.dest)
        if got != served:
            t.dest.unlink(missing_ok=True)
            return False
        logger.warning(
            "%s/%s: the record expects %s bytes, the archive serves %s; kept as stale-record",
            t.dataset_id,
            e.path,
            e.size,
            served,
        )
        e.state, e.bytes, e.error = "stale-record", got, None
        e.message = f"record {e.size} bytes, archive {served}"
        return True

    # -- verify and status --------------------------------------------------

    def verify(
        self,
        *,
        level: str = "size",
        datasets: Sequence[str] | str | None = None,
        n_jobs: int = 8,
    ) -> CorpusReport:
        """Check every expected file against its record, offline, and write ``COMPLETE``.

        ``size`` compares sizes (seconds for a corpus); ``hash`` also reads
        every file whose record carries a git-annex hash. A mismatch marks the
        file ``corrupt`` and removes ``COMPLETE``; a dataset with every file
        present gets ``COMPLETE`` (the level, the time, the plan's fingerprint).
        A file the archive does not hold (``absent``) does not block it. *n_jobs*
        files are checked at a time: hashing reads every byte, and one reader
        managed 19 MB/s on Lustre (46 GB in 40 min, 2026-10-27).
        """
        if level not in ("size", "hash"):
            raise ValueError(f"level must be 'size' or 'hash', got {level!r}")
        t0 = time.monotonic()
        report = CorpusReport("verify")
        for dsid in self._select(datasets):
            rep = DatasetReport(dsid, table_subjects=self._table_subjects(dsid))
            folder = self._folder(dsid)
            rep.served_as = folder if folder != dsid else None
            manifest = _Manifest(self.cache_dir / folder)
            header, entries = manifest.read()
            if not entries:
                rep.problems.append("no manifest: run plan or download first")
                rep.state = "unplanned"
                report.datasets.append(rep)
                continue
            ts = time.monotonic()
            root = self.cache_dir / folder
            with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
                list(
                    pool.map(
                        lambda e: self._verify_entry(root / e.path, e, level),
                        entries.values(),
                    )
                )
            manifest.write(header, entries)
            rep.entries = entries
            if all(e.present or e.state == _ABSENT for e in entries.values()):
                manifest.mark_complete(
                    {
                        "verified": _now(),
                        "level": level,
                        "files": len(entries),
                        "bytes": rep.bytes_present,
                        "planned": header.get("planned"),
                        "fingerprint": header.get("fingerprint"),
                        "subjects": header.get("subjects"),
                    }
                )
            else:
                manifest.clear_complete()
            rep.state = _dataset_state(manifest, entries)
            rep.seconds = time.monotonic() - ts
            c = rep.counts
            logger.info(
                "%s: %d files, %d present, %d absent on the archive, %d missing, %d corrupt (%s)",
                dsid,
                c["files"],
                c["present"],
                c["absent"],
                c["missing"],
                c["corrupt"],
                level,
            )
            report.datasets.append(rep)
        report.seconds = time.monotonic() - t0
        self._write_corpus_file(report)
        return report

    @staticmethod
    def _verify_entry(path: Path, e: FileEntry, level: str) -> None:
        try:
            got = _size_on_disk(path)
        except FileNotFoundError:
            if e.state != _ABSENT:
                e.state = "missing"
            e.bytes, e.when = None, _now()
            return
        e.bytes, e.when = got, _now()
        if e.kind == "directory":
            e.state = "done" if got > 0 else "missing"
            return
        expected = e.expected_bytes
        if expected is not None and got != expected:
            e.state, e.error = "corrupt", "integrity"
            e.message = f"expected {expected} bytes, got {got}"
            return
        if level == "hash" and e.key and e.state != "stale-record":
            try:
                downloader.verify_file(path, e.key, verify_hash=True)
            except OSError as exc:
                e.state, e.error, e.message = "corrupt", "integrity", _short(exc)
                return
        if e.state != "stale-record":
            e.state, e.error, e.message = "done", None, None

    def status(
        self, *, online: bool = False, datasets: Sequence[str] | str | None = None
    ) -> CorpusReport:
        """Per dataset, from the manifests and ``COMPLETE`` markers only.

        With *online*, the plan runs first and each dataset's ``drift`` lists
        the files the catalogue added or dropped since the manifest was written.
        """
        t0 = time.monotonic()
        drift: dict[str, DatasetReport] = {}
        if online:
            drift = {d.dataset_id: d for d in self.plan(datasets=datasets).datasets}
        report = CorpusReport("status")
        for dsid in self._select(datasets):
            rep = DatasetReport(dsid, table_subjects=self._table_subjects(dsid))
            folder = self._folder(dsid)
            rep.served_as = folder if folder != dsid else None
            manifest = _Manifest(self.cache_dir / folder)
            header, entries = manifest.read()
            rep.entries = entries
            rep.state = _dataset_state(manifest, entries)
            if dsid in drift:
                rep.drift = drift[dsid].drift
                rep.problems = list(drift[dsid].problems)
                rep.size_bytes, rep.subjects = (
                    drift[dsid].size_bytes,
                    drift[dsid].subjects,
                )
                if drift[dsid].drift and any(drift[dsid].drift.values()):
                    rep.problems.append(
                        f"catalogue drift: {len(rep.drift['added'])} added, "
                        f"{len(rep.drift['removed'])} removed"
                    )
            elif not entries:
                rep.problems.append("no manifest: run plan or download first")
            report.datasets.append(rep)
        report.seconds = time.monotonic() - t0
        return report

    # -- holding ------------------------------------------------------------

    @property
    def datasets(self) -> dict[str, Any]:
        """``{id: EEGDashDataset(download=False)}`` from the cache, built once, lazily."""
        if self._datasets is None:
            from .dataset import EEGDashDataset  # noqa: PLC0415 (heavy import)

            self._datasets = {
                dsid: EEGDashDataset(
                    cache_dir=self.cache_dir, dataset=self._folder(dsid), download=False
                )
                for dsid in self.ids
            }
        return self._datasets

    def records(self) -> Iterator[Any]:
        """Every recording (:class:`EEGDashRaw`) of every dataset, offline."""
        for ds in self.datasets.values():
            yield from ds.datasets

    def _write_corpus_file(self, report: CorpusReport) -> None:
        """``<cache_dir>/.eegdash/corpus-<name>.json``: ids and the last report; best effort."""
        path = self.cache_dir / _MANIFEST_DIR / f"corpus-{self.name}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        except (OSError, ValueError):
            previous = {}
        previous.update(
            {
                "name": self.name,
                "ids": self.ids,
                "written": _now(),
                f"last_{report.action}": {
                    "when": _now(),
                    "ok": report.ok,
                    "datasets": [d.row() for d in report.datasets],
                },
            }
        )
        try:
            _write_staged(path, (json.dumps(previous, indent=1) + "\n").encode())
        except OSError as exc:
            logger.warning("Could not write %s: %s", path, exc)


def _subject_of(record: Mapping[str, Any]) -> str | None:
    subject = record.get("subject")
    if subject is None:
        subject = (record.get("entities") or {}).get("subject")
    return None if subject is None else str(subject)


def _dataset_state(manifest: _Manifest, entries: Mapping[str, FileEntry]) -> str:
    """The dataset state: complete (COMPLETE written by verify), fetched, partial, planned, unplanned."""
    if not entries:
        return "unplanned"
    if manifest.is_complete():
        return "complete"
    present = sum(e.present for e in entries.values())
    if present + sum(e.state == _ABSENT for e in entries.values()) == len(entries):
        return "fetched"
    return "partial" if present else "planned"


# ---------------------------------------------------------------- CLI


def _corpus_from_arg(arg: str, cache_dir: str | None) -> EEGDashCorpus:
    """A CSV path, comma-separated ids (or one id), or a corpus name."""
    if Path(arg).is_file():
        return EEGDashCorpus.from_table(arg, cache_dir=cache_dir)
    ids = [a.strip() for a in arg.split(",") if a.strip()]
    if all(_DATASET_ID_RE.match(a) for a in ids):
        return EEGDashCorpus(ids, cache_dir=cache_dir)
    return EEGDashCorpus(arg, cache_dir=cache_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eegdash.corpus",
        description="Plan, download, verify and report a corpus of EEGDash datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {}
    for name, doc in (
        ("plan", "query the catalogue, write the manifests, no download"),
        ("download", "fetch every missing file, resumably"),
        ("verify", "check every file against its record, offline; write COMPLETE"),
        ("status", "per dataset, from the manifests only"),
    ):
        p = sub.add_parser(name, help=doc)
        p.add_argument(
            "corpus", help="a corpus name, a CSV with a dataset_id column, or ids a,b,c"
        )
        p.add_argument(
            "--cache-dir",
            help="EEGDash cache (default EEGDASH_CACHE_DIR or ./.eegdash_cache)",
        )
        p.add_argument(
            "--datasets", help="restrict to these ids of the corpus, comma-separated"
        )
        p.add_argument("--json", help="write the report here as JSON")
        parsers[name] = p
    for name in ("plan", "download"):
        parsers[name].add_argument(
            "--subjects",
            help="restrict every dataset to these subjects, comma-separated",
        )
    parsers["download"].add_argument(
        "--n-jobs", type=int, default=8, help="file workers in all (default 8)"
    )
    parsers["download"].add_argument(
        "--n-datasets", type=int, default=2, help="datasets at a time (default 2)"
    )
    parsers["download"].add_argument(
        "--time-budget", type=float, help="seconds after which no new dataset starts"
    )
    parsers["verify"].add_argument("--level", choices=("size", "hash"), default="size")
    parsers["verify"].add_argument(
        "--n-jobs", type=int, default=8, help="files checked at a time (default 8)"
    )
    parsers["status"].add_argument(
        "--online", action="store_true", help="re-plan and report catalogue drift"
    )
    args = parser.parse_args(argv)

    corpus = _corpus_from_arg(args.corpus, args.cache_dir)
    subjects = args.subjects.split(",") if getattr(args, "subjects", None) else None
    if args.command == "plan":
        report = corpus.plan(datasets=args.datasets, subjects=subjects)
    elif args.command == "download":
        report = corpus.download(
            n_jobs=args.n_jobs,
            n_datasets=args.n_datasets,
            datasets=args.datasets,
            subjects=subjects,
            time_budget=args.time_budget,
        )
    elif args.command == "verify":
        report = corpus.verify(
            level=args.level, datasets=args.datasets, n_jobs=args.n_jobs
        )
    else:
        report = corpus.status(online=args.online, datasets=args.datasets)
    print(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8"
        )
    return 0 if report.ok else 1


__all__ = ["EEGDashCorpus", "CorpusReport", "DatasetReport", "FileEntry", "main"]

if __name__ == "__main__":
    sys.exit(main())
