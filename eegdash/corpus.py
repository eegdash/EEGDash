# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Plan, download, verify and hold a corpus of datasets: :class:`EEGDashCorpus`.

A corpus is a list of dataset ids (or a CSV with a ``dataset_id`` column). The
files are fetched by :meth:`EEGDashDataset.download_all`, exactly as a user of
one dataset gets them; this module only adds the list of expected files per
dataset, a check of the files on disk against it, and a report::

    corpus = EEGDashCorpus(["ds004395", "nm000289"], cache_dir="/data/cache")
    corpus.plan()                          # catalogue only: files, bytes, drift
    corpus.download(n_jobs=16, n_datasets=4, time_budget=6 * 3600)
    corpus.verify(level="hash")            # size, then the git-annex hash
    corpus.status()                        # offline, from the manifests
    for rec in corpus.records(): ...       # every recording, offline

Command line: ``python -m eegdash.corpus plan|download|verify|status <csv or
ids a,b,c> [--cache-dir ...]``; the exit code is 1 when anything is missing.

Per dataset the cache holds ``<dataset>/.eegdash/manifest.jsonl`` (a header,
then one line per expected file: path, kind, size and key from the record,
whether it is required, and the state the last check found) and
``<dataset>/.eegdash/COMPLETE``, written when every required file is present.
A file without a git-annex key that the archive does not hold is ``absent``:
reported, never required (on002680 lists an ``.fdt`` beside every ``.set``
that embeds its samples).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from . import downloader
from .dataset._source_inference import nemar_twin_of
from .logging import logger
from .paths import get_default_cache_dir
from .schemas import NEMAR_ROOT_METADATA_FILES

_MANIFEST_DIR = ".eegdash"
_OK_STATES = ("present", "absent")  # absent: keyless, and the archive lacks it


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _short(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}".replace("\n", " ")[:300]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / _MANIFEST_DIR / "manifest.jsonl"
    if not path.exists():
        return {}, {}
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return lines[0], {e["path"]: e for e in lines[1:]}


def _write_manifest(root: Path, header: dict, entries: dict[str, dict]) -> None:
    rows = [header, *entries.values()]
    _atomic_write(
        root / _MANIFEST_DIR / "manifest.jsonl",
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows),
    )


def _plan_hash(entries: dict[str, dict]) -> str:
    """Identity of a plan: its paths and keys. COMPLETE holds for this hash only."""
    pairs = sorted((p, e["key"] or "") for p, e in entries.items())
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


def _subject_of(record: dict) -> str | None:
    subject = record.get("subject") or (record.get("entities") or {}).get("subject")
    return None if subject is None else str(subject)


# ---------------------------------------------------------------- expected files


def _add_expected(
    out: dict[str, dict], rel: str, kind: str, key: str | None, required: bool
) -> None:
    """One expected file; the first record that lists a shared sidecar owns it."""
    size = downloader.annex_key_info(key)[0] if key else None
    entry = {"path": rel, "kind": kind, "size": size, "key": key, "required": required}
    out.setdefault(rel, entry)


def _expected_files(doc: dict, dataset: Any) -> dict[str, dict[str, Any]]:
    """The files ``download_all`` fetches for *dataset*, from its records.

    The recording is always required; a sidecar or dataset-level file only
    when the record carries its git-annex key. Companions the record does not
    list (BrainVision ``.eeg``, EEGLAB ``.fdt``) are checked from the header
    at verify time.
    """
    out: dict[str, dict[str, Any]] = {}
    add = partial(_add_expected, out)
    storage = doc.get("storage") or {}
    keys = storage.get("annex_keys") or {}
    if storage.get("backend") in ("s3", "https"):  # _download_dataset_files
        for rel in [storage.get("raw_key"), *(storage.get("dep_keys") or [])]:
            if rel:
                add(rel, "dataset", keys.get(rel), bool(keys.get(rel)))
    for raw in dataset.datasets:
        root = raw.bids_root
        if raw._storage_backend == "nemar":  # _fetch_nemar_root_metadata
            inline = (raw.record.get("storage") or {}).get("sidecar_inline") or {}
            for name in NEMAR_ROOT_METADATA_FILES:
                if name in inline:
                    add(name, "dataset", None, False)
        add(
            raw.filecache.relative_to(root).as_posix(),
            "recording",
            raw._annex_key_for(raw.filecache),
            True,
        )
        for path in raw._dep_paths:
            key = raw._annex_key_for(path)
            add(path.relative_to(root).as_posix(), "sidecar", key, bool(key))
    return out


def _directory_whole(path: Path) -> bool:
    """A CTF ``.ds`` needs its ``.meg4`` and ``.res4``; a MEF3 ``.mefd`` whole segments."""
    suffix = path.suffix.lower()
    if suffix == ".ds":
        return any(path.glob("*.meg4")) and any(path.glob("*.res4"))
    if suffix == ".mefd":
        from .io.mef import check_session  # noqa: PLC0415 (imports pymef)

        try:
            check_session(path)
        except ValueError:
            return False
        return any(path.glob("*.timd"))
    return any(path.iterdir())


def _data_file_present(path: Path, level: str) -> bool:
    """The file holding the samples is on disk, as ``EEGDashRaw._require_data_file`` names it."""
    suffix = path.suffix.lower()
    if suffix == ".vhdr":
        from .io import brainvision  # noqa: PLC0415

        names = brainvision.companion_names(path)[0]
    elif suffix == ".set" and level == "hash":
        # ponytail: reading a .set loads its matrix, so only the hash level (which
        # reads every byte anyway) checks the .fdt; download reports a missing one.
        from .dataset.io import (  # noqa: PLC0415
            _eeglab_fdt_path,
            _eeglab_load_first_eeg,
        )

        eeg = _eeglab_load_first_eeg(path)
        fdt = eeg and _eeglab_fdt_path(path, eeg)
        if not fdt or fdt.suffix.lower() != ".fdt":
            return True  # the samples are in the .set
        names = (fdt.name,)
    else:
        return True
    return any((path.parent / name).is_file() for name in names)


def _check(root: Path, entry: dict[str, Any], level: str) -> None:
    """Set ``entry["state"]``: present, absent, missing, corrupt or incomplete."""
    path = root / entry["path"]
    entry["bytes"] = None
    if not path.exists():
        entry["state"] = "missing" if entry["required"] else "absent"
        return
    if path.is_dir():
        entry["state"] = "present" if _directory_whole(path) else "incomplete"
        entry["bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return
    entry["bytes"] = path.stat().st_size
    if entry["size"] is not None and entry["bytes"] != entry["size"]:
        entry["state"] = "corrupt"
    elif level == "hash" and entry["key"] and not _hash_ok(path, entry["key"]):
        entry["state"] = "corrupt"
    elif entry["kind"] == "recording" and not _data_file_present(path, level):
        entry["state"] = "incomplete"
    else:
        entry["state"] = "present"


def _hash_ok(path: Path, key: str) -> bool:
    try:
        downloader.verify_file(path, key, verify_hash=True)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- reports


@dataclass
class DatasetReport:
    """What one method found for one dataset."""

    dataset_id: str
    served_as: str | None = None  # the id the catalogue serves it under (ds* -> on*)
    state: str = "unplanned"
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    subjects: int | None = None
    size_bytes: int | None = None
    drift: dict[str, list[str]] | None = None  # plan: added, removed

    @property
    def counts(self) -> dict[str, int]:
        c = Counter(e.get("state", "planned") for e in self.entries.values())
        return {"files": len(self.entries), **c}

    @property
    def ok(self) -> bool:
        return (
            not self.problems
            and bool(self.entries)
            and all(e.get("state") in _OK_STATES for e in self.entries.values())
        )

    def row(self) -> dict[str, Any]:
        c = self.counts
        present = sum(
            e.get("bytes") or 0
            for e in self.entries.values()
            if e.get("state") == "present"
        )
        expected = self.size_bytes or sum(e["size"] or 0 for e in self.entries.values())
        return {
            "dataset_id": self.dataset_id,
            "served_as": self.served_as or "",
            "state": self.state,
            **{
                k: c.get(k, 0)
                for k in (
                    "files",
                    "present",
                    "absent",
                    "missing",
                    "corrupt",
                    "incomplete",
                )
            },
            "gb_present": round(present / 1e9, 2),
            "gb_expected": round(expected / 1e9, 2),
            "subjects": self.subjects,
            "problems": "; ".join(self.problems),
        }


@dataclass
class CorpusReport:
    """The result of plan, download, verify or status."""

    action: str
    datasets: list[DatasetReport] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def problems(self) -> list[str]:
        return [f"{d.dataset_id}: {p}" for d in self.datasets for p in d.problems]

    @property
    def ok(self) -> bool:
        """No problem; and, except after a plan, every required file present."""
        if self.action == "plan":
            return not self.problems
        return all(d.ok for d in self.datasets)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.row() for d in self.datasets])

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "seconds": round(self.seconds, 1),
            "datasets": [
                {
                    **d.row(),
                    "drift": d.drift,
                    "not_present": [
                        e for e in d.entries.values() if e.get("state") != "present"
                    ],
                }
                for d in self.datasets
            ],
        }

    def summary(self) -> str:
        c: Counter = Counter()
        for d in self.datasets:
            c.update(d.counts)
        return (
            f"{self.action}: {len(self.datasets)} datasets, {c['present']} of "
            f"{c['files']} files present, {c['absent']} absent on the archive, "
            f"{c['missing']} missing, {c['corrupt']} corrupt, {c['incomplete']} "
            f"incomplete; {len(self.problems)} problem(s); {self.seconds:.0f} s"
        )

    def __str__(self) -> str:
        lines = [self.to_frame().drop(columns=["problems"]).to_string(index=False)]
        lines += [f"PROBLEM {p}" for p in self.problems]
        return "\n".join([*lines, self.summary()])


# ---------------------------------------------------------------- corpus


class EEGDashCorpus:
    """A corpus of datasets in one EEGDash cache.

    Parameters
    ----------
    source : str | Path | Sequence[str]
        Dataset ids, or the path of a CSV with a ``dataset_id`` column (an
        ``n_subjects`` column is checked against the catalogue when present).
    cache_dir : str | Path | None
        As for :class:`EEGDashDataset` (``EEGDASH_CACHE_DIR``, else
        ``./.eegdash_cache``).
    eeg_dash_instance : EEGDash | None
        The catalogue client; created on first use (offline methods never need one).

    """

    def __init__(
        self,
        source: str | Path | Sequence[str],
        *,
        cache_dir: str | Path | None = None,
        eeg_dash_instance: Any = None,
    ):
        if isinstance(source, (str, Path)):
            self.table = pd.read_csv(source, dtype=str, keep_default_na=False)
            if "dataset_id" not in self.table.columns:
                raise ValueError(f"{source}: a corpus table needs a dataset_id column")
            self.table["dataset_id"] = self.table["dataset_id"].str.strip()
            self.table = self.table[self.table["dataset_id"] != ""]
        else:
            ids = [str(s).strip() for s in source if str(s).strip()]
            self.table = pd.DataFrame({"dataset_id": ids})
        self.ids: list[str] = self.table["dataset_id"].tolist()
        if not self.ids:
            raise ValueError("EEGDashCorpus needs at least one dataset id")
        dup = sorted({d for d in self.ids if self.ids.count(d) > 1})
        if dup:
            raise ValueError(f"duplicate dataset ids in the corpus: {dup}")
        self.cache_dir = Path(cache_dir or get_default_cache_dir())
        self._client = eeg_dash_instance
        self._datasets: dict[str, Any] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from .api import EEGDash  # noqa: PLC0415

            self._client = EEGDash()
        return self._client

    def _select(self, datasets: Sequence[str] | str | None) -> list[str]:
        if datasets is None:
            return list(self.ids)
        wanted = datasets.split(",") if isinstance(datasets, str) else list(datasets)
        wanted = {w.strip() for w in wanted if w.strip()}
        unknown = sorted(wanted - set(self.ids))
        if unknown:
            raise ValueError(f"not in the corpus: {', '.join(unknown)}")
        return [d for d in self.ids if d in wanted]

    def _table_subjects(self, dsid: str) -> int | None:
        if "n_subjects" not in self.table.columns:
            return None
        value = self.table.loc[self.table["dataset_id"] == dsid, "n_subjects"].iloc[0]
        try:
            return int(float(value))
        except ValueError:
            return None

    def _root(self, dsid: str) -> Path:
        """The cache folder of a dataset, offline: its NEMAR twin's when the plan wrote there."""
        twin = nemar_twin_of(dsid)
        if twin and (self.cache_dir / twin / _MANIFEST_DIR / "manifest.jsonl").exists():
            return self.cache_dir / twin
        return self.cache_dir / dsid

    def _from_disk(self, dsid: str) -> DatasetReport:
        root = self._root(dsid)
        header, entries = _read_manifest(root)
        rep = DatasetReport(
            dsid,
            served_as=root.name if root.name != dsid else None,
            entries=entries,
            subjects=header.get("n_subjects"),
            size_bytes=header.get("size_bytes"),
        )
        if not entries:
            rep.problems.append("no manifest: run plan or download first")
        rep.state = _state(root, entries)
        return rep

    # -- plan -------------------------------------------------------------

    def plan(
        self,
        *,
        datasets: Sequence[str] | str | None = None,
        subjects: Sequence[str] | None = None,
    ) -> CorpusReport:
        """Query the catalogue and write every dataset's manifest; no download.

        A catalogue change (files added, dropped or re-keyed) withdraws
        ``COMPLETE``; the report's ``drift`` lists the paths.
        """
        t0 = time.monotonic()
        reps = [self._plan(d, subjects)[0] for d in self._select(datasets)]
        return CorpusReport("plan", reps, time.monotonic() - t0)

    def _plan(
        self, dsid: str, subjects: Sequence[str] | None
    ) -> tuple[DatasetReport, Any]:
        rep = DatasetReport(dsid)
        try:
            doc = self.client.get_dataset(dsid)
            query = {"subject": [str(s) for s in subjects]} if subjects else {}
            records = list(self.client.find(dataset=dsid, **query)) if doc else []
        except Exception as exc:  # one unreachable dataset must not hide the others
            rep.problems.append(f"catalogue query failed: {_short(exc)}")
            rep.state = "problem"
            return rep, None
        if not records:
            rep.problems.append("not served by the catalogue")
            rep.state = "not served"
            return rep, None
        # A retired OpenNeuro id is served as its NEMAR re-host (ds003810 ->
        # on003810, #404); EEGDashDataset caches it under the served id.
        served = str(doc.get("dataset_id") or records[0]["dataset"])
        rep.served_as = served if served != dsid else None
        rep.size_bytes = doc.get("size_bytes")
        rep.subjects = len({s for r in records if (s := _subject_of(r))})
        expected = self._table_subjects(dsid)
        if expected is not None and not subjects and rep.subjects < expected:
            rep.problems.append(
                f"the catalogue lists {rep.subjects} subjects, the table {expected}"
            )
        from .dataset import EEGDashDataset  # noqa: PLC0415 (heavy import)

        try:
            dataset = EEGDashDataset(
                cache_dir=self.cache_dir,
                dataset=served,
                records=records,
                eeg_dash_instance=self.client,
                _suppress_comp_warning=True,
            )
            entries = _expected_files(doc, dataset)
        except Exception as exc:
            rep.problems.append(f"records not usable: {_short(exc)}")
            rep.state = "problem"
            return rep, None
        root = self.cache_dir / served
        # The catalogue is a trust boundary: no path may leave the dataset folder.
        escaped = [
            p
            for p in entries
            if not (root / p).resolve().is_relative_to(root.resolve())
        ]
        if escaped:
            rep.problems.append(f"path outside the dataset folder: {escaped[0]}")
            rep.state = "problem"
            return rep, None
        _, old = _read_manifest(root)
        for path, entry in entries.items():
            prev = old.get(path)
            if prev and prev.get("key") == entry["key"] and "state" in prev:
                entry["state"], entry["bytes"] = prev["state"], prev.get("bytes")
        if old and not subjects:
            rep.drift = {
                "added": sorted(set(entries) - set(old)),
                "removed": sorted(set(old) - set(entries)),
            }
        header = {
            "dataset": served,
            "requested": dsid,
            "planned": _now(),
            "plan_hash": _plan_hash(entries),
            "fingerprint": doc.get("ingestion_fingerprint"),
            "subjects": sorted(str(s) for s in subjects) if subjects else None,
            "n_subjects": rep.subjects,
            "size_bytes": rep.size_bytes,
        }
        _write_manifest(root, header, entries)
        complete = root / _MANIFEST_DIR / "COMPLETE"
        if _read_json(complete).get("plan_hash") != header["plan_hash"]:
            complete.unlink(missing_ok=True)
        rep.entries = entries
        rep.state = _state(root, entries)
        return rep, dataset

    # -- download ---------------------------------------------------------

    def download(
        self,
        *,
        n_jobs: int = 8,
        n_datasets: int = 2,
        datasets: Sequence[str] | str | None = None,
        subjects: Sequence[str] | None = None,
        time_budget: float | None = None,
    ) -> CorpusReport:
        """Plan, fetch what is missing with ``download_all``, then check sizes.

        Datasets go smallest first, *n_datasets* at a time, *n_jobs* workers in
        all. A dataset ``COMPLETE`` for the current plan whose files are all
        still at their sizes is skipped without network. No dataset starts after
        *time_budget* seconds; those left are ``deferred``. Resuming is free:
        ``download_all`` skips every file already at its key's size.
        """
        t0 = time.monotonic()
        planned = [self._plan(d, subjects) for d in self._select(datasets)]
        queue = sorted(
            (p for p in planned if p[1] is not None),
            key=lambda p: (p[0].size_bytes is None, p[0].size_bytes or 0),
        )
        n_datasets = max(1, min(n_datasets, n_jobs))
        workers = max(1, n_jobs // n_datasets)

        deadline = None if time_budget is None else t0 + time_budget
        with ThreadPoolExecutor(max_workers=n_datasets) as pool:
            list(pool.map(lambda p: self._fetch(*p, workers, deadline), queue))
        return CorpusReport("download", [p[0] for p in planned], time.monotonic() - t0)

    def _fetch(
        self, rep: DatasetReport, dataset: Any, workers: int, deadline: float | None
    ) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            rep.state = "deferred"
            rep.problems.append("not started: time budget spent")
            return
        root = self.cache_dir / (rep.served_as or rep.dataset_id)
        if rep.state == "complete":  # stat only; a file deleted since is refetched
            self._check_dataset(rep, root, "size", workers)
            if rep.ok:
                return
        t0 = time.monotonic()
        # download_all checks sizes only: a file verify found corrupt at the
        # right size would be kept, so it goes first and is fetched again.
        for entry in rep.entries.values():
            if entry.get("state") == "corrupt" and (root / entry["path"]).is_file():
                (root / entry["path"]).unlink()
        try:
            dataset.download_all(n_jobs=workers)
        except Exception as exc:  # DataIntegrityError lists the failed recordings
            failures = getattr(exc, "failures", None)
            rep.problems.append(
                f"{len(failures)} recording(s) failed, first: {_short(failures[0][1])}"
                if failures
                else f"download failed: {_short(exc)}"
            )
        self._check_dataset(rep, root, "size", workers)
        c = rep.counts
        logger.info(
            "%s: %d files, %d present, %d absent, %d problem(s), %.0f s",
            rep.dataset_id,
            c["files"],
            c.get("present", 0),
            c.get("absent", 0),
            len(rep.problems),
            time.monotonic() - t0,
        )

    # -- verify and status ------------------------------------------------

    def verify(
        self,
        *,
        level: str = "size",
        datasets: Sequence[str] | str | None = None,
        n_jobs: int = 8,
    ) -> CorpusReport:
        """Check every expected file, offline; write or withdraw ``COMPLETE``.

        ``size`` compares sizes with the git-annex keys; ``hash`` also reads
        every keyed file (one reader managed 19 MB/s on Lustre, hence *n_jobs*
        files at a time). Directory recordings and the BrainVision / EEGLAB
        data files their headers name are checked as well.
        """
        if level not in ("size", "hash"):
            raise ValueError(f"level must be 'size' or 'hash', got {level!r}")
        t0 = time.monotonic()
        reps = []
        for dsid in self._select(datasets):
            rep = self._from_disk(dsid)
            if rep.entries:
                self._check_dataset(rep, self._root(dsid), level, n_jobs)
            reps.append(rep)
        return CorpusReport("verify", reps, time.monotonic() - t0)

    @staticmethod
    def _check_dataset(rep: DatasetReport, root: Path, level: str, n_jobs: int) -> None:
        header, _ = _read_manifest(root)
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
            list(pool.map(lambda e: _check(root, e, level), rep.entries.values()))
        _write_manifest(root, header, rep.entries)
        complete = root / _MANIFEST_DIR / "COMPLETE"
        if rep.ok and level == "size" and complete.exists():
            pass  # keep the record of a hash-level verify
        elif rep.ok:
            info = {"verified": _now(), "level": level, "files": len(rep.entries)}
            info["plan_hash"] = header.get("plan_hash")
            _atomic_write(complete, json.dumps(info, indent=1) + "\n")
        else:
            complete.unlink(missing_ok=True)
        rep.state = _state(root, rep.entries)

    def status(self, *, datasets: Sequence[str] | str | None = None) -> CorpusReport:
        """Per dataset, from the manifests and ``COMPLETE`` markers only (no network, no stat)."""
        t0 = time.monotonic()
        reps = [self._from_disk(d) for d in self._select(datasets)]
        return CorpusReport("status", reps, time.monotonic() - t0)

    # -- holding ----------------------------------------------------------

    @property
    def datasets(self) -> dict[str, Any]:
        """``{id: EEGDashDataset(download=False)}`` from the cache, built once."""
        if self._datasets is None:
            from .dataset import EEGDashDataset  # noqa: PLC0415

            self._datasets = {
                dsid: EEGDashDataset(
                    cache_dir=self.cache_dir,
                    dataset=self._root(dsid).name,
                    download=False,
                )
                for dsid in self.ids
            }
        return self._datasets

    def records(self) -> Iterator[Any]:
        """Every recording (:class:`EEGDashRaw`) of every dataset, offline."""
        for dataset in self.datasets.values():
            yield from dataset.datasets


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _state(root: Path, entries: dict[str, dict]) -> str:
    """Complete (COMPLETE written), fetched, partial, planned or unplanned."""
    if not entries:
        return "unplanned"
    if (root / _MANIFEST_DIR / "COMPLETE").exists():
        return "complete"
    states = [e.get("state") for e in entries.values()]
    if all(s in _OK_STATES for s in states):
        return "fetched"
    return "partial" if "present" in states else "planned"


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eegdash.corpus",
        description="Plan, download, verify and report a corpus of EEGDash datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, doc in (
        ("plan", "query the catalogue, write the manifests, no download"),
        ("download", "fetch every missing file, then check sizes"),
        ("verify", "check every file against its record, offline"),
        ("status", "per dataset, from the manifests only"),
    ):
        p = sub.add_parser(name, help=doc)
        p.add_argument("corpus", help="a CSV with a dataset_id column, or ids a,b,c")
        p.add_argument(
            "--cache-dir", help="default EEGDASH_CACHE_DIR or ./.eegdash_cache"
        )
        p.add_argument("--datasets", help="restrict to these ids of the corpus, a,b")
        p.add_argument("--json", help="write the report here as JSON")
        if name in ("plan", "download"):
            p.add_argument("--subjects", help="restrict every dataset to these, a,b")
        if name in ("download", "verify"):
            p.add_argument("--n-jobs", type=int, default=8)
        if name == "download":
            p.add_argument("--n-datasets", type=int, default=2)
            p.add_argument(
                "--time-budget", type=float, help="seconds; then no new dataset"
            )
        if name == "verify":
            p.add_argument("--level", choices=("size", "hash"), default="size")
    args = parser.parse_args(argv)

    source: Any = args.corpus
    if not Path(source).is_file():
        source = [s for s in source.split(",") if s.strip()]
    corpus = EEGDashCorpus(source, cache_dir=args.cache_dir)
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
        report = corpus.status(datasets=args.datasets)
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1) + "\n")
    return 0 if report.ok else 1


__all__ = ["EEGDashCorpus", "CorpusReport", "DatasetReport", "main"]

if __name__ == "__main__":
    sys.exit(main())
