# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Plan, download, verify and hold a corpus of datasets: :class:`EEGDashCorpus`.

A corpus is a list of dataset ids (or a CSV with a ``dataset_id`` column). The
files are fetched by :meth:`EEGDashDataset.download_all`, exactly as for one
dataset; this module adds the list of expected files per dataset, a check of
the cache against it, and a report::

    corpus = EEGDashCorpus("eegdash-corpus-v0")   # a named corpus: its 77 datasets
    corpus.plan()                        # catalogue only: files, bytes, drift
    corpus.download(n_jobs=16, n_datasets=4, time_budget=6 * 3600)
    corpus.verify(level="hash")          # sizes, then git-annex hashes
    corpus.status()                      # offline, from the manifests
    for raw in corpus.records(): ...     # every recording, offline

Command line: ``python -m eegdash.corpus plan|download|verify|status <csv or
a,b,c> [--cache-dir ...]``; exit code 1 when anything is missing.

Per dataset the cache holds ``<dataset>/.eegdash/manifest.jsonl`` (a header,
then one line per expected file: path, kind, size, key, required, state) and
``<dataset>/.eegdash/COMPLETE``, written when every required file is there.
A keyless file the archive does not hold is ``absent``: reported, not
required (on002680 lists an ``.fdt`` beside every ``.set`` that embeds its
samples).
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
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from . import downloader
from .dataset._source_inference import nemar_twin_of
from .logging import logger
from .paths import get_default_cache_dir
from .schemas import NEMAR_ROOT_METADATA_FILES

_MANIFEST = Path(".eegdash", "manifest.jsonl")
_COMPLETE = Path(".eegdash", "COMPLETE")
_OK = ("present", "absent")
_COUNTED = ("files", "present", "absent", "missing", "corrupt", "incomplete")
_CORPORA = Path(__file__).parent / "corpora"  # <name>.csv: the named corpora


def _read_manifest(root: Path) -> tuple[dict, dict[str, dict]]:
    path = root / _MANIFEST
    if not path.exists():
        return {}, {}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return rows[0], {e["path"]: e for e in rows[1:]}


def _write_manifest(root: Path, header: dict, entries: dict[str, dict]) -> None:
    """Whole file through a unique temporary and a rename: never torn."""
    path = root / _MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    rows = [header, *entries.values()]
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _expected_files(doc: dict, dataset: Any) -> dict[str, dict]:
    """The files ``download_all`` fetches for *dataset*, from its records.

    A recording is required; a sidecar or dataset-level file only when the
    record carries its git-annex key (a keyless one the archive lacks is
    skipped by ``download_all`` too).
    """
    files = []  # (relpath, kind, key, required); a shared sidecar: the first wins
    storage = doc.get("storage") or {}
    keys = storage.get("annex_keys") or {}
    if storage.get("backend") in ("s3", "https"):  # _download_dataset_files
        for rel in filter(
            None, [storage.get("raw_key"), *(storage.get("dep_keys") or [])]
        ):
            files.append((rel, "dataset", keys.get(rel), bool(keys.get(rel))))
    for raw in dataset.datasets:
        if raw._storage_backend == "nemar":  # _fetch_nemar_root_metadata
            inline = raw.record["storage"].get("sidecar_inline") or {}
            files += [
                (n, "dataset", None, False)
                for n in NEMAR_ROOT_METADATA_FILES
                if n in inline
            ]
        for path in (raw.filecache, *raw._dep_paths):
            key = raw._annex_key_for(path)
            kind = "recording" if path == raw.filecache else "sidecar"
            rel = path.relative_to(raw.bids_root).as_posix()
            files.append((rel, kind, key, kind == "recording" or bool(key)))
    out: dict[str, dict] = {}
    for rel, kind, key, required in files:
        size = downloader.annex_key_info(key)[0] if key else None
        out.setdefault(
            rel,
            {"path": rel, "kind": kind, "size": size, "key": key, "required": required},
        )
    return out


def _check(root: Path, entry: dict, level: str) -> None:
    """Set ``entry["state"]``: present, absent, missing, corrupt or incomplete."""
    path = root / entry["path"]
    entry["bytes"] = None
    if not path.exists():
        entry["state"] = "missing" if entry["required"] else "absent"
        return
    if path.is_dir():  # CTF .ds / MEF3 .mefd: the reader's files, not just a folder
        entry["bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if path.suffix.lower() == ".mefd":
            from .io.mef import check_session  # noqa: PLC0415 (imports pymef)

            try:
                check_session(path)
                whole = any(path.glob("*.timd"))
            except ValueError:
                whole = False
        else:
            whole = any(path.glob("*.meg4")) and any(path.glob("*.res4"))
        entry["state"] = "present" if whole else "incomplete"
        return
    entry["bytes"] = path.stat().st_size
    entry["state"] = "present"
    if entry["size"] is not None and entry["bytes"] != entry["size"]:
        entry["state"] = "corrupt"
    elif level == "hash" and entry["key"]:
        try:
            downloader.verify_file(path, entry["key"], verify_hash=True)
        except OSError:
            entry["state"] = "corrupt"
    if entry["state"] == "present" and path.suffix.lower() == ".vhdr":
        # dep_keys may not list the samples file (#427): the header names it.
        from .io.brainvision import companion_names  # noqa: PLC0415

        if not any((path.parent / n).is_file() for n in companion_names(path)[0]):
            entry["state"] = "incomplete"
    # ponytail: a .set's .fdt is not checked here (reading a .set loads its matrix);
    # download_all raises for a missing one (_require_data_file), so download reports it.


def _state(root: Path, entries: dict[str, dict]) -> str:
    """Complete (COMPLETE written), fetched, partial, planned or unplanned."""
    if not entries:
        return "unplanned"
    if (root / _COMPLETE).exists():
        return "complete"
    states = [e.get("state") for e in entries.values()]
    if all(s in _OK for s in states):
        return "fetched"
    return "partial" if "present" in states else "planned"


@dataclass
class DatasetReport:
    """What one method found for one dataset."""

    dataset_id: str
    served_as: str | None = None  # the id the catalogue serves it under (ds* -> on*)
    state: str = "unplanned"
    entries: dict[str, dict] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    subjects: int | None = None
    size_bytes: int | None = None
    drift: dict[str, list[str]] | None = None  # plan: paths added, removed

    @property
    def ok(self) -> bool:
        states = [e.get("state") for e in self.entries.values()]
        return not self.problems and bool(states) and all(s in _OK for s in states)

    def row(self) -> dict[str, Any]:
        c = Counter(e.get("state") for e in self.entries.values())
        c["files"] = len(self.entries)
        present = sum(
            e["bytes"] or 0
            for e in self.entries.values()
            if e.get("state") == "present"
        )
        expected = self.size_bytes or sum(e["size"] or 0 for e in self.entries.values())
        return {
            "dataset_id": self.dataset_id,
            "served_as": self.served_as or "",
            "state": self.state,
            **{k: c[k] for k in _COUNTED},
            "gb_present": round(present / 1e9, 2),
            "gb_expected": round(expected / 1e9, 2),
            "subjects": self.subjects,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "seconds": round(self.seconds, 1),
            "datasets": [
                {
                    **d.row(),
                    "problems": d.problems,
                    "drift": d.drift,
                    "not_present": [
                        e for e in d.entries.values() if e.get("state") != "present"
                    ],
                }
                for d in self.datasets
            ],
        }

    def __str__(self) -> str:
        frame = pd.DataFrame([d.row() for d in self.datasets])
        total = frame[list(_COUNTED)].sum()
        return "\n".join(
            [
                frame.to_string(index=False),
                *(f"PROBLEM {p}" for p in self.problems),
                f"{self.action}: {len(frame)} datasets, "
                + ", ".join(f"{total[k]} {k}" for k in _COUNTED)
                + f"; {len(self.problems)} problem(s); {self.seconds:.0f} s",
            ]
        )


class EEGDashCorpus:
    """A corpus of datasets in one EEGDash cache.

    Parameters
    ----------
    source : str | Path | Sequence[str]
        The name of a released corpus (``"eegdash-corpus-v0"``), dataset ids,
        or the path of a CSV with a ``dataset_id`` column (an ``n_subjects``
        column is checked against the catalogue when present).
    cache_dir : str | Path | None
        As for :class:`EEGDashDataset` (``EEGDASH_CACHE_DIR``, else
        ``./.eegdash_cache``).
    eeg_dash_instance : EEGDash | None
        The catalogue client, created when first needed (verify, status and
        records never need one).

    """

    def __init__(
        self,
        source: str | Path | Sequence[str],
        *,
        cache_dir: str | Path | None = None,
        eeg_dash_instance: Any = None,
    ):
        if isinstance(source, str) and not Path(source).is_file():
            named = _CORPORA / f"{source}.csv"
            if not named.is_file():
                known = sorted(p.stem for p in _CORPORA.glob("*.csv"))
                raise ValueError(f"{source!r} is neither a file nor a corpus: {known}")
            source = named
        if isinstance(source, (str, Path)):
            table = pd.read_csv(source, dtype=str, keep_default_na=False)
            if "dataset_id" not in table.columns:
                raise ValueError(f"{source}: a corpus table needs a dataset_id column")
        else:
            table = pd.DataFrame({"dataset_id": list(source)})
        table["dataset_id"] = table["dataset_id"].astype(str).str.strip()
        self.table = table[table["dataset_id"] != ""].set_index("dataset_id")
        self.ids: list[str] = self.table.index.tolist()
        if not self.ids:
            raise ValueError("EEGDashCorpus needs at least one dataset id")
        if self.table.index.has_duplicates:
            dup = sorted(set(self.table.index[self.table.index.duplicated()]))
            raise ValueError(f"duplicate dataset ids in the corpus: {dup}")
        self.cache_dir = Path(cache_dir or get_default_cache_dir())
        self.client = eeg_dash_instance

    def _root(self, dsid: str) -> Path:
        """The cache folder of *dsid*, offline: its NEMAR twin's when the plan wrote there."""
        twin = nemar_twin_of(dsid)
        if twin and (self.cache_dir / twin / _MANIFEST).exists():
            return self.cache_dir / twin
        return self.cache_dir / dsid

    def _from_disk(self, dsid: str) -> DatasetReport:
        root = self._root(dsid)
        header, entries = _read_manifest(root)
        rep = DatasetReport(
            dsid,
            served_as=root.name if root.name != dsid else None,
            state=_state(root, entries),
            entries=entries,
            subjects=header.get("n_subjects"),
            size_bytes=header.get("size_bytes"),
        )
        if not entries:
            rep.problems.append("no manifest: run plan or download first")
        return rep

    # -- plan -------------------------------------------------------------

    def plan(self, *, subjects: Sequence[str] | None = None) -> CorpusReport:
        """Query the catalogue and write every dataset's manifest; no download.

        A catalogue change (files added, dropped or re-keyed) withdraws
        ``COMPLETE``; the report's ``drift`` lists the paths.
        """
        t0 = time.monotonic()
        reps = [self._plan(d, subjects)[0] for d in self.ids]
        return CorpusReport("plan", reps, time.monotonic() - t0)

    def _plan(
        self, dsid: str, subjects: Sequence[str] | None
    ) -> tuple[DatasetReport, Any]:
        """Plan one dataset; returns its report and the ``EEGDashDataset`` (None on a problem)."""
        from .dataset import EEGDashDataset  # noqa: PLC0415 (heavy import)

        rep = DatasetReport(dsid, state="problem")
        if self.client is None:
            from .api import EEGDash  # noqa: PLC0415

            self.client = EEGDash()
        try:
            doc = self.client.get_dataset(dsid)
            query = {"subject": [str(s) for s in subjects]} if subjects else {}
            records = list(self.client.find(dataset=dsid, **query)) if doc else []
            if not records:
                rep.problems.append("not served by the catalogue")
                return rep, None
            # A retired OpenNeuro id is served as its NEMAR re-host (ds003810 ->
            # on003810, #404); EEGDashDataset caches it under the served id.
            served = str(doc.get("dataset_id") or records[0]["dataset"])
            dataset = EEGDashDataset(
                cache_dir=self.cache_dir,
                dataset=served,
                records=records,
                eeg_dash_instance=self.client,
                _suppress_comp_warning=True,
            )
            entries = _expected_files(doc, dataset)
        except Exception as exc:  # one broken dataset must not hide the others
            rep.problems.append(f"plan failed: {exc!r}"[:300])
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
            return rep, None
        rep.served_as = served if served != dsid else None
        rep.size_bytes = doc.get("size_bytes")
        rep.subjects = len({r["subject"] for r in records if r.get("subject")})
        wanted = self.table.loc[dsid].get("n_subjects", "")
        if wanted and not subjects and rep.subjects < int(float(wanted)):
            rep.problems.append(
                f"the catalogue lists {rep.subjects} subjects, the table {wanted}"
            )
        _, old = _read_manifest(root)
        for path, entry in entries.items():  # keep what the last check found
            prev = old.get(path)
            if prev and prev.get("key") == entry["key"] and "state" in prev:
                entry["state"], entry["bytes"] = prev["state"], prev["bytes"]
        if old and not subjects:
            rep.drift = {
                "added": sorted(set(entries) - set(old)),
                "removed": sorted(set(old) - set(entries)),
            }
        identity = sorted((p, e["key"] or "") for p, e in entries.items())
        header = {
            "dataset": served,
            "requested": dsid,
            "planned": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "plan_hash": hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16],
            "fingerprint": doc.get("ingestion_fingerprint"),
            "subjects": sorted(map(str, subjects)) if subjects else None,
            "n_subjects": rep.subjects,
            "size_bytes": rep.size_bytes,
        }
        _write_manifest(root, header, entries)
        complete = root / _COMPLETE
        if (
            complete.exists()
            and json.loads(complete.read_text()).get("plan_hash") != header["plan_hash"]
        ):
            complete.unlink()  # COMPLETE holds for the plan it was written for only
        rep.entries, rep.state = entries, _state(root, entries)
        return rep, dataset

    # -- download ---------------------------------------------------------

    def download(
        self,
        *,
        n_jobs: int = 8,
        n_datasets: int = 2,
        subjects: Sequence[str] | None = None,
        time_budget: float | None = None,
    ) -> CorpusReport:
        """Plan, fetch what is missing with ``download_all``, then check sizes.

        Datasets go smallest first, *n_datasets* at a time, *n_jobs* workers in
        all. A dataset ``COMPLETE`` for the current plan whose files are all
        still at their sizes is skipped without network. No dataset starts
        after *time_budget* seconds (those left are ``deferred``). Resuming is
        free: ``download_all`` skips every file already at its key's size.
        """
        t0 = time.monotonic()
        planned = [self._plan(d, subjects) for d in self.ids]
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
            self._check_all(rep, root, "size", workers)
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
            failures = getattr(exc, "failures", None) or [(None, exc)]
            rep.problems.append(
                f"{len(failures)} recording(s) failed, first: {failures[0][1]!r}"[:300]
            )
        self._check_all(rep, root, "size", workers)
        logger.info("%s: %s, %.0f s", rep.dataset_id, rep.row(), time.monotonic() - t0)

    # -- verify and status ------------------------------------------------

    def verify(self, *, level: str = "size", n_jobs: int = 8) -> CorpusReport:
        """Check every expected file, offline; write or withdraw ``COMPLETE``.

        ``size`` compares sizes with the git-annex keys; ``hash`` also reads
        every keyed file (*n_jobs* at a time: one reader hashed 19 MB/s on
        Lustre). Directory recordings and the BrainVision data file are checked
        too.
        """
        if level not in ("size", "hash"):
            raise ValueError(f"level must be 'size' or 'hash', got {level!r}")
        t0 = time.monotonic()
        reps = [self._from_disk(d) for d in self.ids]
        for rep in reps:
            if rep.entries:
                self._check_all(rep, self._root(rep.dataset_id), level, n_jobs)
        return CorpusReport("verify", reps, time.monotonic() - t0)

    @staticmethod
    def _check_all(rep: DatasetReport, root: Path, level: str, n_jobs: int) -> None:
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
            list(pool.map(lambda e: _check(root, e, level), rep.entries.values()))
        header, _ = _read_manifest(root)
        _write_manifest(root, header, rep.entries)
        complete = root / _COMPLETE
        if not rep.ok:
            complete.unlink(missing_ok=True)
        elif level == "hash" or not complete.exists():  # keep a hash-level record
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            info = {"verified": now, "level": level, "plan_hash": header["plan_hash"]}
            complete.write_text(json.dumps(info) + "\n")
        rep.state = _state(root, rep.entries)

    def status(self) -> CorpusReport:
        """Per dataset, from the manifests and ``COMPLETE`` markers only (no network, no stat)."""
        t0 = time.monotonic()
        return CorpusReport(
            "status", [self._from_disk(d) for d in self.ids], time.monotonic() - t0
        )

    def records(self) -> Iterator[Any]:
        """Every recording (:class:`EEGDashRaw`) of every dataset, offline."""
        from .dataset import EEGDashDataset  # noqa: PLC0415 (heavy import)

        for dsid in self.ids:
            folder = self._root(dsid).name
            yield from EEGDashDataset(
                cache_dir=self.cache_dir, dataset=folder, download=False
            ).datasets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eegdash.corpus",
        description="Plan, download, verify and report a corpus of EEGDash datasets.",
    )
    parser.add_argument("command", choices=("plan", "download", "verify", "status"))
    parser.add_argument("corpus", help="a CSV with a dataset_id column, or ids a,b,c")
    parser.add_argument(
        "--cache-dir", help="default EEGDASH_CACHE_DIR or ./.eegdash_cache"
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-datasets", type=int, default=2, help="download")
    parser.add_argument(
        "--time-budget", type=float, help="download: seconds, then no new dataset"
    )
    parser.add_argument(
        "--level", choices=("size", "hash"), default="size", help="verify"
    )
    parser.add_argument("--json", help="write the report here as JSON")
    args = parser.parse_args(argv)
    source = args.corpus if Path(args.corpus).is_file() else args.corpus.split(",")
    corpus = EEGDashCorpus(source, cache_dir=args.cache_dir)
    if args.command == "plan":
        report = corpus.plan()
    elif args.command == "download":
        report = corpus.download(
            n_jobs=args.n_jobs, n_datasets=args.n_datasets, time_budget=args.time_budget
        )
    elif args.command == "verify":
        report = corpus.verify(level=args.level, n_jobs=args.n_jobs)
    else:
        report = corpus.status()
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1) + "\n")
    return 0 if report.ok else 1


__all__ = ["EEGDashCorpus", "CorpusReport", "DatasetReport", "main"]

if __name__ == "__main__":
    sys.exit(main())
