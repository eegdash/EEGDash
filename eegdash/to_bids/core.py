# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Bring a source tree to BIDS with an adapter: plan, scan, write, finish.

- **plan**: the adapter's recordings to BIDS paths. Two recordings on one
  path, a source file named twice, a source outside the source tree, an
  entity label that is not alphanumeric, a data format EEGDash has no reader
  for, or (when the source is linked) a source of another format than the
  BIDS data file stop the run before anything is written. Held-out
  recordings (the adapter's table, ``--exclude``) are left out here and
  reported with their reason.
- **scan**: every header is read with the reader the conversion will use
  (``eegdash.io.get_reader(ext)(path, preload=False)`` under
  :func:`eegdash.io.tolerant`), from the file the tree will hold (the source
  when it is linked, the adapter's file when it writes one), so channels.tsv
  lists that reader's own channels, in its order.
- **write**: per recording, the data file is first put at a temporary name (a
  symbolic link, a hard link or a copy of the source, or the adapter's own
  file), then its sidecars are written, then the data file is renamed into
  place; a failure removes what this call wrote. Every file goes through a
  dot-name ``.<stem>.<pid>.tmp<suffix>`` and :func:`os.replace`, sidecars
  only when their content changes; stale temporary files of an interrupted
  run are removed. A data file already in place for the same source is kept;
  one that holds another source is refused. Subjects are spread over
  processes (``n_jobs``) and over job-array tasks (``shard``, ``n_shards``:
  subjects sorted, every n-th): a subject belongs to one worker, so no two
  processes write one folder. The finished subjects' rows are appended to
  ``.eegdash/to_bids/scan-<k>.jsonl`` (``scan-all.jsonl`` without a shard)
  every ``FLUSH_SECONDS``, so a cut job resumes without reading those headers
  again (one append per subject, each flushed to disk, took 0.17 s on a
  parallel file system: longer than writing the subject).
- **finish**, by the call without a shard: the root files (participants,
  dataset_description, README.md, CHANGES, LICENSE, ``sourcedata/``,
  ``task-<t>_events.json``, ``.bidsignore``) and ``.eegdash/to_bids/``
  (``map.csv``, ``report.json``, ``DONE``). A data file in the tree that the
  plan does not name is reported, never deleted.

The tree (every file but ``.eegdash/``) and ``map.csv`` depend on nothing but
the source, the adapter's revision and the options: no clock, host or package
version. ``report.json`` and ``DONE`` also count what this call wrote and what
it kept.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..corpus_convert import PRIMARY_EXT, is_local_id
from ..logging import logger
from .adapter import (
    CHANNEL_TYPES,
    Adapter,
    Channel,
    Event,
    Exclusion,
    Header,
    Recording,
    read_exclusions,
)

LINK_MODES = ("symlink", "hard", "copy")
BIDS_VERSION = "1.9.0"
STATE = Path(".eegdash", "to_bids")  # the converter's own files, beside the tree
LABEL_RE = re.compile(r"[A-Za-z0-9]+")  # a BIDS entity label
VERSION_RE = re.compile(r"(.+)-(\d[a-z0-9.]*)")  # <name>-<version>
TMP_RE = re.compile(r"\..+\.\d+\.tmp(\.[A-Za-z0-9]+)?")  # the names of _tmp()
FLUSH_SECONDS = 30.0  # how often the scan file is appended to, at most
COMPARE_BYTES = 1 << 20  # a copy is the source's when its first and last MiB are
COUNT_KEYS = {
    "EEG": "EEGChannelCount",
    "EOG": "EOGChannelCount",
    "ECG": "ECGChannelCount",
    "EMG": "EMGChannelCount",
    "MISC": "MiscChannelCount",
    "TRIG": "TriggerChannelCount",
}
# A recording's status: written now, kept from an earlier run, planned (a dry
# run), excluded (held out), or a problem: unreadable header, invalid channels,
# refused (its path holds another file), failed (an error while writing).
STATUSES = (
    "written",
    "kept",
    "planned",
    "excluded",
    "unreadable",
    "invalid",
    "refused",
    "failed",
)
IN_TREE = ("written", "kept")  # statuses of a recording the tree holds
MAP_FIELDS = ["source", "bids_path", "size", "link", "status", "reason"]
CSV_FIELDS = [
    "source",
    "bids_path",
    "subject",
    "status",
    "reason",
    "size",
    "sfreq",
    "n_times",
    "duration_s",
    "n_channels",
    "n_unmapped",
    "n_events",
]


# ---------------------------------------------------------------- the plan


def split_id(dataset_id: str) -> tuple[str, str]:
    """``tueg-2.0.1`` -> ``("tueg", "2.0.1")``; a ValueError for an id that is not local."""
    m = VERSION_RE.fullmatch(dataset_id)
    if not is_local_id(dataset_id) or not m:
        raise ValueError(
            f"{dataset_id!r}: a converted dataset is named <name>-<version> "
            "(tueg-2.0.1)"
        )
    return m[1], m[2]


def check_paths(src: Path, out: Path) -> None:
    """Refuse an output inside the source, or a source inside the output."""
    src, out = Path(src).resolve(), Path(out).resolve()
    if not src.is_dir():
        raise ValueError(f"{src}: the source is not a folder")
    if out == src or out.is_relative_to(src):
        raise ValueError(f"refusing: {out} is inside the source {src}")
    if src.is_relative_to(out):
        raise ValueError(f"refusing: the source {src} is inside the output {out}")


def bids_path(rec: Recording, extension: str) -> str:
    """The data file of *rec*, relative to the BIDS root."""
    return (rec.folder / f"{rec.stem}_eeg{extension}").as_posix()


def plan(adapter: Adapter) -> list[Recording]:
    """The adapter's recordings, sorted by BIDS path; a ValueError before any write.

    Refused: a BIDS data format the conversion or EEGDash cannot read, a
    linked source of another format, an entity label that is not
    alphanumeric, a source outside the source tree, a source named twice, two
    sources on one BIDS path.
    """
    from ..io import READERS  # noqa: PLC0415 (imports mne)

    ext = adapter.extension
    if ext not in PRIMARY_EXT or ext not in READERS:
        known = [e for e in PRIMARY_EXT if e in READERS]
        raise ValueError(f"{ext}: the data file must be one of {', '.join(known)}")
    recs = list(adapter.recordings())
    bad, outside, other = [], [], []
    for rec in recs:
        bad += [f"{k}-{v}" for k, v in rec.entities if not LABEL_RE.fullmatch(v)]
        if rec.source.is_absolute() or ".." in rec.source.parts:
            outside.append(rec.source.as_posix())
        if adapter.linkable and rec.source.suffix.lower() != ext:
            other.append(rec.source.as_posix())
    if bad:
        raise ValueError(
            f"{len(bad)} entity label(s) not alphanumeric, first: {bad[0]!r}"
        )
    if outside:
        raise ValueError(f"source outside the source tree: {outside[0]}")
    if other:
        raise ValueError(
            f"{len(other)} source(s) linked as {ext} are not {ext} files, first: "
            f"{other[0]}"
        )
    by_path: dict[str, list[str]] = {}
    for rec in recs:
        by_path.setdefault(bids_path(rec, ext), []).append(rec.source.as_posix())
    twice = {p: s for p, s in by_path.items() if len(s) > 1}
    if twice:
        path, sources = next(iter(sorted(twice.items())))
        raise ValueError(
            f"{len(twice)} BIDS path(s) named by two source files, first: "
            f"{path} <- {', '.join(sources)}"
        )
    sources = Counter(rec.source.as_posix() for rec in recs)
    dup = sorted(s for s, n in sources.items() if n > 1)
    if dup:
        raise ValueError(f"{len(dup)} source file(s) named twice, first: {dup[0]}")
    return sorted(recs, key=lambda r: bids_path(r, ext))


def excluded_by(rec: Recording, rules: Sequence[Exclusion]) -> Exclusion | None:
    """The first rule that leaves *rec* out, or None."""
    return next((r for r in rules if r.matches(rec)), None)


# ---------------------------------------------------------------- the scan


def read_header(path: Path) -> tuple[Header, list[str]]:
    """The header of a data file, read with the conversion's reader; and the reader's warnings."""
    from ..io import READERS, get_reader, tolerant  # noqa: PLC0415 (imports mne)

    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"no reader for {ext!r}")
    with tolerant(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = get_reader(ext)(path, preload=False, verbose="ERROR")
    orig = getattr(raw, "_orig_units", None) or {}  # as MNE-BIDS writes units
    header = Header(
        labels=tuple(raw.ch_names),
        units=tuple(str(orig.get(n, "n/a")) or "n/a" for n in raw.ch_names),
        sfreq=float(raw.info["sfreq"]),
        n_times=int(raw.n_times),
        meas_date=raw.info["meas_date"],
    )
    return header, list(dict.fromkeys(str(w.message) for w in caught))


def check_channels(channels: Sequence[Channel], header: Header) -> None:
    """The rules of a readable channels.tsv; a ValueError says which one is broken.

    One row per channel of the reader's list, in its order (MNE-BIDS sets no
    name from a channels.tsv of another length, and relabels by position):
    each row's ``description`` is the reader's label at that position. Names
    unique, types known.
    """
    if len(channels) != len(header.labels):
        raise ValueError(
            f"{len(channels)} channels.tsv rows for the reader's "
            f"{len(header.labels)} channels"
        )
    moved = [
        k
        for k, (c, label) in enumerate(zip(channels, header.labels))
        if c.description != label
    ]
    if moved:
        k = moved[0]
        raise ValueError(
            f"channels.tsv row {k + 1} describes {channels[k].description!r}, the "
            f"reader's channel {k + 1} is {header.labels[k]!r}"
        )
    names = [c.name for c in channels]
    if any(not n or re.search(r"[\t\n\r]", n) for n in names):
        raise ValueError("an empty channel name, or one with a tab or a line break")
    dup = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup:
        raise ValueError(f"channel name(s) not unique: {', '.join(dup)}")
    kinds = sorted({c.type for c in channels} - set(CHANNEL_TYPES))
    if kinds:
        raise ValueError(f"channel type(s) not in {CHANNEL_TYPES}: {kinds}")


def _num(x: float) -> int | float:
    """An integral float as an int: ``512`` rather than ``512.0`` in a sidecar."""
    return int(x) if float(x).is_integer() else float(x)


def _cell(value: Any) -> str:
    """A TSV cell: ``n/a`` for nothing, no tab or line break."""
    text = "n/a" if value is None or value == "" else str(value)
    return re.sub(r"[\t\n\r]+", " ", text)


def channels_tsv(channels: Sequence[Channel]) -> str:
    """channels.tsv: ``name type units description``, one row per channel."""
    rows = ["name\ttype\tunits\tdescription"]
    rows += [
        "\t".join(map(_cell, (c.name, c.type, c.units, c.description)))
        for c in channels
    ]
    return "\n".join(rows) + "\n"


def events_tsv(events: Sequence[Event]) -> str:
    """events.tsv: ``onset duration trial_type``, by onset."""
    rows = ["onset\tduration\ttrial_type"]
    for e in sorted(events, key=lambda e: (e.onset, e.duration, e.trial_type)):
        rows.append(
            f"{round(float(e.onset), 6)!r}\t{round(float(e.duration), 6)!r}\t"
            f"{_cell(e.trial_type)}"
        )
    return "\n".join(rows) + "\n"


def eeg_json(
    rec: Recording, header: Header, channels: Sequence[Channel], extra: dict
) -> dict[str, Any]:
    """The ``_eeg.json`` sidecar: the core's keys, the adapter's, then the counts."""
    computed = {
        "TaskName": rec.task,
        "SamplingFrequency": _num(header.sfreq),
        "RecordingDuration": round(header.n_times / header.sfreq, 6),
    }
    side = {
        **computed,
        "PowerLineFrequency": "n/a",
        "EEGReference": "n/a",
        "SoftwareFilters": "n/a",
        "RecordingType": "continuous",
    }
    side.update(extra)
    side.update(computed)  # the header's values, whatever the adapter said
    kinds = Counter(c.type for c in channels)
    side.update({key: kinds.get(t, 0) for t, key in COUNT_KEYS.items()})
    return side


# ------------------------------------------------------------- the writing


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _csv_text(rows: Iterable[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _tmp(path: Path) -> Path:
    """``.<stem>.<pid>.tmp<suffix>`` beside *path*.

    A dot-name, which file lists and the conversion's discovery skip, and the
    suffix a reader needs.
    """
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")


def sweep_tmp(folder: Path, *, recursive: bool = True) -> int:
    """Remove the temporary files (:func:`_tmp`'s names only) an interrupted run left."""
    if not folder.is_dir():
        return 0
    found = folder.rglob(".*") if recursive else folder.glob(".*")
    stale = [p for p in found if TMP_RE.fullmatch(p.name) and not p.is_dir()]
    for p in stale:
        p.unlink(missing_ok=True)
    return len(stale)


def write_bytes(path: Path, data: bytes) -> bool:
    """Write *data* through a temporary dot-name and a rename, unless the file already holds it."""
    try:
        if not path.is_symlink() and path.read_bytes() == data:
            return False
    except (FileNotFoundError, IsADirectoryError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def write_text(path: Path, text: str) -> bool:
    """:func:`write_bytes` for UTF-8 text."""
    return write_bytes(path, text.encode("utf-8"))


def _same_bytes(a: Path, b: Path) -> bool:
    """Equal sizes, and equal first and last :data:`COMPARE_BYTES`."""
    size = a.stat().st_size
    if size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(COMPARE_BYTES) != fb.read(COMPARE_BYTES):
            return False
        start = max(0, size - COMPARE_BYTES)
        fa.seek(start)
        fb.seek(start)
        return fa.read() == fb.read()


def link_state(dest: Path, source: Path, link: str, linkable: bool) -> str:
    """``absent``, ``same`` (the data file is the *link* of *source*) or ``other``.

    For a file the adapter writes, ``same`` means only that a regular file is
    there; whether it is current is the earlier run's row to say.
    """
    if not dest.exists() and not dest.is_symlink():
        return "absent"
    if not linkable:
        return "same" if dest.is_file() and not dest.is_symlink() else "other"
    if link == "symlink":
        same = dest.is_symlink() and os.readlink(dest) == str(source)
        return "same" if same else "other"
    if dest.is_symlink() or not dest.is_file():
        return "other"
    if link == "hard":
        return "same" if os.path.samefile(dest, source) else "other"
    same = not os.path.samefile(dest, source) and _same_bytes(dest, source)
    return "same" if same else "other"


def make_link(source: Path, path: Path, link: str) -> None:
    """Create *path* as a symbolic link (absolute) to *source*, a hard link or a copy."""
    path.unlink(missing_ok=True)
    if link == "symlink":
        os.symlink(source, path)
    elif link == "hard":
        os.link(source, path)
    else:
        shutil.copyfile(source, path)  # the bytes only: never the source's mode


@dataclass
class Context:
    """What every worker needs: the adapter (with its plan), the trees, the options."""

    adapter: Adapter
    src: Path  # resolved
    out: Path
    link: str
    dry_run: bool


def _base_row(ctx: Context, rec: Recording) -> dict[str, Any]:
    return {
        "source": rec.source.as_posix(),
        "bids_path": bids_path(rec, ctx.adapter.extension),
        "subject": rec.subject,
        "session": rec.session,
        "task": rec.task,
        "key": rec.key,
        "adapter": f"{ctx.adapter.name}/{ctx.adapter.revision}",
        "status": "",
        "reason": "",
        "size": None,
        "mtime_ns": None,
        "link": "",
        "warnings": [],
        "read_warnings": [],
    }


def _sidecar_paths(rel: str) -> tuple[str, str, str]:
    stem = rel.rsplit("_eeg.", 1)[0]
    return f"{stem}_channels.tsv", f"{stem}_eeg.json", f"{stem}_events.tsv"


def _unchanged(prior: dict | None, row: dict) -> bool:
    """The earlier run's row is for this source, unmodified since, and this adapter revision."""
    keys = ("source", "size", "mtime_ns", "adapter")
    return bool(prior) and all(prior.get(k) == row[k] for k in keys)


def _scan(
    ctx: Context, rec: Recording, path: Path, row: dict
) -> tuple[dict, list[Channel], list[Event] | None, dict] | dict:
    """Read the header of *path* and ask the adapter; the filled row and what to write, or a problem row."""
    adapter = ctx.adapter
    adapter.warnings.clear()
    try:
        header, row["read_warnings"] = read_header(path)
    except Exception as err:  # one unreadable file must not stop the others
        reason = f"{err.__class__.__name__}: {err}"[:300]
        return {**row, "status": "unreadable", "reason": reason, "warnings": []}
    row.update(
        sfreq=_num(header.sfreq),
        n_times=header.n_times,
        duration_s=round(header.n_times / header.sfreq, 3) if header.sfreq else None,
        labels=list(header.labels),
    )
    try:
        channels = adapter.channels(rec, header)
        check_channels(channels, header)
        events = adapter.events(rec, header)
        sidecar = eeg_json(rec, header, channels, adapter.sidecar(rec, header))
    except Exception as err:  # an adapter or rule failure on one file: reported
        warned = list(adapter.warnings)
        return {
            **row,
            "status": "invalid",
            "reason": f"{err}"[:300],
            "warnings": warned,
        }
    row.update(
        names=[c.name for c in channels],
        types=[c.type for c in channels],
        unmapped=[c.description or c.name for c in channels if not c.mapped],
        n_channels=len(channels),
        n_unmapped=sum(not c.mapped for c in channels),
        n_events=None if events is None else len(events),
        trial_types=sorted({e.trial_type for e in events or []}),
        warnings=list(adapter.warnings),
    )
    return row, channels, events, sidecar


def _prune(folder: Path, root: Path) -> None:
    """Remove *folder* and its parents up to *root* while they are empty."""
    while folder != root and root in folder.parents:
        try:
            folder.rmdir()
        except OSError:  # not empty, or not there
            return
        folder = folder.parent


def convert_recording(
    ctx: Context, rec: Recording, prior: dict | None = None
) -> dict[str, Any]:
    """Scan one recording and, unless a dry run, write it; returns its report row.

    See :func:`_convert_recording`. A recording that is not written leaves no
    empty folder behind.
    """
    row = _convert_recording(ctx, rec, prior)
    if not ctx.dry_run and row["status"] not in IN_TREE:
        _prune((ctx.out / row["bids_path"]).parent, ctx.out)
    return row


def _convert_recording(
    ctx: Context, rec: Recording, prior: dict | None = None
) -> dict[str, Any]:
    """Scan one recording and, unless a dry run, write it; returns its report row.

    *prior* is the row an earlier run wrote for the same BIDS path. A path
    that holds another source than the plan's is refused. When the data file
    is still the link of the same source, unmodified, by the same adapter
    revision, with its sidecars, nothing is read or written. A file the
    adapter writes is written again otherwise, and its header read from it.
    """
    adapter = ctx.adapter
    row = _base_row(ctx, rec)
    rel = row["bids_path"]
    source, dest = ctx.src / rec.source, ctx.out / rel
    try:
        st = source.stat()
    except OSError as err:
        return {**row, "status": "unreadable", "reason": f"{err}"[:300]}
    row["size"] = st.st_size if source.is_file() else None
    row["mtime_ns"] = st.st_mtime_ns
    state = "absent"
    if not ctx.dry_run:
        state = link_state(dest, source, ctx.link, adapter.linkable)
        if state != "absent" and prior and prior.get("source") != row["source"]:
            reason = f"holds {prior['source']} from an earlier run, not {row['source']}"
            return {**row, "status": "refused", "reason": reason}
        if state == "other":
            what = ctx.link if adapter.linkable else "file written"
            reason = f"exists and is not the {what} of {row['source']}"
            return {**row, "status": "refused", "reason": reason}
        sidecars = (prior or {}).get("sidecars") or []
        if (
            state == "same"
            and _unchanged(prior, row)
            and sidecars
            and all((ctx.out / p).is_file() for p in sidecars)
        ):
            return {**prior, "status": "kept"}
        if not adapter.linkable:
            state = "absent"  # its source or the adapter changed: written again
    scratch = (
        tempfile.TemporaryDirectory(prefix="eegdash-to-bids-")
        if ctx.dry_run and not adapter.linkable
        else None
    )
    data = None  # the data file at its temporary name
    if scratch is not None:
        data = Path(scratch.name) / dest.name
    elif state == "absent" and not ctx.dry_run:
        data = _tmp(dest)
    sidecars_new: list[Path] = []
    try:
        if data is not None:
            try:
                data.parent.mkdir(parents=True, exist_ok=True)
                if adapter.linkable:
                    make_link(source, data, ctx.link)
                else:
                    adapter.write_data(rec, data)
            except Exception as err:  # a cross-device link, a failing writer: reported
                return {**row, "status": "failed", "reason": f"{err}"[:300]}
        scanned = _scan(ctx, rec, source if adapter.linkable else data, row)
        if isinstance(scanned, dict):
            return scanned
        row, channels, events, sidecar = scanned
        if ctx.dry_run:
            return {**row, "status": "planned"}
        tsv, side, ev = _sidecar_paths(rel)
        texts = {tsv: channels_tsv(channels), side: _dumps(sidecar)}
        if events is not None:
            texts[ev] = events_tsv(events)
        changed = data is not None
        try:
            for name, text in texts.items():
                path = ctx.out / name
                existed = path.exists()
                if write_text(path, text):
                    changed = True
                    if not existed:
                        sidecars_new.append(path)
            if data is not None:
                os.replace(data, dest)
        except OSError as err:  # a full disk: nothing of this call is left behind
            for path in sidecars_new:
                path.unlink(missing_ok=True)
            return {**row, "status": "failed", "reason": f"{err}"[:300]}
        row["sidecars"] = list(texts)
        row["link"] = ctx.link if adapter.linkable else "write"
        return {**row, "status": "written" if changed else "kept"}
    finally:
        if scratch is not None:
            scratch.cleanup()
        elif data is not None and (data.exists() or data.is_symlink()):
            data.unlink()


def convert_subject(
    ctx: Context, recs: Sequence[Recording], prior: dict[str, dict]
) -> list[dict]:
    """The rows of one subject's recordings: one worker writes one subject folder."""
    if not ctx.dry_run and recs:
        sweep_tmp(ctx.out / f"sub-{recs[0].subject}")
    ext = ctx.adapter.extension
    return [convert_recording(ctx, r, prior.get(bids_path(r, ext))) for r in recs]


_CONTEXT: Context | None = None  # a worker process's context, set once


def _init_worker(ctx: Context) -> None:
    global _CONTEXT
    _CONTEXT = ctx


def _pool_job(args: tuple[list[Recording], dict[str, dict]]) -> list[dict]:
    return convert_subject(_CONTEXT, *args)


def prior_rows(out: Path) -> dict[str, dict]:
    """BIDS path -> the last row of an earlier run that put it in the tree (``scan-*.jsonl``)."""
    last: dict[str, dict] = {}
    for path in sorted((out / STATE).glob("scan-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # the last line of a cut job
                continue
            last[row["bids_path"]] = row
    return {p: r for p, r in last.items() if r.get("status") in IN_TREE}


def _append_rows(path: Path, rows: list[dict]) -> None:
    """Finished subjects' rows at the end of the scan file, flushed: what a cut job resumes from."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------- the report


@dataclass
class BidsReport:
    """The result of :func:`eegdash.to_bids.to_bids`: one row per recording, problems, warnings."""

    dataset_id: str
    adapter: str  # <name>/<revision>
    options: dict[str, Any] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    held_out_kept: int = 0  # recordings of the held-out table kept (keep_held_out)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def counts(self) -> dict[str, int]:
        """Recordings per status, subjects in the tree (or planned), and hours."""
        c = Counter(r["status"] for r in self.rows)
        kept = [r for r in self.rows if r["status"] in (*IN_TREE, "planned")]
        return {
            "recordings": len(self.rows),
            **{k: c[k] for k in STATUSES},
            "subjects": len({r["subject"] for r in kept}),
            "hours": round(sum(r.get("duration_s") or 0 for r in kept) / 3600, 3),
        }

    def labels(self) -> dict[str, dict[str, Any]]:
        """Source label -> recordings, its BIDS name and type, whether a rule mapped it."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.rows:
            if "names" not in r:
                continue
            unmapped = set(r["unmapped"])
            for label, name, kind in zip(r["labels"], r["names"], r["types"]):
                e = out.setdefault(
                    label,
                    {
                        "recordings": 0,
                        "name": name,
                        "type": kind,
                        "mapped": label not in unmapped,
                    },
                )
                e["recordings"] += 1
        return dict(sorted(out.items()))

    def to_dict(self) -> dict[str, Any]:
        """Everything but the per-recording rows (the CSV holds those); no clock, no absolute path."""
        configs = Counter(
            tuple(r["names"])
            for r in self.rows
            if r["status"] in (*IN_TREE, "planned") and "names" in r
        )
        labels = self.labels()
        return {
            "dataset_id": self.dataset_id,
            "adapter": self.adapter,
            "ok": self.ok,
            "options": self.options,
            "counts": self.counts,
            "held_out_kept": self.held_out_kept,
            "sfreq": dict(
                sorted(
                    Counter(str(r["sfreq"]) for r in self.rows if "sfreq" in r).items()
                )
            ),
            "unmapped": {
                k: v["recordings"] for k, v in labels.items() if not v["mapped"]
            },
            "labels": labels,
            "configurations": [
                {"n_channels": len(k), "recordings": n, "names": list(k)}
                for k, n in sorted(configs.items(), key=lambda x: (-x[1], x[0]))
            ],
            "excluded": [
                {"source": r["source"], "reason": r["reason"]}
                for r in self.rows
                if r["status"] == "excluded"
            ],
            "problems": self.problems,
            "warnings": self.warnings,
        }

    def write(self, path: str | Path) -> None:
        """The report as JSON at *path*, and one row per recording in a CSV beside it."""
        path = Path(path)
        write_text(path, _dumps(self.to_dict()))
        rows = [{k: r.get(k) for k in CSV_FIELDS} for r in self.rows]
        write_text(path.with_suffix(".csv"), _csv_text(rows, CSV_FIELDS))

    def __str__(self) -> str:
        c = self.counts
        dry = " (dry run, nothing written)" if self.options.get("dry_run") else ""
        shown = self.problems[:20]
        more = len(self.problems) - len(shown)
        unmapped = self.to_dict()["unmapped"]
        return "\n".join(
            [
                *(f"PROBLEM {p}" for p in shown),
                *([f"PROBLEM ... and {more} more"] if more > 0 else []),
                *(
                    [
                        f"UNMAPPED {len(unmapped)} label(s) typed MISC: "
                        f"{', '.join(unmapped)}"
                    ]
                    if unmapped
                    else []
                ),
                f"bids {self.dataset_id} ({self.adapter}): {c['recordings']} recordings, "
                + ", ".join(f"{c[k]} {k}" for k in STATUSES if c[k])
                + f"; {c['subjects']} subjects, {c['hours']:.1f} h; "
                f"{len(self.warnings)} warning(s), {len(self.problems)} problem(s); "
                f"{self.seconds:.0f} s{dry}",
            ]
        )


def _problem(row: dict) -> str | None:
    if row["status"] == "unreadable":
        return f"{row['source']}: unreadable: {row['reason']}"
    if row["status"] in ("invalid", "refused", "failed"):
        return f"{row['bids_path']}: {row['status']}: {row['reason']}"
    return None


# --------------------------------------------------------------- the finish


def _participants_tsv(rows: list[dict]) -> str:
    columns = list(dict.fromkeys(k for r in rows for k in r)) or ["participant_id"]
    lines = ["\t".join(columns)]
    lines += ["\t".join(_cell(r.get(c)) for c in columns) for r in rows]
    return "\n".join(lines) + "\n"


EVENTS_JSON = {
    "onset": {
        "Description": "Onset of the event from the start of the file.",
        "Units": "s",
    },
    "duration": {"Description": "Duration of the event.", "Units": "s"},
    "trial_type": {"Description": "The type of the event."},
}


def strays(out: Path, rows: Sequence[dict]) -> list[str]:
    """Data files in the tree that no row of the plan names (a recording that left it)."""
    known = {r["bids_path"] for r in rows}
    found = (
        p.relative_to(out).as_posix()
        for p in out.glob("sub-*/**/eeg/*_eeg.*")
        if p.suffix in PRIMARY_EXT and not p.name.startswith(".")
    )
    return sorted(p for p in found if p not in known)


def summary(report: BidsReport, plan_warnings: list[str]) -> dict[str, Any]:
    """What :meth:`Adapter.readme` is given: subjects, recordings, hours, warnings."""
    in_tree = [r for r in report.rows if r["status"] in IN_TREE]
    keys = (
        "source",
        "bids_path",
        "subject",
        "n_events",
        "sfreq",
        "duration_s",
        "unmapped",
    )
    return {
        "dataset_id": report.dataset_id,
        "adapter": report.adapter,
        "subjects": sorted({r["subject"] for r in in_tree}),
        "recordings": [{k: r.get(k) for k in keys} for r in in_tree],
        "hours": report.counts["hours"],
        "warnings": [*plan_warnings, *(w for r in report.rows for w in r["warnings"])],
        "excluded": report.to_dict()["excluded"],
    }


def finish(
    adapter: Adapter, out: Path, report: BidsReport, plan_warnings: list[str]
) -> None:
    """The root files and ``.eegdash/to_bids/``, from the rows of the whole run."""
    for folder in (out, out / "sourcedata", out / STATE):
        sweep_tmp(folder, recursive=False)
    in_tree = [r for r in report.rows if r["status"] in IN_TREE]
    subjects = sorted({r["subject"] for r in in_tree})
    rows, sidecar = adapter.participants(subjects)
    by_id = {r["participant_id"]: r for r in rows}
    rows = [by_id.get(f"sub-{s}", {"participant_id": f"sub-{s}"}) for s in subjects]
    write_text(out / "participants.tsv", _participants_tsv(rows))
    if sidecar:
        write_text(out / "participants.json", _dumps(sidecar))
    description = adapter.dataset_description()
    write_text(
        out / "dataset_description.json",
        _dumps(
            {
                "Name": description.get("Name", adapter.dataset_id),
                "BIDSVersion": BIDS_VERSION,
                "DatasetType": "raw",
                **{k: v for k, v in description.items() if k != "Name"},
                "GeneratedBy": [{"Name": "eegdash.to_bids", "Version": report.adapter}],
            }
        ),
    )
    write_text(out / "README.md", adapter.readme(summary(report, plan_warnings)))
    changes = adapter.changes()
    if changes:
        write_text(out / "CHANGES", changes)
    licence = adapter.license_file()
    if licence:
        write_bytes(out / "LICENSE", (adapter.src / licence).read_bytes())
    for name, data in sorted(adapter.sourcedata(subjects).items()):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"sourcedata file outside sourcedata/: {name}")
        write_bytes(out / "sourcedata" / name, data)
    for task in sorted({r["task"] for r in in_tree if r.get("n_events") is not None}):
        write_text(out / f"task-{task}_events.json", _dumps(EVENTS_JSON))
    write_text(out / ".bidsignore", f"{STATE.parts[0]}/\n")
    map_rows = [
        {
            "source": r["source"],
            "bids_path": r["bids_path"],
            "size": r["size"],
            "link": r["link"],
            "status": "written" if r["status"] in IN_TREE else r["status"],
            "reason": r["reason"],
        }
        for r in sorted(report.rows, key=lambda r: r["source"])
    ]
    state = out / STATE
    write_text(state / "map.csv", _csv_text(map_rows, MAP_FIELDS))
    write_text(state / "report.json", _dumps(report.to_dict()))
    done = {
        "adapter": report.adapter,
        **report.counts,
        "problems": len(report.problems),
    }
    write_text(state / "DONE", _dumps(done))


# ------------------------------------------------------------------ the run


def run(
    adapter_cls: type[Adapter],
    dataset_id: str,
    src: str | Path,
    out: str | Path,
    *,
    link: str = "symlink",
    exclude: str | Path | Sequence[Exclusion] | None = None,
    keep_held_out: bool = False,
    dry_run: bool = False,
    n_jobs: int = 1,
    shard: int | None = None,
    n_shards: int | None = None,
) -> BidsReport:
    """Bring the source tree *src* to the BIDS tree *out* with *adapter_cls*.

    See :func:`eegdash.to_bids.to_bids`, which picks the adapter and *out*.
    """
    t0 = time.monotonic()  # the report's duration only, never written to a file
    name, version = split_id(dataset_id)
    if adapter_cls.name != name:
        raise ValueError(f"adapter {adapter_cls.name!r} does not convert {name!r}")
    if version not in adapter_cls.versions:
        raise ValueError(
            f"{dataset_id}: version {version!r} unknown to the {name} adapter "
            f"(knows {', '.join(adapter_cls.versions)})"
        )
    if link not in LINK_MODES:
        raise ValueError(f"link must be one of {LINK_MODES}, got {link!r}")
    if (shard is None) != (n_shards is None):
        raise ValueError("shard and n_shards go together")
    if shard is not None and not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} of {n_shards}")
    src, out = Path(src).resolve(), Path(out)
    check_paths(src, out)
    adapter = adapter_cls(src, version)
    adapter.check_source()
    recs = plan(adapter)
    plan_warnings = list(adapter.warnings)
    adapter.warnings.clear()
    held = adapter.held_out()
    rules = [] if keep_held_out else list(held)
    if isinstance(exclude, (str, Path)):
        rules += read_exclusions(exclude)
    elif exclude:
        rules += list(exclude)
    report = BidsReport(
        dataset_id,
        f"{adapter.name}/{adapter.revision}",
        options={
            "link": link,
            "keep_held_out": keep_held_out,
            "exclude": [asdict(e) for e in rules if e not in held],
            "dry_run": dry_run,
            "shard": shard,
            "n_shards": n_shards,
        },
        warnings=list(plan_warnings),
    )
    if keep_held_out:
        report.held_out_kept = sum(excluded_by(r, held) is not None for r in recs)
    ctx = Context(adapter, src, out, link, dry_run)
    # a subject's shard is decided among all the plan's subjects, excluded ones too
    subjects = sorted({r.subject for r in recs})
    if shard is not None:
        subjects = subjects[shard::n_shards]
    mine = set(subjects)
    todo: dict[str, list[Recording]] = {}
    excluded = []
    for rec in recs:
        if rec.subject not in mine:
            continue
        rule = excluded_by(rec, rules)
        if rule is None:
            todo.setdefault(rec.subject, []).append(rec)
        else:
            excluded.append((rec, rule))
    for rec, rule in excluded:
        row = _base_row(ctx, rec)
        source = src / rec.source
        row.update(status="excluded", reason=rule.reason)
        row["size"] = source.stat().st_size if source.is_file() else None
        report.rows.append(row)
        dest = out / row["bids_path"]
        if not dry_run and (dest.exists() or dest.is_symlink()):
            report.problems.append(
                f"{row['bids_path']}: excluded ({rule.reason}) but in the tree from an "
                "earlier run: remove it"
            )
    scan = out / STATE / f"scan-{'all' if shard is None else shard}.jsonl"
    if not dry_run:
        (out / STATE).mkdir(parents=True, exist_ok=True)
        (out / STATE / "DONE").unlink(missing_ok=True)
    prior = {} if dry_run else prior_rows(out)
    jobs = [
        (
            todo[s],
            {
                p: prior[p]
                for r in todo[s]
                if (p := bids_path(r, adapter.extension)) in prior
            },
        )
        for s in sorted(todo)
    ]
    logger.info(
        "%s: %d recordings of %d subjects to scan, %d excluded",
        dataset_id,
        sum(len(j[0]) for j in jobs),
        len(jobs),
        len(excluded),
    )
    rows: list[dict] = []
    pending: list[dict] = []  # rows not yet in the scan file
    flushed = time.monotonic()
    pool = None
    if n_jobs > 1 and len(jobs) > 1:
        workers = min(n_jobs, len(jobs))
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(ctx,),
        )
        results = pool.map(
            _pool_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))
        )
    else:
        results = (convert_subject(ctx, *job) for job in jobs)
    try:
        for subject_rows in results:  # in subject order, as each finishes
            rows += subject_rows
            pending += subject_rows
            if not dry_run and time.monotonic() - flushed >= FLUSH_SECONDS:
                _append_rows(scan, pending)
                pending, flushed = [], time.monotonic()
    finally:
        if not dry_run and pending:  # what finished before a failure is kept
            _append_rows(scan, pending)
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    report.rows = sorted(report.rows + rows, key=lambda r: r["bids_path"])
    for row in report.rows:
        problem = _problem(row)
        if problem:
            report.problems.append(problem)
        report.warnings += row["warnings"]
        report.warnings += [f"{row['source']}: {w}" for w in row["read_warnings"]]
    if not dry_run:
        compact = sorted(rows, key=lambda r: r["bids_path"])
        write_text(scan, "".join(json.dumps(r, sort_keys=True) + "\n" for r in compact))
        if shard is None:
            report.problems += [
                f"{p}: in the tree but not in the plan: remove it"
                for p in strays(out, report.rows)
            ]
            adapter.warnings.clear()
            finish(adapter, out, report, plan_warnings)
            report.warnings += (
                adapter.warnings
            )  # the root files' own, after report.json
    report.seconds = time.monotonic() - t0
    return report
