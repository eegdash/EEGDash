# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""What an adapter gives the converter: one dataset's layout, channels and metadata.

An adapter walks a source tree and says, for each data file, where it goes in
BIDS (:class:`Recording`), how its channels are named and typed
(:class:`Channel`), which events it holds (:class:`Event`) and what the
dataset-level files say. It never opens a data file: the core reads every
header (:class:`Header`) with the reader the conversion uses and hands it over.

:class:`ChannelTable` is the shared helper for :meth:`Adapter.channels`: an
ordered list of regular expressions of the source label, each giving a BIDS
name and type; a label no row matches is ``MISC`` and counted as unmapped.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Literal, Mapping, Sequence

TABLES = Path(__file__).parent / "tables"  # <name>-<version>.json, .channels.csv, ...
# The channel types channels.tsv may carry, upper case.
CHANNEL_TYPES = ("EEG", "ECG", "EOG", "EMG", "MISC", "TRIG", "RESP")
EXCLUSION_FIELDS = ["key", "match", "reason", "hashed"]


@dataclass(frozen=True)
class Recording:
    """One source data file and where it goes in the BIDS tree."""

    source: Path  # relative to the source root
    subject: str
    task: str
    session: str | None = None
    acquisition: str | None = None
    run: str | None = None
    key: str = ""  # what an exclusion matches; default: the source's stem
    extra: Mapping[str, str] = field(default_factory=dict)  # the adapter's own

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if not self.key:
            object.__setattr__(self, "key", self.source.stem)

    @property
    def entities(self) -> list[tuple[str, str]]:
        """The BIDS entities, in file-name order (``sub ses task acq run``)."""
        pairs = [
            ("sub", self.subject),
            ("ses", self.session),
            ("task", self.task),
            ("acq", self.acquisition),
            ("run", self.run),
        ]
        return [(k, v) for k, v in pairs if v is not None]

    @property
    def stem(self) -> str:
        """``sub-<s>[_ses-<e>]_task-<t>[_acq-<a>][_run-<r>]``."""
        return "_".join(f"{k}-{v}" for k, v in self.entities)

    @property
    def folder(self) -> Path:
        """``sub-<s>/[ses-<e>/]eeg``, relative to the BIDS root."""
        parts = [f"sub-{self.subject}"]
        if self.session is not None:
            parts.append(f"ses-{self.session}")
        return Path(*parts, "eeg")


@dataclass(frozen=True)
class Header:
    """A data file's header, read by the core, never by the adapter."""

    labels: tuple[str, ...]  # the reader's channel list, in its order
    units: tuple[str, ...]
    sfreq: float
    n_times: int
    meas_date: datetime | None


@dataclass(frozen=True)
class Channel:
    """One row of channels.tsv."""

    name: str
    type: str
    units: str = "n/a"
    description: str = ""  # the reader's label at this row's position (checked)
    mapped: bool = True  # False: no rule of the channel table matched the label


@dataclass(frozen=True)
class Event:
    """One row of events.tsv, seconds from the start of the file."""

    onset: float
    duration: float
    trial_type: str


@dataclass(frozen=True)
class Exclusion:
    """A recording, or every recording of a subject, left out of the tree."""

    key: str  # a Recording.key or a subject label; its SHA-256 when hashed
    match: Literal["recording", "subject"]
    reason: str
    hashed: bool = False

    def matches(self, rec: Recording) -> bool:
        value = rec.key if self.match == "recording" else rec.subject
        if self.hashed:
            value = hashlib.sha256(value.encode("utf-8")).hexdigest()
            return value == self.key.lower()
        return value == self.key


def read_exclusions(path: str | Path) -> list[Exclusion]:
    """An exclusion table, ``key,match,reason[,hashed]``: a held-out table or ``--exclude``."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"key", "match", "reason"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: an exclusion table needs {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            match = (row["match"] or "").strip()
            if match not in ("recording", "subject"):
                raise ValueError(
                    f"{path}:{line}: match must be recording or subject, got {match!r}"
                )
            hashed = (row.get("hashed") or "").strip().lower() in ("1", "true", "yes")
            out.append(
                Exclusion(row["key"].strip(), match, row["reason"].strip(), hashed)
            )
    return out


class ChannelTable:
    r"""Source label -> BIDS name and type, by the first matching rule.

    Parameters
    ----------
    rules : str | Path | Iterable[tuple[str, str, str]]
        A CSV with the columns ``pattern,name,type`` (the packaged
        ``tables/<id>.channels.csv``), or the rows themselves. ``pattern`` is
        a regular expression matched against the whole label, ``name`` its
        replacement (``\g<0>`` keeps the label, ``\1`` a group), ``type`` one
        of :data:`CHANNEL_TYPES`.

    """

    def __init__(self, rules: str | Path | Iterable[tuple[str, str, str]]):
        if isinstance(rules, (str, Path)):
            with open(rules, newline="", encoding="utf-8") as f:
                rows = [(r["pattern"], r["name"], r["type"]) for r in csv.DictReader(f)]
        else:
            rows = [tuple(r) for r in rules]
        self.rules = []
        for pattern, name, kind in rows:
            if kind not in CHANNEL_TYPES:
                raise ValueError(f"channel type {kind!r} is not one of {CHANNEL_TYPES}")
            self.rules.append((re.compile(pattern), name, kind))

    def lookup(self, label: str) -> tuple[str, str, bool]:
        """(name, type, mapped) of *label*; ``(label, "MISC", False)`` when no rule matches."""
        for pattern, name, kind in self.rules:
            m = pattern.fullmatch(label)
            if m:
                return m.expand(name), kind, True
        return label, "MISC", False

    def channels(self, header: Header) -> list[Channel]:
        """One :class:`Channel` per label of *header*, in its order."""
        out = []
        for label, unit in zip(header.labels, header.units, strict=True):
            name, kind, mapped = self.lookup(label)
            out.append(Channel(name, kind, unit, label, mapped))
        return out


class Adapter:
    """One dataset's layout and metadata, for :func:`eegdash.to_bids.to_bids`.

    A subclass sets the class attributes and implements :meth:`recordings`,
    :meth:`channels`, :meth:`sidecar`, :meth:`participants`,
    :meth:`dataset_description` and :meth:`readme`. It may append to
    ``self.warnings`` (through :meth:`warn`) in any method: the core collects
    them per recording into the report. An adapter is pickled to the worker
    processes after :meth:`recordings` ran, so what it planned there is
    available to the per-recording methods.

    Parameters
    ----------
    src : Path
        The source tree, never written to.
    version : str
        The source release, one of :attr:`versions`.

    """

    name: ClassVar[str]  # "tueg"
    versions: ClassVar[tuple[str, ...]]  # ("2.0.1",)
    revision: ClassVar[int] = 1  # bumped when the output changes
    extension: ClassVar[str] = ".edf"  # of the BIDS data file
    linkable: ClassVar[bool] = True  # the source file is a valid BIDS data file

    def __init__(self, src: Path, version: str) -> None:
        self.src = Path(src)
        self.version = version
        self.warnings: list[str] = []

    @property
    def dataset_id(self) -> str:
        """``<name>-<version>``: the tree's folder in the cache."""
        return f"{self.name}-{self.version}"

    def table(self, suffix: str) -> Path:
        """The packaged table ``tables/<name>-<version><suffix>``."""
        return TABLES / f"{self.dataset_id}{suffix}"

    def warn(self, message: str) -> None:
        """Record a warning for the report (and the README, if the adapter writes them there)."""
        self.warnings.append(message)

    def check_source(self) -> None:
        """Raise ``ValueError`` when the source is not this dataset."""

    def recordings(self) -> Iterator[Recording]:
        """Every data file of the source: a walk that opens no data file."""
        raise NotImplementedError

    def channels(self, rec: Recording, header: Header) -> list[Channel]:
        """One channel per label of *header*, in its order."""
        raise NotImplementedError

    def sidecar(self, rec: Recording, header: Header) -> dict[str, Any]:
        """The adapter's keys of ``_eeg.json``: ``EEGReference``, ``PowerLineFrequency``, ..."""
        raise NotImplementedError

    def events(self, rec: Recording, header: Header) -> list[Event] | None:
        """The recording's events; None writes no events.tsv."""
        return None

    def held_out(self) -> list[Exclusion]:
        """Recordings and subjects left out: ``tables/<name>-<version>.heldout.csv`` when there is one."""
        path = self.table(".heldout.csv")
        return read_exclusions(path) if path.is_file() else []

    def participants(self, subjects: Sequence[str]) -> tuple[list[dict], dict]:
        """participants.tsv rows (``participant_id`` first) for *subjects*, and participants.json."""
        return [{"participant_id": f"sub-{s}"} for s in subjects], {}

    def dataset_description(self) -> dict[str, Any]:
        """``Name``, ``License``, ``Authors``, DOI and links of dataset_description.json."""
        raise NotImplementedError

    def readme(self, summary: dict[str, Any]) -> str:
        """README.md, from the run's summary (subjects, recordings, hours, warnings)."""
        raise NotImplementedError

    def changes(self) -> str | None:
        """CHANGES, or None for no file."""
        return None

    def license_file(self) -> Path | None:
        """A file of the source (relative to it) copied to ``LICENSE``, or None."""
        return None

    def sourcedata(self, subjects: Sequence[str]) -> dict[str, bytes]:
        """Small text files for ``sourcedata/``: name -> content, about *subjects* only.

        *subjects* are those with a recording in the tree: a held-out subject's
        files, or its lines in a shared table, are left out.
        """
        return {}

    def write_data(self, rec: Recording, dest: Path) -> None:
        """Write the BIDS data file of *rec* to *dest* (``linkable = False`` only)."""
        raise NotImplementedError
