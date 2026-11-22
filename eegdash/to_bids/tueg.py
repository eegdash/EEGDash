# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""The TUH EEG Corpus (TUEG) 2.0.1 (NEDC, EDF) as BIDS.

A user obtains the corpus from the Neural Engineering Data Consortium (NEDC)
under NEDC's terms, which forbid redistribution, and makes the tree from that
copy. The source is the release root (``AAREADME.txt``, ``DOCS/``, ``edf/``);
every EDF under ``edf/`` is laid out as
``<NNN>/<subject>/s<SSS>_<YYYY>/<montage>/<subject>_s<SSS>_t<TTT>.edf`` and
enters the tree as it is (linked or copied):

- ``sub-<subject>``;
- ``ses-<SSS>y<YYYY>``: the session number alone is not unique, as one subject
  has ``s012_2014`` and ``s012_2016`` with the same file names;
- ``task-clinical``;
- ``acq-ar``, ``acq-le``, ``acq-ara``, ``acq-lea`` from the montage folders
  ``01_tcp_ar``, ``02_tcp_le``, ``03_tcp_ar_a``, ``04_tcp_le_a``;
- ``run-<TTT>`` from ``t<TTT>``.

A recording's key (what the held-out table matches) is the file stem. Channel
names drop a leading ``EEG `` and a trailing ``-REF`` / ``-LE``
(``tables/tueg-2.0.1.channels.csv``); power line 60 Hz; ``EEGReference`` is
``n/a``, as the release defines neither ``ar`` nor ``le``. TUAB recordings and
the recordings of TUEV training subjects are left out
(``tables/tueg-2.0.1.heldout.csv``, SHA-256 of each key). participants.tsv
holds ``participant_id`` only; there are no events.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .adapter import TABLES, Adapter, Channel, ChannelTable, Header, Recording

TABLE_ID = "tueg-2.0.1"
SESSION_RE = re.compile(r"s(\d+)_(\d{4})")
SUBJECT_RE = re.compile(r"[A-Za-z0-9]+")
GROUP_RE = re.compile(r"\d+")
MAX_README_WARNINGS = 100  # the README lists these; the report holds them all


@functools.lru_cache(maxsize=1)
def _meta() -> dict[str, Any]:
    """The dataset's metadata table, ``tables/tueg-2.0.1.json``."""
    return json.loads((TABLES / f"{TABLE_ID}.json").read_text("utf-8"))


@functools.lru_cache(maxsize=1)
def _channel_table() -> ChannelTable:
    """The channel rules, ``tables/tueg-2.0.1.channels.csv``, read once."""
    return ChannelTable(TABLES / f"{TABLE_ID}.channels.csv")


def parse_path(rel: Path, montages: dict[str, str]) -> dict[str, str] | str:
    """The entities of an EDF path relative to ``edf/``, or why it is not in the layout.

    ``000/aaaaaaaa/s012_2014/01_tcp_ar/aaaaaaaa_s012_t000.edf`` ->
    ``{subject: aaaaaaaa, session: 012y2014, acquisition: ar, run: 000}``.
    """
    parts = rel.parts
    if len(parts) != 5:
        return "not at edf/<NNN>/<subject>/s<NNN>_<YYYY>/<montage>/"
    group, subject, session, montage, name = parts
    if not GROUP_RE.fullmatch(group):
        return f"group folder {group!r} is not a number"
    if not SUBJECT_RE.fullmatch(subject):
        return f"subject folder {subject!r} is not alphanumeric"
    m = SESSION_RE.fullmatch(session)
    if not m:
        return f"session folder {session!r} is not s<NNN>_<YYYY>"
    if montage not in montages:
        return f"montage folder {montage!r} is not one of {', '.join(montages)}"
    number, year = m.groups()
    f = re.fullmatch(rf"{re.escape(subject)}_s{number}_t(\d+)", Path(name).stem)
    if not f:
        return f"file name {name!r} is not {subject}_s{number}_t<NNN>.edf"
    return {
        "subject": subject,
        "session": f"{number}y{year}",
        "acquisition": montages[montage],
        "run": f[1],
    }


def walk_edf(root: Path) -> list[Path]:
    """Every ``.edf`` file under *root* (any case), relative to it, sorted."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath).relative_to(root)
        found += [base / f for f in filenames if f.lower().endswith(".edf")]
    return sorted(found)


class TUEG(Adapter):
    """The TUH EEG Corpus (TUEG) 2.0.1 (NEDC): see the module docstring."""

    name = "tueg"
    versions = ("2.0.1",)
    revision = 1
    extension = ".edf"
    linkable = True

    def __init__(self, src: Path, version: str) -> None:
        super().__init__(src, version)
        self.meta = _meta()
        self.channel_table = _channel_table()

    def check_source(self) -> None:
        missing = [
            what
            for what, ok in (
                ("edf/", (self.src / "edf").is_dir()),
                ("AAREADME.txt", (self.src / "AAREADME.txt").is_file()),
            )
            if not ok
        ]
        if missing:
            raise ValueError(
                f"{self.src} has no {' or '.join(missing)}: not the TUH EEG release"
            )

    def recordings(self) -> Iterator[Recording]:
        montages = self.meta["montages"]
        for rel in walk_edf(self.src / "edf"):
            parsed = parse_path(rel, montages)
            if isinstance(parsed, str):
                self.warn(f"edf/{rel.as_posix()}: {parsed}; left out")
                continue
            yield Recording(Path("edf", rel), task=self.meta["task"], **parsed)

    def channels(self, rec: Recording, header: Header) -> list[Channel]:
        return self.channel_table.channels(header)

    def sidecar(self, rec: Recording, header: Header) -> dict[str, Any]:
        return {
            "PowerLineFrequency": self.meta["power_line_frequency"],
            "EEGReference": self.meta["eeg_reference"],
            **self.meta["sidecar"],
        }

    def dataset_description(self) -> dict[str, Any]:
        return dict(self.meta["dataset_description"])

    def readme(self, summary: dict[str, Any]) -> str:
        meta = self.meta
        recs = summary["recordings"]
        held = {e.reason for e in self.held_out()}
        reasons = Counter(e["reason"] for e in summary["excluded"])
        unmapped = Counter(label for r in recs for label in r["unmapped"] or [])
        acqs = Counter(
            re.search(r"_acq-([A-Za-z0-9]+)_", r["bids_path"])[1] for r in recs
        )
        rows = [
            f"| `{folder}/` | `acq-{acq}` |" for folder, acq in meta["montages"].items()
        ]
        lines = [
            f"# TUH EEG Corpus (TUEG) {self.version}",
            "",
            "Clinical scalp electroencephalography (EEG) recordings from the archive of "
            "Temple University Hospital, distributed by the Neural Engineering Data "
            "Consortium (NEDC) of Temple University. NEDC's terms forbid redistribution: "
            f"every user obtains the corpus from NEDC ({meta['url']}) and makes this "
            "tree from their own copy. Please cite the paper that describes the corpus: "
            f"{meta['paper']}.",
            "",
            "## This BIDS version",
            "",
            f"Written by EEGDash's converter (`eegdash.to_bids`, adapter `{self.name}`, "
            f"revision {self.revision}) from the release's `edf/` folder: {len(recs)} "
            f"recordings of {len(summary['subjects'])} subjects, "
            f"{summary['hours']:.1f} hours. Every EDF file enters the tree unchanged "
            "(linked or copied): no channel was resampled or filtered.",
            "",
            "| release | BIDS |",
            "|---|---|",
            "| `edf/<NNN>/<subject>/` | `sub-<subject>` |",
            "| `s<NNN>_<YYYY>/` | `ses-<NNN>y<YYYY>`: the year is part of the label, "
            "as one subject has two session folders with the same number in different "
            "years and the same file names |",
            *rows,
            "| `<subject>_s<NNN>_t<NNN>.edf` | `run-<NNN>` from `t<NNN>` |",
            "",
            f"The task is `{meta['task']}` throughout. Recordings per montage: "
            + (", ".join(f"`acq-{k}` {v}" for k, v in sorted(acqs.items())) or "none")
            + ".",
            "",
            "Channel names are the EDF labels without a leading `EEG ` and a trailing "
            "`-REF` or `-LE`, case kept. `EKG` channels are ECG, `LOC` and `ROC` EOG, "
            "`EMG` EMG, `PHOTIC` TRIG; `IBI`, `BURSTS`, `SUPPR` and the numbered spare "
            "channels are MISC; 10-20 and 10-10 electrode names (with `A1`, `A2`, "
            "`T1`, `T2`, `SP1`, `SP2`) are EEG. A label no rule matches keeps its EDF "
            "name and is typed MISC: "
            + (
                ", ".join(
                    f"`{k}` ({v} recordings)" for k, v in sorted(unmapped.items())
                )
                if unmapped
                else "none in this tree"
            )
            + ".",
            "",
            f"`PowerLineFrequency` is {meta['power_line_frequency']} Hz. `EEGReference` "
            "is `n/a`: the montage folders name the reference set-up (`ar`, `le` and "
            "their `_a` variants), but the release does not define them (its "
            "`AAREADME.txt` does not), so no reference is stated. participants.tsv "
            "lists the subjects only, and there are no events.",
            "",
            "## Recordings left out",
            "",
            "The TUH Abnormal EEG Corpus (TUAB) and the TUH EEG Events Corpus (TUEV) "
            "are benchmark tasks drawn from this corpus. EEGDash's held-out table for "
            "this release (`tueg-2.0.1.heldout.csv`, the SHA-256 of each key) leaves "
            "out every TUAB recording, matched by file name, and every recording of a "
            "subject of TUEV's training set. TUEV's evaluation files carry no subject "
            "code and cannot be matched to this corpus, so they are not left out: a "
            "model pretrained on this tree may have seen them, and no TUEV score "
            "should be reported from it.",
            "",
        ]
        if reasons:
            lines += [
                "| reason | recordings |",
                "|---|---:|",
                *(
                    f"| {k}{'' if k in held else ' (asked by the user)'} | {v} |"
                    for k, v in sorted(reasons.items())
                ),
            ]
        else:
            lines.append(
                "None in this run: the held-out table was not applied, or none of its "
                "recordings is in the source."
            )
        warnings = summary["warnings"]
        lines += ["", "## Warnings", ""]
        if warnings:
            lines += [f"- {w}" for w in warnings[:MAX_README_WARNINGS]]
            more = len(warnings) - MAX_README_WARNINGS
            if more > 0:
                lines.append(f"- ... and {more} more (the converter's report)")
        else:
            lines.append("None.")
        return "\n".join(lines) + "\n"
