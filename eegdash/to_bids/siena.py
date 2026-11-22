# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""The Siena Scalp EEG Database 1.0.0 (PhysioNet, CC BY 4.0, EDF) as BIDS.

One run per EDF in file order (``run-01``...; ``PN01.edf`` and ``PN12-1.2.edf``
are run-01), task ``monitoring``, no sessions. Each EDF enters the tree as it
is (linked or copied); channels keep their header names (``EEG Fp1``, which
the position bank reads through its prefix strip), ``EKG*`` is ECG, ``EEG xx``
labels and bare 10-20 names are EEG, the rest MISC
(``tables/siena-1.0.0.channels.csv``); power line 50 Hz; no electrode
positions (the release has none measured). Seizures from
``Seizures-list-PNxx.txt`` become events.tsv (trial_type ``seizure``; onset =
seizure start - registration start, modulo 24 h, checked against the EDF
header's clock). participants.tsv from ``subject_info.csv``; the dataset's
metadata, sidecar constants and participants.json are in
``tables/siena-1.0.0.json``; the source text files are kept under
``sourcedata/``.

The parsers are pure functions (text in, dicts out); nothing here imports MNE.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from .adapter import Adapter, Channel, ChannelTable, Event, Header, Recording

DAY_S = 86400
CLOCK_TOLERANCE_S = 2.0  # EDF header start vs the seizure list's registration start
SEX = {"male": "M", "female": "F", "m": "M", "f": "F"}
CLOCK = re.compile(r"(\d{1,2})\s*[.:]\s*(\d{1,2})\s*[.:]\s*(\d{1,2})")
# (key, pattern); "registration" lines are tested before plain start/end lines
SEIZURE_LINES = [
    ("seizure", re.compile(r"^\s*seizure\s+n\.?\s*(\d+)", re.I)),
    ("file", re.compile(r"^\s*file\s*name\s*:\s*(\S+)", re.I)),
    ("reg_start", re.compile(r"^\s*registration\s+start\s+time\s*:\s*(.*)$", re.I)),
    ("reg_end", re.compile(r"^\s*registration\s+end\s+time\s*:\s*(.*)$", re.I)),
    ("start", re.compile(r"^\s*(?:seizure\s+)?start\s+time\s*:\s*(.*)$", re.I)),
    ("end", re.compile(r"^\s*(?:seizure\s+)?end\s+time\s*:\s*(.*)$", re.I)),
    ("channel", re.compile(r"^\s*channel\s+(\d+)\s*:\s*(.*?)\s*$", re.I)),
    ("sfreq", re.compile(r"sampling\s+rate\s*:\s*([\d.]+)", re.I)),
]
TABLE_ID = "siena-1.0.0"


def _meta() -> dict[str, Any]:
    """The dataset's metadata table, ``tables/siena-1.0.0.json``."""
    return json.loads(
        (Path(__file__).parent / "tables" / f"{TABLE_ID}.json").read_text("utf-8")
    )


@functools.lru_cache(maxsize=1)
def _channel_table() -> ChannelTable:
    """The channel rules, ``tables/siena-1.0.0.channels.csv``, read once."""
    return ChannelTable(Path(__file__).parent / "tables" / f"{TABLE_ID}.channels.csv")


PARTICIPANT_COLUMNS = list(_meta()["participants_json"])


# ---------------------------------------------------------------- the parsers


def parse_clock(text: str) -> int:
    """Seconds since midnight of ``hh.mm.ss`` or ``hh:mm:ss`` (the lists mix both)."""
    m = CLOCK.search(text)
    if not m:
        raise ValueError(f"not a clock time: {text!r}")
    h, mi, s = map(int, m.groups())
    if h > 23 or mi > 59 or s > 59:
        raise ValueError(f"not a clock time: {text!r}")
    return h * 3600 + mi * 60 + s


def clock_diff(later: int, earlier: int) -> int:
    """Seconds from *earlier* to *later* on a 24 h clock: a recording may cross midnight."""
    return (later - earlier) % DAY_S


def seizure_row(n: int, cur: dict, start: int, end: int) -> dict:
    """One seizure: its file, the clock times, onset and duration in seconds from the file's start."""
    row = {
        "n": n,
        "file": cur["file"],
        "registration_start": cur["reg_start"],
        "registration_end": cur["reg_end"],
        "start": start,
        "end": end,
        "onset": float(clock_diff(start, cur["reg_start"])),
        "duration": float(clock_diff(end, start)),
    }
    row["registration_length"] = (
        float(clock_diff(cur["reg_end"], cur["reg_start"]))
        if cur["reg_end"] is not None
        else None
    )
    return row


def parse_seizure_list(text: str) -> dict:
    """A ``Seizures-list-PNxx.txt`` -> ``{sfreq, channels, seizures, warnings}``.

    Both layouts: A repeats ``File name`` and the registration times in every
    seizure block; B gives them once, then several ``Seizure n k`` blocks with
    ``Start time`` / ``End time``. A seizure is emitted when its end time is
    read, with the file and registration start in force at that moment.
    """
    out: dict = {"sfreq": None, "channels": {}, "seizures": [], "warnings": []}
    cur = {"file": None, "reg_start": None, "reg_end": None}
    pending: dict | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        for key, pat in SEIZURE_LINES:
            m = pat.search(line) if key == "sfreq" else pat.match(line)
            if not m:
                continue
            if key == "seizure":
                if pending and pending.get("start") is not None:
                    out["warnings"].append(
                        f"line {lineno}: seizure {pending['n']} has a start time "
                        "and no end time"
                    )
                pending = {"n": int(m.group(1)), "start": None}
            elif key == "file":
                cur["file"] = m.group(1).strip()
            elif key == "channel":
                out["channels"][int(m.group(1))] = m.group(2)
            elif key == "sfreq":
                out["sfreq"] = float(m.group(1))
            else:
                try:
                    t = parse_clock(m.group(1))
                except ValueError as e:
                    out["warnings"].append(f"line {lineno}: {e}")
                    break
                extra = CLOCK.sub("", m.group(1), count=1).strip(" \t;,")
                if (
                    extra
                ):  # "11.41.04 oppure 11.40.43", "15.43.53 (CLINICAL ONSET); ..."
                    out["warnings"].append(
                        f"line {lineno}: first time kept, the rest of the line "
                        f"ignored: {extra!r}"
                    )
                if key in ("reg_start", "reg_end"):
                    cur[key] = t
                elif key == "start":
                    if pending is None:
                        pending = {"n": len(out["seizures"]) + 1, "start": None}
                    pending["start"] = t
                else:
                    if pending is None or pending.get("start") is None:
                        out["warnings"].append(
                            f"line {lineno}: an end time without a start time"
                        )
                        break
                    if cur["file"] is None or cur["reg_start"] is None:
                        out["warnings"].append(
                            f"line {lineno}: seizure {pending['n']} before any file "
                            "name or registration start; skipped"
                        )
                    else:
                        out["seizures"].append(
                            seizure_row(pending["n"], cur, pending["start"], t)
                        )
                    pending = None
            break
    if pending and pending.get("start") is not None:
        out["warnings"].append(
            f"seizure {pending['n']} has a start time and no end time"
        )
    return out


def run_order_key(fname: str) -> tuple:
    """Sort key of an EDF within its patient: ``PN01.edf`` first, then ``PN00-1`` < ``PN12-1.2`` < ``PN12-3``.

    ``1.2`` is one file, the first of PN12, so the numbers after the dash
    compare as a tuple.
    """
    stem = Path(fname).stem
    suffix = stem.split("-", 1)[1] if "-" in stem else ""
    return tuple(int(x) for x in re.findall(r"\d+", suffix)) or (0,), fname


def assign_runs(fnames: list[str]) -> list[tuple[str, str]]:
    """``[(run label "01", file name)]`` in file order."""
    return [(f"{i:02d}", f) for i, f in enumerate(sorted(fnames, key=run_order_key), 1)]


def parse_records(text: str) -> dict[str, list[str]]:
    """PhysioNet's RECORDS (one ``PNxx/<file>.edf`` per line) -> ``{patient: [file names]}``."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.lower().endswith(".edf") and "/" in line:
            patient, name = line.rsplit("/", 1)
            out.setdefault(patient.split("/")[-1], []).append(name)
    return out


def _number(v: str) -> int | float | str:
    """An integer, a float, or ``n/a`` for an empty cell."""
    v = v.strip()
    if not v:
        return "n/a"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_subject_info(text: str) -> list[dict]:
    """``subject_info.csv`` (spaces after the commas) -> participants rows in :data:`PARTICIPANT_COLUMNS`."""
    rows = []
    for r in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("patient_id"):
            continue
        rows.append(
            {
                "participant_id": f"sub-{r['patient_id']}",
                "age": _number(r.get("age_years", "")),
                "sex": SEX.get(r.get("gender", "").lower(), "n/a"),
                "seizure_type": r.get("seizure") or "n/a",
                "localization": r.get("localization") or "n/a",
                "lateralization": r.get("lateralization") or "n/a",
                "n_seizures": _number(r.get("number_seizures", "")),
                "recording_minutes": _number(r.get("rec_time_minutes", "")),
                "eeg_channels": _number(r.get("eeg_channel", "")),
            }
        )
    return rows


def channel_type(name: str) -> str:
    """``ecg`` for the EKG/ECG channels, ``eeg`` for ``EEG xx`` labels and bare 10-20 names, ``misc`` for the rest.

    The Siena headers also carry SPO2, HR, PLET, MK and numbered or lettered
    spare channels (dry run of 2026-10-12): MISC.
    """
    return _channel_table().lookup(name)[1].lower()


def norm_name(name: str) -> str:
    """A file name as the seizure lists misspell it: case, ``.edf``, the letter O for 0 (``PNO6-1``)."""
    stem = name.strip().lower().removesuffix(".edf")
    return re.sub(r"^pn[o0-9]+", lambda m: m.group(0).replace("o", "0"), stem)


def match_file(name: str, files: list[str]) -> tuple[str | None, str | None]:
    """(the EDF a seizure list names, a note when the match needed a repair).

    Exact after :func:`norm_name`; a name that matches nothing goes to the
    patient's only file when there is one (``PN01.edf`` for ``PN01-1.edf``,
    ``PN11-.edf``).
    """
    want = norm_name(name)
    for f in files:
        if norm_name(f) == want:
            same = f.lower().removesuffix(".edf") == name.strip().lower().removesuffix(
                ".edf"
            )
            return f, (None if same else f"{name} read as {f}")
    if len(files) == 1:
        return files[0], f"{name} taken as {files[0]}, the patient's only file"
    return None, None


def fmt_clock(t: int) -> str:
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def place_seizures(
    seizures: list[dict], meas: Any, length: float, tag: str, warn: list[str]
) -> list[dict]:
    """Seizure onsets within the file.

    The lists' registration start is typed by hand and is sometimes wrong
    (PN10's ``1 6.49.25``, PN14-3 three hours off: dry run of 2026-10-12); the
    EDF header's clock is the machine's. When the two disagree by more than the
    tolerance, the onset is taken from the header's clock if that puts the
    seizure inside the file. A seizure that still ends after the file is
    clipped (PN00-3's end time is an hour past the file's end), one that starts
    after it is dropped; each case is a warning.
    """
    kept = []
    edf_clock = (
        None if meas is None else meas.hour * 3600 + meas.minute * 60 + meas.second
    )
    said = False
    for s in seizures:
        s = dict(s)
        if edf_clock is not None:
            off = min(
                clock_diff(edf_clock, s["registration_start"]),
                clock_diff(s["registration_start"], edf_clock),
            )
            if off > CLOCK_TOLERANCE_S:
                from_edf = float(clock_diff(s["start"], edf_clock))
                use_edf = (
                    from_edf + s["duration"] <= length
                    or s["onset"] + s["duration"] > length
                )
                if not said:
                    warn.append(
                        f"{tag}: EDF header starts at {meas:%H:%M:%S}, the seizure "
                        f"list's registration start is "
                        f"{fmt_clock(s['registration_start'])} ({off} s apart); onsets "
                        f"{'from the header clock' if use_edf else 'from the list'}"
                    )
                    said = True
                if use_edf:
                    s["onset"] = from_edf
        if s["onset"] >= length:
            warn.append(
                f"{tag}: seizure {s['n']} starts at {s['onset']:.0f} s, after the end "
                f"of the file ({length:.0f} s); dropped"
            )
            continue
        if s["onset"] + s["duration"] > length:
            warn.append(
                f"{tag}: seizure {s['n']} ends after the file (list: "
                f"{fmt_clock(s['start'])} to {fmt_clock(s['end'])}); duration clipped "
                f"from {s['duration']:.0f} s to {length - s['onset']:.0f} s"
            )
            s["duration"] = float(length - s["onset"])
        kept.append(s)
    return kept


def plan_siena(src: Path) -> dict:
    """Everything decided before an EDF is opened: participants, runs, seizures per run, warnings."""
    warn: list[str] = []
    participants = parse_subject_info(
        (src / "subject_info.csv").read_text(encoding="utf-8", errors="replace")
    )
    records = (
        parse_records((src / "RECORDS").read_text())
        if (src / "RECORDS").exists()
        else {}
    )
    if not records:
        warn.append(
            "RECORDS missing or empty: EDF files found by listing the patient folders"
        )
    runs: list[dict] = []
    for patient in [p["participant_id"].removeprefix("sub-") for p in participants]:
        folder = src / patient
        on_disk = (
            sorted(f.name for f in folder.glob("*.edf")) if folder.is_dir() else []
        )
        listed = records.get(patient, on_disk)
        if records and sorted(listed) != on_disk:
            warn.append(
                f"{patient}: RECORDS lists {sorted(listed)}, the folder holds {on_disk}"
            )
        files = [f for f in listed if (folder / f).exists()]
        for f in sorted(set(listed) - set(files)):
            warn.append(f"{patient}/{f}: listed but missing; skipped")
        if not files:
            warn.append(f"{patient}: no EDF file; patient skipped")
            continue
        lists = sorted(folder.glob("Seizures-list-*.txt"))
        parsed = (
            parse_seizure_list(lists[0].read_text(encoding="utf-8", errors="replace"))
            if lists
            else {
                "seizures": [],
                "warnings": ["no Seizures-list file"],
                "sfreq": None,
                "channels": {},
            }
        )
        warn += [f"{patient} seizure list: {w}" for w in parsed["warnings"]]
        by_file: dict[str, list[dict]] = {f: [] for f in files}
        for s in parsed["seizures"]:
            f, note = match_file(s["file"], files)
            if f is None:
                warn.append(
                    f"{patient}: seizure {s['n']} names {s['file']}, not among "
                    f"{files}; skipped"
                )
                continue
            if note:
                warn.append(f"{patient}: seizure {s['n']}: {note}")
            if (
                s["registration_length"] is not None
                and s["onset"] + s["duration"] > s["registration_length"]
            ):
                warn.append(
                    f"{patient}/{f}: seizure {s['n']} ends after the registration "
                    "end time"
                )
            by_file[f].append(s)
        n_listed = sum(len(v) for v in by_file.values())
        expected = next(
            p["n_seizures"]
            for p in participants
            if p["participant_id"] == f"sub-{patient}"
        )
        if isinstance(expected, int) and expected != n_listed:
            warn.append(
                f"{patient}: subject_info.csv says {expected} seizures, the seizure "
                f"list gives {n_listed}"
            )
        eeg_expected = next(
            p["eeg_channels"]
            for p in participants
            if p["participant_id"] == f"sub-{patient}"
        )
        for run, f in assign_runs(files):
            if not by_file[f]:
                warn.append(
                    f"{patient}/{f}: no seizure in the list (run {run} has no "
                    "events.tsv)"
                )
            runs.append(
                {
                    "subject": patient,
                    "run": run,
                    "file": f,
                    "seizures": by_file[f],
                    "list_sfreq": parsed["sfreq"],
                    "eeg_expected": eeg_expected,
                }
            )
    kept = {r["subject"] for r in runs}
    participants = [
        p for p in participants if p["participant_id"].removeprefix("sub-") in kept
    ]
    return {"participants": participants, "runs": runs, "warnings": warn}


# ----------------------------------------------------------------- the README


def readme_text(
    meta: dict,
    participants: list[dict],
    runs: list[dict],
    notes: dict[str, list[str]],
    revision: int,
    issues: list[str] | None = None,
    hours: float | None = None,
) -> str:
    """The dataset-first README: the data, then how this BIDS copy was made, the runs and the repairs.

    *runs*: ``{subject, file, bids, seizures}``; *notes*: ``{file: [channel
    notes]}``; *issues*: the warnings about the seizure lists.
    """
    types_desc = meta["seizure_types"]
    sexes = Counter(p["sex"] for p in participants)
    ages = [p["age"] for p in participants if isinstance(p["age"], (int, float))]
    minutes = sum(
        p["recording_minutes"]
        for p in participants
        if isinstance(p["recording_minutes"], (int, float))
    )
    types = Counter(p["seizure_type"] for p in participants)
    n_seiz = sum(r["seizures"] for r in runs)
    age_text = f", aged {min(ages)} to {max(ages)} years" if ages else ""
    type_text = "; ".join(
        f"{k} ({types_desc.get(k, k)}): {v} patient{'s' if v != 1 else ''}"
        for k, v in sorted(types.items())
    )
    hours_text = (
        f"{hours:.0f} hours of EEG (the recording times of `subject_info.csv` sum to "
        f"{minutes / 60:.0f} hours)"
        if hours
        else f"about {minutes / 60:.0f} hours of EEG"
    )
    task = meta["task"]
    lines = [
        "# Siena Scalp EEG Database",
        "",
        f"The Siena Scalp EEG Database holds long-term scalp electroencephalography (EEG) "
        f"recordings of {len(participants)} patients with epilepsy, acquired at the Unit "
        f"of Neurology and Neurophysiology of the University of Siena, Italy: "
        f"{sexes.get('M', 0)} men and {sexes.get('F', 0)} women{age_text}. The patients "
        "were monitored with video-EEG at 512 Hz, with reusable silver/gold cup "
        "electrodes placed according to the international 10-20 system and EB Neuro "
        "and Natus Quantum LTM amplifiers, and were asked to stay in bed as much as "
        "possible, asleep or awake. Every recording also carries an electrocardiogram "
        "(EKG) channel, and most carry pulse oximetry, heart rate and plethysmography "
        "traces. The diagnosis of epilepsy and the classification of every seizure "
        "according to the criteria of the International League Against Epilepsy "
        "(ILAE) were made by an expert clinician after review of each patient's "
        "clinical and electrophysiological data.",
        "",
        "Antiepileptic drugs fail to control seizures in roughly 30 % of patients and "
        "surgery is not always an option, so predicting a seizure early enough to warn "
        "the patient or trigger a treatment is a goal in itself. The data were "
        "collected during PANACEE, a regional research project of the University of "
        "Siena for noninvasive, patient-specific, low-cost devices that predict "
        "seizures, and were used to validate seizure-prediction methods based on EEG "
        "synchronization and on heart-rate variability.",
        "",
        f"The dataset holds {len(runs)} recordings and {n_seiz} annotated seizures, "
        f"{hours_text}, recorded on the same or on different days; the dates in the "
        f"EDF headers are de-identified. Seizure types: {type_text}. For every patient "
        "the release gives age, sex, seizure type, localization and lateralization of "
        "the epileptogenic zone, and the number of EEG channels, seizures and "
        "recording minutes (participants.tsv); for every seizure, its start and end "
        "time within the recording (events.tsv).",
        "",
        f"Published on PhysioNet as version 1.0.0 ({meta['url']}, {meta['doi']}) under "
        f"CC BY 4.0 (see LICENSE); PhysioNet credits {meta['contributor']} as the "
        "contributor, and the data paper's three authors are listed as the dataset's. "
        "Please cite:",
        "",
        f"- {meta['paper']}",
        f"- {meta['dataset_citation']}",
        f"- {meta['physionet_paper']}",
        "",
        "## This BIDS version",
        "",
        f"The PhysioNet release in the Brain Imaging Data Structure (BIDS), written by "
        f"EEGDash's converter (`eegdash.to_bids`, adapter `siena`, revision {revision}) "
        f"so that BIDS tools, EEGDash and NEMAR can read it. Every original EDF file is "
        f"one run of the task `{task}` (long-term video-EEG monitoring), byte for byte "
        "the original: no channel was renamed, resampled or filtered, and there are no "
        "sessions. Runs are numbered in the order of the original files (table below).",
        "",
        "Channel types follow the EDF headers: `EEG`-prefixed channels are EEG, `EKG` "
        "is ECG, the others (SPO2, HR, PLET, the marker MK, spare channels named by "
        "numbers or letters) are MISC. The power line frequency is 50 Hz. No electrode "
        "positions are given: the release measured none, and template positions are "
        "not written.",
        "",
        "Seizures come from the release's `Seizures-list-PNxx.txt` files: onset is the "
        "seizure start minus the registration start of the file (modulo 24 hours, as "
        "recordings cross midnight), duration is end minus start, `trial_type` is "
        "`seizure`. The lists are typed by hand, so the conversion checks them against "
        "the EDF headers and `subject_info.csv`: a registration start that disagrees "
        "with the header's clock is replaced by the clock when that places the seizure "
        "inside the file, a seizure ending after its file is clipped, a misspelt file "
        "name is matched to the patient's files; every repair is listed under "
        '"Source inconsistencies" below. participants.tsv comes from '
        "`subject_info.csv`, the original text files are kept under `sourcedata/`, and "
        "the amplifier, electrode and institution details from the PhysioNet page are "
        "in each run's sidecar.",
        "",
        "### Original files and BIDS runs",
        "",
        "| original file | BIDS recording | seizures |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['subject']}/{r['file']} | {r['bids']} | {r['seizures']} |" for r in runs
    ]
    lines += ["", "### Channel notes", ""]
    flat = sorted({n for ns in notes.values() for n in ns})
    if flat:
        lines.append("Channels kept under the name in the EDF header, with their type:")
        lines.append("")
        for n in flat:
            files = sorted(f for f, ns in notes.items() if n in ns)
            where = (
                f"all {len(notes)} files"
                if len(files) == len(notes)
                else f"{len(files)} files: {', '.join(files)}"
            )
            lines.append(f"- {n} ({where})")
    else:
        lines.append("Every channel is an EEG or ECG channel.")
    lines += ["", "### Source inconsistencies", ""]
    if issues:
        lines.append(
            "The release's seizure lists are typed by hand. The conversion checks them "
            "against the EDF headers and `subject_info.csv`; what it found and how each "
            "case was handled:"
        )
        lines.append("")
        lines += [f"- {i}" for i in issues]
    else:
        lines.append(
            "None found: every seizure list agrees with the EDF headers and "
            "`subject_info.csv`."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the adapter


class Siena(Adapter):
    """The Siena Scalp EEG Database 1.0.0 (PhysioNet): see the module docstring."""

    name = "siena"
    versions = ("1.0.0",)
    revision = 1
    extension = ".edf"
    linkable = True

    def __init__(self, src: Path, version: str) -> None:
        super().__init__(src, version)
        self.meta = _meta()
        self.channel_table = _channel_table()
        self.runs: dict[str, dict] = {}  # Recording.source (posix) -> its plan
        self.subjects: list[dict] = []  # participants rows of the planned subjects

    def check_source(self) -> None:
        if not (self.src / "subject_info.csv").is_file():
            raise ValueError(
                f"{self.src} has no subject_info.csv: not the Siena release"
            )

    def recordings(self) -> Iterator[Recording]:
        plan = plan_siena(self.src)
        for w in plan["warnings"]:
            self.warn(w)
        self.subjects = plan["participants"]
        for run in plan["runs"]:
            rec = Recording(
                Path(run["subject"], run["file"]),
                subject=run["subject"],
                task=self.meta["task"],
                run=run["run"],
            )
            self.runs[rec.source.as_posix()] = run
            yield rec

    def channels(self, rec: Recording, header: Header) -> list[Channel]:
        channels = self.channel_table.channels(header)
        run = self.runs[rec.source.as_posix()]
        tag = rec.source.as_posix()
        n_eeg = sum(c.type == "EEG" for c in channels)
        if isinstance(run["eeg_expected"], int) and n_eeg < run["eeg_expected"]:
            self.warn(
                f"{tag}: {n_eeg} EEG channels, subject_info.csv says "
                f"{run['eeg_expected']}"
            )
        if run["list_sfreq"] and abs(header.sfreq - run["list_sfreq"]) > 1e-6:
            self.warn(
                f"{tag}: {header.sfreq} Hz, the seizure list says {run['list_sfreq']} Hz"
            )
        return channels

    def sidecar(self, rec: Recording, header: Header) -> dict[str, Any]:
        n_misc = sum(self.channel_table.lookup(x)[1] == "MISC" for x in header.labels)
        return {
            "PowerLineFrequency": self.meta["power_line_frequency"],
            "EEGReference": self.meta["eeg_reference"],
            **self.meta["sidecar"],
            "MISCChannelCount": n_misc,
        }

    def events(self, rec: Recording, header: Header) -> list[Event] | None:
        length = header.n_times / header.sfreq
        placed = place_seizures(
            self.runs[rec.source.as_posix()]["seizures"],
            header.meas_date,
            length,
            rec.source.as_posix(),
            self.warnings,
        )
        if not placed:
            return None
        return [Event(s["onset"], s["duration"], "seizure") for s in placed]

    def participants(self, subjects: Sequence[str]) -> tuple[list[dict], dict]:
        rows = self.subjects or parse_subject_info(
            (self.src / "subject_info.csv").read_text(
                encoding="utf-8", errors="replace"
            )
        )
        wanted = {f"sub-{s}" for s in subjects}
        return [r for r in rows if r["participant_id"] in wanted], self.meta[
            "participants_json"
        ]

    def dataset_description(self) -> dict[str, Any]:
        return dict(self.meta["dataset_description"])

    def readme(self, summary: dict[str, Any]) -> str:
        participants, _ = self.participants(summary["subjects"])
        runs = [
            {
                "subject": r["subject"],
                "file": Path(r["source"]).name,
                "bids": r["bids_path"],
                "seizures": r["n_events"] or 0,
            }
            for r in summary["recordings"]
        ]
        notes = {
            r["source"]: [
                f"channel '{label}' typed MISC (not EEG or ECG)"
                for label in r["unmapped"] or []
            ]
            for r in summary["recordings"]
        }
        return readme_text(
            self.meta,
            participants,
            runs,
            notes,
            self.revision,
            summary["warnings"],
            summary["hours"],
        )

    def changes(self) -> str:
        return (
            f"1.0.0 {self.meta['published']}\n"
            f"  - The Siena Scalp EEG Database v1.0.0 (PhysioNet, {self.meta['doi']}) in "
            "BIDS: EDF files unchanged, seizure lists as events.tsv, subject_info.csv as "
            "participants.tsv.\n"
        )

    def license_file(self) -> Path | None:
        return Path("LICENSE.txt") if (self.src / "LICENSE.txt").is_file() else None

    def sourcedata(self, subjects: Sequence[str]) -> dict[str, bytes]:
        """The seizure lists of *subjects*, and subject_info.csv and RECORDS cut to them."""
        kept = set(subjects)
        out: dict[str, bytes] = {}
        info = self.src / "subject_info.csv"
        if info.is_file():
            lines = info.read_text(encoding="utf-8", errors="replace").splitlines(True)
            rows = [x for x in lines[1:] if x.split(",", 1)[0].strip() in kept]
            cut = len(rows) < len([x for x in lines[1:] if x.strip()])
            out[info.name] = (
                "".join(lines[:1] + rows).encode("utf-8") if cut else info.read_bytes()
            )
        records = self.src / "RECORDS"
        if records.is_file():
            lines = records.read_text(encoding="utf-8").splitlines(True)
            rows = [x for x in lines if x.strip().split("/", 1)[0] in kept]
            cut = len(rows) < len([x for x in lines if x.strip()])
            out[records.name] = (
                "".join(rows).encode("utf-8") if cut else records.read_bytes()
            )
        for path in sorted(self.src.glob("PN*/Seizures-list-*.txt")):
            if path.parent.name in kept:
                out[path.name] = path.read_bytes()
        return out
