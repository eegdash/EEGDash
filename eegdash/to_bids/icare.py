# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""I-CARE 2.0 (BDSP, WFDB) as BIDS.

A user obtains the International Cardiac Arrest REsearch consortium (I-CARE)
database from the Brain Data Science Platform (BDSP) under its restricted
licence and data use agreement, which forbid sharing it, and makes the tree
from that copy. The source is the release root: ``training/``, ``validation/``
and ``test/`` (the 2023 PhysioNet Challenge partitions; a copy that holds only
some of them is read as it is), one folder per patient
``<split>/<patient>/`` with the patient's clinical file ``<patient>.txt`` and
WFDB records ``<patient>_<segment>_<hour>_<group>.hea`` + ``.mat`` (MATLAB v4,
int16), one record per signal group (``EEG``, ``ECG``, ``REF``, ``OTHER``) of a
segment. Times count from the return of spontaneous circulation (ROSC), as
BDSP's page states; a segment ends on the hour, or at the end of the
recording. PhysioNet's I-CARE 2.1 (the 607 training patients) used the same
record and file names.

Every segment with an ``EEG`` record is one recording, written as an EDF file
(``linkable = False``): the EEG, ECG, REF and OTHER records of the segment
merged, in that order; a group whose sampling rate or length differs from the
EEG record's is left out with a warning, and a segment without an EEG record
is left out. The entities are those of the BIDS tree of PhysioNet's 2.1 that
EEGDash held:

- ``sub-<patient>``;
- ``task-icu``;
- ``acq-h<hour>``: the hour, as in the file name;
- ``run-<segment>``: the segment number, as in the file name (two segments
  can share an hour).

A recording's key is ``<patient>_<segment>_<hour>``; its source is the EEG
record's ``.hea``: a run is written again when that file changes or the
adapter's revision does, not when only another group's record changes.

**Samples.** The EDF file holds WFDB's 16-bit digital values unchanged, but
for the format's missing-sample code (-32768), which is replaced by the
channel's baseline (physical 0) and counted per channel in the sidecar. A
channel's EDF physical range is derived from its gain and baseline
(:func:`physical_range`), so the EDF physical value of a code is WFDB's
``(code - baseline) / gain``: exactly, as a decimal, when EDF's 8-character
fields can hold such a range (the full int16 range, or one that encloses the
channel's codes); otherwise the full int16 range with its ends rounded to
8 characters (a physical value then differs from WFDB's by at most half a unit
of the last decimal place, 0.005 uV for a range under 10 mV), and the channel
is listed in the sidecar. A unit WFDB declares (``uV``, ``mV``, ``V``) is
kept; ``nu`` (no unit, PhysioNet's 2.1) or none is read as microvolts, as the
earlier BIDS tree did: the values match microvolts in amplitude, which the
release does not state. A range too wide for 8 characters in microvolts is
written in millivolts or volts, the codes unchanged.

**Channels** are named by their WFDB label; the EDF label is ``<group>
<label>``, so the type follows the group (``tables/icare-2.0.channels.csv``):
EEG-group 10-20 names are EEG (a bipolar pair is MISC), ECG-group ECG,
REF-group (``A1``, ``M1``, ``C2``, ...) MISC, OTHER-group EMG (``EMG*``,
``LEG*``, ``LAT*``), EOG (``LOC``, ``ROC``) or MISC (``SpO2``). participants.tsv
holds the clinical file's fields and the split. ``wfdb`` is needed to write
data (``pip install wfdb``) and is imported only then.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .adapter import TABLES, Adapter, Channel, ChannelTable, Header, Recording

TABLE_ID = "icare-2.0"
SUBJECT_RE = re.compile(r"[A-Za-z0-9]+")
MISSING = -32768  # WFDB format 16: an invalid sample
INT16 = (-32768, 32767)
# A voltage unit and its size in microvolts; a range is tried from the declared one up.
VOLTAGES = (("uV", Fraction(1)), ("mV", Fraction(1000)), ("V", Fraction(10**6)))
VOLTAGE_NAMES = {"uV": "uV", "µV": "uV", "μV": "uV", "mV": "mV", "V": "V"}
NO_UNIT = (None, "nu", "NU")  # read as microvolts
EDF_FIELD = 8  # characters of an EDF number field
EDF_LABEL = 16
RECORD_SECONDS_MAX = 10  # the longest EDF data record tried when 1 s does not divide
WRITE_RECORDS = 64  # EDF data records written per block
MAX_README_WARNINGS = 100
NA_VALUES = ("", "nan", "NaN", "NAN", "n/a", "None")


@functools.lru_cache(maxsize=1)
def _meta() -> dict[str, Any]:
    """The dataset's metadata table, ``tables/icare-2.0.json``."""
    return json.loads((TABLES / f"{TABLE_ID}.json").read_text("utf-8"))


@functools.lru_cache(maxsize=1)
def _channel_table() -> ChannelTable:
    """The channel rules, ``tables/icare-2.0.channels.csv``, read once."""
    return ChannelTable(TABLES / f"{TABLE_ID}.channels.csv")


def _wfdb():
    """The ``wfdb`` package, which only writing data needs."""
    try:
        import wfdb  # noqa: PLC0415 (an optional dependency)
    except ImportError as err:
        raise ImportError(
            "the I-CARE adapter reads WFDB records with the wfdb package: "
            "pip install wfdb (or eegdash[icare])"
        ) from err
    return wfdb


# ------------------------------------------------------------- the layout


def parse_name(name: str, patient: str) -> tuple[str, str, str] | str:
    """``(segment, hour, group)`` of a header file name of *patient*, or why it is not one.

    ``0284_001_004_EEG.hea`` -> ``("001", "004", "EEG")``.
    """
    m = re.fullmatch(rf"{re.escape(patient)}_(\d+)_(\d+)_([A-Za-z]+)\.hea", name)
    if not m:
        return f"not {patient}_<segment>_<hour>_<group>.hea"
    return m[1], m[2], m[3]


def read_patient_file(path: Path) -> dict[str, str]:
    """The ``Key: value`` lines of a patient's clinical file, in their order."""
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    return out


def header_comments(path: Path) -> dict[str, str]:
    """The ``#Key: value`` comments of a WFDB header (``Utility frequency``, ``Start time``, ...)."""
    out = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep:
                out[key.strip()] = value.strip()
    return out


def declared_units(path: Path) -> list[str | None]:
    """The unit of each signal line of a WFDB header as written; None where there is none.

    The wfdb package reads a missing unit as ``mV``; here it stays None.
    """
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="latin-1").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    out = []
    for line in lines[1:]:
        fields = line.split()
        gain = fields[2] if len(fields) > 2 else ""
        out.append(gain.split("/", 1)[1] if "/" in gain else None)
    return out


# ---------------------------------------------------- the EDF number fields


def edf_exact(x: Fraction) -> str | None:
    """*x* as a decimal of at most 8 characters, or None when it has none."""
    for places in range(EDF_FIELD):
        scaled = x * 10**places
        if scaled.denominator != 1:
            continue
        n = scaled.numerator
        digits = str(abs(n)).rjust(places + 1, "0")
        text = f"{digits[:-places]}.{digits[-places:]}" if places else digits
        text = f"-{text}" if n < 0 else text
        return text if len(text) <= EDF_FIELD else None
    return None


def edf_rounded(x: Fraction) -> str | None:
    """*x* rounded to the most decimals 8 characters hold; None when its integer part does not fit."""
    for places in range(EDF_FIELD - 1, -1, -1):
        text = f"{float(x):.{places}f}"
        if float(text) == 0:
            text = text.lstrip("-")
        if len(text) <= EDF_FIELD:
            return text
    return None


def lattice_step(per: Fraction, places: int) -> int:
    """The code step on which ``code / per`` is a decimal of *places* places.

    ``(code - baseline) * Q * 10**places / P`` is an integer, with *per* =
    ``P / Q`` codes per physical unit, when ``P / gcd(P, Q * 10**places)``
    divides ``code - baseline``.
    """
    p, q = per.numerator, per.denominator
    return p // math.gcd(p, q * 10**places)


def physical_range(
    gain: float, baseline: int, lo: int, hi: int, unit: str | None = None
) -> dict[str, Any]:
    """The EDF fields of a channel whose codes run from *lo* to *hi*; physical ``(code - baseline) / gain``.

    *unit* is the unit WFDB declares (None or ``nu``: microvolts). Returns
    ``unit``, ``dmin``, ``dmax``, ``pmin``, ``pmax`` (texts of at most 8
    characters) and ``exact``. Tried in order, in the declared voltage unit and
    then each larger one: the full int16 range; then, from 7 decimal places
    down, the narrowest range enclosing *lo* to *hi* whose ends are codes with
    a decimal physical value. When none is exact: the full int16 range rounded
    to 8 characters, in the first unit it fits.
    """
    if not gain > 0:
        raise ValueError(f"gain {gain!r}: not a calibrated channel")
    g = Fraction(repr(float(gain)))  # the header's decimal, as WFDB read it
    b = int(baseline)
    name = "uV" if unit in NO_UNIT else VOLTAGE_NAMES.get(unit)
    if name is None:  # a unit that is not a voltage: kept, no other tried
        units = [(unit, Fraction(1))]
        base = Fraction(1)
    else:
        start = [n for n, _ in VOLTAGES].index(name)
        units = list(VOLTAGES[start:])
        base = VOLTAGES[start][1]
    for uname, size in units:
        per = g * size / base  # codes per physical unit of uname
        ranges = [INT16]
        for places in range(EDF_FIELD - 1, -1, -1):
            step = lattice_step(per, places)
            if step > INT16[1] - INT16[0]:
                continue
            dmin = b + step * math.floor(Fraction(lo - b, step))
            dmax = b + step * math.ceil(Fraction(hi - b, step))
            if dmax == dmin:
                dmax += step
            if INT16[0] <= dmin and dmax <= INT16[1]:
                ranges.append((dmin, dmax))
        for dmin, dmax in ranges:
            pmin, pmax = edf_exact((dmin - b) / per), edf_exact((dmax - b) / per)
            if pmin and pmax:
                return dict(
                    unit=uname, dmin=dmin, dmax=dmax, pmin=pmin, pmax=pmax, exact=True
                )
    for uname, size in units:
        per = g * size / base
        pmin = edf_rounded((INT16[0] - b) / per)
        pmax = edf_rounded((INT16[1] - b) / per)
        if pmin and pmax and float(pmin) < float(pmax):
            return dict(
                unit=uname,
                dmin=INT16[0],
                dmax=INT16[1],
                pmin=pmin,
                pmax=pmax,
                exact=False,
            )
    raise ValueError(
        f"gain {gain!r}, baseline {baseline}: no EDF range in 8 characters"
    )


def record_layout(n_times: int, sfreq: float) -> tuple[int, int, str, int]:
    """``(samples per EDF data record, records, the duration field, samples dropped)``.

    One second when it divides *n_times*; else the longest record of at most
    :data:`RECORD_SECONDS_MAX` seconds that divides it and whose duration is
    an exact field (the reader's rate, samples / duration, is *sfreq*); else
    one-second records and the trailing part second dropped.
    """
    rate = Fraction(repr(float(sfreq)))
    first = [int(rate)] if rate.denominator == 1 else []
    top = min(n_times, int(RECORD_SECONDS_MAX * rate))
    for spr in [*first, *range(top, 0, -1)]:
        if spr <= 0 or n_times % spr:
            continue
        duration = edf_exact(spr / rate)
        if duration and spr / float(duration) == float(sfreq):
            return spr, n_times // spr, duration, 0
    if rate.denominator == 1 and n_times >= rate:
        spr = int(rate)
        return spr, n_times // spr, "1", n_times % spr
    raise ValueError(f"{n_times} samples at {sfreq} Hz: no EDF data record fits")


def _field(value: Any, width: int) -> bytes:
    """An EDF header field: *value* left-aligned in *width* ASCII characters, never cut."""
    text = str(value)
    if len(text) > width:
        raise ValueError(f"EDF header field {text!r} is longer than {width}")
    return text.ljust(width).encode("ascii")


def write_edf(
    path: Path,
    labels: Sequence[str],
    ranges: Sequence[dict[str, Any]],
    data: np.ndarray,
    sfreq: float,
) -> int:
    """Write *data* (samples x channels, int16) as an EDF file; returns the samples dropped.

    Plain EDF, no annotation signal; start date 01.01.85 (EDF+'s date for an
    unknown one: I-CARE's times count from ROSC and are in the sidecar).
    """
    n_times, ns = data.shape
    spr, records, duration, dropped = record_layout(n_times, sfreq)
    header = b"".join(
        [
            _field("0", 8),
            _field("X", 80),
            _field("X", 80),
            _field("01.01.85", 8),
            _field("00.00.00", 8),
            _field(256 * (ns + 1), 8),
            _field("", 44),
            _field(records, 8),
            _field(duration, 8),
            _field(ns, 4),
        ]
    )
    per_signal = [
        (EDF_LABEL, list(labels)),
        (80, [""] * ns),  # transducer
        (8, [r["unit"] for r in ranges]),
        (8, [r["pmin"] for r in ranges]),
        (8, [r["pmax"] for r in ranges]),
        (8, [r["dmin"] for r in ranges]),
        (8, [r["dmax"] for r in ranges]),
        (80, [""] * ns),  # prefiltering
        (8, [spr] * ns),
        (32, [""] * ns),
    ]
    for width, values in per_signal:
        header += b"".join(_field(v, width) for v in values)
    with open(path, "wb") as f:
        f.write(header)
        for start in range(0, records, WRITE_RECORDS):
            stop = min(records, start + WRITE_RECORDS)
            block = data[start * spr : stop * spr].reshape(stop - start, spr, ns)
            f.write(np.ascontiguousarray(block.transpose(0, 2, 1), "<i2").tobytes())
    return dropped


def _na(value: str) -> str | None:
    return None if value.strip() in NA_VALUES else value.strip()


def _snake(key: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_") or "field"


class ICARE(Adapter):
    """I-CARE 2.0 (BDSP, WFDB): see the module docstring."""

    name = "icare"
    versions = ("2.0",)
    revision = 1
    extension = ".edf"
    linkable = False

    def __init__(self, src: Path, version: str) -> None:
        super().__init__(src, version)
        self.meta = _meta()
        self.channel_table = _channel_table()
        self.splits: dict[
            str, str
        ] = {}  # patient -> its split folder, from recordings()
        self._written: dict[str, dict[str, Any]] = {}  # key -> what write_data found

    def check_source(self) -> None:
        splits = self.meta["splits"]
        if not any((self.src / s).is_dir() for s in splits):
            raise ValueError(
                f"{self.src} has no {', '.join(f'{s}/' for s in splits)}: "
                "not the I-CARE release"
            )

    def recordings(self) -> Iterator[Recording]:
        groups = self.meta["groups"]
        no_eeg: Counter = Counter()
        for split in self.meta["splits"]:
            root = self.src / split
            if not root.is_dir():
                continue
            for entry in sorted(os.scandir(root), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                patient = entry.name
                if not SUBJECT_RE.fullmatch(patient):
                    self.warn(f"{split}/{patient}/: not alphanumeric; left out")
                    continue
                if patient in self.splits:
                    self.warn(
                        f"{split}/{patient}/: also in {self.splits[patient]}/; left out"
                    )
                    continue
                self.splits[patient] = split
                segments: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
                for name in sorted(os.listdir(entry.path)):
                    if not name.endswith(".hea"):
                        continue
                    parsed = parse_name(name, patient)
                    if isinstance(parsed, str):
                        self.warn(f"{split}/{patient}/{name}: {parsed}; left out")
                        continue
                    segment, hour, group = parsed
                    if group not in groups:
                        self.warn(
                            f"{split}/{patient}/{name}: group {group!r} is not one of "
                            f"{', '.join(groups)}; left out"
                        )
                        continue
                    segments[(segment, hour)][group] = name
                for (segment, hour), found in sorted(segments.items()):
                    if "EEG" not in found:
                        no_eeg[split] += 1
                        continue
                    yield Recording(
                        Path(split, patient, found["EEG"]),
                        subject=patient,
                        task=self.meta["task"],
                        acquisition=f"h{hour}",
                        run=segment,
                        key=f"{patient}_{segment}_{hour}",
                        extra={"groups": ",".join(g for g in groups if g in found)},
                    )
        for split, n in sorted(no_eeg.items()):
            self.warn(f"{split}/: {n} segment(s) without an EEG record; left out")

    def channels(self, rec: Recording, header: Header) -> list[Channel]:
        return self.channel_table.channels(header)

    def write_data(self, rec: Recording, dest: Path) -> None:
        """The segment's records merged into one EDF file, the 16-bit codes kept."""
        wfdb = _wfdb()
        self._written.pop(rec.key, None)
        folder = self.src / rec.source.parent
        signals, labels, ranges = [], [], []
        missing, rounded, assumed, left_out, records = {}, [], [], [], []
        flat: list[str] = []
        eeg = None
        for group in rec.extra["groups"].split(","):
            base = folder / f"{rec.key}_{group}"
            if eeg is not None and not base.with_name(f"{base.name}.mat").is_file():
                left_out.append(f"{base.name}: a header without its sample file")
                continue
            record = wfdb.rdrecord(str(base), physical=False, return_res=16)
            if eeg is None:
                eeg = record
            elif record.fs != eeg.fs or record.sig_len != eeg.sig_len:
                left_out.append(
                    f"{base.name}: {record.fs} Hz, {record.sig_len} samples, the EEG "
                    f"record {eeg.fs} Hz, {eeg.sig_len} samples"
                )
                continue
            fmts = sorted(set(record.fmt))
            if fmts != ["16"]:
                raise ValueError(f"{base.name}: WFDB format {fmts}, not 16")
            data = np.asarray(record.d_signal, dtype=np.int16)
            units = declared_units(base.with_name(f"{base.name}.hea"))
            for k, label in enumerate(record.sig_name):
                name = f"{group} {label}"
                if len(name) > EDF_LABEL or not name.isascii():
                    raise ValueError(f"{base.name}: label {label!r} too long for EDF")
                col = data[:, k].copy()
                gap = col == MISSING
                if gap.any():
                    col[gap] = np.clip(round(record.baseline[k]), *INT16)
                    missing[label] = int(gap.sum())
                unit = units[k] if k < len(units) else None
                gain, baseline = record.adc_gain[k], record.baseline[k]
                if not gain > 0 and col.min() == col.max():
                    # one code throughout with no usable gain (BDSP's 2.0 writes
                    # a negative one): a flat channel, its code read as physical 0
                    gain, baseline = 1.0, int(col[0])
                    flat.append(label)
                fields = physical_range(
                    gain, baseline, int(col.min()), int(col.max()), unit
                )
                if not fields["exact"]:
                    rounded.append(label)
                if unit in NO_UNIT:
                    assumed.append(label)
                signals.append(col)
                labels.append(name)
                ranges.append(fields)
            records.append(base.name)
        dropped = write_edf(dest, labels, ranges, np.column_stack(signals), eeg.fs)
        self._written[rec.key] = dict(
            records=records,
            missing=missing,
            rounded=rounded,
            flat=flat,
            assumed=assumed,
            left_out=left_out,
            dropped=dropped,
        )

    def sidecar(self, rec: Recording, header: Header) -> dict[str, Any]:
        comments = header_comments(self.src / rec.source)
        found = self._written.pop(rec.key, {})
        try:
            line = int(float(comments.get("Utility frequency", "")))
        except ValueError:
            line = "n/a"
        side: dict[str, Any] = {
            "PowerLineFrequency": line,
            "EEGReference": "n/a",
            **self.meta["sidecar"],
            "HourAfterArrest": int(rec.acquisition[1:]),
            "SegmentIndex": int(rec.run),
        }
        for key, name in (
            ("Start time", "StartTimeAfterArrest"),
            ("End time", "EndTimeAfterArrest"),
        ):
            if key in comments:
                side[name] = comments[key]
        if found.get("records"):
            side["SourceRecords"] = found["records"]
        if found.get("missing"):
            side["MissingSamples"] = found["missing"]
            side["MissingSamplesNote"] = (
                "Samples with WFDB's missing-sample code (-32768) per channel, "
                "written as the channel's baseline (physical 0)."
            )
        if found.get("rounded"):
            side["PhysicalRangeRounded"] = found["rounded"]
            side["PhysicalRangeRoundedNote"] = (
                "Channels whose physical range, from the WFDB gain and baseline, has no "
                "exact 8-character EDF value: the ends of the EDF range are rounded to "
                "their last decimal place, so a physical value differs from WFDB's by "
                "at most half a unit of that place; the 16-bit codes are WFDB's."
            )
        if found.get("flat"):
            side["FlatChannels"] = found["flat"]
            side["FlatChannelsNote"] = (
                "Channels that hold one sample value throughout and whose WFDB header "
                "gives no positive gain: written flat, the value as physical 0."
            )
        if found.get("assumed"):
            side["UnitsNote"] = (
                "The WFDB records declare no unit ('nu' or none) for "
                f"{len(found['assumed'])} channel(s); their gain and baseline give "
                "values of microvolt amplitude, so they are written in uV. The release "
                "does not state the unit."
            )
        for text in found.get("left_out", []):
            self.warn(f"{rec.source.parent.as_posix()}/{text}; left out of the run")
        if found.get("dropped"):
            self.warn(
                f"{rec.source.as_posix()}: the last {found['dropped']} sample(s) dropped: "
                "no EDF data record divides the record"
            )
            side["TrailingSamplesDropped"] = found["dropped"]
        return side

    def participants(self, subjects: Sequence[str]) -> tuple[list[dict], dict]:
        meta = self.meta["participants"]
        known = list(dict.fromkeys(meta["columns"].values()))
        rows, extra = [], {}
        for s in subjects:
            split = self.splits.get(s)
            row: dict[str, Any] = {"participant_id": f"sub-{s}", "split": split}
            path = self.src / (split or "") / s / f"{s}.txt"
            if not path.is_file():
                self.warn(
                    f"{(split or '?')}/{s}/{s}.txt: missing; participants.tsv n/a"
                )
                rows.append(row)
                continue
            for key, value in read_patient_file(path).items():
                if key == "Patient":
                    continue
                column = meta["columns"].get(key)
                if column is None:
                    column = _snake(key)
                    extra.setdefault(column, key)
                value = _na(value)
                if column == "sex" and value is not None:
                    value = meta["sex"].get(value, value)
                row[column] = value
            rows.append(row)
        columns = ["participant_id", "split", *known, *sorted(extra)]
        present = [c for c in columns if any(c in r for r in rows)]
        rows = [{c: r.get(c) for c in present} for r in rows]
        sidecar = {c: meta["json"][c] for c in present if c in meta["json"]}
        sidecar.update(
            {
                c: {"Description": f"The release's patient file field `{k}`."}
                for c, k in sorted(extra.items())
                if c in present
            }
        )
        return rows, sidecar

    def dataset_description(self) -> dict[str, Any]:
        return dict(self.meta["dataset_description"])

    def readme(self, summary: dict[str, Any]) -> str:
        meta = self.meta
        recs = summary["recordings"]
        splits = Counter(self.splits.get(s, "?") for s in summary["subjects"])
        rates = Counter(r["sfreq"] for r in recs)
        unmapped = Counter(label for r in recs for label in r["unmapped"] or [])
        lines = [
            f"# I-CARE {self.version} (BDSP)",
            "",
            "The International Cardiac Arrest REsearch consortium (I-CARE) "
            "electroencephalography (EEG) database: continuous EEG, with "
            "electrocardiography (ECG) and other signals where recorded, of comatose "
            "adults after cardiac arrest, from seven hospitals in the United States and "
            "Europe; times are hours after the return of spontaneous circulation. "
            "Distributed by the Brain Data Science Platform (BDSP) under its restricted "
            "licence and data use agreement, which forbid sharing the data: every user "
            f"obtains it from BDSP ({meta['url']}) and makes this tree from their own "
            f"copy. Please cite the database ({meta['doi']}) and the paper that "
            f"describes it: {meta['paper']}.",
            "",
            "## This BIDS version",
            "",
            f"Written by EEGDash's converter (`eegdash.to_bids`, adapter `{self.name}`, "
            f"revision {self.revision}) from the release's WFDB records: {len(recs)} "
            f"recordings of {len(summary['subjects'])} patients "
            f"({', '.join(f'{k} {v}' for k, v in sorted(splits.items())) or 'none'}), "
            f"{summary['hours']:.1f} hours. Sampling rates: "
            + (", ".join(f"{k} Hz ({v})" for k, v in sorted(rates.items())) or "none")
            + ". No channel was resampled or filtered.",
            "",
            "| release | BIDS |",
            "|---|---|",
            "| `<split>/<patient>/` | `sub-<patient>`; the split is a column of "
            "participants.tsv |",
            "| `<patient>_<segment>_<hour>_<group>.hea` + `.mat` | `task-icu`, "
            "`acq-h<hour>` (the hour from ROSC), `run-<segment>`: one EDF file "
            "per segment |",
            "| `EEG`, `ECG`, `REF`, `OTHER` records of a segment | merged into its EDF "
            "file, in that order; a record of another rate or length than the EEG "
            "record, or without its sample file, is left out (warnings) |",
            "| `<patient>.txt` | participants.tsv |",
            "",
            "The EDF files hold the release's 16-bit samples unchanged. Each channel's "
            "EDF physical range is derived from its WFDB gain and baseline, so a sample's "
            "physical value is WFDB's (sample - baseline) / gain; where EDF's 8-character "
            "fields cannot hold that range exactly, it is rounded and the channel listed "
            "under `PhysicalRangeRounded` in the recording's sidecar. Samples with WFDB's "
            "missing-sample code are written as the channel's baseline and counted "
            "under `MissingSamples`. A channel that holds one value throughout and whose "
            "header gives no positive gain is written flat at physical 0 "
            "(`FlatChannels`). A channel whose record declares no unit (`nu` or "
            "none) is written in uV, as its values have microvolt amplitude; the "
            "release does not state the unit (`UnitsNote`).",
            "",
            "Channel names are the WFDB labels; the EDF label adds the record's group "
            "(`EEG Fp1`), which gives the type: 10-20 names of the EEG record are EEG, "
            "the ECG record's channels ECG, the REF record's (`A1`, `A2`, `M1`, `M2`, "
            "`C2`, ...) MISC, and of the OTHER record `EMG*`, `LEG*` and `LAT*` EMG, "
            "`LOC` and `ROC` EOG, `SpO2` MISC. A label no rule matches keeps its "
            "name and is typed MISC: "
            + (
                ", ".join(
                    f"`{k}` ({v} recordings)" for k, v in sorted(unmapped.items())
                )
                if unmapped
                else "none in this tree"
            )
            + ". `PowerLineFrequency` is the record's utility frequency; "
            "`EEGReference` is `n/a`, as the release does not state it.",
        ]
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
