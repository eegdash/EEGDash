"""Iso-hour landscape: subjects per dataset against hours per subject.

Every catalogue dataset is one point: x is the recorded hours per subject,
y the number of subjects, both on log scales. The product of the two is the
dataset's total recorded hours, so the curves ``subjects x hours = H`` of the
original fMRI figure (Naselaris, Allen & Kay, Curr. Opin. Behav. Sci. 2021)
become straight diagonals ("iso-hour lines"). The chart answers a question
the subject-count and file-count bubbles cannot: does the catalogue sample
many people briefly, or few people densely?

Data come from ``/datasets/summary`` (``demographics.subjects_count`` and
``total_duration_s``). The catalogue is cleaned before plotting, and every
dropped dataset keeps its reason in the ``status`` column so the selection
can be audited:

``non_human``
    Animal recordings listed in :data:`NON_HUMAN` (``demographics.species``
    is empty in the catalogue today, so the list is curated by hand), or any
    dataset whose species field names a non-human species.
``not_a_recording``
    Phantom recordings and placeholder deposits (:data:`NOT_RECORDINGS`).
``no_duration``
    ``total_duration_s`` missing or zero.
``duplicate``
    Dropped only with proof that the entry holds recordings of another kept
    entry: identical git-annex keys of every signal-bearing file (the ``.eeg``
    of a BrainVision header, the ``.fdt`` of an EEGLAB header, every member of
    a CTF/MEF directory) for every recording (a mirror, or a strict subset), or
    a listed verified re-encoding
    (:data:`HBN_REENCODINGS`) checked record by record (path, duration within
    1 s). Every fetched record must have a path, a duration and a unique path,
    or nothing is proven. ``duplicate_of`` and ``evidence`` record the proof.

Rows that stay in the figure can carry a ``flag``:

``possible_duplicate``
    Same accession, DOI, normalised name, or same subjects, sampling rates and
    duration, but the same recordings are not proven. Equal paths and
    durations are not proof (WRCC2023 MI-A/MI-B: 720 of 810 payloads differ).
``short_duration`` / ``epoched_short_duration``
    Under one minute per subject; the second when most records are stored
    epoched. Kept: the label does not show that one epoch was counted.

``duration_source`` is ``"records"`` when the dataset-level aggregate was
replaced by the sum of record-level ``duration_seconds`` (see
:func:`correct_durations_from_records`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from .utils import build_and_export_html, get_dataset_url
except ImportError:  # pragma: no cover - direct-script execution
    from utils import build_and_export_html, get_dataset_url  # type: ignore

__all__ = [
    "ISO_HOURS",
    "LANDMARKS",
    "MODALITY_STYLE",
    "NON_HUMAN",
    "NOT_RECORDINGS",
    "build_isohour_table",
    "correct_durations_from_records",
    "fetch_catalogue",
    "fetch_records",
    "generate_isohour_landscape",
    "summarise_isohour",
]

_DEFAULT_API_BASE = "https://data.eegdash.org/api"
_DEFAULT_DATABASE = "eegdash"
_CACHE_TTL_S = 6 * 3600

#: Iso-hour lines drawn on the landscape (total hours per dataset).
ISO_HOURS: tuple[int, ...] = (1, 10, 100, 1_000, 10_000)

#: Animal datasets (checked against each README, 2026-10-23).
NON_HUMAN: Mapping[str, str] = {
    "ds003380": "pig (corticothalamic recordings under analgesia)",
    "on003380": "pig (corticothalamic recordings under analgesia)",
    "on004127": "rat somatosensory cortex",
    "on004598": "TgF344-AD rats",
    "on005688": "mice (diagnostic ultrasound, Tan et al. 2024)",
    "on006269": "Syngap1 rats",
    "on006366": "mice (MSSV sleep staging)",
    "on006890": "Macaca fuscata ECoG",
    "on007028": "macaque auditory cortex",
    "on008610": "non-human primate VPL",
}

#: Deposits that hold no human recording session.
NOT_RECORDINGS: Mapping[str, str] = {
    "on002885": "DBS phantom",
    "on004738": "MEG phantom",
    "on004784": "EEG phantom",
    **{
        ds: "placeholder deposit"
        for ds in (
            "on004849",
            "on004850",
            "on004852",
            "on004853",
            "on004854",
            "on004855",
        )
    },
}

_HBN_COPY = re.compile(r"^EEG2025r\d+(mini)?$")
_HUMAN_SPECIES = {"human", "humans", "homo sapiens", "homo_sapiens"}

#: Okabe-Ito colours plus a marker per modality (redundant encoding).
MODALITY_STYLE: Mapping[str, dict[str, str]] = {
    "eeg": {"label": "EEG", "color": "#0072B2", "marker": "o", "symbol": "circle"},
    "meg": {"label": "MEG", "color": "#CC79A7", "marker": "s", "symbol": "square"},
    "ieeg": {
        "label": "iEEG",
        "color": "#D55E00",
        "marker": "^",
        "symbol": "triangle-up",
    },
    "fnirs": {"label": "fNIRS", "color": "#009E73", "marker": "D", "symbol": "diamond"},
    "emg": {"label": "EMG", "color": "#7F7F7F", "marker": "P", "symbol": "cross"},
}

#: Short labels for well-known datasets, drawn directly on the landscape.
LANDMARKS: Mapping[str, str] = {
    "nm000181": "NMT",
    "nm000225": "PhysioNet 2018",
    "nm000289": "I-CARE",
    "on005873": "SeizeIT2",
    "on004395": "PEERS",
    "nm000185": "Sleep-EDF",
    "on005178": "EESM23",
    "nm000110": "CHB-MIT",
    "nm000358": "AJILE12",
    "nm000232": "THINGS-EEG2",
    "ds004212": "THINGS-MEG",
    "nm000229": "MEG-MASC",
    "nm000180": "Alice (Brennan)",
    "nm000179": "LEMON",
    "on004362": "PhysioNet MI",
    "nm000139": "BCI IV 2a",
    "on004789": "Delayed free recall (iEEG)",
    "on005515": "HBN R10",
    "on006317": "Chisco-2.0",
    "nm000238": "SparrKULee",
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    root = (
        os.environ.get("EEGDASH_DOCS_CACHE") or Path.home() / ".cache" / "eegdash-docs"
    )
    path = Path(root) / "isohour"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_json(url: str, *, timeout: int = 120) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _cached_get_json(url: str, ttl_s: float = _CACHE_TTL_S) -> dict[str, Any]:
    path = _cache_dir() / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
    if path.exists() and time.time() - path.stat().st_mtime < ttl_s:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        payload = _get_json(url)
    except OSError:
        if path.exists():  # stale cache beats a failed docs build
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_catalogue(
    api_base: str | None = None,
    database: str | None = None,
    *,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Return every dataset document from ``/datasets/summary``."""
    api_base = (
        api_base or os.environ.get("EEGDASH_API_BASE", _DEFAULT_API_BASE)
    ).rstrip("/")
    database = database or os.environ.get("EEGDASH_DATABASE", _DEFAULT_DATABASE)
    out: list[dict[str, Any]] = []
    skip = 0
    while True:
        url = f"{api_base}/{database}/datasets/summary?limit={page_size}&skip={skip}"
        payload = _cached_get_json(url)
        rows = payload.get("data") or []
        out.extend(rows)
        skip += page_size
        if not rows or skip >= int(payload.get("total") or 0):
            return out


def fetch_records(
    dataset_id: str, api_base: str | None = None, database: str | None = None
) -> list[dict[str, Any]]:
    """Every record of one dataset from ``/records`` (cached on disk)."""
    api_base = (
        api_base or os.environ.get("EEGDASH_API_BASE", _DEFAULT_API_BASE)
    ).rstrip("/")
    database = database or os.environ.get("EEGDASH_DATABASE", _DEFAULT_DATABASE)
    flt = urllib.parse.quote(json.dumps({"dataset": dataset_id}))
    out: list[dict[str, Any]] = []
    skip = 0
    while True:
        url = f"{api_base}/{database}/records?filter={flt}&limit=1000&skip={skip}"
        rows = _cached_get_json(url).get("data") or []
        out.extend(rows)
        if len(rows) < 1000:
            return out
        skip += 1000


def _record_duration_sum(
    dataset_id: str, api_base: str | None = None, database: str | None = None
) -> float:
    """Sum of record-level ``duration_seconds`` for one dataset."""
    api_base = (
        api_base or os.environ.get("EEGDASH_API_BASE", _DEFAULT_API_BASE)
    ).rstrip("/")
    database = database or os.environ.get("EEGDASH_DATABASE", _DEFAULT_DATABASE)
    flt = urllib.parse.quote(json.dumps({"dataset": dataset_id}))
    total, skip = 0.0, 0
    while True:
        url = f"{api_base}/{database}/records?filter={flt}&limit=1000&skip={skip}"
        rows = _cached_get_json(url).get("data") or []
        total += sum(float(r.get("duration_seconds") or 0.0) for r in rows)
        if len(rows) < 1000:
            return total
        skip += 1000


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _norm_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _primary_modality(value: Any) -> str:
    if isinstance(value, str):
        value = [value]
    items = [str(v).lower() for v in (value or [])]
    if items and all(len(v) == 1 for v in items):  # ['e', 'e', 'g'] -> 'eeg'
        items = ["".join(items)]
    for item in items:
        if item in MODALITY_STYLE:
            return item
    return items[0] if items else "unknown"


def _display_name(doc: Mapping[str, Any]) -> str:
    name = doc.get("name") or doc.get("computed_title")
    if not name:
        canonical = doc.get("canonical_name")
        name = canonical[0] if isinstance(canonical, list) and canonical else canonical
    return str(name or doc.get("dataset_id")).strip()


def _sfreq_signature(doc: Mapping[str, Any]) -> tuple:
    counts = doc.get("sfreq_counts") or []
    return tuple(
        sorted(
            (round(float(c.get("val") or 0), 1), int(c.get("count") or 0))
            for c in counts
        )
    )


def _species_is_non_human(doc: Mapping[str, Any]) -> bool:
    species = (doc.get("demographics") or {}).get("species")
    if not species:
        return False
    values = species if isinstance(species, list) else [species]
    return any(str(v).strip().lower() not in _HUMAN_SPECIES for v in values)


#: ``records(dataset_id)`` -> record dicts with at least ``bids_relpath``,
#: ``ntimes`` and ``recording_type`` (see :func:`fetch_records`).
RecordFetcher = Callable[[str], list[dict[str, Any]]]

_ACCESSION = re.compile(r"(?:^|[^a-z0-9])(?:ds|on)(\d{6})(?![0-9])")
_NAME_SUFFIX = re.compile(r"\(bdf converted\)|\bmini\b")


def _accession(text: str) -> str | None:
    m = _ACCESSION.search(str(text or "").lower())
    return m.group(1) if m else None


def _name_key(name: str) -> str:
    return _norm_name(_NAME_SUFFIX.sub("", str(name or "").lower()))


#: Per-recording duration tolerance: a whole-second BDF re-encoding trims
#: < 1 s (HBN EEG2025r1 vs on005505: at most 0.998 s over 1,342 recordings).
_SAME_RECORDING_TOL_S = 1.0

#: Verified re-encodings whose bytes necessarily differ from the original, so
#: payload hashes cannot match. Each is still checked record by record (same
#: path, duration within 1 s). Source: the EEG2025 competition release of
#: HBN-EEG: BDF at 100 Hz of the 500 Hz releases ds005505-ds005516 (dataset_doi
#: of r1-r9 and their minis names the ds release; r10/r11 by release number),
#: and the 20-subject "mini" subset of each release.
HBN_REENCODINGS: Mapping[str, str] = {
    **{
        f"EEG2025r{n}{suffix}": f"on0055{acc}"
        for n, acc in zip(
            range(1, 12),
            ("05", "06", "07", "08", "09", "10", "11", "12", "14", "15", "16"),
        )
        for suffix in ("", "mini")
    }
}


#: Signal-bearing companions of multi-file formats: the entry file alone (a
#: BrainVision or EEGLAB header) does not identify the samples.
_SIGNAL_COMPANIONS = {".vhdr": (".eeg",), ".set": (".fdt",)}
_DIRECTORY_FORMATS = (".ds", ".mefd")


def _payload_key(record: dict[str, Any]) -> tuple | None:
    """Annex keys of every file holding the recording's samples, or ``None``.

    Single-file formats (EDF, BDF, FIF, ...): the raw file's key. BrainVision:
    the ``.vhdr`` and its ``.eeg`` (required). EEGLAB: the ``.set`` and its
    ``.fdt`` when one is referenced. CTF ``.ds`` / MEF ``.mefd`` directories:
    every member. A referenced signal file without a key -> ``None`` (unproven):
    on003421/on002833 share all 80 ``.vhdr`` hashes but 2 ``.eeg`` payloads
    differ in size and hash.
    """
    storage = record.get("storage") or {}
    keys = storage.get("annex_keys") or {}
    raw = str(storage.get("raw_key") or "")
    deps = [str(d) for d in storage.get("dep_keys") or []]
    ext = "." + raw.rsplit(".", 1)[-1].lower() if "." in raw else ""
    if ext in _DIRECTORY_FORMATS:
        members = sorted(m for m in {*deps, *keys} if m.startswith(raw + "/"))
        if not members or any(m not in keys for m in members):
            return None
        return tuple((m[len(raw) :], keys[m]) for m in members)
    if raw not in keys:
        return None
    files = [raw]
    for comp in _SIGNAL_COMPANIONS.get(ext, ()):
        found = sorted({f for f in (*deps, *keys) if f.lower().endswith(comp)})
        if ext == ".vhdr" and not found:
            return None  # BrainVision samples live in the .eeg
        files += found
    if any(f not in keys for f in files):
        return None
    return tuple((f.rsplit(".", 1)[-1].lower(), keys[f]) for f in files)


def _signature(
    recs: list[dict[str, Any]],
) -> dict[str, tuple[float, tuple | None]] | None:
    """``path without extension -> (duration s, signal payload keys)`` of every record.

    ``None`` when any record lacks a path or a duration, or two records share a
    path without extension: then "every recording" cannot be checked.
    """
    out: dict[str, tuple[float, str | None]] = {}
    for r in recs:
        dur = r.get("duration_seconds")
        if not dur and r.get("ntimes") and r.get("sampling_frequency"):
            dur = r["ntimes"] / r["sampling_frequency"]
        path = str(r.get("bids_relpath") or "").rsplit(".", 1)[0]
        if not path or not dur or path in out:
            return None
        out[path] = (float(dur), _payload_key(r))
    return out or None


def _covers(small: dict, big: dict, *, same_payload: bool) -> bool:
    """Every recording of ``small`` is in ``big``: same signal payload keys, or same duration."""
    for k, (dur, key) in small.items():
        if k not in big:
            return False
        other_dur, other_key = big[k]
        if same_payload and not (key and key == other_key):
            return False
        if abs(dur - other_dur) > _SAME_RECORDING_TOL_S:
            return False
    return True


def _pairs_within(groups: Iterable[list[int]], why: str) -> list[tuple[int, int, str]]:
    return [(a, b, why) for g in groups for a in g for b in g if a < b]


def _candidate_pairs(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Pairs of kept rows that may hold the same recordings, with the first reason."""
    kept = df[df["status"] == "kept"]
    by_acc: dict[str, list[int]] = {}
    by_name: dict[str, list[int]] = {}
    by_agg: dict[tuple, list[int]] = {}
    for idx, row in kept.iterrows():
        acc = _accession(row["dataset_id"])
        if acc:
            by_acc.setdefault(acc, []).append(idx)
        key = _name_key(row["name"])
        if len(key) >= 8:
            by_name.setdefault(key, []).append(idx)
        if row["_sfreq"] and row["duration_s"] > 0:
            by_agg.setdefault((row["n_subjects"], row["_sfreq"]), []).append(idx)
    found: list[tuple[int, int, str]] = []
    for idx, row in kept.iterrows():
        acc = _accession(row["_doi"])
        found += [
            (min(idx, o), max(idx, o), f"dataset_doi names {acc}")
            for o in by_acc.get(acc or "", [])
            if o != idx
        ]
    found += _pairs_within(by_acc.values(), "same accession number")
    found += _pairs_within(by_name.values(), "same normalised name")
    found += [
        (a, b, why)
        for a, b, why in _pairs_within(
            by_agg.values(), "same subjects, sampling rates and duration"
        )
        if abs(df.at[a, "duration_s"] - df.at[b, "duration_s"]) < 1.0
    ]
    row_of = {df.at[i, "dataset_id"]: i for i in kept.index}
    found += [
        (
            min(row_of[c], row_of[o]),
            max(row_of[c], row_of[o]),
            "verified HBN re-encoding",
        )
        for c, o in HBN_REENCODINGS.items()
        if c in row_of and o in row_of
    ]
    first: dict[tuple[int, int], str] = {}
    for a, b, why in found:
        first.setdefault((a, b), why)
    return [(a, b, why) for (a, b), why in sorted(first.items())]


def _keep_order(df: pd.DataFrame, idx: int) -> tuple:
    """Sort key: originals before HBN re-encodings, then most subjects, then id."""
    ds = df.at[idx, "dataset_id"]
    return (bool(_HBN_COPY.match(ds)), -df.at[idx, "n_subjects"], ds)


def _proof(df: pd.DataFrame, a: int, b: int, sa: dict | None, sb: dict | None):
    """``(keep, drop, evidence)`` when a and b provably hold the same recordings."""
    if not sa or not sb:
        return None
    ida, idb = df.at[a, "dataset_id"], df.at[b, "dataset_id"]
    for copy, orig, sc, so in ((a, b, sa, sb), (b, a, sb, sa)):
        if HBN_REENCODINGS.get(df.at[copy, "dataset_id"]) == df.at[orig, "dataset_id"]:
            if _covers(sc, so, same_payload=False):
                return (
                    orig,
                    copy,
                    (
                        f"verified HBN re-encoding; {len(sc)} of {len(so)} recordings "
                        "matched (path, duration within 1 s)"
                    ),
                )
            return None
    if sa.keys() == sb.keys() and _covers(sa, sb, same_payload=True):
        keep, drop = sorted((a, b), key=lambda i: _keep_order(df, i))
        return keep, drop, f"same {len(sa)} recordings (path, every signal file's key)"
    for small, big, s_small, s_big in ((a, b, sa, sb), (b, a, sb, sa)):
        if len(s_small) < len(s_big) and _covers(s_small, s_big, same_payload=True):
            return (
                big,
                small,
                (
                    f"subset: {len(s_small)} of {len(s_big)} recordings (path, every signal file's key)"
                ),
            )
    del ida, idb
    return None


def _resolve_duplicates(df: pd.DataFrame, records: RecordFetcher | None) -> None:
    """Drop proven duplicates; flag unproven candidates as ``possible_duplicate``.

    Proof is identical annex keys of every signal-bearing file of every
    recording (:func:`_payload_key`; a mirror or a subset), or a listed verified re-encoding checked record by record. Equal
    paths and durations alone are not proof: separately collected experiments
    share BIDS paths and fixed trial lengths (WRCC2023 MI-A/MI-B: 720 of 810
    payloads differ).
    """
    cache: dict[str, dict | None] = {}
    for a, b, why in _candidate_pairs(df):
        if "kept" not in (df.at[a, "status"], df.at[b, "status"]):
            continue
        sigs = []
        for i in (a, b):
            ds = df.at[i, "dataset_id"]
            if ds not in cache:
                try:
                    cache[ds] = _signature(records(ds)) if records else None
                except OSError:
                    cache[ds] = None
            sigs.append(cache[ds])
        proof = _proof(df, a, b, *sigs)
        if proof is None:
            other, flagged = sorted((a, b), key=lambda i: _keep_order(df, i))
            if df.at[flagged, "status"] == "kept" and pd.isna(df.at[flagged, "flag"]):
                df.at[flagged, "flag"] = "possible_duplicate"
                df.at[flagged, "duplicate_of"] = df.at[other, "dataset_id"]
                df.at[flagged, "evidence"] = f"{why}; same recordings not proven"
            continue
        keep, drop, evidence = proof
        if df.at[drop, "status"] != "kept":
            continue
        df.at[drop, "status"] = "duplicate"
        df.at[drop, "duplicate_of"] = df.at[keep, "dataset_id"]
        df.at[drop, "flag"] = None
        df.at[drop, "evidence"] = f"{why}; {evidence}"


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Mark still-kept rows selected by ``mask`` as dropped for ``reason``."""
    df.loc[mask & (df["status"] == "kept"), "status"] = reason


def build_isohour_table(
    docs: Iterable[Mapping[str, Any]],
    records: RecordFetcher | None = None,
) -> pd.DataFrame:
    """Turn ``/datasets/summary`` documents into the cleaned landscape table.

    Returns one row per catalogue dataset. ``status == "kept"`` marks the
    rows to plot; every other row carries the reason it was dropped. Without
    ``records`` (offline), no row is dropped as a duplicate or as epoched:
    candidates are only flagged.
    """
    rows: list[dict[str, Any]] = []
    for doc in docs:
        tags = doc.get("tags") or {}
        n_subj = (doc.get("demographics") or {}).get("subjects_count")
        dur_s = doc.get("total_duration_s")
        rows.append(
            {
                "dataset_id": str(doc.get("dataset_id")),
                "name": _display_name(doc),
                "modality": _primary_modality(
                    doc.get("recording_modality") or doc.get("datatypes")
                ),
                "n_subjects": int(n_subj) if n_subj else 0,
                "duration_s": float(dur_s) if dur_s else 0.0,
                "n_sessions": len(doc.get("sessions") or []),
                "types": list(tags.get("type") or []),
                "pathology": list(tags.get("pathology") or []),
                "source": doc.get("source"),
                "_sfreq": _sfreq_signature(doc),
                "_non_human_species": _species_is_non_human(doc),
                "status": "kept",
                "duplicate_of": None,
                "flag": None,
                "evidence": None,
                "_doi": str(doc.get("dataset_doi") or "").lower(),
                "duration_source": "dataset",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    _drop(df, df["dataset_id"].isin(NON_HUMAN) | df["_non_human_species"], "non_human")
    _drop(df, df["dataset_id"].isin(NOT_RECORDINGS), "not_a_recording")
    _drop(df, (df["duration_s"] <= 0) | (df["n_subjects"] <= 0), "no_duration")

    _resolve_duplicates(df, records)

    df = df.drop(columns=["_sfreq", "_non_human_species", "_doi"])
    return _classify_short(_finalise(df), records)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df["hours_total"] = df["duration_s"] / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["hours_per_subject"] = np.where(
            df["n_subjects"] > 0,
            df["hours_total"] / df["n_subjects"].clip(lower=1),
            np.nan,
        )
    short = (df["status"] == "kept") & (df["hours_per_subject"] * 60 < 1.0)
    df.loc[short & df["flag"].isna(), "flag"] = "short_duration"
    df.loc[
        ~short & df["flag"].isin(["short_duration", "epoched_short_duration"]), "flag"
    ] = None
    return df


def _classify_short(df: pd.DataFrame, records: RecordFetcher | None) -> pd.DataFrame:
    """Flag short datasets whose records are mostly epoched; nothing is dropped.

    ``RecordingType: epoched`` describes how the data are stored, not that the
    catalogue length counts one epoch (nm000355: 26 epochs back to back, 33.8 s,
    all counted), so these stay in the figure with ``epoched_short_duration``.
    """
    if records is None:
        return df
    for idx in df.index[(df["status"] == "kept") & (df["flag"] == "short_duration")]:
        try:
            recs = records(df.at[idx, "dataset_id"])
        except OSError:
            continue
        n_ep = sum(1 for r in recs if r.get("recording_type") == "epoched")
        if recs and 2 * n_ep >= len(recs):
            df.at[idx, "flag"] = "epoched_short_duration"
            df.at[idx, "evidence"] = f"{n_ep}/{len(recs)} records epoched"
    return df


def correct_durations_from_records(
    df: pd.DataFrame,
    record_sum: Callable[[str], float] | None = None,
    *,
    max_minutes_per_subject: float = 5.0,
    min_ratio: float = 1.5,
    records: RecordFetcher | None = None,
) -> pd.DataFrame:
    """Replace implausibly short dataset durations by the record-level sum.

    The dataset aggregate ``total_duration_s`` disagrees with the sum of its
    records' ``duration_seconds`` for a few datasets (e.g. ``on005106``:
    42 s in the aggregate, 5.5 h in the records). For every dataset under
    ``max_minutes_per_subject`` minutes per subject, the record sum replaces
    the aggregate when it is at least ``min_ratio`` times larger. The
    one-minute-per-subject cut is then re-applied.
    """
    record_sum = record_sum or _record_duration_sum
    df = df.copy()
    candidates = (df["status"] == "kept") & (
        df["hours_per_subject"] * 60 < max_minutes_per_subject
    )
    for idx in df.index[candidates]:
        try:
            total = float(record_sum(df.at[idx, "dataset_id"]))
        except OSError:
            continue
        if total >= min_ratio * df.at[idx, "duration_s"]:
            df.at[idx, "duration_s"] = total
            df.at[idx, "duration_source"] = "records"
    return _classify_short(_finalise(df), records)


def summarise_isohour(df: pd.DataFrame) -> dict[str, Any]:
    """Numbers quoted in the caption and the docs text."""
    kept = df[df["status"] == "kept"]
    hours = kept["hours_total"].sort_values(ascending=False)
    dense = kept[(kept["n_subjects"] >= 100) & (kept["hours_per_subject"] >= 10)]
    rho = pd.Series(np.log(kept["n_subjects"])).corr(
        pd.Series(np.log(kept["hours_per_subject"])), method="spearman"
    )
    return {
        "n_catalogue": int(len(df)),
        "n_kept": int(len(kept)),
        "dropped": df.loc[df["status"] != "kept", "status"].value_counts().to_dict(),
        "flagged": kept["flag"].value_counts().to_dict(),
        "hours_total": float(hours.sum()),
        "subjects_total": int(kept["n_subjects"].sum()),
        "median_subjects": float(kept["n_subjects"].median()),
        "median_hours_per_subject": float(kept["hours_per_subject"].median()),
        "share_hours_top10": float(hours.head(10).sum() / hours.sum()),
        "n_datasets_half_hours": int((hours.cumsum() < hours.sum() / 2).sum() + 1),
        "dense": dense.sort_values("hours_total", ascending=False)[
            ["dataset_id", "name", "n_subjects", "hours_per_subject"]
        ].to_dict("records"),
        "spearman_subjects_vs_hours": float(rho),
        "corrected_from_records": kept.loc[
            kept["duration_source"] == "records", "dataset_id"
        ].tolist(),
    }


# ---------------------------------------------------------------------------
# Interactive chart (docs)
# ---------------------------------------------------------------------------


def _iso_label_point(hours: float, x_range: tuple[float, float]) -> tuple[float, float]:
    """Label point of an iso-hour line: near the bottom edge, else the right edge."""
    y = 1.45
    x = hours / y
    if x > x_range[1] / 1.7:
        x = x_range[1] / 1.7
        y = hours / x
    return x, y


def generate_isohour_landscape(
    df: Any | None = None,
    out_html: str | Path = "dataset_isohour.html",
    *,
    table: pd.DataFrame | None = None,
    height: int = 680,
    correct_from_records: bool = True,
) -> Path:
    """Render the interactive iso-hour landscape for the docs."""
    del df  # signature parity with the other dataset-summary charts
    if table is None:
        table = build_isohour_table(fetch_catalogue(), records=fetch_records)
        if correct_from_records:
            table = correct_durations_from_records(table, records=fetch_records)
    kept = table[table["status"] == "kept"]
    stats = summarise_isohour(table)

    x_range = (1 / 3600, 400.0)
    y_range = (0.75, 4500.0)
    fig = go.Figure()

    xs = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 120)
    for hours in ISO_HOURS:
        ys = hours / xs
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(100,116,139,0.18)", width=9),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        lx, ly = _iso_label_point(hours, x_range)
        fig.add_annotation(
            x=np.log10(lx),
            y=np.log10(ly),
            text=f"{hours:,} h",
            showarrow=False,
            font=dict(size=12, color="#64748b"),
            bgcolor="rgba(255,255,255,0.85)",
        )

    n_dense = len(stats["dense"])
    # Shapes take data units on log axes (annotations take log10 units).
    fig.add_shape(
        type="rect",
        x0=10,
        x1=x_range[1],
        y0=100,
        y1=y_range[1],
        xref="x",
        yref="y",
        fillcolor="rgba(241,245,249,0.8)",
        line_width=0,
        layer="below",
    )
    fig.add_annotation(
        x=np.log10(x_range[1]),
        y=np.log10(y_range[1]),
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="right",
        text=f"≥100 subjects × ≥10 h<br>{n_dense} datasets",
        font=dict(size=12, color="#64748b"),
    )

    for mod, style in MODALITY_STYLE.items():
        sub = kept[kept["modality"] == mod]
        if sub.empty:
            continue
        custom = np.array(
            [
                [
                    r.name,
                    r.dataset_id,
                    get_dataset_url(r.dataset_id),
                    r.hours_total,
                    ", ".join(r.types) or "-",
                ]
                for r in sub.itertuples()
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["hours_per_subject"],
                y=sub["n_subjects"],
                mode="markers",
                name=f"{style['label']} ({len(sub)})",
                marker=dict(
                    color=style["color"],
                    symbol=style["symbol"],
                    size=8,
                    opacity=0.75,
                    line=dict(width=0.6, color="white"),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "%{y:,} subjects × %{x:.2f} h = %{customdata[3]:,.0f} h<br>"
                    "Type: %{customdata[4]}<br><i>Click to open the dataset page</i>"
                    "<extra></extra>"
                ),
            )
        )

    for ds, label in LANDMARKS.items():
        row = kept[kept["dataset_id"] == ds]
        if row.empty:
            continue
        r = row.iloc[0]
        fig.add_annotation(
            x=np.log10(r["hours_per_subject"]),
            y=np.log10(r["n_subjects"]),
            text=label,
            showarrow=False,
            xanchor="left",
            xshift=7,
            yshift=6,
            font=dict(size=11, color="#1f2937"),
        )

    fig.update_xaxes(
        type="log",
        range=[np.log10(x_range[0]), np.log10(x_range[1])],
        title_text="Recorded hours per subject",
        tickvals=[1 / 3600, 1 / 60, 1, 10, 100],
        ticktext=["1 s", "1 min", "1 h", "10 h", "100 h"],
        showgrid=False,
        ticks="outside",
    )
    fig.update_yaxes(
        type="log",
        range=[np.log10(y_range[0]), np.log10(y_range[1])],
        title_text="Subjects per dataset",
        tickvals=[1, 10, 100, 1000],
        ticktext=["1", "10", "100", "1,000"],
        showgrid=False,
        ticks="outside",
    )
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=70, r=30, t=60, b=60),
        title=dict(
            text=(
                f"{stats['n_kept']} datasets, {stats['subjects_total']:,} subjects, "
                f"{stats['hours_total']:,.0f} h. Grey diagonals: total hours per dataset."
            ),
            font=dict(size=14),
            x=0.01,
        ),
        legend=dict(x=0.01, xanchor="left", y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family="Inter, system-ui, Helvetica, Arial, sans-serif", size=14),
    )

    extra_html = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.getElementById('dataset-isohour');
    if (!plot || typeof plot.on !== 'function') { return; }
    plot.on('plotly_click', function(evt) {
        var p = evt && evt.points && evt.points[0];
        var url = p && p.customdata && p.customdata[2];
        if (url) { window.open(url, '_blank', 'noopener'); }
    });
});
</script>
"""
    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-isohour",
        height=height,
        extra_html=extra_html,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
