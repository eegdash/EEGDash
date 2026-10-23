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
    One study present more than once. Three rules, applied in order:
    (1) the ``EEG2025r*`` entries are BDF re-encodings (and 20-subject
    ``mini`` subsets) of the HBN releases ``on005505``-``on005516``;
    (2) entries with the same normalised name are versions or mirrors of one
    study, and the most complete one (most subjects, then most hours) is kept;
    (3) entries with the same subject count, the same sampling-rate profile
    and a total duration equal to within one second hold the same recordings.
    ``duplicate_of`` names the entry that was kept.
``duration_unreliable``
    Under one minute per subject after correction. Most are epoched EEGLAB
    files whose ``ntimes`` counts one epoch, so the catalogue duration is the
    length of a single trial.

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


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Mark still-kept rows selected by ``mask`` as dropped for ``reason``."""
    df.loc[mask & (df["status"] == "kept"), "status"] = reason


def build_isohour_table(docs: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Turn ``/datasets/summary`` documents into the cleaned landscape table.

    Returns one row per catalogue dataset. ``status == "kept"`` marks the
    rows to plot; every other row carries the reason it was dropped.
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
                "duration_source": "dataset",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    _drop(df, df["dataset_id"].isin(NON_HUMAN) | df["_non_human_species"], "non_human")
    _drop(df, df["dataset_id"].isin(NOT_RECORDINGS), "not_a_recording")
    _drop(df, (df["duration_s"] <= 0) | (df["n_subjects"] <= 0), "no_duration")

    # Rule 1: HBN BDF re-encodings and their mini subsets.
    hbn = df["dataset_id"].str.match(_HBN_COPY)
    _drop(df, hbn, "duplicate")
    df.loc[hbn & (df["status"] == "duplicate"), "duplicate_of"] = "on005505-on005516"

    # Rule 2: same normalised name -> keep the most complete entry.
    kept = df[df["status"] == "kept"].copy()
    kept["_key"] = kept["name"].map(_norm_name)
    for key, group in kept[kept["_key"].str.len() >= 4].groupby("_key"):
        if len(group) < 2:
            continue
        order = group.sort_values(
            ["n_subjects", "duration_s", "dataset_id"], ascending=[False, False, True]
        )
        winner = order.index[0]
        losers = order.index[1:]
        df.loc[losers, "status"] = "duplicate"
        df.loc[losers, "duplicate_of"] = df.at[winner, "dataset_id"]

    # Rule 3: same subjects, same sampling-rate profile, same duration (+-1 s).
    kept = df[df["status"] == "kept"].sort_values("dataset_id")
    seen: dict[tuple, list[tuple[float, str]]] = {}
    for idx, row in kept.iterrows():
        key = (row["n_subjects"], row["_sfreq"])
        match = next(
            (ds for dur, ds in seen.get(key, []) if abs(dur - row["duration_s"]) < 1.0),
            None,
        )
        if match is not None and row["_sfreq"]:
            df.at[idx, "status"] = "duplicate"
            df.at[idx, "duplicate_of"] = match
        else:
            seen.setdefault(key, []).append((row["duration_s"], row["dataset_id"]))

    df = df.drop(columns=["_sfreq", "_non_human_species"])
    return _finalise(df)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df["hours_total"] = df["duration_s"] / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        df["hours_per_subject"] = np.where(
            df["n_subjects"] > 0,
            df["hours_total"] / df["n_subjects"].clip(lower=1),
            np.nan,
        )
    short = (df["status"] == "kept") & (df["hours_per_subject"] * 60 < 1.0)
    df.loc[short, "status"] = "duration_unreliable"
    return df


def correct_durations_from_records(
    df: pd.DataFrame,
    record_sum: Callable[[str], float] | None = None,
    *,
    max_minutes_per_subject: float = 5.0,
    min_ratio: float = 1.5,
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
    candidates = df["status"].isin(["kept", "duration_unreliable"]) & (
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
    df.loc[df["status"] == "duration_unreliable", "status"] = "kept"
    return _finalise(df)


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
        table = build_isohour_table(fetch_catalogue())
        if correct_from_records:
            table = correct_durations_from_records(table)
    kept = table[table["status"] == "kept"]
    stats = summarise_isohour(table)

    x_range = (1 / 60, 400.0)
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
        tickvals=[1 / 60, 10 / 60, 1, 10, 100],
        ticktext=["1 min", "10 min", "1 h", "10 h", "100 h"],
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
