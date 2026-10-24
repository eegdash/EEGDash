"""Recorded hours for the catalog counter, measured by NEMAR where it can.

The catalog page's hours card covers every EEGDash dataset. For each one it
takes the hours NEMAR measured from the recordings themselves
(``total_recording_duration`` in seconds on ``https://api.nemar.org/datasets``,
the same value as the sum of ``recording_duration`` in the dataset's
``records.json``), and falls back to the EEGDash catalog value
(``duration_hours_total``) for datasets NEMAR does not serve or has not
measured: the EEG2025 competition releases and OpenNeuro datasets without an
``on*`` mirror.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

CATALOG_URL = "https://api.nemar.org/datasets"
DASHBOARD_URL = "https://dashboard.nemar.org/observability"
# Set to "0" to build offline with the EEGDash catalog values only.
ENV_TOGGLE = "EEGDASH_NEMAR_HOURS"
_PAGE = 200
_MAX_DATASETS = 20_000


def nemar_id(dataset_id: str) -> str:
    """NEMAR id for an EEGDash dataset id (``ds`` OpenNeuro ids map to ``on``)."""
    if re.fullmatch(r"ds\d{6}", dataset_id):
        return "on" + dataset_id[2:]
    return dataset_id


def _seconds(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _get(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "eegdash-docs"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_catalog_pages(pages: list[dict]) -> dict[str, float] | None:
    """``{dataset_id: hours}`` from catalog pages; ``None`` if inconsistent.

    A page set that does not add up (count mismatch, duplicate id, bad
    duration) is rejected whole: a partial map would silently move datasets
    to the fallback.
    """
    hours: dict[str, float] = {}
    total = None
    for page in pages:
        rows = page.get("datasets") if isinstance(page, dict) else None
        count = page.get("total_count") if isinstance(page, dict) else None
        if not isinstance(rows, list) or not isinstance(count, int):
            return None
        if total is not None and count != total:
            return None
        total = count
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("dataset_id"), str):
                return None
            dataset_id = row["dataset_id"]
            if dataset_id in hours:
                return None
            seconds = row.get("total_recording_duration")
            if seconds is None:
                hours[dataset_id] = 0.0
            elif _seconds(seconds):
                hours[dataset_id] = float(seconds) / 3600
            else:
                return None
    if total is None or len(hours) != total:
        return None
    return hours


def fetch_catalog_hours(
    url: str = CATALOG_URL, timeout: float = 30.0
) -> dict[str, float] | None:
    """Measured hours per public NEMAR dataset; ``None`` on any failure."""
    if os.environ.get(ENV_TOGGLE, "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    pages: list[dict] = []
    try:
        offset = 0
        while offset < _MAX_DATASETS:
            query = urllib.parse.urlencode({"limit": _PAGE, "offset": offset})
            page = _get(f"{url}?{query}", timeout)
            pages.append(page)
            rows = page.get("datasets") if isinstance(page, dict) else None
            if not isinstance(rows, list) or not rows:
                break
            offset += len(rows)
            if offset >= page.get("total_count", 0):
                break
    except Exception as exc:  # network, HTTP, JSON: fall back, never fail the build
        print(f"[nemar-hours] catalog unavailable ({exc}); using EEGDash values")
        return None
    hours = parse_catalog_pages(pages)
    if hours is None:
        print("[nemar-hours] inconsistent NEMAR catalog; using EEGDash values")
    return hours


def combined_hours(
    df_raw: pd.DataFrame, nemar_hours: dict[str, float] | None
) -> dict[str, Any]:
    """Total hours over every EEGDash dataset, NEMAR-measured where possible.

    Returns the total plus how many datasets each source supplied, so the
    build log and ``summary_stats.json`` say where the number came from.
    """
    nemar_hours = nemar_hours or {}
    ids = (
        df_raw["dataset"]
        if "dataset" in df_raw.columns
        else df_raw.get("dataset_id", pd.Series(dtype=str))
    )
    if "duration_hours_total" in df_raw.columns:
        own = pd.to_numeric(df_raw["duration_hours_total"], errors="coerce")
    elif "total_duration_s" in df_raw.columns:
        own = pd.to_numeric(df_raw["total_duration_s"], errors="coerce") / 3600
    else:
        own = pd.Series([float("nan")] * len(df_raw), index=df_raw.index)

    total = 0.0
    counts = {"nemar": 0, "eegdash": 0, "none": 0}
    for dataset_id, eegdash_value in zip(ids.astype(str), own):
        measured = nemar_hours.get(nemar_id(dataset_id.strip()))
        if measured:
            total += measured
            counts["nemar"] += 1
        elif pd.notna(eegdash_value) and eegdash_value > 0:
            total += float(eegdash_value)
            counts["eegdash"] += 1
        else:
            counts["none"] += 1
    return {
        "hours": total,
        "datasets_from_nemar": counts["nemar"],
        "datasets_from_eegdash": counts["eegdash"],
        "datasets_without_hours": counts["none"],
    }
