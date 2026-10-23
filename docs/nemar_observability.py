"""Recorded-hours counter from the NEMAR observability dashboard.

The catalog page's hours card used to sum the ``total_duration_s`` of every
EEGDash dataset document, which is whatever the digest managed to read (71
datasets had none on 2026-10-23). NEMAR measures hours per recording from its
Zarr index and publishes them, binned by channel count, at
https://dashboard.nemar.org/observability (JSON:
``/observability/api/snapshot``, section ``recordings``, ``channel_hours``).

``HOURS_VIEW`` reproduces one shareable view of that page,
``#hours=<modality>:<min_channels>``, with the page's own arithmetic
(``atLeast`` in the dashboard script): hours and recordings are the sum of the
bins with at least ``min_channels`` channels; datasets are counted from the
dataset peaks, so a dataset is counted once.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Any

OBSERVABILITY_URL = "https://dashboard.nemar.org/observability"
SNAPSHOT_URL = f"{OBSERVABILITY_URL}/api/snapshot"
# https://dashboard.nemar.org/observability#hours=eeg:16
HOURS_VIEW: tuple[str, int] = ("eeg", 16)
# Set to "0" to build offline with the EEGDash catalog sum.
ENV_TOGGLE = "EEGDASH_NEMAR_HOURS"


def hours_view_url(modality: str, min_channels: int) -> str:
    """Link to the dashboard view a counter was computed from."""
    return f"{OBSERVABILITY_URL}#hours={modality}:{min_channels}"


def fetch_snapshot(url: str = SNAPSHOT_URL, timeout: float = 20.0) -> dict | None:
    """Fetch the public observability snapshot; ``None`` on any failure."""
    if os.environ.get(ENV_TOGGLE, "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "eegdash-docs"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except Exception as exc:  # network, HTTP, JSON: fall back, never fail the build
        print(f"[nemar-hours] snapshot unavailable ({exc}); using the EEGDash sum")
        return None
    return data if isinstance(data, dict) else None


def _amount(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def hours_at_least(
    snapshot: dict | None, modality: str, min_channels: int
) -> dict[str, Any] | None:
    """Hours recorded in ``modality`` with ``min_channels`` channels or more.

    Returns ``{"hours", "recordings", "datasets", "generated_at"}``, or ``None``
    when the snapshot lacks the section or any bin is malformed (a partial sum
    would understate the archive without saying so).
    """
    if not isinstance(snapshot, dict) or min_channels < 1:
        return None
    sections = snapshot.get("sections")
    if not isinstance(sections, list):
        return None
    section = next(
        (s for s in sections if isinstance(s, dict) and s.get("key") == "recordings"),
        None,
    )
    channel_hours = (section or {}).get("channel_hours")
    if not isinstance(channel_hours, dict):
        return None
    modalities = channel_hours.get("modalities")
    if not isinstance(modalities, list):
        return None
    entry = next(
        (
            m
            for m in modalities
            if isinstance(m, dict)
            and str(m.get("modality", "")).lower() == modality.lower()
        ),
        None,
    )
    if entry is None:
        return None
    bins, peaks = entry.get("bins"), entry.get("dataset_peaks")
    if not isinstance(bins, list) or not isinstance(peaks, list) or not bins:
        return None

    hours = 0.0
    recordings = 0
    datasets = 0
    for b in bins:
        if not (
            isinstance(b, dict)
            and isinstance(b.get("channels"), int)
            and _amount(b.get("hours"))
            and isinstance(b.get("recordings"), int)
        ):
            return None
        if b["channels"] >= min_channels:
            hours += float(b["hours"])
            recordings += b["recordings"]
    for p in peaks:
        if not (
            isinstance(p, dict)
            and isinstance(p.get("channels"), int)
            and isinstance(p.get("datasets"), int)
        ):
            return None
        if p["channels"] >= min_channels:
            datasets += p["datasets"]

    return {
        "hours": hours,
        "recordings": recordings,
        "datasets": datasets,
        "generated_at": snapshot.get("generated_at"),
    }


def hours_counter(snapshot: dict | None = None) -> dict[str, Any] | None:
    """The catalog hours counter for ``HOURS_VIEW``, or ``None`` to fall back."""
    modality, min_channels = HOURS_VIEW
    if snapshot is None:
        snapshot = fetch_snapshot()
    result = hours_at_least(snapshot, modality, min_channels)
    if result is None:
        return None
    return {
        **result,
        "modality": modality,
        "min_channels": min_channels,
        "label": f"{modality.upper()} hours (≥{min_channels} channels)",
        "url": hours_view_url(modality, min_channels),
        "source": "nemar-observability",
    }
