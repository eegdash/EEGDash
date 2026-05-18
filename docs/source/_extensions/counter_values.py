"""Sphinx extension: substitute dataset-counter placeholders into RST source.

The ``dataset_summary`` page uses placeholder tokens like
``|datasets_total|`` so the numeric counters render even when JavaScript
is disabled. The values come from
``_static/dataset_generated/summary_stats.json`` (produced by
``prepare_summary_tables.py``) with a CSV fallback. This extension wires
a ``source-read`` hook that replaces the placeholders before Sphinx
parses the document.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path


def _split_tokens(value: str | None) -> set[str]:
    tokens: set[str] = set()
    if not value:
        return tokens
    for part in value.split(","):
        cleaned = part.strip()
        if cleaned:
            tokens.add(cleaned)
    return tokens


def _compute_dataset_counter_defaults() -> dict[str, int]:
    # 1. Try to load from the generated JSON (produced by prepare_summary_tables.py)
    # Note: prepare_summary_tables.py runs before sphinx-build in the Makefile
    stats_path = (
        Path(__file__).resolve().parent.parent
        / "_static"
        / "dataset_generated"
        / "summary_stats.json"
    )
    if stats_path.exists():
        try:
            with open(stats_path, "r") as f:
                data = json.load(f)
            return {
                "datasets": data.get("datasets_total", 0),
                "subjects": data.get("subjects_total", 0),
                "recording": data.get("recording_total", 0),
                "duration_hours": data.get("duration_hours", 0),
                "modalities": data.get("modalities_total", 0),
                "sources": data.get("sources_total", 0),
            }
        except Exception:
            pass

    # 2. Fallback to legacy CSV logic
    csv_path = Path(importlib.import_module("eegdash.dataset").__file__).with_name(
        "dataset_summary.csv"
    )
    if not csv_path.exists():
        return {}

    dataset_ids: set[str] = set()
    modalities: set[str] = set()
    sources: set[str] = set()
    subject_total = 0

    with csv_path.open(encoding="utf-8") as handle:
        filtered = (
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered)
        for row in reader:
            dataset = (row.get("dataset") or row.get("Dataset") or "").strip()
            if dataset:
                dataset_ids.add(dataset)

            try:
                subject_total += int(float(row.get("n_subjects", "0") or 0))
            except (TypeError, ValueError):
                pass

            modalities.update(_split_tokens(row.get("record_modality")))
            sources.add((row.get("source") or "unknown").strip())

    return {
        "datasets": len(dataset_ids),
        "subjects": subject_total,
        "recording": 0,
        "modalities": len(modalities),
        "sources": len(sources),
    }


_DATASET_COUNTER_DEFAULTS = _compute_dataset_counter_defaults()


def _format_counter(key: str) -> str:
    value = _DATASET_COUNTER_DEFAULTS.get(key, 0)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"
    return str(value)


_DATASET_COUNTER_PLACEHOLDERS = {
    "|datasets_total|": _format_counter("datasets"),
    "|subjects_total|": _format_counter("subjects"),
    "|recording_total|": _format_counter("recording"),
    "|duration_hours|": _format_counter("duration_hours"),
    "|modalities_total|": _format_counter("modalities"),
    "|sources_total|": _format_counter("sources"),
}


def _inject_counter_values(app, docname, source) -> None:
    if docname != "dataset_summary":
        return

    text = source[0]
    for token, value in _DATASET_COUNTER_PLACEHOLDERS.items():
        text = text.replace(token, value)
    source[0] = text


def setup(app) -> dict:
    app.connect("source-read", _inject_counter_values)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
