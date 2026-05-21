from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Allow execution as a script or module
    from .colours import RECORDING_MODALITY_COLORS
    from .utils import (
        RECORDING_MODALITY_MAP,
        build_and_export_html,
        get_dataset_url,
        read_dataset_csv,
        safe_int,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import RECORDING_MODALITY_COLORS  # type: ignore
    from utils import (  # type: ignore
        RECORDING_MODALITY_MAP,
        build_and_export_html,
        get_dataset_url,
        read_dataset_csv,
        safe_int,
    )

__all__ = ["generate_api_study_explorer"]

_UNKNOWN_TOKENS = {"", "nan", "none", "null", "unknown", "nothing"}
_EXCLUDED_DATASETS = {"test", "ds003380"}
_FACET_GROUPS = (
    ("recording", "Recording"),
    ("modality", "Experiment"),
    ("type", "Type"),
    ("pathology", "Population"),
)
_RECORDING_ORDER = ["EEG", "MEG", "iEEG", "fNIRS", "EMG", "fMRI", "MRI", "ECG"]
_VIEWS = ("volume", "matrix", "both")
_FACET_SEPARATOR = "::"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in _UNKNOWN_TOKENS else text


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        text = _clean_text(value)
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            raw_values = parsed
        else:
            for sep in ("|", ";", "/"):
                text = text.replace(sep, ",")
            raw_values = text.split(",")

    values: list[str] = []
    for item in raw_values:
        cleaned = _clean_text(item)
        if cleaned:
            values.append(" ".join(cleaned.replace("_", " ").split()))
    return list(dict.fromkeys(values))


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if _split_values(value):
            return value
    return ""


def _normalize_recording(value: str) -> str:
    lowered = value.strip().lower()
    canonical = RECORDING_MODALITY_MAP.get(lowered)
    if canonical:
        return canonical
    return value.strip().upper() if len(value) <= 4 else value


def _normalize_label(value: str) -> str:
    cleaned = " ".join(value.replace("_", " ").split())
    lowered = cleaned.lower()
    aliases = {
        "rest": "Resting State",
        "resting state": "Resting State",
        "resting-state": "Resting State",
        "decision making": "Decision-making",
        "clinical": "Clinical",
        "intervention": "Intervention",
        "healthy controls": "Healthy",
        "control": "Healthy",
    }
    if lowered in aliases:
        return aliases[lowered]
    if cleaned.isupper() and len(cleaned) <= 5:
        return cleaned
    return cleaned[:1].upper() + cleaned[1:]


def _canonical_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [_clean_text(v) for v in raw if _clean_text(v)]
    text = _clean_text(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [_clean_text(v) for v in parsed if _clean_text(v)]
    return [v for v in _split_values(text) if v]


def _dataset_title(row: pd.Series, dataset_id: str) -> str:
    for col in ("dataset_title", "computed_title", "name", "title"):
        text = _clean_text(row.get(col))
        if text:
            return text
    return dataset_id


def _count_items(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return len(_split_values(value))


def _format_number(value: int | float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if isinstance(value, float) and value < 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{int(round(value)):,}"


def _format_hours(value: float | None) -> str:
    if value is None or value <= 0:
        return "-"
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in value]
    return "-".join(part for part in "".join(chars).split("-") if part)


def _record_from_row(row: pd.Series) -> dict[str, Any] | None:
    dataset_id = _clean_text(row.get("dataset")) or _clean_text(row.get("dataset_id"))
    if not dataset_id or dataset_id.lower() in _EXCLUDED_DATASETS:
        return None

    demographics = _as_mapping(row.get("demographics"))
    tags = _as_mapping(row.get("tags"))
    clinical = _as_mapping(row.get("clinical"))
    paradigm = _as_mapping(row.get("paradigm"))

    subjects = safe_int(row.get("n_subjects"), default=0) or safe_int(
        demographics.get("subjects_count"), default=0
    )
    records = safe_int(row.get("n_records"), default=0) or safe_int(
        row.get("total_files"), default=0
    )
    tasks = safe_int(row.get("n_tasks"), default=0) or _count_items(row.get("tasks"))
    sessions = safe_int(row.get("n_sessions"), default=0) or _count_items(
        row.get("sessions")
    )
    hours = _to_float(row.get("duration_hours_total"))
    if hours is None and _to_float(row.get("total_duration_s")) is not None:
        hours = (_to_float(row.get("total_duration_s")) or 0) / 3600
    size_bytes = safe_int(row.get("size_bytes"), default=0)

    recording_raw = _first_nonempty(
        row.get("record_modality"),
        row.get("recording_modality"),
        row.get("datatypes"),
    )
    recording = [_normalize_recording(v) for v in _split_values(recording_raw)]
    if not recording:
        recording = ["Unknown"]

    modality_raw = _first_nonempty(
        row.get("modality of exp"),
        tags.get("modality"),
        row.get("experimental_modalities"),
        paradigm.get("modality"),
    )
    modality = [_normalize_label(v) for v in _split_values(modality_raw)]

    type_raw = _first_nonempty(
        row.get("type of exp"),
        tags.get("type"),
        paradigm.get("cognitive_domain"),
    )
    exp_type = [_normalize_label(v) for v in _split_values(type_raw)]

    pathology_raw = _first_nonempty(row.get("Type Subject"), tags.get("pathology"))
    if not _split_values(pathology_raw):
        is_clinical = clinical.get("is_clinical")
        if is_clinical:
            pathology_raw = clinical.get("purpose") or "Clinical"
        elif is_clinical is False:
            pathology_raw = "Healthy"
    pathology = [_normalize_label(v) for v in _split_values(pathology_raw)]

    source = _clean_text(row.get("source")) or "unknown"
    license_text = _clean_text(row.get("license"))
    aliases = _canonical_names(row.get("canonical_name"))
    author_year = _clean_text(row.get("author_year"))
    if author_year and author_year not in aliases:
        aliases.insert(0, author_year)

    facets = {
        "recording": recording,
        "modality": modality,
        "type": exp_type,
        "pathology": pathology,
        "source": [source],
    }
    facet_tokens = [
        f"{group}{_FACET_SEPARATOR}{label}"
        for group, labels in facets.items()
        for label in labels
        if label
    ]

    title = _dataset_title(row, dataset_id)
    search_text = " ".join(
        [
            dataset_id,
            title,
            source,
            license_text,
            " ".join(aliases),
            " ".join(recording),
            " ".join(modality),
            " ".join(exp_type),
            " ".join(pathology),
        ]
    ).lower()

    hours_per_subject = (
        hours / subjects if hours is not None and hours > 0 and subjects > 0 else None
    )
    return {
        "id": dataset_id,
        "title": title,
        "url": get_dataset_url(dataset_id),
        "source": source,
        "license": license_text,
        "aliases": aliases,
        "subjects": subjects,
        "records": records,
        "tasks": tasks,
        "sessions": sessions,
        "hours": hours,
        "hours_per_subject": hours_per_subject,
        "size_bytes": size_bytes,
        "facets": facets,
        "facet_tokens": facet_tokens,
        "search": search_text,
    }


def _prepare_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prepared = _record_from_row(row)
        if prepared is not None:
            records.append(prepared)
    records.sort(key=lambda item: item["id"].lower())
    return records


def _facet_counts(records: list[dict[str, Any]]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {group: Counter() for group, _ in _FACET_GROUPS}
    counts["source"] = Counter()
    for record in records:
        for group, labels in record["facets"].items():
            counts.setdefault(group, Counter()).update(labels)
    return counts


def _order_recording(labels: Counter) -> list[str]:
    ordered = [label for label in _RECORDING_ORDER if label in labels]
    ordered.extend(sorted(set(labels) - set(ordered)))
    return ordered


def _selected_columns(counts: dict[str, Counter]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for group, _ in _FACET_GROUPS:
        group_counts = counts.get(group, Counter())
        if group == "recording":
            selected[group] = _order_recording(group_counts)[:10]
            continue
        ordered = sorted(group_counts, key=lambda label: (-group_counts[label], label))
        selected[group] = ordered[:12]
    return selected


def _log_ticks(min_value: float, max_value: float) -> list[float]:
    if min_value <= 0 or max_value <= 0:
        return []
    start = math.floor(math.log10(min_value))
    stop = math.ceil(math.log10(max_value))
    ticks: list[float] = []
    for power in range(start, stop + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10**power
            if min_value <= value <= max_value:
                ticks.append(value)
    return ticks


def _axis_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g}K"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def _scatter_svg(records: list[dict[str, Any]]) -> str:
    points = [
        record
        for record in records
        if record["hours_per_subject"] is not None and record["subjects"] > 0
    ]
    if not points:
        return (
            '<div class="api-empty" role="status">'
            "No duration metadata is available for the current API payload."
            "</div>"
        )

    width, height = 1460, 760
    left, right, top, bottom = 92, 38, 42, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_values = [p["hours_per_subject"] for p in points if p["hours_per_subject"]]
    y_values = [p["subjects"] for p in points if p["subjects"] > 0]
    x_min = max(min(x_values) * 0.75, 0.01)
    x_max = max(x_values) * 1.35
    y_min = max(min(y_values) * 0.75, 1)
    y_max = max(y_values) * 1.35
    log_x_min, log_x_max = math.log10(x_min), math.log10(x_max)
    log_y_min, log_y_max = math.log10(y_min), math.log10(y_max)

    def x_pos(value: float) -> float:
        return left + (math.log10(value) - log_x_min) / (log_x_max - log_x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (log_y_max - math.log10(value)) / (log_y_max - log_y_min) * plot_h

    max_hours = max((p["hours"] or 0) for p in points) or 1
    lines: list[str] = [
        (
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            'aria-label="Dataset volume scatterplot">'
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" class="api-svg-bg"/>',
    ]
    for tick in _log_ticks(x_min, x_max):
        x = x_pos(tick)
        lines.append(
            f'<line class="api-grid" x1="{x:.1f}" y1="{top}" '
            f'x2="{x:.1f}" y2="{height - bottom}"/>'
        )
        lines.append(
            f'<text class="api-axis-tick" x="{x:.1f}" y="{height - 42}" '
            f'text-anchor="middle">{html.escape(_axis_label(tick))}</text>'
        )
    for tick in _log_ticks(y_min, y_max):
        y = y_pos(tick)
        lines.append(
            f'<line class="api-grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{width - right}" y2="{y:.1f}"/>'
        )
        lines.append(
            f'<text class="api-axis-tick" x="{left - 16}" y="{y + 4:.1f}" '
            f'text-anchor="end">{html.escape(_axis_label(tick))}</text>'
        )

    lines.extend(
        [
            f'<line class="api-axis" x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="api-axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}"/>',
            f'<text class="api-axis-label" x="{width / 2:.1f}" y="{height - 12}" '
            'text-anchor="middle">Known recording hours per subject</text>',
            (
                f'<text class="api-axis-label" transform="translate(24 '
                f'{height / 2:.1f}) rotate(-90)" text-anchor="middle">Subjects</text>'
            ),
        ]
    )

    for record in points:
        hours = record["hours"] or 0
        radius = 4 + 15 * math.log1p(hours) / math.log1p(max_hours)
        recording = record["facets"]["recording"][0]
        color = RECORDING_MODALITY_COLORS.get(recording, "#64748b")
        attrs = _data_attrs(record)
        lines.append(
            f'<circle class="api-study-point rec-{_slug(recording)}" '
            f'cx="{x_pos(record["hours_per_subject"]):.1f}" '
            f'cy="{y_pos(record["subjects"]):.1f}" r="{radius:.1f}" '
            f'fill="{html.escape(color)}" {attrs} tabindex="0"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _data_attrs(record: dict[str, Any]) -> str:
    alias_text = ", ".join(record["aliases"])
    facet_text = "|".join(record["facet_tokens"])
    attrs = {
        "data-dataset": record["id"],
        "data-name": record["id"],
        "data-description": record["title"],
        "data-url": record["url"],
        "data-aliases": alias_text,
        "data-facets": facet_text,
        "data-search": record["search"],
        "data-subjects": str(record["subjects"]),
        "data-records": str(record["records"]),
        "data-hours": "" if record["hours"] is None else f"{record['hours']:.6f}",
    }
    return " ".join(
        f'{name}="{html.escape(value, quote=True)}"' for name, value in attrs.items()
    )


def _option_html(counts: dict[str, Counter]) -> str:
    groups = [("source", "Source"), *_FACET_GROUPS]
    chunks = ['<option value="">Choose an API facet...</option>']
    for group, label in groups:
        group_counts = counts.get(group, Counter())
        if not group_counts:
            continue
        chunks.append(f'<optgroup label="{html.escape(label)}">')
        ordered = sorted(group_counts, key=lambda value: (-group_counts[value], value))
        for value in ordered:
            token = f"{group}{_FACET_SEPARATOR}{value}"
            text = f"{value} ({group_counts[value]:,})"
            chunks.append(
                f'<option value="{html.escape(token, quote=True)}">'
                f"{html.escape(text)}</option>"
            )
        chunks.append("</optgroup>")
    return "\n".join(chunks)


def _legend_html(counts: dict[str, Counter]) -> str:
    recording_counts = counts.get("recording", Counter())
    parts = []
    for label in _order_recording(recording_counts):
        color = RECORDING_MODALITY_COLORS.get(label, "#64748b")
        token = f"recording{_FACET_SEPARATOR}{label}"
        parts.append(
            '<button class="api-legend-chip" type="button" '
            f'data-filter-token="{html.escape(token, quote=True)}">'
            f'<i style="background:{html.escape(color)}"></i>'
            f"<span>{html.escape(label)}</span>"
            f"<b>{recording_counts[label]:,}</b>"
            "</button>"
        )
    return "\n".join(parts)


def _matrix_html(records: list[dict[str, Any]], selected: dict[str, list[str]]) -> str:
    facet_columns = [
        (group, label, group_label)
        for group, group_label in _FACET_GROUPS
        for label in selected[group]
    ]
    head_1 = (
        "<tr>"
        '<th class="api-study-col" rowspan="2">Dataset</th>'
        '<th rowspan="2">Source</th>'
        '<th rowspan="2">Rec.</th>'
        '<th rowspan="2">Subjects</th>'
        '<th rowspan="2">Records</th>'
        '<th rowspan="2">Hours</th>'
    )
    for group, group_label in _FACET_GROUPS:
        span = len(selected[group])
        if span:
            head_1 += f'<th colspan="{span}">{html.escape(group_label)}</th>'
    head_1 += "</tr>"
    head_2 = "<tr>"
    for group, label, group_label in facet_columns:
        token = f"{group}{_FACET_SEPARATOR}{label}"
        head_2 += (
            f'<th class="api-facet-col api-{html.escape(group)}-col" '
            f'data-filter-token="{html.escape(token, quote=True)}">'
            f'<button type="button" title="Filter {html.escape(group_label)}: '
            f'{html.escape(label, quote=True)}">{html.escape(label)}</button></th>'
        )
    head_2 += "</tr>"

    body = []
    for record in records:
        attrs = _data_attrs(record)
        recording = ", ".join(record["facets"]["recording"])
        facet_sets = {
            group: set(record["facets"].get(group, [])) for group, _ in _FACET_GROUPS
        }
        row = [
            f'<tr class="api-study-row rec-{_slug(recording)}" {attrs}>',
            (
                '<th class="api-study-col api-study-name">'
                f'<a href="{html.escape(record["url"], quote=True)}">'
                f'{html.escape(record["id"])}</a></th>'
            ),
            f'<td>{html.escape(record["source"])}</td>',
            f"<td>{html.escape(recording)}</td>",
            f'<td class="num">{record["subjects"]:,}</td>',
            f'<td class="num">{record["records"]:,}</td>',
            f'<td class="num">{html.escape(_format_hours(record["hours"]))}</td>',
        ]
        for group, label, _ in facet_columns:
            tick = label in facet_sets.get(group, set())
            row.append('<td class="tick">+</td>' if tick else "<td></td>")
        row.append("</tr>")
        body.append("".join(row))

    return (
        '<div class="api-table-wrap">'
        '<table class="api-matrix-table">'
        f"<thead>{head_1}{head_2}</thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
        "</div>"
    )


def _build_html(
    records: list[dict[str, Any]],
    *,
    api_url: str,
    database: str,
    view: str,
    root_id: str,
) -> str:
    total_subjects = sum(record["subjects"] for record in records)
    total_records = sum(record["records"] for record in records)
    known_hours = sum(record["hours"] or 0 for record in records)
    duration_count = sum(1 for record in records if record["hours_per_subject"])
    source_count = len({record["source"] for record in records})
    fields = "dataset_id, demographics, recording_modality, tags, tasks, total_files, total_duration_s"
    counts = _facet_counts(records)
    selected = _selected_columns(counts)

    panels = []
    if view in {"volume", "both"}:
        panels.append(
            f"""
  <section class="api-panel">
    <header>
      <h4>Dataset Volume From API Metadata</h4>
      <p>Scatter points use API summary fields only: {html.escape(fields)}.</p>
    </header>
    <div class="api-legend" aria-label="Recording modality legend">{_legend_html(counts)}</div>
    <div class="api-scatter-wrap">{_scatter_svg(records)}</div>
  </section>"""
        )
    if view in {"matrix", "both"}:
        panels.append(
            f"""
  <section class="api-panel">
    <header>
      <h4>Metadata Coverage Matrix</h4>
      <p>Columns are API facets, not inferred event classes. Click a column header or legend chip to filter.</p>
    </header>
    {_matrix_html(records, selected)}
  </section>"""
        )

    return f"""
<div id="{html.escape(root_id, quote=True)}" class="eegdash-api-volume-explorer">
  <div class="api-study-tooltip" hidden></div>
  <div class="api-contract-rail">
    <span>API source</span>
    <code>{html.escape(api_url.rstrip('/'))}/{html.escape(database)}/datasets/chart-data</code>
    <span>{len(records):,} datasets</span>
    <span>{duration_count:,} with duration metadata</span>
  </div>
  <div class="api-controls" aria-label="Dataset volume filters">
    <label class="api-select-field">
      <span>Facet</span>
      <select class="api-facet-filter">{_option_html(counts)}</select>
    </label>
    <label class="api-search-field">
      <span>Search</span>
      <input class="api-text-filter" type="search" placeholder="Dataset, task, source, modality">
    </label>
    <span class="api-filter-status">All API facets</span>
    <button class="api-clear-filters" type="button" disabled>Clear</button>
  </div>
  <div class="api-summary" aria-live="polite">
    <div><strong class="api-summary-datasets">{len(records):,}</strong><span>datasets</span></div>
    <div><strong class="api-summary-subjects">{total_subjects:,}</strong><span>subjects</span></div>
    <div><strong class="api-summary-records">{total_records:,}</strong><span>records</span></div>
    <div><strong class="api-summary-hours">{_format_number(known_hours)}</strong><span>known hours</span></div>
    <div><strong>{source_count:,}</strong><span>sources</span></div>
  </div>
{''.join(panels)}
</div>
"""


_STYLE = """
<style>
.eegdash-api-volume-explorer {
  --api-ink: #111827;
  --api-muted: #64748b;
  --api-rule: rgba(15, 23, 42, 0.12);
  --api-soft: #f8fafc;
  --api-paper: #ffffff;
  --api-mark: #2563eb;
  color: var(--api-ink);
  font-family: var(--pst-font-family-base, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
  width: 100%;
}
html[data-theme="dark"] .eegdash-api-volume-explorer {
  --api-ink: #e5e7eb;
  --api-muted: #94a3b8;
  --api-rule: rgba(148, 163, 184, 0.26);
  --api-soft: rgba(15, 23, 42, 0.58);
  --api-paper: rgba(15, 23, 42, 0.12);
  --api-mark: #60a5fa;
}
.eegdash-api-volume-explorer .api-contract-rail {
  align-items: center;
  border-block: 1px solid var(--api-rule);
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  padding: 10px 0;
  font-size: 12px;
  color: var(--api-muted);
}
.eegdash-api-volume-explorer code {
  color: var(--api-mark);
  font-size: 12px;
  white-space: normal;
}
.eegdash-api-volume-explorer .api-controls {
  align-items: center;
  background: var(--api-paper);
  border-bottom: 1px solid var(--api-rule);
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 12px 0;
  position: sticky;
  top: 0;
  z-index: 5;
}
.eegdash-api-volume-explorer label {
  color: var(--api-muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.eegdash-api-volume-explorer select,
.eegdash-api-volume-explorer input,
.eegdash-api-volume-explorer button {
  border: 1px solid var(--api-rule);
  border-radius: 6px;
  color: var(--api-ink);
  background: var(--api-paper);
  font: inherit;
  min-height: 34px;
  padding: 6px 9px;
}
.eegdash-api-volume-explorer button {
  cursor: pointer;
}
.eegdash-api-volume-explorer button:disabled {
  color: var(--api-muted);
  cursor: default;
}
.eegdash-api-volume-explorer .api-select-field,
.eegdash-api-volume-explorer .api-search-field {
  align-items: center;
  display: inline-flex;
  gap: 8px;
  text-transform: uppercase;
}
.eegdash-api-volume-explorer .api-select-field select {
  min-width: min(280px, 64vw);
}
.eegdash-api-volume-explorer .api-search-field input {
  min-width: min(320px, 68vw);
}
.eegdash-api-volume-explorer .api-filter-status {
  background: var(--api-soft);
  border: 1px solid var(--api-rule);
  border-radius: 999px;
  color: var(--api-muted);
  font-size: 12px;
  padding: 7px 11px;
}
.eegdash-api-volume-explorer .api-summary {
  display: grid;
  grid-template-columns: repeat(5, minmax(110px, 1fr));
  border-bottom: 1px solid var(--api-rule);
}
.eegdash-api-volume-explorer .api-summary div {
  border-right: 1px solid var(--api-rule);
  padding: 14px 12px;
}
.eegdash-api-volume-explorer .api-summary div:last-child {
  border-right: 0;
}
.eegdash-api-volume-explorer .api-summary strong {
  display: block;
  font-size: clamp(20px, 3vw, 31px);
  line-height: 1.05;
}
.eegdash-api-volume-explorer .api-summary span {
  color: var(--api-muted);
  font-size: 12px;
  text-transform: uppercase;
}
.eegdash-api-volume-explorer .api-panel {
  margin-top: 20px;
}
.eegdash-api-volume-explorer .api-panel header {
  align-items: baseline;
  border-bottom: 1px solid var(--api-rule);
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  justify-content: space-between;
  padding-bottom: 8px;
}
.eegdash-api-volume-explorer .api-panel h4 {
  font-size: 16px;
  margin: 0;
}
.eegdash-api-volume-explorer .api-panel p {
  color: var(--api-muted);
  font-size: 13px;
  margin: 0;
}
.eegdash-api-volume-explorer .api-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 0;
}
.eegdash-api-volume-explorer .api-legend-chip {
  align-items: center;
  display: inline-flex;
  gap: 7px;
  min-height: 30px;
}
.eegdash-api-volume-explorer .api-legend-chip i {
  border: 1px solid rgba(15, 23, 42, 0.28);
  border-radius: 999px;
  display: inline-block;
  height: 11px;
  width: 11px;
}
.eegdash-api-volume-explorer .api-legend-chip b {
  color: var(--api-muted);
  font-size: 11px;
  font-weight: 600;
}
.eegdash-api-volume-explorer .api-scatter-wrap,
.eegdash-api-volume-explorer .api-table-wrap {
  border: 1px solid var(--api-rule);
  overflow: auto;
}
.eegdash-api-volume-explorer svg {
  display: block;
  min-width: 980px;
  width: 100%;
}
.eegdash-api-volume-explorer .api-svg-bg {
  fill: var(--api-paper);
}
.eegdash-api-volume-explorer .api-grid {
  stroke: var(--api-rule);
  stroke-width: 1;
}
.eegdash-api-volume-explorer .api-axis {
  stroke: var(--api-ink);
  stroke-width: 1.4;
}
.eegdash-api-volume-explorer .api-axis-label {
  fill: var(--api-ink);
  font-size: 15px;
  font-weight: 700;
}
.eegdash-api-volume-explorer .api-axis-tick {
  fill: var(--api-muted);
  font-size: 12px;
}
.eegdash-api-volume-explorer .api-study-point {
  cursor: pointer;
  opacity: 0.72;
  stroke: rgba(255, 255, 255, 0.82);
  stroke-width: 1.1;
}
.eegdash-api-volume-explorer .api-study-point:hover,
.eegdash-api-volume-explorer .api-study-point:focus {
  opacity: 1;
  stroke: var(--api-ink);
  stroke-width: 1.5;
}
.eegdash-api-volume-explorer .api-matrix-table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 12px;
  min-width: 1380px;
  width: 100%;
}
.eegdash-api-volume-explorer th,
.eegdash-api-volume-explorer td {
  border-bottom: 1px solid var(--api-rule);
  border-right: 1px solid var(--api-rule);
  padding: 7px 8px;
  text-align: center;
  white-space: nowrap;
}
.eegdash-api-volume-explorer thead th {
  background: var(--api-soft);
  position: sticky;
  top: 0;
  z-index: 2;
}
.eegdash-api-volume-explorer .api-study-col {
  background: var(--api-paper);
  left: 0;
  position: sticky;
  text-align: left;
  z-index: 3;
}
.eegdash-api-volume-explorer thead .api-study-col {
  background: var(--api-soft);
}
.eegdash-api-volume-explorer .api-study-name a {
  color: var(--api-mark);
  text-decoration: none;
}
.eegdash-api-volume-explorer .api-study-name a:hover {
  text-decoration: underline;
}
.eegdash-api-volume-explorer .api-facet-col {
  writing-mode: vertical-rl;
}
.eegdash-api-volume-explorer .api-facet-col button {
  background: transparent;
  border: 0;
  cursor: pointer;
  min-height: 0;
  padding: 0;
  writing-mode: vertical-rl;
}
.eegdash-api-volume-explorer .api-facet-col.active,
.eegdash-api-volume-explorer .api-facet-col.active button {
  color: #b45309;
  background: #fffbeb;
}
html[data-theme="dark"] .eegdash-api-volume-explorer .api-facet-col.active,
html[data-theme="dark"] .eegdash-api-volume-explorer .api-facet-col.active button {
  color: #fbbf24;
  background: rgba(146, 64, 14, 0.28);
}
.eegdash-api-volume-explorer .num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.eegdash-api-volume-explorer .tick {
  color: #15803d;
  font-weight: 800;
}
.eegdash-api-volume-explorer [hidden] {
  display: none !important;
}
.eegdash-api-volume-explorer .api-study-tooltip {
  background: #111827;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 7px;
  color: #fff;
  font-size: 13px;
  line-height: 1.4;
  max-width: 420px;
  padding: 10px 12px;
  pointer-events: auto;
  position: fixed;
  z-index: 1000;
}
.eegdash-api-volume-explorer .api-study-tooltip .tt-title {
  font-size: 15px;
  font-weight: 800;
  margin-bottom: 5px;
}
.eegdash-api-volume-explorer .api-study-tooltip a {
  color: #93c5fd;
  display: inline-block;
  margin-top: 7px;
}
@media (max-width: 760px) {
  .eegdash-api-volume-explorer .api-summary {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }
  .eegdash-api-volume-explorer .api-summary div {
    border-bottom: 1px solid var(--api-rule);
  }
  .eegdash-api-volume-explorer .api-controls {
    position: static;
  }
}
</style>
"""


_SCRIPT = """
<script>
(function() {
  function initApiExplorer(root) {
    if (!root || root.dataset.apiExplorerReady === "1") return;
    root.dataset.apiExplorerReady = "1";

  const filterSelect = root.querySelector(".api-facet-filter");
  const textFilter = root.querySelector(".api-text-filter");
  const clearButton = root.querySelector(".api-clear-filters");
  const status = root.querySelector(".api-filter-status");
  const tooltip = root.querySelector(".api-study-tooltip");
  const studyRows = root.querySelectorAll(".api-study-row");
  const studyPoints = root.querySelectorAll(".api-study-point");
  const summaryItems = studyRows.length ? studyRows : studyPoints;
  const filterables = [...studyRows, ...studyPoints];
  const summaryDatasets = root.querySelector(".api-summary-datasets");
  const summarySubjects = root.querySelector(".api-summary-subjects");
  const summaryRecords = root.querySelector(".api-summary-records");
  const summaryHours = root.querySelector(".api-summary-hours");
  const active = new Set();
  let tooltipTimer = null;

  function hasFacet(el, token) {
    return (el.dataset.facets || "").split("|").includes(token);
  }

  function matches(el) {
    const q = (textFilter && textFilter.value || "").trim().toLowerCase();
    if (q && !(el.dataset.search || "").includes(q)) return false;
    for (const token of active) {
      if (!hasFacet(el, token)) return false;
    }
    return true;
  }

  function formatValue(value) {
    if (!Number.isFinite(value)) return "";
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2).replace(/\\.00$/, "");
  }

  function updateSummary() {
    let datasets = 0;
    let subjects = 0;
    let records = 0;
    let hours = 0;
    summaryItems.forEach((row) => {
      if (row.hasAttribute("hidden")) return;
      datasets += 1;
      subjects += Number(row.dataset.subjects || 0);
      records += Number(row.dataset.records || 0);
      const h = Number(row.dataset.hours || 0);
      if (Number.isFinite(h)) hours += h;
    });
    if (summaryDatasets) summaryDatasets.textContent = datasets.toLocaleString();
    if (summarySubjects) summarySubjects.textContent = subjects.toLocaleString();
    if (summaryRecords) summaryRecords.textContent = records.toLocaleString();
    if (summaryHours) summaryHours.textContent = formatValue(hours);
  }

  function updateStatus() {
    const bits = Array.from(active).map((token) => token.replace("::", ": "));
    const q = (textFilter && textFilter.value || "").trim();
    if (q) bits.push(`search: ${q}`);
    if (status) status.textContent = bits.length ? bits.join(" | ") : "All API facets";
    if (clearButton) clearButton.disabled = bits.length === 0;
    if (filterSelect) {
      Array.from(filterSelect.options).forEach((option) => {
        option.disabled = option.value !== "" && active.has(option.value);
      });
      filterSelect.value = "";
    }
    root.querySelectorAll("[data-filter-token]").forEach((el) => {
      el.classList.toggle("active", active.has(el.dataset.filterToken || ""));
    });
  }

  function applyFilters() {
    filterables.forEach((el) => {
      if (matches(el)) el.removeAttribute("hidden");
      else el.setAttribute("hidden", "");
    });
    updateStatus();
    updateSummary();
  }

  function moveTooltip(event) {
    if (!tooltip || tooltip.hasAttribute("hidden")) return;
    if (typeof event.clientX !== "number" || typeof event.clientY !== "number") {
      tooltip.style.left = "18px";
      tooltip.style.top = "18px";
      return;
    }
    const margin = 14;
    const rect = tooltip.getBoundingClientRect();
    let left = event.clientX + margin;
    let top = event.clientY + margin;
    if (left + rect.width > window.innerWidth) left = event.clientX - rect.width - margin;
    if (top + rect.height > window.innerHeight) top = event.clientY - rect.height - margin;
    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function showTooltip(event) {
    if (!tooltip) return;
    const el = event.currentTarget;
    clearTimeout(tooltipTimer);
    tooltip.innerHTML = "";
    const title = document.createElement("div");
    title.className = "tt-title";
    title.textContent = el.dataset.name || "";
    tooltip.appendChild(title);
    const desc = document.createElement("div");
    desc.textContent = el.dataset.description || "No description available.";
    tooltip.appendChild(desc);
    if (el.dataset.aliases) {
      const aliases = document.createElement("div");
      aliases.textContent = `Aliases: ${el.dataset.aliases}`;
      tooltip.appendChild(aliases);
    }
    const stats = document.createElement("div");
    const hours = el.dataset.hours ? Number(el.dataset.hours).toLocaleString(undefined, { maximumFractionDigits: 1 }) : "not reported";
    stats.textContent = `${Number(el.dataset.subjects || 0).toLocaleString()} subjects | ${Number(el.dataset.records || 0).toLocaleString()} records | ${hours} h`;
    tooltip.appendChild(stats);
    if (el.dataset.url) {
      const link = document.createElement("a");
      link.href = el.dataset.url;
      link.textContent = "Open dataset page";
      tooltip.appendChild(link);
    }
    tooltip.removeAttribute("hidden");
    moveTooltip(event);
  }

  function hideTooltip() {
    tooltipTimer = setTimeout(() => {
      if (tooltip) tooltip.setAttribute("hidden", "");
    }, 120);
  }

  root.querySelectorAll("[data-filter-token]").forEach((el) => {
    el.addEventListener("click", () => {
      const token = el.dataset.filterToken || "";
      if (!token) return;
      if (active.has(token)) active.delete(token);
      else active.add(token);
      applyFilters();
    });
  });

  if (filterSelect) {
    filterSelect.addEventListener("change", () => {
      if (filterSelect.value) {
        active.add(filterSelect.value);
        applyFilters();
      }
    });
  }
  if (textFilter) {
    textFilter.addEventListener("input", applyFilters);
  }
  if (clearButton) {
    clearButton.addEventListener("click", () => {
      active.clear();
      if (textFilter) textFilter.value = "";
      applyFilters();
    });
  }

  root.querySelectorAll("[data-description]").forEach((el) => {
    el.addEventListener("mouseenter", showTooltip);
    el.addEventListener("mousemove", moveTooltip);
    el.addEventListener("mouseleave", hideTooltip);
    el.addEventListener("focus", showTooltip);
    el.addEventListener("blur", hideTooltip);
  });
  root.querySelectorAll(".api-study-point").forEach((el) => {
    el.addEventListener("click", () => {
      if (el.dataset.url) window.open(el.dataset.url, "_self");
    });
  });
  if (tooltip) {
    tooltip.addEventListener("mouseenter", () => clearTimeout(tooltipTimer));
    tooltip.addEventListener("mouseleave", hideTooltip);
  }
  applyFilters();
  }

  document
    .querySelectorAll(".eegdash-api-volume-explorer")
    .forEach(initApiExplorer);
})();
</script>
"""


def generate_api_study_explorer(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    api_url: str = "https://data.eegdash.org/api",
    database: str = "eegdash",
    view: str = "both",
) -> Path:
    """Generate the API-oriented dataset volume explorer."""
    if view not in _VIEWS:
        raise ValueError(f"view must be one of: {_VIEWS}")
    records = _prepare_records(df)
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root_id = {
        "volume": "eegdash-api-volume-scatter",
        "matrix": "eegdash-api-coverage-matrix",
        "both": "eegdash-api-volume-explorer",
    }[view]

    if not records:
        html_content = (
            f'<div id="{root_id}" class="eegdash-api-volume-explorer">'
            '<div class="api-empty">No API dataset records available.</div>'
            "</div>"
        )
    else:
        html_content = _build_html(
            records,
            api_url=api_url,
            database=database,
            view=view,
            root_id=root_id,
        )

    return build_and_export_html(
        fig=None,
        out_path=out_path,
        div_id=root_id,
        height=900,
        extra_style=_STYLE,
        extra_html=_SCRIPT,
        include_default_style=False,
        html_content=html_content,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the API study explorer.")
    parser.add_argument("source", type=Path, help="Path to dataset summary CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_api_study_explorer.html"),
        help="Output HTML file",
    )
    parser.add_argument("--api-url", default="https://data.eegdash.org/api")
    parser.add_argument("--database", default="eegdash")
    parser.add_argument(
        "--view",
        choices=list(_VIEWS),
        default="both",
        help="Which API explorer panel to render.",
    )
    args = parser.parse_args()

    output_path = generate_api_study_explorer(
        read_dataset_csv(args.source),
        args.output,
        api_url=args.api_url,
        database=args.database,
        view=args.view,
    )
    print(f"API study explorer saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
