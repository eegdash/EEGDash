from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

try:  # Allow execution as a script or module
    from .colours import CANONICAL_MAP, RECORDING_MODALITY_COLORS
    from .utils import (
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_modality,
        primary_recording_modality,
        read_dataset_csv,
        safe_int,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import CANONICAL_MAP, RECORDING_MODALITY_COLORS  # type: ignore
    from utils import (  # type: ignore
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_modality,
        primary_recording_modality,
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


def _normalize_label(value: str, column_key: str | None = None) -> str:
    cleaned = " ".join(value.replace("_", " ").split())
    lowered = cleaned.lower()
    if column_key:
        canonical = CANONICAL_MAP.get(column_key, {})
        if lowered in canonical:
            return canonical[lowered]
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
    recording = [primary_recording_modality(v) for v in _split_values(recording_raw)]
    if not recording:
        recording = ["Unknown"]

    modality_raw = _first_nonempty(
        row.get("modality of exp"),
        tags.get("modality"),
        row.get("experimental_modalities"),
        paradigm.get("modality"),
    )
    modality = [primary_modality(v) for v in _split_values(modality_raw)]

    type_raw = _first_nonempty(
        row.get("type of exp"),
        tags.get("type"),
        paradigm.get("cognitive_domain"),
    )
    exp_type = [_normalize_label(v, "type of exp") for v in _split_values(type_raw)]

    pathology_raw = _first_nonempty(row.get("Type Subject"), tags.get("pathology"))
    if not _split_values(pathology_raw):
        is_clinical = clinical.get("is_clinical")
        if is_clinical:
            pathology_raw = clinical.get("purpose") or "Clinical"
        elif is_clinical is False:
            pathology_raw = "Healthy"
    pathology = [
        _normalize_label(v, "Type Subject") for v in _split_values(pathology_raw)
    ]

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


def _build_volume_figure(
    records: list[dict[str, Any]], *, height: int = 720
) -> go.Figure:
    points = [
        record
        for record in records
        if record["hours_per_subject"] is not None
        and record["hours_per_subject"] > 0
        and record["subjects"] > 0
    ]
    fig = go.Figure()

    if not points:
        fig.add_annotation(
            text="No duration metadata is available for the current API payload.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#64748b"),
        )
        fig.update_layout(
            height=height,
            template="plotly_white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    recording_counts = Counter(
        record["facets"]["recording"][0]
        for record in points
        if record["facets"]["recording"]
    )
    size_values = [max(math.log1p(record["hours"] or 0), 0.5) for record in points]
    sizeref = (2.0 * max(size_values)) / (36.0**2)

    for recording in _order_recording(recording_counts):
        group = [
            record
            for record in points
            if record["facets"]["recording"]
            and record["facets"]["recording"][0] == recording
        ]
        if not group:
            continue
        marker_sizes = [max(math.log1p(record["hours"] or 0), 0.5) for record in group]
        customdata = [
            [
                record["id"],
                record["title"],
                recording,
                record["records"],
                _format_hours(record["hours"]),
                _format_number(record["hours_per_subject"]),
                record["source"],
                record["url"],
                human_readable_size(record["size_bytes"]),
                ", ".join(record["aliases"]),
            ]
            for record in group
        ]
        fig.add_trace(
            go.Scatter(
                x=[record["hours_per_subject"] for record in group],
                y=[record["subjects"] for record in group],
                mode="markers",
                name=recording,
                marker=dict(
                    color=RECORDING_MODALITY_COLORS.get(recording, "#64748b"),
                    line=dict(width=1, color="rgba(255,255,255,0.82)"),
                    opacity=0.72,
                    size=marker_sizes,
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=7,
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Recording: %{customdata[2]}<br>"
                    "Subjects: %{y:,}<br>"
                    "Records: %{customdata[3]:,}<br>"
                    "Known hours: %{customdata[4]}<br>"
                    "Hours / subject: %{customdata[5]}<br>"
                    "Source: %{customdata[6]}<br>"
                    "Size: %{customdata[8]}<br>"
                    "<i>Click point to open dataset page</i>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(
        title_text="Hours per subject",
        type="log",
        tickvals=[0.001, 0.01, 0.1, 1, 10, 100],
        ticktext=["0.001", "0.01", "0.1", "1", "10", "100"],
        automargin=True,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Subjects",
        type="log",
        tickvals=[1, 10, 100, 1000],
        ticktext=["1", "10", "100", "1000"],
        automargin=True,
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
    )
    fig.update_layout(
        height=height,
        autosize=True,
        template="plotly_white",
        margin=dict(l=64, r=28, t=32, b=62),
        legend=dict(
            title=None,
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(15, 23, 42, 0.08)",
            borderwidth=1,
            font=dict(size=12),
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
            color="#0f172a",
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(15, 23, 42, 0.2)",
            font=dict(size=12),
        ),
    )
    return fig


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
                f"{html.escape(record['id'])}</a></th>"
            ),
            f"<td>{html.escape(record['source'])}</td>",
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
    volume_html: str = "",
) -> str:
    total_subjects = sum(record["subjects"] for record in records)
    total_records = sum(record["records"] for record in records)
    known_hours = sum(record["hours"] or 0 for record in records)
    duration_count = sum(1 for record in records if record["hours_per_subject"])
    source_count = len({record["source"] for record in records})
    counts = _facet_counts(records)
    selected = _selected_columns(counts)

    panels = []
    if view in {"volume", "both"}:
        panels.append(
            f"""
  <section class="api-panel">
    <header>
      <h4>Dataset Volume From API Metadata</h4>
      <p>Scatter points use API summary fields only; use the Plotly legend to toggle recording modalities.</p>
    </header>
    <div class="api-plot-wrap">{volume_html}</div>
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
  <div class="api-contract-rail">
    <span>API source</span>
    <code>{html.escape(api_url.rstrip("/"))}/{html.escape(database)}/datasets/chart-data</code>
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
{"".join(panels)}
</div>
"""


_VOLUME_STYLE = """
.eegdash-api-volume-meta {
  align-items: center;
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  color: #64748b;
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
  font-size: 12px;
  margin: 0 0 12px;
  padding: 0 0 10px;
}
.eegdash-api-volume-meta code {
  color: #2563eb;
  font-size: 12px;
  overflow-wrap: anywhere;
  white-space: normal;
}
html[data-theme="dark"] .eegdash-api-volume-meta {
  border-color: rgba(148, 163, 184, 0.18);
  color: #94a3b8;
}
html[data-theme="dark"] .eegdash-api-volume-meta code {
  color: #93c5fd;
}
"""


def _volume_pre_html(
    records: list[dict[str, Any]],
    *,
    api_url: str,
    database: str,
) -> str:
    duration_count = sum(1 for record in records if record["hours_per_subject"])
    return (
        '<div class="eegdash-api-volume-meta">'
        "<span>API source</span>"
        f"<code>{html.escape(api_url.rstrip('/'))}/"
        f"{html.escape(database)}/datasets/chart-data</code>"
        f"<span>{len(records):,} datasets</span>"
        f"<span>{duration_count:,} with duration metadata</span>"
        "</div>"
    )


def _plotly_click_script(div_id: str) -> str:
    return f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  const plot = document.getElementById({json.dumps(div_id)});
  function resizeSoon() {{
    [0, 60, 250].forEach(function(delay) {{
      window.setTimeout(function() {{
        if (plot && plot.offsetWidth > 0 && typeof Plotly !== 'undefined') {{
          Plotly.Plots.resize(plot);
        }}
      }}, delay);
    }});
  }}
  function hook(attempts) {{
    if (!plot || typeof plot.on !== 'function') {{
      if (attempts < 40) {{
        window.setTimeout(function() {{ hook(attempts + 1); }}, 60);
      }}
      return;
    }}
    plot.on('plotly_click', function(evt) {{
      const point = evt && evt.points && evt.points[0];
      const url = point && point.customdata && point.customdata[7];
      if (url) {{
        window.open(url, '_blank', 'noopener');
      }}
    }});
    resizeSoon();
  }}
  document.querySelectorAll('.sd-tab-set label, .sd-tab-set input').forEach(function(el) {{
    el.addEventListener('click', resizeSoon);
    el.addEventListener('change', resizeSoon);
  }});
  window.addEventListener('resize', resizeSoon);
  if (typeof IntersectionObserver !== 'undefined' && plot) {{
    new IntersectionObserver(function(entries) {{
      if (entries[0] && entries[0].isIntersecting) {{
        resizeSoon();
      }}
    }}).observe(plot);
  }}
  hook(0);
  resizeSoon();
}});
</script>
"""


def _plotly_config(filename: str) -> dict[str, Any]:
    return {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 720,
            "width": 1260,
            "scale": 2,
        },
    }


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
.eegdash-api-volume-explorer .api-plot-wrap,
.eegdash-api-volume-explorer .api-table-wrap {
  border: 1px solid var(--api-rule);
  overflow: auto;
}
.eegdash-api-volume-explorer .api-plot-wrap {
  margin-top: 12px;
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
  const studyRows = root.querySelectorAll(".api-study-row");
  const summaryItems = studyRows;
  const filterables = [...studyRows];
  const summaryDatasets = root.querySelector(".api-summary-datasets");
  const summarySubjects = root.querySelector(".api-summary-subjects");
  const summaryRecords = root.querySelector(".api-summary-records");
  const summaryHours = root.querySelector(".api-summary-hours");
  const active = new Set();

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
    config = _plotly_config("api_dataset_volume")

    if view == "volume":
        fig = _build_volume_figure(records)
        return build_and_export_html(
            fig,
            out_path=out_path,
            div_id=root_id,
            height=720,
            extra_style=_VOLUME_STYLE,
            pre_html=_volume_pre_html(records, api_url=api_url, database=database),
            extra_html=_plotly_click_script(root_id),
            config=config,
        )

    if not records:
        html_content = (
            f'<div id="{root_id}" class="eegdash-api-volume-explorer">'
            '<div class="api-empty">No API dataset records available.</div>'
            "</div>"
        )
        fig = None
        extra_html = _SCRIPT
    else:
        fig = None
        volume_html = ""
        extra_html = _SCRIPT
        if view == "both":
            fig = _build_volume_figure(records)
            volume_div_id = f"{root_id}-plot"
            volume_html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                config=config,
                div_id=volume_div_id,
            )
            extra_html = _plotly_click_script(volume_div_id) + _SCRIPT
        html_content = _build_html(
            records,
            api_url=api_url,
            database=database,
            view=view,
            root_id=root_id,
            volume_html=volume_html,
        )

    return build_and_export_html(
        fig=fig,
        out_path=out_path,
        div_id=root_id,
        height=900,
        extra_style=_STYLE,
        extra_html=extra_html,
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
