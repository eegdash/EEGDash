"""Plotly rendering utilities for EEG-DaSh dataset summary bubble charts.

The Phase 5 redesign focuses on a single record-count metric, deterministic grid
placement, and streamlined docs integration without external image exporters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from eegdash.plotting.dataset_summary_adapter import (
    BubbleSizeMetric,
    DatasetSummaryRecord,
    LegendConfig,
    SummaryBubbleArtifacts,
    SummaryBubbleConfig,
    summary_to_bubble_kwargs,
)

__all__ = ["generate_plotly_dataset_bubble"]

_SUBJECT_ALPHA_SERIES: tuple[float, ...] = (0.8, 0.65, 0.5, 0.35, 0.2)
_DEFAULT_MARKER_PX_SCALE = 22.0
_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b>"
    "<br>Subjects: %{customdata[1]}"
    "<br>Records: %{customdata[2]}"
    "<br>Duration: %{customdata[3]}"
    "<br>Tasks: %{customdata[4]}"
    "<br>Type Subject: %{customdata[5]}"
    "<extra></extra>"
)


def _get_hexa_grid(
    n: int,
    diameter: float,
    center: tuple[float, float],
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(n) - n // 2
    if seed is None:
        offset_x = 0.0
        offset_y = 0.0
    else:
        rng = np.random.default_rng(seed)
        offset_x = float(rng.uniform(-0.25, 0.25))
        offset_y = float(rng.uniform(-0.25, 0.25))

    x, y = np.meshgrid(base + offset_x, base + offset_y)
    x = x.flatten()
    y = y.flatten()
    return (
        np.concatenate([x, x + 0.5]) * diameter + center[0],
        np.concatenate([y, y + 0.5]) * diameter * np.sqrt(3) + center[1],
    )


def _get_bubble_coordinates(
    n: int,
    diameter: float,
    center: tuple[float, float],
    *,
    layout_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _get_hexa_grid(n, diameter, center, seed=layout_seed)
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    sort_idx = np.argsort(dist)
    x = x[sort_idx]
    y = y[sort_idx]
    return x[:n], y[:n]


def _compute_diameter(size_value: float, scale: float) -> float:
    safe_value = max(float(size_value), math.e**1e-3)
    log_value = math.log(safe_value)
    diameter = log_value * scale
    if diameter <= 0:
        diameter = max(scale * 0.05, 1e-3)
    return diameter


@dataclass(slots=True)
class _DatasetCluster:
    """Container with prepared per-dataset payloads for Plotly rendering."""

    record: DatasetSummaryRecord
    modality: str
    color: str
    center: tuple[float, float]
    x: list[float]
    y: list[float]
    marker_size: list[float]
    opacity: list[float]
    customdata: list[list[str]]
    url: str | None


def generate_plotly_dataset_bubble(
    records: Sequence[DatasetSummaryRecord],
    config: SummaryBubbleConfig,
    metric: BubbleSizeMetric = BubbleSizeMetric.N_RECORDS,
    *,
    outfile: Path | None = None,
) -> go.Figure:
    """Generate a grouped dataset bubble chart using Plotly."""
    if not records:
        raise ValueError("generate_plotly_dataset_bubble requires at least one record.")

    cfg = _ensure_config(config, metric)
    artifacts = summary_to_bubble_kwargs(records, config=cfg)
    if not artifacts.bubble_kwargs:
        raise ValueError(
            "Adapter did not return any plotting payloads for Plotly export."
        )

    centers = _compute_cluster_centers(artifacts, cfg)
    clusters = _build_clusters(artifacts, cfg, centers)
    grouped = _group_clusters_by_modality(clusters)
    legend_config = _build_legend(artifacts.records, cfg)

    fig = go.Figure()
    for modality, payload in grouped.items():
        marker_dict = {
            "size": payload["marker_size"],
            "color": payload["color"],
            "sizemode": "diameter",
            "opacity": payload["opacity"],
            "line": {"width": 1, "color": "rgba(17,24,39,0.25)"},
        }
        fig.add_trace(
            go.Scatter(
                name=modality,
                x=payload["x"],
                y=payload["y"],
                mode="markers",
                marker=marker_dict,
                customdata=payload["customdata"],
                hovertemplate=_HOVER_TEMPLATE,
            )
        )

    annotations = _legend_annotations(legend_config)

    fig.update_layout(
        title=dict(
            text="EEG-DaSh Dataset Summary – Grouped Bubbles",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        legend=dict(title="Modality", yanchor="top", y=0.99),
        margin=dict(l=60, r=200, t=70, b=60),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1.0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        clickmode="event+select",
        annotations=annotations,
    )

    if outfile is not None:
        _write_interactive_html(fig, outfile)

    return fig


def _compute_cluster_centers(
    artifacts: SummaryBubbleArtifacts,
    cfg: SummaryBubbleConfig,
) -> list[tuple[float, float]]:
    diameters: list[float] = []
    for payload in artifacts.bubble_kwargs:
        size_value = float(payload.get("size_override", cfg.min_size_override))
        diameters.append(max(_compute_diameter(size_value, cfg.scale), 1.0))

    n_clusters = len(diameters)
    if n_clusters == 0:
        return []

    columns = max(1, int(math.ceil(math.sqrt(n_clusters))))
    rows = int(math.ceil(n_clusters / columns))

    assignments: list[tuple[int, int, int]] = []
    index = 0
    for row in range(rows):
        for column in range(columns):
            if index >= n_clusters:
                break
            assignments.append((index, row, column))
            index += 1

    column_widths = [0.0] * columns
    row_heights = [0.0] * rows
    for idx, row, column in assignments:
        diameter = diameters[idx]
        column_widths[column] = max(column_widths[column], diameter)
        row_heights[row] = max(row_heights[row], diameter)

    gap = float(cfg.meta_gap)
    column_offsets: list[float] = []
    running = 0.0
    for width in column_widths:
        column_offsets.append(running + width / 2.0)
        running += width + gap
    total_width = max(running - gap, 1.0)

    row_offsets: list[float] = []
    running = 0.0
    for height in row_heights:
        row_offsets.append(-(running + height / 2.0))
        running += height + gap
    total_height = max(running - gap, 1.0)

    x_adjust = total_width / 2.0
    y_adjust = -(total_height / 2.0)

    centers: list[tuple[float, float]] = [(0.0, 0.0)] * n_clusters
    for idx, row, column in assignments:
        x_pos = column_offsets[column] - x_adjust
        y_pos = row_offsets[row] - y_adjust
        centers[idx] = (float(x_pos), float(y_pos))
    return centers


def _build_clusters(
    artifacts: SummaryBubbleArtifacts,
    cfg: SummaryBubbleConfig,
    centers: Sequence[tuple[float, float]],
) -> list[_DatasetCluster]:
    clusters: list[_DatasetCluster] = []
    for index, payload in enumerate(artifacts.bubble_kwargs):
        record = artifacts.records[index]
        center = centers[index]
        n_subjects = max(int(payload["n_subjects"]), 1)
        layout_seed = payload.get("layout_seed")

        size_value = float(payload.get("size_override", cfg.min_size_override))
        diameter = _compute_diameter(size_value, cfg.scale)
        marker_px = _diameter_to_marker_px(diameter)

        x_coords, y_coords = _get_bubble_coordinates(
            n_subjects,
            diameter,
            center,
            layout_seed=layout_seed,
        )
        x_list = [float(value) for value in x_coords]
        y_list = [float(value) for value in y_coords]

        opacity_value = _sessions_to_opacity(int(payload["n_sessions"]))
        opacity = [opacity_value] * n_subjects

        custom_row = _build_customdata_row(record, payload)
        customdata = [custom_row[:] for _ in range(n_subjects)]

        url = payload.get("dataset_url")
        clusters.append(
            _DatasetCluster(
                record=record,
                modality=record.modality,
                color=artifacts.color_map.get(record.modality, "#94a3b8"),
                center=center,
                x=x_list,
                y=y_list,
                marker_size=[marker_px] * n_subjects,
                opacity=opacity,
                customdata=customdata,
                url=url if isinstance(url, str) and url.strip() else None,
            )
        )
    return clusters


def _group_clusters_by_modality(
    clusters: Sequence[_DatasetCluster],
) -> Mapping[str, dict[str, list[float] | str]]:
    grouped: dict[str, dict[str, list[float] | str]] = {}
    for cluster in clusters:
        payload = grouped.setdefault(
            cluster.modality,
            {
                "color": cluster.color,
                "x": [],
                "y": [],
                "marker_size": [],
                "opacity": [],
                "customdata": [],
            },
        )
        payload["x"].extend(cluster.x)
        payload["y"].extend(cluster.y)
        payload["marker_size"].extend(cluster.marker_size)
        payload["opacity"].extend(cluster.opacity)
        payload["customdata"].extend(cluster.customdata)
    return grouped


def _build_legend(
    records: Sequence[DatasetSummaryRecord],
    cfg: SummaryBubbleConfig,
) -> LegendConfig | None:
    legend_cfg = replace(cfg, size_metric=BubbleSizeMetric.N_RECORDS)
    legend_artifacts = summary_to_bubble_kwargs(records, config=legend_cfg)
    return legend_artifacts.legend_config


def _legend_annotations(legend: LegendConfig | None) -> list[go.layout.Annotation]:
    if not legend or not legend.size_bins:
        return []
    lines: list[str] = [f"<b>{legend.heading or 'Record volume'}</b>"]
    for entry in legend.size_bins:
        lines.append(f"&bull; {entry.label}")
    if legend.type_subjects:
        lines.append(" ")
        lines.append("<b>Type Subject</b>")
        for badge in legend.type_subjects:
            swatch_color = badge.color or "#1f2937"
            swatch = (
                '<span style="display:inline-block;width:0.85em;height:0.85em;'
                f'background:{swatch_color};border-radius:0.2em;margin-right:0.4em;"></span>'
            )
            lines.append(f"{swatch}{badge.label}")
    text = "<br>".join(lines)
    return [
        go.layout.Annotation(
            x=1.05,
            y=0.9,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            text=text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(15,23,42,0.2)",
            borderwidth=1,
            borderpad=6,
            font=dict(size=12),
        )
    ]


def _ensure_config(
    config: SummaryBubbleConfig, metric: BubbleSizeMetric
) -> SummaryBubbleConfig:
    target = config
    if target.size_metric is not metric:
        target = replace(target, size_metric=metric)
    if target.dataset_url_resolver is None:
        target = replace(target, dataset_url_resolver=_default_dataset_url)
    return target


def _default_dataset_url(record: DatasetSummaryRecord) -> str:
    dataset_key = record.dataset.strip().upper()
    return f"api/dataset/eegdash.dataset.{dataset_key}.html"


def _sessions_to_opacity(n_sessions: int) -> float:
    clamped = max(1, int(n_sessions))
    index = min(clamped, len(_SUBJECT_ALPHA_SERIES)) - 1
    return _SUBJECT_ALPHA_SERIES[index]


def _build_customdata_row(
    record: DatasetSummaryRecord, payload: Mapping[str, object]
) -> list[str]:
    subjects = f"{record.n_subjects:,}"
    records_text = f"{record.n_records:,}"
    if record.duration_hours_total is not None and record.duration_hours_total > 0:
        duration = f"{record.duration_hours_total:.2f} h"
    else:
        duration = "N/A"
    if record.n_tasks is not None and record.n_tasks > 0:
        tasks = f"{record.n_tasks:,}"
    else:
        tasks = "N/A"
    type_subject = record.type_subject or "Unspecified"
    url = str(payload.get("dataset_url") or "")
    return [
        record.dataset,
        subjects,
        records_text,
        duration,
        tasks,
        type_subject,
        url,
    ]


def _diameter_to_marker_px(diameter: float) -> float:
    safe = max(diameter, 0.1)
    return max(safe * _DEFAULT_MARKER_PX_SCALE, 6.0)


def _write_interactive_html(fig: go.Figure, outfile: Path) -> None:
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  const plots = document.querySelectorAll('.plotly-graph-div');
  plots.forEach(function(plot) {
    plot.on('plotly_click', function(evt) {
      const point = evt.points && evt.points[0];
      const url = point && point.customdata && point.customdata[6];
      if (url) {
        window.open(url, '_blank', 'noopener');
      }
    });
  });
});
</script>
"""
    html = html.replace("</body>", f"{script}\n</body>")
    outfile.write_text(html, encoding="utf-8")
