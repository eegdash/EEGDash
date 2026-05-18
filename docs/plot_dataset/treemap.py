"""Dataset treemap powered by ``/aggregations/treemap``.

Pre-migration this module did two levels of ``df.groupby(...).agg(sum)``
on (population_type, experimental_modality, dataset_name), exploded
multi-modality datasets, and applied per-cell tokenisation. The server
endpoint ``/aggregations/treemap?path=source,modality,clinical`` now
returns the pre-computed three-level rollup with
``subjects/hours/records/datasets`` per node.

Behavioural shift: the server takes the first ``recording_modality``
and the first ``population_type`` per dataset, so each dataset
contributes once. The legacy client exploded multi-modality rows
across all branches, double-counting subjects/hours. The new totals
match what the rest of the documentation cards now report (single-
modality, single-population per dataset).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

try:
    from ._live import AggregationFetchError, fetch_aggregation
    from .colours import (
        MODALITY_COLOR_MAP,
        MODALITY_EMOJI,
        PATHOLOGY_PASTEL_OVERRIDES,
    )
    from .utils import build_and_export_html
except ImportError:  # pragma: no cover - direct-script execution
    from _live import AggregationFetchError, fetch_aggregation  # type: ignore
    from colours import (  # type: ignore
        MODALITY_COLOR_MAP,
        MODALITY_EMOJI,
        PATHOLOGY_PASTEL_OVERRIDES,
    )
    from utils import build_and_export_html  # type: ignore

__all__ = ["generate_dataset_treemap"]

_DEFAULT_COLOR = "#94a3b8"


def _abbreviate(value: float | int | None) -> str:
    """Pretty integer/short-scale formatter for tile labels."""
    try:
        num = float(value or 0)
    except (TypeError, ValueError):
        return "0"
    if not math.isfinite(num) or num == 0:
        return "0"
    if abs(num) < 1000:
        return f"{int(round(num)):,}"
    for divisor, suffix in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")]:
        if abs(num) >= divisor:
            scaled = round(num / divisor, 1)
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{int(num):,}"


def _format_label(name: str, subjects: float, hours: float, *, font_px: int) -> str:
    subjects_text = _abbreviate(subjects)
    secondary = f"{hours:.0f} h" if hours and hours > 0 else "records unavailable"
    return (
        f"{name}<br>"
        f"<span style='font-size:{font_px}px;'>{subjects_text} subj | {secondary}</span>"
    )


def _pathology_color(name: str) -> str:
    """Return the pastel tone used for level-1 (clinical) tiles."""
    base = PATHOLOGY_PASTEL_OVERRIDES.get(name)
    if not base:
        base = PATHOLOGY_PASTEL_OVERRIDES.get(name.title(), _DEFAULT_COLOR)
    return base


def _empty_figure(reason: str, out_html: str | Path) -> Path:
    fig = go.Figure()
    fig.add_annotation(text=reason, showarrow=False)
    fig.update_layout(template="plotly_white", height=600)
    return build_and_export_html(
        fig, out_html, div_id="dataset-treemap-plot", height=600
    )


def _color_for_node(node: dict[str, Any]) -> str:
    """Pick a fill colour based on the node's depth and label."""
    parts = str(node.get("id", "")).split("/")
    depth = len(parts) - 1 if node.get("id") else 0
    label = str(node.get("label", ""))
    if depth == 0:
        # Top-level source nodes — neutral pastels.
        return PATHOLOGY_PASTEL_OVERRIDES.get("Unknown", _DEFAULT_COLOR)
    if depth == 1:
        # Modality level.
        return MODALITY_COLOR_MAP.get(label, _DEFAULT_COLOR)
    # Deeper (clinical / population_type) — pathology palette.
    return _pathology_color(label)


def _build_figure(payload: dict[str, Any]) -> go.Figure:
    nodes = payload.get("nodes") or []
    if not nodes:
        raise ValueError("Treemap aggregation returned no nodes")

    ids = [str(n.get("id") or "") for n in nodes]
    labels = [str(n.get("label") or "") for n in nodes]
    parents = [str(n.get("parent") or "") for n in nodes]
    subjects = [float(n.get("subjects") or 0) for n in nodes]
    hours = [float(n.get("hours") or 0) for n in nodes]

    # Use emoji prefixes for modality nodes when the tile is big enough
    # to read; falls through to the plain label otherwise. Keeps the
    # legacy visual character.
    display_labels: list[str] = []
    for label, subj in zip(labels, subjects):
        emoji = MODALITY_EMOJI.get(label)
        display_labels.append(f"{emoji} {label}" if (emoji and subj >= 120) else label)

    text = [
        _format_label(lbl, s, h, font_px=16)
        for lbl, s, h in zip(display_labels, subjects, hours)
    ]
    colors = [_color_for_node(n) for n in nodes]
    hover = [
        f"{lbl}<br>{int(s):,} subjects<br>"
        f"{h:.0f} hours<br>"
        f"{int(n.get('records') or 0):,} records<br>"
        f"{int(n.get('datasets') or 0):,} datasets"
        for lbl, s, h, n in zip(display_labels, subjects, hours, nodes)
    ]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=display_labels,
            parents=parents,
            values=subjects,
            text=text,
            customdata=[[h] for h in hover],
            branchvalues="total",
            marker=dict(
                colors=colors,
                line=dict(color="white", width=1),
                pad=dict(t=15, r=15, b=15, l=15),
            ),
            textinfo="text",
            hovertemplate="%{customdata[0]}<extra></extra>",
            pathbar=dict(
                visible=True, edgeshape="/", thickness=34, textfont=dict(size=14)
            ),
            textfont=dict(size=20),
            insidetextfont=dict(size=20),
            tiling=dict(pad=8, packing="squarify"),
            root=dict(color="rgba(255,255,255,0.98)"),
        )
    )
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif"),
        uniformtext=dict(minsize=18, mode="hide"),
        margin=dict(t=60, l=28, r=28, b=36),
        hoverlabel=dict(font=dict(size=14), align="left"),
        height=880,
        autosize=True,
        paper_bgcolor="#ffffff",
    )
    return fig


def generate_dataset_treemap(
    df: Any | None = None,
    out_html: str | Path = "dataset_treemap.html",
    *,
    width: int = 1260,
) -> Path:
    """Render the dataset treemap from the server aggregation."""
    del df, width

    try:
        response = fetch_aggregation("treemap", {"path": "source,modality,clinical"})
    except AggregationFetchError as exc:
        return _empty_figure(f"Treemap data unavailable: {exc}", out_html)

    try:
        fig = _build_figure(response.payload)
    except ValueError as exc:
        return _empty_figure(str(exc), out_html)

    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-treemap-plot",
        height=880,
    )
