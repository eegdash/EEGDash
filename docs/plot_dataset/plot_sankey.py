"""Source-to-modality Sankey diagram powered by ``/aggregations/sankey``.

Pre-migration this module ran a ``df.groupby([col_from, col_to]).agg(
subject_sum=..., dataset_count=...)`` to build links between (Type Subject,
modality of exp, type of exp). It also exploded multi-modality datasets
across each branch, double-counting subjects.

The server's ``/aggregations/sankey?levels=source,modality`` endpoint
returns one node per source (openneuro, nemar) and one per recording
modality, with ``link.value == subject_sum`` and ``link.dataset_count``
exposed alongside. This module is now a thin go.Sankey renderer over
that response.

Behavioural note: the new diagram visualises Source -> Recording
Modality (the canonical two-level rollup the server owns), not the
three-column Type-Subject / Modality / Type pipeline the legacy
client tried to compute. The server takes the first ``recording_modality``
per dataset so each dataset contributes once; the legacy chart's
multi-modality double counting is intentionally dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

try:
    from ._live import AggregationFetchError, fetch_aggregation
    from .colours import (
        MODALITY_COLOR_MAP,
        PATHOLOGY_PASTEL_OVERRIDES,
        hex_to_rgba,
    )
    from .utils import build_and_export_html
except ImportError:  # pragma: no cover - direct-script execution
    from _live import AggregationFetchError, fetch_aggregation  # type: ignore
    from colours import (  # type: ignore
        MODALITY_COLOR_MAP,
        PATHOLOGY_PASTEL_OVERRIDES,
        hex_to_rgba,
    )
    from utils import build_and_export_html  # type: ignore

__all__ = ["generate_dataset_sankey"]

# Default colour for unknown source / modality nodes (legacy slate-400).
_DEFAULT_NODE_COLOR = "#94a3b8"

# Source-level colours. Two known sources today; anything else falls
# back to slate. PATHOLOGY_PASTEL_OVERRIDES carries the "Healthy"
# pastel-green we want for openneuro, and the slate-300 we want for
# the catch-all.
_SOURCE_COLORS = {
    "openneuro": PATHOLOGY_PASTEL_OVERRIDES.get("Healthy", "#86efac"),
    "nemar": PATHOLOGY_PASTEL_OVERRIDES.get("Clinical", "#fecaca"),
}


def _node_color(label: str, level: int) -> str:
    if level == 0:
        return _SOURCE_COLORS.get(label.lower(), _DEFAULT_NODE_COLOR)
    return MODALITY_COLOR_MAP.get(label, _DEFAULT_NODE_COLOR)


def _empty_figure(reason: str, out_html: str | Path) -> Path:
    fig = go.Figure()
    fig.add_annotation(text=reason, showarrow=False)
    fig.update_layout(template="plotly_white", height=600)
    extra_style = (
        '<div id="dataset-sankey-wrapper" style="width: 100%; height: 600px;">'
    )
    extra_html = "</div>"
    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-sankey",
        extra_style=extra_style,
        extra_html=extra_html,
        include_default_style=False,
    )


def _build_figure(payload: dict[str, Any], *, height: int) -> go.Figure:
    nodes = payload.get("nodes") or []
    links = payload.get("links") or []
    if not nodes or not links:
        raise ValueError("Sankey aggregation returned no nodes or links")

    node_labels: list[str] = []
    node_colors: list[str] = []
    # Annotate level-0 nodes with the rollup subjects (link.value sums
    # to the source's subjects across modalities by construction).
    source_total_subjects = {
        node["id"]: node.get("value", 0) for node in nodes if node.get("level") == 0
    }
    total_subjects = sum(source_total_subjects.values()) or 1

    for node in nodes:
        label = str(node.get("label", ""))
        level = int(node.get("level", 0) or 0)
        color = _node_color(label, level)
        if level == 0:
            subj = int(node.get("value", 0))
            pct = (subj / total_subjects) * 100
            node_labels.append(f"{label}<br>({subj:,} subjects, {pct:.1f}%)")
        else:
            node_labels.append(label)
        node_colors.append(color)

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    link_hover: list[str] = []

    label_by_id = {node["id"]: node.get("label", "") for node in nodes}
    for link in links:
        src_id = link["source"]
        tgt_id = link["target"]
        subject_sum = int(link.get("subject_sum", link.get("value", 0)) or 0)
        dataset_count = int(link.get("dataset_count", 0) or 0)
        sources.append(src_id)
        targets.append(tgt_id)
        # link.value == subject_sum, fixed in server commit cc9785f.
        values.append(subject_sum)
        src_label = str(label_by_id.get(src_id, "")).lower()
        link_colors.append(
            hex_to_rgba(_SOURCE_COLORS.get(src_label, _DEFAULT_NODE_COLOR))
        )
        link_hover.append(
            f"{label_by_id.get(src_id, '')} → {label_by_id.get(tgt_id, '')}:<br>"
            f"{subject_sum:,} subjects in {dataset_count:,} datasets"
        )

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=18,
            label=node_labels,
            color=node_colors,
            align="left",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover,
        ),
    )

    fig = go.Figure(sankey)
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0f172a"),
        height=height,
        autosize=True,
        margin=dict(t=60, b=60, l=40, r=40),
        paper_bgcolor="#ffffff",
        annotations=[
            dict(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Source",
                showarrow=False,
                font=dict(size=16, color="black"),
            ),
            dict(
                x=1,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Recording Modality",
                showarrow=False,
                font=dict(size=16, color="black"),
            ),
        ],
    )
    return fig


def generate_dataset_sankey(
    df: Any | None = None,
    out_html: str | Path = "dataset_sankey.html",
    *,
    columns: Any | None = None,
    width: int = 1260,
    height: int = 1100,
) -> Path:
    """Render the dataset Sankey from the server aggregation."""
    del df, columns, width  # Server owns the cross-tab; args retained.

    try:
        response = fetch_aggregation("sankey", {"levels": "source,modality"})
    except AggregationFetchError as exc:
        return _empty_figure(f"Sankey data unavailable: {exc}", out_html)

    try:
        fig = _build_figure(response.payload, height=height)
    except ValueError as exc:
        return _empty_figure(str(exc), out_html)

    extra_style = (
        f'<div id="dataset-sankey-wrapper" style="width: 100%; height: {height}px;">'
    )
    extra_html = """</div>
<script>
    window.addEventListener('load', function() {
        window.dispatchEvent(new Event('resize'));
    });
</script>
"""
    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-sankey",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        extra_style=extra_style,
        extra_html=extra_html,
        include_default_style=False,
    )
