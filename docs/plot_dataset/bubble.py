"""Dataset landscape bubble chart powered by ``/aggregations/bubble``.

Pre-migration this module computed per-dataset numerics from the
catalog DataFrame: ``pd.to_numeric`` over n_subjects/n_records, log-
scaled bubble sizes from duration-per-subject, a primary-modality
extractor over comma-separated cells, and a 12-column ``customdata``
array driving the hover. The server's
``/aggregations/bubble?x=subjects&y=files&color=modality`` endpoint
returns one point per dataset with the same numbers pre-computed.

Behavioural shift: the server returns one point per dataset, taking
the first ``recording_modality``. The legacy chart filtered NaN
client-side; the server emits ``null`` x/y for unmeasured datasets,
which this module drops at the input boundary. Hover content is
slimmed to what the server exposes (label, x, y, color); deeper
catalog facts (size, sampling rate, etc.) now live on the dataset
detail pages linked from the bubble.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go

try:
    from ._live import AggregationFetchError, fetch_aggregation
    from .colours import RECORDING_MODALITY_COLORS
    from .utils import build_and_export_html, get_dataset_url
except ImportError:  # pragma: no cover - direct-script execution
    from _live import AggregationFetchError, fetch_aggregation  # type: ignore
    from colours import RECORDING_MODALITY_COLORS  # type: ignore
    from utils import build_and_export_html, get_dataset_url  # type: ignore

__all__ = ["generate_dataset_bubble"]

_BACKGROUND_CATS = {"Other", "Unknown", "other", "unknown"}
_DEFAULT_BUBBLE_COLOR = "#94a3b8"


def _empty_figure(reason: str, out_html: str | Path) -> Path:
    fig = go.Figure()
    fig.add_annotation(text=reason, showarrow=False)
    fig.update_layout(template="plotly_white", height=720)
    return build_and_export_html(fig, out_html, div_id="dataset-bubble", height=720)


def _coerce_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _filter_points(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop points with non-positive x/y; bubble axes are log-scaled."""
    filtered: list[dict[str, Any]] = []
    for p in points:
        x = _coerce_float(p.get("x"))
        y = _coerce_float(p.get("y"))
        if x is None or y is None or x <= 0 or y <= 0:
            continue
        filtered.append(
            {
                "id": str(p.get("id") or ""),
                "label": str(p.get("label") or p.get("id") or ""),
                "color": str(p.get("color") or "Unknown"),
                "x": x,
                "y": y,
                "size": _coerce_float(p.get("size")),
            }
        )
    return filtered


def _build_modality_traces(
    points: list[dict[str, Any]],
    *,
    sizeref: float,
    color_map: dict[str, str],
) -> list[go.Scatter]:
    """One trace per modality, neutrals drawn first so coloured dots sit on top."""
    by_modality: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        by_modality.setdefault(p["color"], []).append(p)

    modalities = list(by_modality.keys())
    bg = [m for m in modalities if m in _BACKGROUND_CATS]
    fg = [m for m in modalities if m not in _BACKGROUND_CATS]
    ordered = bg + fg

    traces: list[go.Scatter] = []
    for modality in ordered:
        subset = by_modality[modality]
        xs = [p["x"] for p in subset]
        ys = [p["y"] for p in subset]
        sizes = [p["size"] if p["size"] and p["size"] > 0 else 1.0 for p in subset]
        # Bubble area encodes the optional ``size`` field; we fall back to
        # unit weights so all points are still visible when the server
        # omits a size column.
        log_sizes = [np.log1p(max(s, 0.0)) for s in sizes]
        labels = [p["label"] for p in subset]
        ids = [p["id"] for p in subset]
        urls = [get_dataset_url(p["id"]) for p in subset]
        is_bg = modality in _BACKGROUND_CATS
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=modality,
                marker=dict(
                    size=log_sizes,
                    color=color_map.get(modality, _DEFAULT_BUBBLE_COLOR),
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=5,
                    line=dict(width=1, color="rgba(255,255,255,0.8)"),
                    opacity=0.35 if is_bg else 0.7,
                ),
                customdata=np.array(
                    [[lab, urls[i], ids[i]] for i, lab in enumerate(labels)],
                    dtype=object,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "x: %{x:,}<br>y: %{y:,}"
                    f"<br>Modality: {modality}"
                    "<br><i>Click bubble to open dataset page</i>"
                    "<extra></extra>"
                ),
                visible="legendonly" if is_bg else True,
                legendgroup=f"mod-{modality}",
            )
        )
    return traces


def generate_dataset_bubble(
    df: Any | None = None,
    out_html: str | Path = "dataset_bubble.html",
    *,
    x_var: str = "subjects",
    max_width: int = 1280,
    height: int = 720,
) -> Path:
    """Render the dataset landscape bubble from the server aggregation."""
    del df

    # Map legacy x_var names to the canonical server fields.
    y_var = "files" if x_var == "subjects" else "subjects"
    try:
        response = fetch_aggregation(
            "bubble",
            {"x": x_var, "y": y_var, "color": "modality"},
        )
    except AggregationFetchError as exc:
        return _empty_figure(f"Bubble data unavailable: {exc}", out_html)

    points = _filter_points(response.payload.get("points") or [])
    if not points:
        return _empty_figure("No dataset records available for plotting.", out_html)

    sizes = [(p["size"] or 1.0) for p in points]
    max_size = max(sizes) if sizes else 1.0
    sizeref = (2.0 * np.log1p(max_size)) / (34.0**2) if max_size > 0 else 1.0

    traces = _build_modality_traces(
        points,
        sizeref=sizeref,
        color_map=RECORDING_MODALITY_COLORS,
    )

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    axis_labels = {
        "records": "#Records",
        "files": "#Files",
        "subjects": "#Subjects",
        "duration_h": "Duration (hours)",
        "size_gb": "Size (GB)",
        "tasks": "#Tasks",
    }
    fig.update_xaxes(
        title_text=axis_labels.get(x_var, x_var),
        type="log",
        title_font=dict(size=18),
        tickfont=dict(size=15),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        zeroline=False,
        dtick=1,
    )
    fig.update_yaxes(
        title_text=axis_labels.get(y_var, y_var),
        type="log",
        title_font=dict(size=18),
        tickfont=dict(size=15),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        zeroline=False,
        dtick=1,
    )

    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=60, r=80, t=50, b=60),
        template="plotly_white",
        legend=dict(
            title=None,
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            font=dict(size=13),
            tracegroupgap=0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.06)",
            borderwidth=1,
            itemsizing="constant",
            valign="middle",
        ),
        font=dict(
            family=(
                "Inter, system-ui, -apple-system, Segoe UI, Roboto, "
                "Helvetica, Arial, sans-serif"
            ),
            size=16,
        ),
    )

    extra_style = """.dataset-loading {
    display: flex; justify-content: center; align-items: center;
    height: 720px; font-family: Inter, system-ui, sans-serif; color: #6b7280;
}"""

    extra_html = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.getElementById('dataset-bubble');
    if (!plot) { return; }
    if (typeof plot.on === 'function') {
        plot.on('plotly_click', function(evt) {
            var point = evt && evt.points && evt.points[0];
            var url = point && point.customdata && point.customdata[1];
            if (url) { window.open(url, '_blank', 'noopener'); }
        });
    }
});
</script>
"""

    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-bubble",
        height=height,
        extra_style=extra_style,
        extra_html=extra_html,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "dataset_landscape",
                "height": height,
                "width": max_width,
                "scale": 2,
            },
        },
    )
