"""Cumulative growth chart powered by ``/aggregations/growth``.

Pre-migration this module ran ``df.groupby("Modality").cumsum`` over the
catalog DataFrame to derive the year-by-year curve. The server now owns
that pass — see ``aggregations/growth?bucket=year&field=dataset_created_at``
— and returns one cumulative series per modality. This module is a
thin renderer over that response.

Behavioural note: the legacy client also extracted publication years
from the ``author_year`` column for datasets missing
``dataset_created_at``. The server applies the same heuristic on its
side; the migration leaves any residual divergence to be reconciled
server-side (it is the source of truth now).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

try:
    from ._live import AggregationFetchError, fetch_aggregation
    from .colours import MODALITY_COLOR_MAP
    from .utils import build_and_export_html
except ImportError:  # pragma: no cover - direct-script execution
    from _live import AggregationFetchError, fetch_aggregation  # type: ignore
    from colours import MODALITY_COLOR_MAP  # type: ignore
    from utils import build_and_export_html  # type: ignore

__all__ = ["generate_dataset_growth"]


def _empty_figure(reason: str, out_html: str | Path) -> Path:
    """Render an empty placeholder when there is genuinely nothing to plot."""
    fig = go.Figure()
    fig.add_annotation(text=reason, showarrow=False)
    fig.update_layout(template="plotly_white", height=550)
    return build_and_export_html(
        fig, out_html, div_id="dataset-growth-plot", height=550
    )


def _series_to_frame(series: Mapping[str, list[dict]]) -> pd.DataFrame:
    """Flatten the server's ``{modality: [{period, count, cumulative}]}`` block."""
    rows: list[dict] = []
    for modality, entries in series.items():
        for entry in entries:
            try:
                year = int(str(entry.get("period", "")))
            except ValueError:
                continue
            rows.append(
                {
                    "modality": modality,
                    # Pin the bucket label on July 1 of the bucket year — the
                    # midpoint convention the legacy chart used for missing
                    # exact dates. Plotly renders this as a date axis.
                    "date": pd.Timestamp(year=year, month=7, day=1),
                    "count": int(entry.get("count", 0) or 0),
                    "cumulative": int(entry.get("cumulative", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def generate_dataset_growth(
    df: pd.DataFrame | None = None,
    out_html: str | Path = "dataset_growth.html",
    *,
    width: int = 1260,
    snapshot: object | None = None,
) -> Path:
    """Render the cumulative growth chart from the server aggregation."""
    del df  # The catalog DataFrame is no longer the source of truth.
    del width  # Layout owns its own sizing; arg retained for caller parity.

    try:
        response = fetch_aggregation(
            "growth",
            {"bucket": "year", "field": "dataset_created_at"},
        )
    except AggregationFetchError as exc:
        return _empty_figure(f"Growth data unavailable: {exc}", out_html)

    series = response.payload.get("series") or {}
    frame = _series_to_frame(series)
    if frame.empty:
        return _empty_figure("No growth data available", out_html)

    # Sum across modalities for the headline "all datasets" curve. The
    # server returns per-modality cumulatives; the all-up curve is just
    # the per-bucket sum of new datasets, then cumulated.
    bucket_counts: dict[pd.Timestamp, int] = {}
    for _, row in frame.iterrows():
        bucket_counts[row["date"]] = bucket_counts.get(row["date"], 0) + int(
            row["count"]
        )
    bucket_dates = sorted(bucket_counts)
    cumulative = 0
    cumulative_rows: list[dict] = []
    for ts in bucket_dates:
        cumulative += bucket_counts[ts]
        cumulative_rows.append({"date": ts, "cumulative": cumulative})
    totals = pd.DataFrame(cumulative_rows)

    modalities = sorted(frame["modality"].unique())
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=totals["date"],
            y=totals["cumulative"],
            mode="lines",
            name="All datasets",
            legendgroup="all",
            visible=True,
            line=dict(width=3.5, color="#0f172a"),
            hovertemplate=("<b>%{x|%Y}</b><br>Total datasets: %{y:,}<extra></extra>"),
        )
    )
    for mod in modalities:
        grp = frame[frame["modality"] == mod].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["cumulative"],
                mode="lines",
                name=mod,
                legendgroup=mod,
                visible=True,
                line=dict(width=2, color=MODALITY_COLOR_MAP.get(mod, "#999999")),
                hovertemplate=(
                    f"<b>{mod}</b><br>%{{x|%Y}}<br>Cumulative: %{{y:,}}<extra></extra>"
                ),
            )
        )

    latest_x = totals["date"].iloc[-1]
    latest_y = int(totals["cumulative"].iloc[-1])
    fig.add_annotation(
        x=latest_x,
        y=latest_y,
        text=f"<b>{latest_y:,}</b>",
        showarrow=False,
        xshift=14,
        yshift=4,
        font=dict(size=13, color="#0f172a"),
    )

    fig.update_layout(
        yaxis=dict(
            title=dict(text="Cumulative datasets", font=dict(size=13)),
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            rangemode="tozero",
        ),
        xaxis=dict(
            title=dict(text="Publication date", font=dict(size=13)),
            gridcolor="rgba(15, 23, 42, 0.04)",
            showline=True,
            linecolor="rgba(15, 23, 42, 0.2)",
        ),
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0f172a"),
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(15, 23, 42, 0.1)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(15, 23, 42, 0.2)",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
        ),
        hovermode="x unified",
        margin=dict(t=60, l=60, r=40, b=60),
        height=550,
        autosize=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-growth-plot",
        height=550,
    )
