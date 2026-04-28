"""Utilities to generate the dataset growth plot."""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from .colours import MODALITY_COLOR_MAP
    from .utils import (
        build_and_export_html,
        detect_modality_column,
        normalize_modality_string,
    )
except ImportError:
    from colours import MODALITY_COLOR_MAP  # type: ignore
    from utils import (  # type: ignore
        build_and_export_html,
        detect_modality_column,
        normalize_modality_string,
    )

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _author_year_to_midyear(v: object) -> pd.Timestamp:
    """Parse a ``FirstAuthorYYYY`` tag into a July-1 UTC Timestamp.

    Used as a fallback when ``dataset_created_at`` is missing: the original
    publication year is the best signal we have for "when did this dataset
    land in the ecosystem" on the growth curve.
    """
    match = _YEAR_RE.search(str(v or ""))
    if not match:
        return pd.NaT
    year = int(match.group(0))
    return pd.Timestamp(f"{year}-07-01", tz="UTC")


def generate_dataset_growth(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    width: int = 1260,
) -> Path:
    """Generate cumulative growth plot."""
    df = df.copy()

    # Extract creation date. Accept both the API field and legacy alias.
    date_col = None
    for col in ["dataset_created_at", "created", "created_at"]:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        # Create empty plot with annotation
        fig = px.line(
            title="Cumulative Number of Dataset Publications (Data Unavailable)"
        )
        fig.add_annotation(text="Timestamp data missing", showarrow=False)
        return build_and_export_html(
            fig, out_html, div_id="dataset-growth-plot", height=550
        )

    # Coerce dates; empty strings and malformed values become NaT. The
    # source column mixes pure ISO dates ("2024-10-04") and full ISO
    # timestamps with Z suffix ("2026-03-17T11:47:46.636Z"). Pandas' single-
    # format parser fails on the mix; format="mixed" handles both.
    df["date"] = pd.to_datetime(
        df[date_col].astype(str).str.strip(), errors="coerce", format="mixed", utc=True
    )

    # Fallback: many NEMAR mirrors lack dataset_created_at but carry a
    # publication year inside author_year ("Shirazi2017"). Use that year's
    # midpoint (July 1) so those 170+ datasets appear on the growth curve
    # instead of being silently dropped. This is fuzzy but honest — it
    # reflects the *original publication* year, which is what a catalog-
    # growth chart is trying to communicate.
    if "author_year" in df.columns:
        missing_mask = df["date"].isna()
        if missing_mask.any():
            fallback_dates = df.loc[missing_mask, "author_year"].map(
                _author_year_to_midyear
            )
            df.loc[missing_mask, "date"] = fallback_dates

    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.sort_values("date")

    # Ensure subjects column is numeric
    subj_col = "n_subjects" if "n_subjects" in df.columns else "subjects"
    if subj_col in df.columns:
        df["n_subjects_clean"] = pd.to_numeric(df[subj_col], errors="coerce").fillna(0)
    else:
        df["n_subjects_clean"] = 0

    mod_col = detect_modality_column(df)

    if mod_col:
        df["Modality"] = df[mod_col].apply(normalize_modality_string)
    else:
        df["Modality"] = "Unknown"

    parts = []
    for mod, group in df.groupby("Modality"):
        group = group.sort_values("date")
        group["cumulative_datasets"] = range(1, len(group) + 1)
        group["cumulative_subjects"] = group["n_subjects_clean"].cumsum()
        parts.append(group)

    if not parts:
        fig = px.line(title="Cumulative Number of Dataset Publications (No Data)")
        return build_and_export_html(
            fig, out_html, div_id="dataset-growth-plot", height=550
        )

    df_plot = pd.concat(parts)

    # We use Graph Objects directly to support better updatemenu control or update px figure?
    # Px is easier to generate the traces. We can generate two sets of traces?
    # Or just use updatemenus to switch y-data.

    # Let's use plotly express to create the initial figure (datasets)
    fig = px.line(
        df_plot,
        x="date",
        y="cumulative_datasets",
        color="Modality",
        title="Cumulative Growth",
        labels={
            "date": "Publication Date",
            "cumulative_datasets": "Number of Datasets",
        },
    )

    fig.update_layout(template="plotly_white", font=dict(size=14))
    fig.update_traces(line=dict(width=3))

    # To implement the button, we need to supply the 'cumulative_subjects' data to the traces.
    # However, px splits traces by color (modality).
    # We need to construct the update dictionary carefully.
    # It's safer to build with graph objects if we want precision, or just stick to switching the whole data source (via transforms which are deprecated/complex)
    # Easiest way: re-assign y-values in the JS update.
    # But we need to know WHICH trace corresponds to WHICH modality to grab the right y-values.

    # ALTERNATIVE: Create two sets of traces, one visible, one hidden, and toggle visibility.
    # This is standard for dropdowns.

    fig = go.Figure()

    modalities = sorted(df_plot["Modality"].unique())
    # Also compute the "All datasets" overall cumulative line across
    # modalities, so the main story — how the whole catalog grew — is
    # immediately visible instead of buried under per-modality splits.
    df_all = df.sort_values("date").copy()
    df_all["cumulative_datasets"] = range(1, len(df_all) + 1)
    df_all["cumulative_subjects"] = df_all["n_subjects_clean"].cumsum()

    # Trace 0: All datasets combined (shown by default, highlighted).
    fig.add_trace(
        go.Scatter(
            x=df_all["date"],
            y=df_all["cumulative_datasets"],
            mode="lines",
            name="All datasets",
            legendgroup="all",
            visible=True,
            line=dict(width=3.5, color="#0f172a"),
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>Total datasets: %{y:,}<extra></extra>"
            ),
        )
    )

    # Trace group 1: Datasets per modality
    for mod in modalities:
        grp = df_plot[df_plot["Modality"] == mod].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["cumulative_datasets"],
                mode="lines",
                name=mod,
                legendgroup=mod,
                visible=True,
                line=dict(width=2, color=MODALITY_COLOR_MAP.get(mod, "#999999")),
                hovertemplate=(
                    f"<b>{mod}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Cumulative: %{y:,}<extra></extra>"
                ),
            )
        )

    # Trace 0': All subjects
    fig.add_trace(
        go.Scatter(
            x=df_all["date"],
            y=df_all["cumulative_subjects"],
            mode="lines",
            name="All subjects",
            legendgroup="all",
            visible=False,
            line=dict(width=3.5, color="#0f172a"),
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>Total subjects: %{y:,}<extra></extra>"
            ),
        )
    )

    # Trace group 2: Subjects per modality
    for mod in modalities:
        grp = df_plot[df_plot["Modality"] == mod].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["cumulative_subjects"],
                mode="lines",
                name=mod,
                legendgroup=mod,
                visible=False,
                line=dict(width=2, color=MODALITY_COLOR_MAP.get(mod, "#999999")),
                showlegend=False,
                hovertemplate=(
                    f"<b>{mod}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Cumulative subjects: %{y:,}<extra></extra>"
                ),
            )
        )

    # Update Menus — trace layout:
    #   [0]           All datasets
    #   [1..n_mods]   Per-modality datasets
    #   [n_mods+1]    All subjects
    #   [n_mods+2..]  Per-modality subjects
    n_mods = len(modalities)
    datasets_block = [True] * (1 + n_mods) + [False] * (1 + n_mods)
    subjects_block = [False] * (1 + n_mods) + [True] * (1 + n_mods)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            {"visible": datasets_block},
                            {"yaxis.title.text": "Cumulative datasets"},
                        ],
                        label="Datasets",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"visible": subjects_block},
                            {"yaxis.title.text": "Cumulative subjects"},
                        ],
                        label="Subjects",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.14,
                yanchor="top",
                bgcolor="#ffffff",
                bordercolor="rgba(15, 23, 42, 0.12)",
                borderwidth=1,
                font=dict(size=12),
            ),
        ]
    )

    # Annotate the latest point with the current total so the "where we are
    # today" datum is readable without hover.
    latest_x = df_all["date"].iloc[-1]
    latest_y = int(df_all["cumulative_datasets"].iloc[-1])
    fig.add_annotation(
        x=latest_x,
        y=latest_y,
        text=f"<b>{latest_y:,}</b>",
        showarrow=False,
        xshift=14,
        yshift=4,
        font=dict(size=13, color="#0f172a"),
    )

    # The bold headline lives in the .rst figure-title block above; the
    # chart itself ships untitled so it doesn't fight the rst H3.
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
