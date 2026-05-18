"""Export EEGDash dataset figures to PDF for paper inclusion.

Usage:
    cd docs
    python export_paper_figures.py [--output-dir paper_figures]

Generates publication-quality PDF exports of all Plotly-based charts.
Requires kaleido: pip install kaleido
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure eegdash package is importable (this script lives in docs/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_dataset import (
    generate_clinical_stacked_bar,
    generate_dataset_bubble,
    generate_dataset_growth,
    generate_dataset_sankey,
    generate_dataset_treemap,
    generate_moabb_bubble,
)
from plot_dataset.ridgeline import _build_ridgeline_traces
from plot_dataset.utils import (
    get_figure_registry,
    primary_modality,
    primary_recording_modality,
)

from eegdash.dataset.snapshot import DatasetSnapshot

# API Configuration
API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"

# Paper figure dimensions (pixels at scale=1 → 1:1 with PDF points)
# Kaleido converts px to pt at 72/96 ratio. To get 183mm (519pt) native PDF width:
# 519pt * (96/72) = 693px. This ensures fonts render at true size in the final PDF.
LANDSCAPE_W, LANDSCAPE_H = 693, 453
TALL_W, TALL_H = 933, 733
PDF_SCALE = 1

# Publication font — Helvetica/Arial for Scientific Data compliance
PUB_FONT = "Helvetica, Arial, sans-serif"


def _load_local_csv() -> pd.DataFrame:
    """Load the local dataset_summary.csv for enrichment."""
    csv_path = (
        Path(__file__).resolve().parent.parent
        / "eegdash"
        / "dataset"
        / "dataset_summary.csv"
    )
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=False, header=0, skipinitialspace=True)
    return pd.DataFrame()


def load_data() -> pd.DataFrame:
    """Fetch dataset data from the EEGDash API."""
    print("Fetching data from API...")
    snapshot = DatasetSnapshot.build(
        api_base=API_BASE_URL, database=DEFAULT_DATABASE, limit=1000
    )
    print(
        f"  DatasetSnapshot source={snapshot.source} "
        f"dataset_count={snapshot.dataset_count}"
    )
    df_raw = snapshot.rows()

    # Enrich with local CSV for modality info
    fallback_df = _load_local_csv()
    if not fallback_df.empty and not df_raw.empty:
        if "dataset" not in df_raw.columns and "dataset_id" in df_raw.columns:
            df_raw["dataset"] = df_raw["dataset_id"]
        enrich_cols = ["dataset", "record_modality", "modality of exp"]
        enrich_df = fallback_df[
            [c for c in enrich_cols if c in fallback_df.columns]
        ].copy()
        if not enrich_df.empty and "dataset" in enrich_df.columns:
            df_raw = df_raw.merge(
                enrich_df, on="dataset", how="left", suffixes=("", "_csv")
            )
            if "record_modality_csv" in df_raw.columns:
                if "recording_modality" not in df_raw.columns:
                    df_raw["recording_modality"] = pd.Series(
                        index=df_raw.index, dtype="object"
                    )
                df_raw["recording_modality"] = df_raw[
                    "record_modality_csv"
                ].combine_first(df_raw["recording_modality"])
                df_raw["record_modality"] = df_raw["record_modality_csv"]

    print(f"  Loaded {len(df_raw)} datasets")
    return df_raw


def clean_for_print(fig: go.Figure) -> go.Figure:
    """Remove interactive elements and enforce publication styling."""
    fig_dict = fig.to_dict()
    fig_dict.get("layout", {}).pop("updatemenus", None)
    fig_dict.get("layout", {}).pop("sliders", None)
    fig_dict["layout"]["paper_bgcolor"] = "white"
    fig_dict["layout"]["plot_bgcolor"] = "white"
    cleaned = go.Figure(fig_dict)

    # Enforce publication font family across all text
    cleaned.update_layout(font=dict(family=PUB_FONT))

    # Bump axis label and tick sizes for print readability
    cleaned.update_xaxes(
        title_font=dict(size=12, family=PUB_FONT),
        tickfont=dict(size=10, family=PUB_FONT),
    )
    cleaned.update_yaxes(
        title_font=dict(size=12, family=PUB_FONT),
        tickfont=dict(size=10, family=PUB_FONT),
    )

    # Enforce legend font
    cleaned.update_layout(
        legend=dict(font=dict(size=9, family=PUB_FONT)),
    )

    return cleaned


def export_pdf(fig: go.Figure, path: Path, *, width: int, height: int) -> None:
    """Export a Plotly figure to PDF."""
    fig.write_image(
        str(path), width=width, height=height, scale=PDF_SCALE, format="pdf"
    )
    print(f"  -> {path.name}")


def _extract_visibility(fig: go.Figure, button_idx: int) -> list | None:
    """Extract visibility array from updatemenus button args."""
    try:
        menus = fig.layout.updatemenus
        if not menus:
            return None
        btn = menus[0].buttons[button_idx]
        args = btn.args
        if isinstance(args, (list, tuple)):
            # restyle method: args = ["visible", [True, False, ...]]
            if len(args) >= 2 and args[0] == "visible":
                return list(args[1])
            # update method: args = [{"visible": [...]}, ...]
            if isinstance(args[0], dict) and "visible" in args[0]:
                return list(args[0]["visible"])
        return None
    except (IndexError, AttributeError, TypeError):
        return None


def _extract_y_data(fig: go.Figure, button_idx: int) -> list | None:
    """Extract y-data arrays from updatemenus button args (for restyle)."""
    try:
        menus = fig.layout.updatemenus
        if not menus:
            return None
        btn = menus[0].buttons[button_idx]
        args = btn.args
        if (
            isinstance(args, (list, tuple))
            and isinstance(args[0], dict)
            and "y" in args[0]
        ):
            return args[0]["y"]
        return None
    except (IndexError, AttributeError, TypeError):
        return None


def _set_visibility(fig: go.Figure, vis_list: list) -> None:
    """Set trace visibility from a list."""
    for i, vis in enumerate(vis_list):
        if i < len(fig.data):
            fig.data[i].visible = True if vis is True or vis == "legendonly" else vis


def _fix_bubble_for_print(fig: go.Figure) -> None:
    """Bubble-chart-specific print fixes: legend below, lighter gridlines, outlier labels."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=9, family=PUB_FONT),
        ),
        margin=dict(l=60, r=80, t=40, b=90),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")


def _collect_bubble_outliers(
    fig_orig: go.Figure, vis_list: list, n_outliers: int = 5
) -> list[dict]:
    """Identify top outlier datasets from the original (non-serialized) figure.

    Must be called on the *original* figure before ``copy.deepcopy``,
    because deep-copying a Plotly figure serializes trace arrays into
    binary dicts that cannot be iterated.

    Returns a list of dicts with keys: name, x, y, subjects, records.
    """
    points: list[dict] = []
    for trace_idx, trace in enumerate(fig_orig.data):
        # Determine effective visibility from the button vis_list
        if vis_list and trace_idx < len(vis_list):
            vis = vis_list[trace_idx]
            if vis is False:
                continue
        elif trace.visible is False:
            continue

        if not hasattr(trace, "customdata") or trace.customdata is None:
            continue
        if trace.x is None or trace.y is None:
            continue

        # Convert to plain lists (handles numpy arrays & pandas Series)
        x_arr = np.asarray(trace.x).tolist()
        y_arr = np.asarray(trace.y).tolist()
        cd_arr = list(trace.customdata)
        size_arr: list = []
        try:
            size_arr = np.asarray(trace.marker.size).tolist()
        except (TypeError, AttributeError):
            pass

        for i, cd in enumerate(cd_arr):
            if cd is None or len(cd) < 3:
                continue
            if i >= len(x_arr) or i >= len(y_arr):
                continue
            try:
                x_val = float(x_arr[i])
                y_val = float(y_arr[i])
            except (TypeError, ValueError):
                continue
            if x_val <= 0 or y_val <= 0:
                continue
            try:
                subjects = float(cd[1]) if cd[1] else 0
                records = float(cd[2]) if cd[2] else 0
            except (TypeError, ValueError):
                subjects, records = 0, 0
            bsize = 0.0
            if i < len(size_arr):
                try:
                    bsize = float(size_arr[i])
                except (TypeError, ValueError):
                    pass
            points.append(
                {
                    "name": str(cd[0]),
                    "x": x_val,
                    "y": y_val,
                    "subjects": subjects,
                    "records": records,
                    "bubble_size": bsize,
                }
            )

    if not points:
        return []

    df_pts = pd.DataFrame(points)

    # Select diverse outliers across different dimensions to ensure spatial
    # spread.  Each criterion adds at most one new dataset until we reach
    # the target count.
    selected_names: list[str] = []
    criteria = [
        ("subjects", 1),  # most subjects (top-right on chart)
        ("records", 1),  # most records (top of chart)
        ("bubble_size", 1),  # longest duration per subject (biggest bubble)
    ]
    # Also add records/subject ratio to catch extreme upper-left outliers
    df_pts["_rec_per_subj"] = df_pts["records"] / df_pts["subjects"].clip(lower=1)
    criteria.append(("_rec_per_subj", 1))

    # Round-robin through criteria until we have enough
    while len(selected_names) < n_outliers:
        added_any = False
        for col, _ in criteria:
            ranked = df_pts.nlargest(n_outliers * 2, col)
            for name in ranked["name"]:
                if name not in selected_names:
                    selected_names.append(name)
                    added_any = True
                    break
            if len(selected_names) >= n_outliers:
                break
        if not added_any:
            break

    selected = df_pts[df_pts["name"].isin(selected_names)].drop_duplicates(
        subset="name", keep="first"
    )
    return selected.to_dict("records")


def _add_outlier_annotations(fig: go.Figure, outliers: list[dict]) -> None:
    """Add dataset-name annotations for pre-computed outlier bubbles.

    Coordinates are placed in log10 space to match the bubble chart's
    logarithmic axes.  Each label is placed in a quadrant chosen to
    avoid the dense data-cloud centre and the existing "One record per
    subject" annotation that occupies the upper-middle area of the plot.
    """
    if not outliers:
        return

    # Sort by descending y (records) so upper points are labelled first
    pts = sorted(outliers, key=lambda p: p["y"], reverse=True)

    # Assign label placement per-point based on its chart quadrant.
    # The "One record per subject" text lives around log_x ~2.4,
    # log_y ~3.5, so labels near that zone are pushed sideways.
    used_sectors: list[tuple[int, int]] = []
    length = 55  # leader-line length in pixels

    for pt in pts:
        px, py = math.log10(pt["x"]), math.log10(pt["y"])

        # Choose offset direction based on where the point sits.
        # Key constraints to avoid:
        #   - "One record per subject" text near log_x~2.4, log_y~3.6
        #   - Bubble-size legend in the right-centre (~log_x>2.5, log_y~2.5-3.2)
        if px > 2.9:
            # Far right — label to the upper-right
            ax, ay = length, -30
        elif px < 1.2:
            # Far left — label to the left and up
            ax, ay = -length, -15
        elif py > 4.0:
            # Very top — label upward
            ax, ay = 10, -length
        elif py > 3.6 and px < 2.5:
            # Upper-left quadrant — push up and slightly left
            ax, ay = -30, -length
        elif py > 3.4 and px >= 2.5:
            # Upper-right near the "1:1" annotation — push down-left
            ax, ay = -length, 20
        elif py > 3.4:
            # Middle-upper — push left
            ax, ay = -length, -10
        else:
            # Default — push up and left
            ax, ay = -length, -25

        # Nudge if another label already occupies a similar sector
        # (quantised to 25px grid).
        sector = (ax // 25, ay // 25)
        while sector in used_sectors:
            ay += 22  # shift down
            sector = (ax // 25, ay // 25)
        used_sectors.append(sector)

        fig.add_annotation(
            x=px,
            y=py,
            text=pt["name"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.8,
            arrowcolor="#555",
            ax=ax,
            ay=ay,
            axref="pixel",
            ayref="pixel",
            font=dict(size=8, family=PUB_FONT, color="#222"),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
            xref="x",
            yref="y",
        )


def export_bubble(registry: dict, out_dir: Path) -> None:
    """Export bubble chart — both experiment and recording modality views."""
    fig_orig = registry.get("dataset-bubble")
    if fig_orig is None:
        print("  Bubble chart: not found in registry, skipping")
        return

    exp_vis = _extract_visibility(fig_orig, 0)
    rec_vis = _extract_visibility(fig_orig, 1)

    if exp_vis:
        # Collect outliers from original figure (numpy arrays still intact)
        exp_outliers = _collect_bubble_outliers(fig_orig, exp_vis)
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, exp_vis)
        fig = clean_for_print(fig)
        _fix_bubble_for_print(fig)
        _add_outlier_annotations(fig, exp_outliers)
        export_pdf(
            fig,
            out_dir / "bubble_exp_modality.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )

    # Recording modality (alternate view — button 1)
    if rec_vis:
        rec_outliers = _collect_bubble_outliers(fig_orig, rec_vis)
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, rec_vis)
        fig = clean_for_print(fig)
        _fix_bubble_for_print(fig)
        _add_outlier_annotations(fig, rec_outliers)
        export_pdf(
            fig,
            out_dir / "bubble_rec_modality.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )


def _export_svg_to_pdf(
    fig: go.Figure,
    pdf_path: Path,
    *,
    width: int,
    height: int,
) -> None:
    """Export a Plotly figure to PDF via SVG intermediate.

    Kaleido's Chromium-based PDF renderer duplicates text labels 3-5x at
    slight offsets in Sankey diagrams, creating an illegible "bold blur".
    Exporting to SVG first and converting to PDF with CairoSVG (or
    rsvg-convert) avoids the Chromium text-rendering bug entirely.
    """
    import shutil
    import subprocess

    svg_path = pdf_path.with_suffix(".svg")

    # Step 1: Plotly -> SVG (Kaleido SVG output is clean)
    fig.write_image(
        str(svg_path), width=width, height=height, scale=PDF_SCALE, format="svg"
    )

    # Step 2: SVG -> PDF via rsvg-convert (preferred) or cairosvg fallback
    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    converted = False

    if Path(rsvg).is_file():
        try:
            subprocess.run(
                [rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
                check=True,
                capture_output=True,
            )
            converted = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not converted:
        try:
            import cairosvg

            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
            converted = True
        except Exception as exc:
            print(f"  WARNING: cairosvg fallback failed: {exc}")

    if not converted:
        # Last resort: fall back to direct Kaleido PDF (with the artifact)
        print("  WARNING: SVG->PDF conversion unavailable, falling back to Kaleido PDF")
        fig.write_image(
            str(pdf_path), width=width, height=height, scale=PDF_SCALE, format="pdf"
        )

    # Clean up intermediate SVG
    svg_path.unlink(missing_ok=True)
    print(f"  -> {pdf_path.name}")


def export_sankey(registry: dict, out_dir: Path) -> None:
    """Export Sankey diagram.

    Uses SVG intermediate export to avoid Kaleido/Chromium 5x text
    rendering artifact that makes small labels illegible in direct PDF.
    """
    fig_orig = registry.get("dataset-sankey")
    if fig_orig is None:
        print("  Sankey: not found in registry, skipping")
        return
    fig = copy.deepcopy(fig_orig)
    fig = clean_for_print(fig)
    # Fix Sankey for print: smaller font, clean text rendering
    for trace in fig.data:
        if hasattr(trace, "node") and trace.node:
            trace.node.label = [l if l else "" for l in (trace.node.label or [])]
            trace.textfont = dict(size=8, family=PUB_FONT, color="black")
        if hasattr(trace, "link") and trace.link:
            trace.link.label = [""] * len(trace.link.label or [])
    fig.update_layout(
        font=dict(size=8, family=PUB_FONT),
        margin=dict(l=5, r=5, t=30, b=30),
    )
    # Use SVG->PDF pipeline to bypass Chromium text duplication bug
    _export_svg_to_pdf(fig, out_dir / "sankey.pdf", width=TALL_W, height=TALL_H)


def _strip_emojis(text: str) -> str:
    """Remove emoji characters from text (keep ASCII + common unicode letters)."""
    import re

    # Remove characters outside basic multilingual plane + common symbols
    return re.sub(
        r"[\U0001F300-\U0001FFFF\u2600-\u27BF\u2B50\u2B55\u231A-\u23FF]",
        "",
        text,
    ).strip()


def export_treemap(registry: dict, out_dir: Path) -> None:
    """Export treemap."""
    fig_orig = registry.get("dataset-treemap-plot")
    if fig_orig is None:
        print("  Treemap: not found in registry, skipping")
        return
    fig = copy.deepcopy(fig_orig)
    fig = clean_for_print(fig)
    # Remove emojis from all trace names and legend entries
    for trace in fig.data:
        if hasattr(trace, "name") and trace.name:
            trace.name = _strip_emojis(trace.name)
        # Treemap text/labels
        if hasattr(trace, "text") and trace.text:
            if isinstance(trace.text, (list, tuple)):
                trace.text = [_strip_emojis(str(t)) for t in trace.text]
        if hasattr(trace, "labels") and trace.labels:
            if isinstance(trace.labels, (list, tuple)):
                trace.labels = [_strip_emojis(str(l)) for l in trace.labels]
    # Show more labels: lower the minimum text size threshold
    fig.update_layout(
        uniformtext=dict(minsize=7, mode="hide"),
        treemapcolorway=None,
    )
    # Increase treemap internal text size for print
    for trace in fig.data:
        if hasattr(trace, "textfont"):
            trace.textfont = dict(size=9, family=PUB_FONT)
        if hasattr(trace, "insidetextfont"):
            trace.insidetextfont = dict(size=9, family=PUB_FONT)
    export_pdf(fig, out_dir / "treemap.pdf", width=TALL_W, height=600)


_DASH_MAP = {
    "EEG": "solid",
    "EMG": "dash",
    "MEG": "dot",
    "fNIRS": "dashdot",
    "iEEG": "longdash",
}


def _apply_dash_patterns(figure):
    """Apply distinct dash patterns per modality for B&W printing."""
    for trace in figure.data:
        if trace.visible is True or trace.visible is None:
            dash = _DASH_MAP.get(trace.name, "solid")
            trace.update(line_dash=dash, line_width=2)


def _style_growth(figure, y_title):
    """Style growth chart for subfigure display with log y-axis."""
    y_title = y_title + " (log scale)"
    figure.update_layout(
        yaxis_title=dict(text=y_title, font=dict(size=20, family=PUB_FONT)),
        xaxis_title=dict(font=dict(size=20, family=PUB_FONT)),
        font=dict(size=18, family=PUB_FONT),
        legend=dict(font=dict(size=14, family=PUB_FONT)),
        margin=dict(l=70, r=20, t=20, b=60),
    )
    figure.update_xaxes(tickfont=dict(size=16, family=PUB_FONT))
    figure.update_yaxes(tickfont=dict(size=16, family=PUB_FONT))
    figure.update_yaxes(type="log", dtick=1)


def export_growth(registry: dict, out_dir: Path) -> None:
    """Export growth chart — both datasets and subjects views."""
    fig_orig = registry.get("dataset-growth-plot")
    if fig_orig is None:
        print("  Growth chart: not found in registry, skipping")
        return

    ds_vis = _extract_visibility(fig_orig, 0)
    subj_vis = _extract_visibility(fig_orig, 1)

    if ds_vis:
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, ds_vis)
        fig = clean_for_print(fig)
        _apply_dash_patterns(fig)
        _style_growth(fig, "Cumulative Number of Datasets")
        export_pdf(
            fig, out_dir / "growth_datasets.pdf", width=LANDSCAPE_W, height=LANDSCAPE_H
        )

    if subj_vis:
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, subj_vis)
        # Fix subjects traces: solid lines + show legend (they were hidden/dotted by design)
        for trace in fig.data:
            if trace.visible is True or trace.visible is None:
                if hasattr(trace, "line") and trace.line and trace.line.dash:
                    trace.line.dash = "solid"
                trace.showlegend = True
        fig = clean_for_print(fig)
        _apply_dash_patterns(fig)
        _style_growth(fig, "Cumulative Number of Subjects")
        export_pdf(
            fig, out_dir / "growth_subjects.pdf", width=LANDSCAPE_W, height=LANDSCAPE_H
        )


def _collapse_clinical_categories(fig: go.Figure) -> go.Figure:
    """Merge rare pathology categories into 'Other Clinical' for print clarity.

    Also applies high-contrast Okabe-Ito-inspired colours so that the
    categories are distinguishable in print, on screen, and for readers
    with colour-vision deficiency.
    """
    KEEP = {"Healthy", "Epilepsy", "Parkinson's", "Unknown", "Development", "Other"}

    # Okabe-Ito palette mapped to the kept categories + merged bucket
    _OI_COLORS = {
        "Healthy": "#009E73",  # bluish green
        "Epilepsy": "#E69F00",  # orange
        "Parkinson's": "#CC79A7",  # reddish purple
        "Unknown": "#999999",  # grey
        "Development": "#56B4E9",  # sky blue
        "Other": "#F0E442",  # yellow
        "Other Clinical": "#D55E00",  # vermilion
    }

    merge_traces = []
    keep_traces = []
    for trace in fig.data:
        if trace.name in KEEP:
            keep_traces.append(trace)
        else:
            merge_traces.append(trace)

    if not merge_traces:
        # Still apply Okabe-Ito colours even when nothing needs merging
        for t in keep_traces:
            if t.name in _OI_COLORS:
                t.marker.color = _OI_COLORS[t.name]
        return fig

    # Create merged "Other Clinical" trace by summing y-values
    base_y = np.zeros(len(merge_traces[0].x))
    for t in merge_traces:
        y_arr = np.array([float(v) if v is not None else 0.0 for v in t.y])
        base_y += y_arr

    other_trace = go.Bar(
        x=list(merge_traces[0].x),
        y=base_y.tolist(),
        name="Other Clinical",
        marker=dict(color=_OI_COLORS["Other Clinical"]),
    )

    new_fig = go.Figure()
    for t in keep_traces:
        if t.name in _OI_COLORS:
            t.marker.color = _OI_COLORS[t.name]
        new_fig.add_trace(t)
    new_fig.add_trace(other_trace)
    new_fig.update_layout(fig.layout)
    new_fig.update_layout(barmode="stack")
    return new_fig


def _annotate_bar_totals(fig: go.Figure) -> None:
    """Add total-count annotations on top of each stacked bar."""
    if not fig.data:
        return
    x_cats = list(fig.data[0].x)
    totals = {x: 0.0 for x in x_cats}
    for trace in fig.data:
        for x_val, y_val in zip(trace.x, trace.y):
            totals[x_val] += float(y_val) if y_val is not None else 0.0

    for x_val, total in totals.items():
        if total > 0:
            fig.add_annotation(
                x=x_val,
                y=total,
                text=str(int(total)),
                showarrow=False,
                font=dict(size=14, family=PUB_FONT, color="black"),
                yshift=10,
                xanchor="center",
            )


def _split_clinical_panels(
    fig: go.Figure,
) -> tuple[go.Figure, go.Figure]:
    """Split a clinical stacked-bar figure into EEG-only and non-EEG panels.

    Returns (fig_eeg, fig_noneeg).  Each panel keeps only the relevant
    modality bars; the non-EEG panel gets a tighter y-range so the
    smaller bars are fully readable.
    """
    x_cats = list(fig.data[0].x) if fig.data else []
    eeg_idx = x_cats.index("EEG") if "EEG" in x_cats else None

    # --- Panel (a): EEG only ---
    fig_eeg = go.Figure()
    for trace in fig.data:
        y_vals = list(trace.y)
        if eeg_idx is not None:
            new_y = [y_vals[eeg_idx]]
        else:
            new_y = y_vals
        fig_eeg.add_trace(
            go.Bar(
                x=["EEG"],
                y=new_y,
                name=trace.name,
                marker=dict(color=trace.marker.color),
                showlegend=True,
            )
        )
    fig_eeg.update_layout(fig.layout)
    fig_eeg.update_layout(barmode="stack")

    # --- Panel (b): non-EEG modalities ---
    non_eeg_cats = [c for c in x_cats if c != "EEG"]
    non_eeg_idxs = [x_cats.index(c) for c in non_eeg_cats]

    fig_noneeg = go.Figure()
    for trace in fig.data:
        y_vals = list(trace.y)
        new_y = [y_vals[i] for i in non_eeg_idxs]
        fig_noneeg.add_trace(
            go.Bar(
                x=non_eeg_cats,
                y=new_y,
                name=trace.name,
                marker=dict(color=trace.marker.color),
                showlegend=True,
            )
        )
    fig_noneeg.update_layout(fig.layout)
    fig_noneeg.update_layout(barmode="stack")

    # Compute tight y-range for non-EEG panel
    stack_totals = {c: 0.0 for c in non_eeg_cats}
    for trace in fig_noneeg.data:
        for x_val, y_val in zip(trace.x, trace.y):
            stack_totals[x_val] += float(y_val) if y_val is not None else 0.0
    max_stack = max(stack_totals.values()) if stack_totals else 0.0
    y_ceil = max(10, int(max_stack * 1.15 / 10 + 1) * 10)  # round up, +15%
    fig_noneeg.update_yaxes(range=[0, y_ceil])

    return fig_eeg, fig_noneeg


_CLIN_FONT = dict(size=18, family=PUB_FONT)


def _style_clinical(figure, y_title):
    """Style clinical chart for subfigure display."""
    figure = _collapse_clinical_categories(figure)
    figure.update_layout(
        yaxis_title=dict(text=y_title, font=dict(size=20, family=PUB_FONT)),
        xaxis_title=dict(font=dict(size=20, family=PUB_FONT)),
        font=_CLIN_FONT,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=14, family=PUB_FONT),
        ),
        margin=dict(l=70, r=20, t=30, b=110),
    )
    figure.update_xaxes(tickfont=dict(size=16, family=PUB_FONT))
    figure.update_yaxes(tickfont=dict(size=16, family=PUB_FONT))
    return figure


def export_clinical(registry: dict, out_dir: Path) -> None:
    """Export clinical breakdown as two-panel figures.

    EEG dominates (700+ studies) while other modalities have <70 each,
    making a single-panel chart unreadable. We export:
      - clinical_studies_eeg.pdf / clinical_subjects_eeg.pdf   (Panel a)
      - clinical_studies_noneeg.pdf / clinical_subjects_noneeg.pdf (Panel b)
    The paper combines them as subfigures.

    Also exports the combined single-panel version for backward compat.
    """
    fig_orig = registry.get("dataset-clinical-plot")
    if fig_orig is None:
        print("  Clinical chart: not found in registry, skipping")
        return

    y_counts = _extract_y_data(fig_orig, 0)
    y_subjects = _extract_y_data(fig_orig, 1)

    for y_data, view_name, y_title in [
        (y_counts, "studies", "Number of Studies"),
        (y_subjects, "subjects", "Number of Subjects"),
    ]:
        fig = copy.deepcopy(fig_orig)
        if y_data:
            for i, y_arr in enumerate(y_data):
                if i < len(fig.data):
                    fig.data[i].y = y_arr
        fig = clean_for_print(fig)
        fig = _style_clinical(fig, y_title)

        fig_eeg, fig_noneeg = _split_clinical_panels(fig)
        for panel in [fig_eeg, fig_noneeg]:
            panel.update_layout(
                yaxis_title=dict(text=y_title, font=dict(size=20, family=PUB_FONT)),
                xaxis_title=dict(font=dict(size=20, family=PUB_FONT)),
                font=_CLIN_FONT,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=14, family=PUB_FONT),
                ),
                margin=dict(l=70, r=20, t=30, b=110),
            )
            panel.update_xaxes(tickfont=dict(size=16, family=PUB_FONT))
            panel.update_yaxes(tickfont=dict(size=16, family=PUB_FONT))
            _annotate_bar_totals(panel)

        export_pdf(
            fig_eeg,
            out_dir / f"clinical_{view_name}_eeg.pdf",
            width=LANDSCAPE_W // 2,
            height=LANDSCAPE_H,
        )
        export_pdf(
            fig_noneeg,
            out_dir / f"clinical_{view_name}_noneeg.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )
        _annotate_bar_totals(fig)
        export_pdf(
            fig,
            out_dir / f"clinical_{view_name}.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )


def export_moabb(
    df_raw: pd.DataFrame, out_dir: Path, moabb_html: Path | None = None
) -> None:
    """Export MOABB circle-packing chart via Playwright headless browser.

    The MOABB chart uses D3.js canvas rendering, so we need a real browser
    to render it and capture the result as a high-resolution PNG.
    """
    # Generate the HTML if not provided
    if moabb_html is None or not moabb_html.exists():
        print("  Generating MOABB HTML first...")
        with tempfile.TemporaryDirectory() as tmp:
            moabb_html = Path(tmp) / "moabb.html"
            generate_moabb_bubble(df_raw.copy(), moabb_html, color_by="modality")

            if not moabb_html.exists():
                print("  MOABB HTML generation failed, skipping")
                return

            _capture_moabb(moabb_html, out_dir)
    else:
        _capture_moabb(moabb_html, out_dir)


def _capture_moabb(html_path: Path, out_dir: Path) -> None:
    """Capture MOABB canvas as publication-quality hero figure.

    Uses Playwright to render the D3.js canvas, then injects:
    - Clean removal of all interactive UI elements
    - Summary stats banner at the top
    - Publication-quality legend styling
    - Tight crop with DPI metadata
    """
    from PIL import Image as PILImage
    from playwright.sync_api import sync_playwright

    W, H = 1950, 1350  # wider viewport for more breathing room

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})

        page.goto(f"file://{html_path.resolve()}", wait_until="networkidle")
        page.wait_for_selector("canvas", timeout=15000)
        page.wait_for_timeout(4000)

        # Inject publication styling via JS
        page.evaluate("""() => {
    document.querySelectorAll('input, button, .eegdash-branding').forEach(el => el.remove());
    var tt = document.getElementById('moabb-tooltip');
    if (tt) tt.remove();
    document.querySelectorAll('div, p, span').forEach(el => {
        if (el.children.length === 0 && el.innerText && (el.innerText.includes('Hover') || el.innerText.includes('Scroll') || el.innerText.includes('Click to'))) el.style.visibility = 'hidden';
    });
    document.querySelectorAll('div').forEach(el => {
        if (el.children.length <= 3 && el.innerText && el.innerText.includes('Hover to highlight') && !el.querySelector('canvas')) el.style.visibility = 'hidden';
    });
    var leg = document.querySelector('[class*="legend"]');
    if (leg) { leg.style.fontFamily = 'Helvetica, Arial, sans-serif'; leg.style.fontSize = '13px'; leg.style.border = 'none'; leg.style.boxShadow = 'none'; leg.style.background = 'rgba(255,255,255,0.95)'; leg.style.borderRadius = '0'; }
    document.querySelectorAll('*').forEach(el => { try { var cs = getComputedStyle(el); if (cs.fontFamily.includes('system-ui') || cs.fontFamily.includes('Inter')) el.style.fontFamily = 'Helvetica, Arial, sans-serif'; } catch(e) {} });
    var b = document.createElement('div');
    b.style.cssText = 'position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:9999;font-family:Helvetica,Arial,sans-serif;font-size:15px;font-weight:600;color:#374151;letter-spacing:0.5px;background:rgba(255,255,255,0.92);padding:6px 24px;border-radius:4px;border:1px solid rgba(0,108,163,0.15)';
    b.innerHTML = '<span style="color:#006CA3;font-weight:700">700+</span> datasets \\u00B7 <span style="color:#006CA3;font-weight:700">35,000+</span> subjects \\u00B7 <span style="color:#006CA3;font-weight:700">65,000+</span> hours \\u00B7 <span style="color:#006CA3;font-weight:700">5</span> modalities';
    document.body.appendChild(b);
}""")

        page.wait_for_timeout(500)

        # Screenshot and crop
        raw_path = out_dir / "moabb_circle_packing_raw.png"
        page.screenshot(path=str(raw_path), full_page=False)
        browser.close()

    # Crop and set DPI
    img = PILImage.open(str(raw_path))
    import numpy as _np

    arr = _np.array(img)
    non_white = _np.any(arr < 245, axis=2)
    rows = _np.any(non_white, axis=1)
    cols = _np.any(non_white, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = _np.where(rows)[0][[0, -1]]
        cmin, cmax = _np.where(cols)[0][[0, -1]]
        pad = 15
        rmin = max(0, rmin - pad)
        # Chop the bottom 40px to remove any residual footer text
        rmax = min(arr.shape[0], rmax - 40)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1], cmax + pad)
        img = img.crop((cmin, rmin, cmax, rmax))

    png_path = out_dir / "moabb_circle_packing.png"
    img.save(str(png_path), dpi=(300, 300))
    raw_path.unlink(missing_ok=True)
    print(f"  -> {png_path.name} ({img.size[0]}x{img.size[1]}, 300 DPI)")


def export_ridgeline(df_raw: pd.DataFrame, out_dir: Path) -> None:
    """Export ridgeline — both experiment and recording modality views.

    The ridgeline uses custom JS rendering (fig=None in build_and_export_html),
    so we rebuild the figure from the trace-builder function.
    """
    data = df_raw[df_raw["dataset"].str.lower() != "test"].copy()
    data["n_subjects"] = pd.to_numeric(data["n_subjects"], errors="coerce")
    data = data.dropna(subset=["n_subjects"])

    if data.empty:
        print("  Ridgeline: no data, skipping")
        return

    rng = np.random.default_rng(42)
    amplitude = 0.6
    row_spacing = 0.95

    for view_name, col, func, filename in [
        (
            "Experiment Modality",
            "modality of exp",
            primary_modality,
            "ridgeline_experimental.pdf",
        ),
        (
            "Recording Modality",
            "record_modality",
            primary_recording_modality,
            "ridgeline_recording.pdf",
        ),
    ]:
        if col not in data.columns:
            print(f"  Ridgeline ({view_name}): column '{col}' missing, skipping")
            continue

        traces, order, _ = _build_ridgeline_traces(data, col, func, rng, view_name)
        if not traces:
            print(f"  Ridgeline ({view_name}): no traces, skipping")
            continue

        fig = go.Figure()
        for trace in traces:
            trace.visible = True
            fig.add_trace(trace)

        h_px = max(340, 50 * len(order))
        fig.update_layout(
            height=h_px,
            width=LANDSCAPE_W,
            template="plotly_white",
            font=dict(size=10, family=PUB_FONT),
            xaxis=dict(
                type="log",
                title=dict(
                    text="Number of Participants (Log Scale)", font=dict(size=12)
                ),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.12)",
                zeroline=False,
                dtick=1,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title=dict(text="Modality", font=dict(size=12)),
                tickmode="array",
                tickvals=[idx * row_spacing for idx in range(len(order))],
                ticktext=order,
                showgrid=False,
                range=[
                    -0.25,
                    max(0.35, (len(order) - 1) * row_spacing + amplitude + 0.25),
                ],
                tickfont=dict(size=10),
            ),
            showlegend=False,
            margin=dict(l=80, r=20, t=15, b=45),
            paper_bgcolor="white",
            plot_bgcolor="white",
            annotations=[],  # Remove any hardcoded annotations
        )

        export_pdf(fig, out_dir / filename, width=LANDSCAPE_W, height=h_px)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export EEGDash figures to PDF for paper."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_figures",
        help="Output directory for PDF files (default: docs/paper_figures/)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data from API
    df_raw = load_data()
    if df_raw.empty:
        print("ERROR: No data fetched from API.")
        sys.exit(1)

    # Prepare bubble DataFrame
    df_bubble = df_raw.copy()
    if "subjects" not in df_bubble.columns and "n_subjects" in df_bubble.columns:
        df_bubble["subjects"] = df_bubble["n_subjects"]
    if "records" not in df_bubble.columns and "n_records" in df_bubble.columns:
        df_bubble["records"] = df_bubble["n_records"]

    # Generate all charts to a temp directory (populates the figure registry)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print("\nGenerating charts (to populate figure registry)...")

        generators = [
            (
                "Bubble",
                generate_dataset_bubble,
                df_bubble,
                "bubble.html",
                {"x_var": "subjects"},
            ),
            ("Sankey", generate_dataset_sankey, df_raw, "sankey.html", {}),
            ("Treemap", generate_dataset_treemap, df_raw, "treemap.html", {}),
            ("Growth", generate_dataset_growth, df_raw, "growth.html", {}),
            ("Clinical", generate_clinical_stacked_bar, df_raw, "clinical.html", {}),
        ]

        for name, gen_func, df, filename, kwargs in generators:
            try:
                gen_func(df.copy(), tmp_dir / filename, **kwargs)
                print(f"  {name}: OK")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    # Retrieve figures from registry
    registry = get_figure_registry()
    print(f"\nFigures in registry: {list(registry.keys())}")

    # Export each figure to PDF
    print("\nExporting PDFs...")

    print("Bubble chart:")
    export_bubble(registry, out_dir)

    print("Sankey diagram:")
    export_sankey(registry, out_dir)

    print("Treemap:")
    export_treemap(registry, out_dir)

    print("Growth chart:")
    export_growth(registry, out_dir)

    print("Clinical breakdown:")
    export_clinical(registry, out_dir)

    print("Ridgeline (KDE):")
    export_ridgeline(df_raw, out_dir)

    print("MOABB circle-packing:")
    export_moabb(df_raw, out_dir)

    # Summary
    pdfs = sorted(out_dir.glob("*.pdf"))
    pngs = sorted(out_dir.glob("*.png"))
    all_files = sorted(pdfs + pngs, key=lambda f: f.name)
    print(f"\nDone! {len(all_files)} figures exported to {out_dir}/")
    for f in all_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
