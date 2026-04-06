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

from eegdash.dataset.registry import fetch_chart_data_from_api

# API Configuration
API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"

# Paper figure dimensions (pixels, then scaled)
LANDSCAPE_W, LANDSCAPE_H = 1200, 720
TALL_W, TALL_H = 1200, 1100
PDF_SCALE = 3


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
    df_raw, _ = fetch_chart_data_from_api(API_BASE_URL, DEFAULT_DATABASE, limit=1000)

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
    """Remove interactive elements for static export."""
    # Remove updatemenus via dict manipulation (more reliable than update_layout)
    fig_dict = fig.to_dict()
    fig_dict.get("layout", {}).pop("updatemenus", None)
    fig_dict.get("layout", {}).pop("sliders", None)
    fig_dict["layout"]["paper_bgcolor"] = "white"
    fig_dict["layout"]["plot_bgcolor"] = "white"
    cleaned = go.Figure(fig_dict)
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


def export_bubble(registry: dict, out_dir: Path) -> None:
    """Export bubble chart — both experiment and recording modality views."""
    fig_orig = registry.get("dataset-bubble")
    if fig_orig is None:
        print("  Bubble chart: not found in registry, skipping")
        return

    # Experiment modality (default view — button 0)
    exp_vis = _extract_visibility(fig_orig, 0)
    rec_vis = _extract_visibility(fig_orig, 1)

    if exp_vis:
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, exp_vis)
        fig = clean_for_print(fig)
        export_pdf(
            fig,
            out_dir / "bubble_exp_modality.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )

    # Recording modality (alternate view — button 1)
    if rec_vis:
        fig = copy.deepcopy(fig_orig)
        _set_visibility(fig, rec_vis)
        fig = clean_for_print(fig)
        export_pdf(
            fig,
            out_dir / "bubble_rec_modality.pdf",
            width=LANDSCAPE_W,
            height=LANDSCAPE_H,
        )


def export_sankey(registry: dict, out_dir: Path) -> None:
    """Export Sankey diagram."""
    fig_orig = registry.get("dataset-sankey")
    if fig_orig is None:
        print("  Sankey: not found in registry, skipping")
        return
    fig = copy.deepcopy(fig_orig)
    fig = clean_for_print(fig)
    export_pdf(fig, out_dir / "sankey.pdf", width=TALL_W, height=TALL_H)


def export_treemap(registry: dict, out_dir: Path) -> None:
    """Export treemap."""
    fig_orig = registry.get("dataset-treemap-plot")
    if fig_orig is None:
        print("  Treemap: not found in registry, skipping")
        return
    fig = copy.deepcopy(fig_orig)
    fig = clean_for_print(fig)
    export_pdf(fig, out_dir / "treemap.pdf", width=TALL_W, height=880)


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
        fig.update_layout(yaxis_title="Number of Datasets")
        export_pdf(fig, out_dir / "growth_datasets.pdf", width=LANDSCAPE_W, height=550)

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
        fig.update_layout(yaxis_title="Number of Subjects")
        export_pdf(fig, out_dir / "growth_subjects.pdf", width=LANDSCAPE_W, height=550)


def export_clinical(registry: dict, out_dir: Path) -> None:
    """Export clinical breakdown — both studies and subjects views."""
    fig_orig = registry.get("dataset-clinical-plot")
    if fig_orig is None:
        print("  Clinical chart: not found in registry, skipping")
        return

    # The clinical chart uses restyle (changes y-data, not visibility)
    y_counts = _extract_y_data(fig_orig, 0)
    y_subjects = _extract_y_data(fig_orig, 1)

    # Studies view (default — button 0)
    fig = copy.deepcopy(fig_orig)
    if y_counts:
        for i, y_arr in enumerate(y_counts):
            if i < len(fig.data):
                fig.data[i].y = y_arr
    fig = clean_for_print(fig)
    fig.update_layout(yaxis_title="Number of Studies")
    export_pdf(fig, out_dir / "clinical_studies.pdf", width=LANDSCAPE_W, height=650)

    # Subjects view (alternate — button 1)
    fig = copy.deepcopy(fig_orig)
    if y_subjects:
        for i, y_arr in enumerate(y_subjects):
            if i < len(fig.data):
                fig.data[i].y = y_arr
    fig = clean_for_print(fig)
    fig.update_layout(yaxis_title="Number of Subjects")
    export_pdf(fig, out_dir / "clinical_subjects.pdf", width=LANDSCAPE_W, height=650)


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
    """Capture MOABB canvas via Playwright at high resolution."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        # Load the HTML file
        page.goto(f"file://{html_path.resolve()}", wait_until="networkidle")

        # Wait for the canvas to render
        page.wait_for_selector("canvas", timeout=15000)
        # Give D3 simulation time to settle
        page.wait_for_timeout(4000)

        # Gently hide only interactive/UI elements (not canvas or legend)
        page.evaluate("""() => {
            // Hide search input
            document.querySelectorAll('input[type="text"], input[type="search"]').forEach(
                el => { el.style.visibility = 'hidden'; }
            );
            // Hide tooltip
            const tt = document.getElementById('moabb-tooltip');
            if (tt) tt.style.visibility = 'hidden';
            // Hide zoom buttons (+/- and reset)
            document.querySelectorAll('button').forEach(el => {
                el.style.visibility = 'hidden';
            });
            // Hide instruction footer — target only leaf text nodes
            document.querySelectorAll('div, p, span').forEach(el => {
                if (el.childElementCount === 0 && el.innerText &&
                    el.innerText.includes('Hover to highlight')) {
                    el.style.visibility = 'hidden';
                }
            });
            // Hide branding
            document.querySelectorAll('.eegdash-branding').forEach(
                el => { el.style.visibility = 'hidden'; }
            );
        }""")

        page.wait_for_timeout(300)

        # Full-page screenshot at native resolution
        png_path = out_dir / "moabb_circle_packing.png"
        page.screenshot(path=str(png_path), full_page=False)
        print(f"  -> {png_path.name}")

        browser.close()


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

        fig.update_layout(
            height=max(650, 150 * len(order)),
            width=LANDSCAPE_W,
            template="plotly_white",
            xaxis=dict(
                type="log",
                title=dict(
                    text="Number of Participants (Log Scale)", font=dict(size=18)
                ),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                zeroline=False,
                dtick=1,
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title=dict(text="Modality", font=dict(size=18)),
                tickmode="array",
                tickvals=[idx * row_spacing for idx in range(len(order))],
                ticktext=order,
                showgrid=False,
                range=[
                    -0.25,
                    max(0.35, (len(order) - 1) * row_spacing + amplitude + 0.25),
                ],
                tickfont=dict(size=14),
            ),
            showlegend=False,
            margin=dict(l=120, r=40, t=50, b=60),
            font=dict(size=16),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        h = max(650, 150 * len(order))
        export_pdf(fig, out_dir / filename, width=LANDSCAPE_W, height=h)


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
