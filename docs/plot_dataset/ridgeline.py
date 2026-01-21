from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

try:  # Allow execution as a script or module
    from .colours import MODALITY_COLOR_MAP, hex_to_rgba
    from .utils import get_dataset_url, primary_modality, primary_recording_modality
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import MODALITY_COLOR_MAP, hex_to_rgba  # type: ignore
    from utils import (  # type: ignore
        get_dataset_url,
        primary_modality,
        primary_recording_modality,
    )

__all__ = ["generate_modality_ridgeline"]


def _build_ridgeline_traces(
    data: pd.DataFrame,
    modality_column: str,
    modality_func: callable,
    rng: np.random.Generator,
    trace_group: str,
) -> tuple[list[go.Scatter], list[str], dict[str, float]]:
    """Build ridgeline traces for a given modality type.

    Returns:
        traces: List of Plotly Scatter traces
        order: List of modality labels in order
        medians: Dict mapping label -> median participant count

    """
    amplitude = 0.6
    row_spacing = 0.95

    work = data.copy()
    work["modality_label"] = work[modality_column].apply(modality_func)
    work = work[work["modality_label"] != "Other"]
    work = work[work["modality_label"] != "Unknown"]
    # Filter out datasets with 0 or negative participants (causes log10 issues)
    work = work[work["n_subjects"] > 0]

    if work.empty:
        return [], [], {}

    median_participants = (
        work.groupby("modality_label")["n_subjects"].median().sort_values()
    )
    order = [
        label
        for label in median_participants.index
        if label in work["modality_label"].unique()
    ]
    if not order:
        return [], [], {}

    traces = []
    for idx, label in enumerate(order):
        subset = work[work["modality_label"] == label].copy()
        values = subset["n_subjects"].astype(float).dropna()
        if len(values) < 3:
            continue

        subset["dataset_url"] = subset["dataset"].apply(get_dataset_url)
        log_vals = np.log10(values)
        grid = np.linspace(log_vals.min() - 0.25, log_vals.max() + 0.25, 240)
        kde = gaussian_kde(log_vals)
        density = kde(grid)
        if density.max() <= 0:
            continue

        density_norm = density / density.max()
        baseline = idx * row_spacing
        y_curve = baseline + density_norm * amplitude
        x_curve = 10**grid

        color = MODALITY_COLOR_MAP.get(label, "#6b7280")
        fill = hex_to_rgba(color, 0.28)

        # Convert numpy arrays to lists to avoid Plotly binary encoding
        x_fill = np.concatenate([x_curve, x_curve[::-1]]).tolist()
        y_fill = np.concatenate([y_curve, np.full_like(y_curve, baseline)]).tolist()
        x_line = x_curve.tolist()
        y_line = y_curve.tolist()

        # Fill trace
        traces.append(
            go.Scatter(
                x=x_fill,
                y=y_fill,
                name=label,
                fill="toself",
                fillcolor=fill,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                visible=True,
                meta={"group": trace_group},
            )
        )

        # Line trace
        traces.append(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{label}</b><br>#Participants: %{{x:.0f}}<extra></extra>",
                showlegend=False,
                visible=True,
                meta={"group": trace_group},
            )
        )

        # Scatter points
        jitter = rng.uniform(0.02, amplitude * 0.5, size=len(values))
        median_val = float(median_participants.get(label, np.nan))
        custom_data = np.column_stack(
            [subset["dataset"].to_numpy(), subset["dataset_url"].to_numpy()]
        ).tolist()
        x_scatter = values.tolist()
        y_scatter = (np.full_like(values, baseline) + jitter).tolist()
        traces.append(
            go.Scatter(
                x=x_scatter,
                y=y_scatter,
                mode="markers",
                name=label,
                marker=dict(color=color, size=8, opacity=0.6),
                customdata=custom_data,
                hovertemplate="<b><a href='%{customdata[1]}' target='_parent'>%{customdata[0]}</a></b><br>#Participants: %{x}<br><i>Click to view dataset details</i><extra></extra>",
                showlegend=False,
                visible=True,
                meta={"group": trace_group},
            )
        )

        # Median line
        if np.isfinite(median_val) and median_val > 0:
            traces.append(
                go.Scatter(
                    x=[median_val, median_val],
                    y=[baseline, baseline + amplitude],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=(
                        f"<b>{label}</b><br>Median participants: {median_val:.0f}<extra></extra>"
                    ),
                    showlegend=False,
                    visible=True,
                    meta={"group": trace_group},
                )
            )

    return traces, order, dict(median_participants)


def generate_modality_ridgeline(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    rng_seed: int = 42,
) -> Path | None:
    """Generate a ridgeline (KDE) plot showing participants per modality.

    Includes a toggle button to switch between:
    - Recording Modality (EEG, MEG, iEEG, fNIRS, EMG)
    - Experiment Modality (Visual, Auditory, Motor, Resting State, etc.)
    """
    data = df[df["dataset"].str.lower() != "test"].copy()
    data["n_subjects"] = pd.to_numeric(data["n_subjects"], errors="coerce")
    data = data.dropna(subset=["n_subjects"])

    if data.empty:
        return None

    rng = np.random.default_rng(rng_seed)

    # Build traces for experiment modality (default view)
    exp_traces, exp_order, _ = _build_ridgeline_traces(
        data, "modality of exp", primary_modality, rng, "experiment"
    )

    # Build traces for recording modality
    rec_traces, rec_order, _ = _build_ridgeline_traces(
        data, "record_modality", primary_recording_modality, rng, "recording"
    )

    if not exp_traces and not rec_traces:
        return None

    # Create figure with experiment modality visible by default
    fig = go.Figure()

    # Track trace counts for visibility toggling
    n_exp_traces = len(exp_traces)
    n_rec_traces = len(rec_traces)

    # Add experiment modality traces (visible by default)
    for trace in exp_traces:
        trace.visible = True
        fig.add_trace(trace)

    # Add recording modality traces (hidden by default)
    for trace in rec_traces:
        trace.visible = False
        fig.add_trace(trace)

    # Use the longer order for height calculation
    max_order = max(len(exp_order), len(rec_order))
    kde_height = max(650, 150 * max_order)
    amplitude = 0.6
    row_spacing = 0.95
    date_stamp = datetime.now().strftime("%d/%m/%Y")

    # Build visibility arrays for toggle buttons
    # [True for exp traces] + [False for rec traces] = show experiment
    # [False for exp traces] + [True for rec traces] = show recording
    exp_visible = [True] * n_exp_traces + [False] * n_rec_traces
    rec_visible = [False] * n_exp_traces + [True] * n_rec_traces

    # Create update menus (toggle buttons)
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.0,
            xanchor="left",
            y=1.12,
            yanchor="top",
            buttons=[
                dict(
                    label="Experiment Modality",
                    method="update",
                    args=[
                        {"visible": exp_visible},
                        {
                            "yaxis.tickvals": [
                                idx * row_spacing for idx in range(len(exp_order))
                            ],
                            "yaxis.ticktext": exp_order,
                            "yaxis.range": [
                                -0.25,
                                max(
                                    0.35,
                                    (len(exp_order) - 1) * row_spacing
                                    + amplitude
                                    + 0.25,
                                ),
                            ],
                        },
                    ],
                ),
                dict(
                    label="Recording Modality",
                    method="update",
                    args=[
                        {"visible": rec_visible},
                        {
                            "yaxis.tickvals": [
                                idx * row_spacing for idx in range(len(rec_order))
                            ],
                            "yaxis.ticktext": rec_order,
                            "yaxis.range": [
                                -0.25,
                                max(
                                    0.35,
                                    (len(rec_order) - 1) * row_spacing
                                    + amplitude
                                    + 0.25,
                                ),
                            ],
                        },
                    ],
                ),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            bgcolor="white",
            bordercolor="#d1d5db",
            font=dict(size=13),
        )
    ]

    fig.update_layout(
        height=kde_height,
        width=None,
        autosize=True,
        template="plotly_white",
        updatemenus=updatemenus,
        xaxis=dict(
            type="log",
            title=dict(text="Number of Participants (Log Scale)", font=dict(size=18)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            dtick=1,
            minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="Modality", font=dict(size=18)),
            tickmode="array",
            tickvals=[idx * row_spacing for idx in range(len(exp_order))],
            ticktext=exp_order,
            showgrid=False,
            range=[
                -0.25,
                max(0.35, (len(exp_order) - 1) * row_spacing + amplitude + 0.25),
            ],
            tickfont=dict(size=14),
        ),
        showlegend=False,
        margin=dict(l=120, r=40, t=130, b=80),
        title=dict(
            text=f"<br><sub>Based on EEG-Dash datasets available at {date_stamp}.</sub>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=20),
        ),
        font=dict(size=16),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.02,
        text="Visual studies consistently use the<br>largest sample sizes, typically 20-30 participants",
        showarrow=False,
        font=dict(size=14, color="#111827"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(17,24,39,0.3)",
        borderwidth=1,
        borderpad=8,
        xanchor="right",
        yanchor="bottom",
    )

    plot_config = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "participant_kde",
            "height": kde_height,
            "width": 1200,
            "scale": 2,
        },
    }

    # Convert figure to dict and ensure arrays are lists (not binary encoded)
    fig_dict = fig.to_dict()

    def convert_arrays(obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_arrays(item) for item in obj]
        return obj

    fig_data = convert_arrays(fig_dict.get("data", []))
    fig_layout = convert_arrays(fig_dict.get("layout", {}))

    data_json = json.dumps(fig_data)
    layout_json = json.dumps(fig_layout)
    config_json = json.dumps(plot_config)

    styled_html = f"""
<style>
#dataset-kde-modalities {{
    width: 100% !important;
    max-width: 1200px;
    height: {kde_height}px !important;
    min-height: {kde_height}px;
    margin: 0 auto;
    display: none;
}}
#dataset-kde-modalities.plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
.kde-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {kde_height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}
</style>
<div class="kde-loading" id="kde-loading">Loading participant distribution...</div>
<div id="dataset-kde-modalities" class="plotly-graph-div"></div>
<script>
(function() {{
  const TARGET_ID = 'dataset-kde-modalities';
  const FIG_DATA = {data_json};
  const FIG_LAYOUT = {layout_json};
  const FIG_CONFIG = {config_json};

  function onReady(callback) {{
    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', callback, {{ once: true }});
    }} else {{
      callback();
    }}
  }}

  function renderPlot() {{
    const container = document.getElementById(TARGET_ID);
    if (!container) {{
      return;
    }}

    const draw = () => {{
      if (!window.Plotly) {{
        window.requestAnimationFrame(draw);
        return;
      }}

      window.Plotly.newPlot(TARGET_ID, FIG_DATA, FIG_LAYOUT, FIG_CONFIG).then((plot) => {{
        const loading = document.getElementById('kde-loading');
        if (loading) {{
          loading.style.display = 'none';
        }}
        container.style.display = 'block';

        plot.on('plotly_click', (event) => {{
          const point = event.points && event.points[0];
          if (!point || !point.customdata) {{
            return;
          }}
          const url = point.customdata[1];
          if (url) {{
            const resolved = new URL(url, window.location.href);
            window.open(resolved.href, '_self');
          }}
        }});
      }});
    }};

    draw();
  }}

  onReady(renderPlot);
}})();
</script>
"""

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(styled_html, encoding="utf-8")
    return out_path


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, header=0, skipinitialspace=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the modality ridgeline plot from a dataset summary CSV."
    )
    parser.add_argument("source", type=Path, help="Path to dataset summary CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_kde_modalities.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling jitter placement",
    )
    args = parser.parse_args()

    df = _read_dataset(args.source)
    output_path = generate_modality_ridgeline(df, args.output, rng_seed=args.seed)
    if output_path is None:
        print("Ridgeline plot could not be generated (insufficient data).")
    else:
        print(f"Ridgeline plot saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
