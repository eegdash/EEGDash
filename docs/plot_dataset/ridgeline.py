from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy.stats import gaussian_kde

try:  # Allow execution as a script or module
    from .colours import MODALITY_COLOR_MAP, hex_to_rgba
    from .utils import get_dataset_url, primary_modality
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import MODALITY_COLOR_MAP, hex_to_rgba  # type: ignore
    from utils import get_dataset_url, primary_modality  # type: ignore

__all__ = ["generate_modality_ridgeline"]


def generate_modality_ridgeline(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    rng_seed: int = 42,
) -> Path | None:
    """Generate a ridgeline (KDE) plot showing participants per modality."""
    data = df[df["dataset"].str.lower() != "test"].copy()
    data["modality_label"] = data["modality of exp"].apply(primary_modality)
    data["n_subjects"] = pd.to_numeric(data["n_subjects"], errors="coerce")
    data = data.dropna(subset=["n_subjects"])
    data = data[data["modality_label"] != "Other"]

    if data.empty:
        return None

    median_participants = (
        data.groupby("modality_label")["n_subjects"].median().sort_values()
    )
    order = [
        label
        for label in median_participants.index
        if label in data["modality_label"].unique()
    ]
    if not order:
        return None

    fig = go.Figure()
    rng = np.random.default_rng(rng_seed)
    amplitude = 0.6
    row_spacing = 0.95

    for idx, label in enumerate(order):
        subset = data[data["modality_label"] == label].copy()
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

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_curve, x_curve[::-1]]),
                y=np.concatenate([y_curve, np.full_like(y_curve, baseline)]),
                name=label,
                fill="toself",
                fillcolor=fill,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{label}</b><br>#Participants: %{{x:.0f}}<extra></extra>",
                showlegend=False,
            )
        )

        jitter = rng.uniform(0.02, amplitude * 0.5, size=len(values))
        median_val = float(median_participants.get(label, np.nan))
        custom_data = np.column_stack(
            [subset["dataset"].to_numpy(), subset["dataset_url"].to_numpy()]
        )
        fig.add_trace(
            go.Scatter(
                x=values,
                y=np.full_like(values, baseline) + jitter,
                mode="markers",
                name=label,
                marker=dict(color=color, size=8, opacity=0.6),
                customdata=custom_data,
                hovertemplate="<b><a href='%{customdata[1]}' target='_parent'>%{customdata[0]}</a></b><br>#Participants: %{x}<br><i>Click to view dataset details</i><extra></extra>",
                showlegend=False,
            )
        )

        if np.isfinite(median_val) and median_val > 0:
            fig.add_trace(
                go.Scatter(
                    x=[median_val, median_val],
                    y=[baseline, baseline + amplitude],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=(
                        f"<b>{label}</b><br>Median participants: {median_val:.0f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    if not fig.data:
        return None

    kde_height = max(650, 150 * len(order))
    date_stamp = datetime.now().strftime("%d/%m/%Y")
    fig.update_layout(
        height=kde_height,
        width=1200,
        template="plotly_white",
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
            tickvals=[idx * row_spacing for idx in range(len(order))],
            ticktext=order,
            showgrid=False,
            range=[-0.25, max(0.35, (len(order) - 1) * row_spacing + amplitude + 0.25)],
            tickfont=dict(size=14),
        ),
        showlegend=False,
        margin=dict(l=120, r=40, t=108, b=80),
        title=dict(
            text=f"<br><sub>Based on EEG-Dash datasets available at {date_stamp}.</sub>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=20),
        ),
        autosize=True,
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

    fig_spec = fig.to_plotly_json()
    data_json = json.dumps(fig_spec.get("data", []), cls=PlotlyJSONEncoder)
    layout_json = json.dumps(fig_spec.get("layout", {}), cls=PlotlyJSONEncoder)
    config_json = json.dumps(plot_config, cls=PlotlyJSONEncoder)

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
