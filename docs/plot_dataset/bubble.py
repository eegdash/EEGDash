from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:  # Allow execution as a script or module
    from .colours import (
        EXPERIMENTAL_MODALITY_COLORS,
        RECORDING_MODALITY_COLORS,
    )
    from .utils import (
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_modality,
        primary_recording_modality,
        safe_int,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import (  # type: ignore
        EXPERIMENTAL_MODALITY_COLORS,
        RECORDING_MODALITY_COLORS,
    )
    from utils import (  # type: ignore
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_modality,
        primary_recording_modality,
        safe_int,
    )

__all__ = ["generate_dataset_bubble"]


def _to_numeric_median_list(val: Any) -> float | None:
    """Compute median from nchans/sfreq data, handling API aggregation format.

    Supports:
    - API aggregation: [{"val": 64, "count": 10}, ...]
    - Simple lists: [64, 128, 256]
    - JSON strings of the above
    - Comma/space-separated strings: "64, 128, 256"

    Returns:
        Median value as float, or None if no valid data.

    """
    # Parse JSON string if needed
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass  # Fall through to string parsing below

    # Handle API aggregation format: [{"val": 64, "count": 10}, ...]
    if isinstance(val, list) and val and isinstance(val[0], dict) and "val" in val[0]:
        vals: list[float] = []
        weights: list[int] = []
        for item in val:
            v = item.get("val")
            if v is not None:
                try:
                    vals.append(float(v))
                    weights.append(item.get("count", 1))
                except (ValueError, TypeError):
                    continue  # Skip non-numeric values
        if not vals:
            return None
        # Use np.repeat for memory-efficient weighted median
        return float(np.median(np.repeat(vals, weights)))

    # Handle literal collections (list, array, Series)
    if isinstance(val, (list, np.ndarray, pd.Series)):
        if len(val) == 0:
            return None
        try:
            return float(np.nanmedian(val))
        except (ValueError, TypeError):
            pass  # Non-numeric collection, try other methods

    # Handle scalar NaNs safely
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        if np.asarray(pd.isna(val)).any():
            return None

    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        pass  # Not directly convertible to float

    # Fall back to string parsing for simple comma/space-separated lists
    s = str(val).strip().strip("[]")
    if not s:
        return None

    try:
        sep = "," if "," in s else None
        nums = [float(x) for x in s.split(sep) if str(x).strip()]
        if not nums:
            return None
        return float(np.median(nums))
    except (ValueError, TypeError):
        return None


def _format_int(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def _build_hover_template(x_field: str, y_field: str) -> tuple[str, str]:
    x_map = {
        "duration_h": "Duration (x): %{x:.2f} h",
        "size_gb": "Size (x): %{x:.2f} GB",
        "tasks": "Tasks (x): %{x:,}",
        "subjects": "Subjects (x): %{x:,}",
    }
    y_map = {
        "subjects": "Subjects (y): %{y:,}",
    }
    x_hover = x_map.get(x_field, "Records (x): %{x:,}")
    y_hover = y_map.get(y_field, "Records (y): %{y:,}")
    return x_hover, y_hover


def generate_dataset_bubble(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    x_var: str = "records",
    max_width: int = 1280,
    height: int = 720,
) -> Path:
    """Generate the dataset landscape bubble chart."""
    data = df.copy()
    data = data[data["dataset"].str.lower() != "test"]

    data["duration_h"] = pd.to_numeric(
        data.get("duration_hours_total"), errors="coerce"
    )
    data["subjects"] = pd.to_numeric(data.get("n_subjects"), errors="coerce")
    data["records"] = pd.to_numeric(data.get("n_records"), errors="coerce")
    data["tasks"] = pd.to_numeric(data.get("n_tasks"), errors="coerce")
    data["size_bytes"] = pd.to_numeric(data.get("size_bytes"), errors="coerce")

    data["sfreq"] = data["sampling_freqs"].map(_to_numeric_median_list)
    data["nchans"] = data["nchans_set"].map(_to_numeric_median_list)

    # Compute experiment modality label (Visual, Auditory, Motor, etc.)
    data["exp_modality_label"] = data.get("modality of exp").apply(primary_modality)

    # Compute recording modality label (EEG, MEG, iEEG, fNIRS, etc.)
    rec_col = None
    for col in ["record_modality", "recording_modality"]:
        if col in data.columns:
            rec_col = col
            break
    if rec_col:
        data["rec_modality_label"] = data[rec_col].apply(primary_recording_modality)
    else:
        data["rec_modality_label"] = "EEG"  # Default fallback

    GB = 1024**3
    data["size_gb"] = data["size_bytes"] / GB

    x_field = (
        x_var
        if x_var in {"records", "duration_h", "size_gb", "tasks", "subjects"}
        else "records"
    )
    axis_labels = {
        "records": "#Records",
        "duration_h": "Duration (hours)",
        "size_gb": "Size (GB)",
        "tasks": "#Tasks",
        "subjects": "#Subjects",
    }
    x_label = f"{axis_labels[x_field]} (log scale)"
    y_field = "subjects" if x_field != "subjects" else "records"
    y_label = f"{axis_labels[y_field]} (log scale)"
    x_hover, y_hover = _build_hover_template(x_field, y_field)

    required_columns = {x_field, y_field, "size_gb"}
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=list(required_columns))
    data = data[(data[x_field] > 0) & (data[y_field] > 0)]

    data["dataset_url"] = data["dataset"].apply(get_dataset_url)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data.empty:
        empty_html = """
<div class="dataset-loading" id="dataset-loading">No dataset records available for plotting.</div>
"""
        out_path.write_text(empty_html, encoding="utf-8")
        return out_path

    size_max = data["size_gb"].max()
    if not np.isfinite(size_max) or size_max <= 0:
        size_max = 1.0
    sizeref = (2.0 * size_max) / (40.0**2)

    sfreq_str = data["sfreq"].map(_format_int)
    nchans_str = data["nchans"].map(_format_int)
    size_str = data["size_bytes"].map(
        lambda bytes_: human_readable_size(safe_int(bytes_, 0))
    )

    # Build custom data array for hover
    custom_data = np.column_stack(
        [
            data["dataset"].values,
            data["subjects"].values,
            data["records"].values,
            data["tasks"].values,
            nchans_str.values,
            sfreq_str.values,
            size_str.values,
            data["exp_modality_label"].values,
            data["rec_modality_label"].values,
            data["dataset_url"].values,
        ]
    )

    # Build traces for experiment modality (default view)
    exp_traces = []
    exp_modalities = [
        label
        for label in EXPERIMENTAL_MODALITY_COLORS.keys()
        if label in data["exp_modality_label"].unique()
    ]
    for modality in exp_modalities:
        mask = data["exp_modality_label"] == modality
        if not mask.any():
            continue
        subset = data[mask]
        exp_traces.append(
            go.Scatter(
                x=subset[x_field],
                y=subset[y_field],
                mode="markers",
                name=modality,
                marker=dict(
                    size=subset["size_gb"],
                    color=EXPERIMENTAL_MODALITY_COLORS.get(modality, "#94a3b8"),
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=6,
                    line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
                    opacity=0.75,
                ),
                customdata=custom_data[mask.values],
                hovertemplate=None,  # Will be set later
                visible=True,
                legendgroup="exp",
            )
        )

    # Build traces for recording modality (hidden by default)
    rec_traces = []
    rec_modalities = [
        label
        for label in RECORDING_MODALITY_COLORS.keys()
        if label in data["rec_modality_label"].unique()
    ]
    for modality in rec_modalities:
        mask = data["rec_modality_label"] == modality
        if not mask.any():
            continue
        subset = data[mask]
        rec_traces.append(
            go.Scatter(
                x=subset[x_field],
                y=subset[y_field],
                mode="markers",
                name=modality,
                marker=dict(
                    size=subset["size_gb"],
                    color=RECORDING_MODALITY_COLORS.get(modality, "#94a3b8"),
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=6,
                    line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
                    opacity=0.75,
                ),
                customdata=custom_data[mask.values],
                hovertemplate=None,  # Will be set later
                visible=False,
                legendgroup="rec",
            )
        )

    # Create figure with experiment modality traces visible by default
    fig = go.Figure()

    # Add experiment modality traces (visible by default)
    for trace in exp_traces:
        fig.add_trace(trace)

    # Add recording modality traces (hidden by default)
    for trace in rec_traces:
        fig.add_trace(trace)

    # Track trace counts for visibility toggling
    n_exp_traces = len(exp_traces)
    n_rec_traces = len(rec_traces)

    # Update axis labels
    fig.update_xaxes(title_text=x_label, type="log")
    fig.update_yaxes(title_text=y_label, type="log")

    # ---------- Reference line, OLS fit, and arrow (all robust in log space)
    numeric_x = pd.to_numeric(data[x_field], errors="coerce")
    numeric_y = pd.to_numeric(data[y_field], errors="coerce")
    mask = (
        np.isfinite(numeric_x)
        & np.isfinite(numeric_y)
        & (numeric_x > 0)
        & (numeric_y > 0)
    )

    fit_annotation_text = None
    if mask.sum() >= 2:
        log_x = np.log10(numeric_x[mask])
        log_y = np.log10(numeric_y[mask])
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)

        # Draw 1:1 line as an underlying shape, using actual data bounds
        lx_min = max(log_x.min(), log_y.min())
        lx_max = min(log_x.max(), log_y.max())
        if lx_min < lx_max:
            x0 = 10**lx_min
            x1 = 10**lx_max
            fig.add_shape(
                type="line",
                x0=x0,
                y0=x0,
                x1=x1,
                y1=x1,
                xref="x",
                yref="y",
                layer="below",
                line=dict(color="#9ca3af", width=1.5, dash="dash"),
            )

        # Red dotted OLS line (computed in log space), using actual data bounds
        if np.ptp(log_x) > 0 and np.ptp(log_y) > 0 and ss_tot > 0:
            slope, intercept = np.polyfit(log_x, log_y, 1)
            line_log_x = np.linspace(log_x.min(), log_x.max(), 200)
            line_x = 10**line_log_x
            line_y = 10 ** (slope * line_log_x + intercept)
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="log-log fit",
                    line=dict(color="#dc2626", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.35,
                )
            )
            residuals = log_y - (slope * log_x + intercept)
            r_squared = 1 - np.sum(residuals**2) / ss_tot
            fit_annotation_text = f"<span style='color:#dc2626'>Red dotted line: log-log OLS fit R² = {r_squared:.3f}</span>"

        # Arrow label ~60% along the 1:1 segment for stable placement
        if lx_min < lx_max:
            t = 0.82  # control the position along the line
            annot_log = (1 - t) * lx_min + t * lx_max
            annot_xy = np.log10(10**annot_log)
            fig.add_annotation(
                x=annot_xy,
                y=annot_xy,
                text="One record per subject",
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor="#6b7280",
                ax=110,
                ay=90,
                axref="pixel",
                ayref="pixel",
                font=dict(size=20, color="#374151"),
                align="left",
            )

    # ---------- Build visibility arrays for toggle buttons ----------
    # Must be done AFTER all traces are added (including OLS line)
    # Count total traces: exp + rec + OLS line (if present)
    n_total_traces = len(fig.data)
    n_ols_traces = n_total_traces - n_exp_traces - n_rec_traces

    # Build visibility: exp traces visible, rec traces hidden, OLS always visible
    exp_visible = [True] * n_exp_traces + [False] * n_rec_traces + [True] * n_ols_traces
    rec_visible = [False] * n_exp_traces + [True] * n_rec_traces + [True] * n_ols_traces

    # ---------- Hover and styling ----------
    x_hover, y_hover = _build_hover_template(x_field, y_field)
    hover_template = (
        "<b>%{customdata[0]}</b>"
        f"<br>{x_hover}"
        f"<br>{y_hover}"
        "<br>Subjects (total): %{customdata[1]:,}"
        "<br>Records (total): %{customdata[2]:,}"
        "<br>Tasks: %{customdata[3]:,}"
        "<br>Channels: %{customdata[4]}"
        "<br>Sampling: %{customdata[5]} Hz"
        "<br>Size: %{customdata[6]}"
        "<br>Experiment: %{customdata[7]}"
        "<br>Recording: %{customdata[8]}"
        "<br><i>Click bubble to open dataset page</i>"
        "<extra></extra>"
    )

    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "markers" not in mode:
            continue
        trace.hovertemplate = hover_template

    # Create update menus (toggle buttons)
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.0,
            xanchor="left",
            y=1.18,
            yanchor="top",
            buttons=[
                dict(
                    label="Experiment Modality",
                    method="restyle",
                    args=["visible", exp_visible],
                ),
                dict(
                    label="Recording Modality",
                    method="restyle",
                    args=["visible", rec_visible],
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
        height=height,
        width=max_width + 200,  # Wider figure
        margin=dict(l=60, r=40, t=140, b=60),
        template="plotly_white",
        updatemenus=updatemenus,
        legend=dict(
            title=None,  # No title for inline style
            orientation="h",  # Horizontal inline
            yanchor="bottom",
            y=1.08,  # Position above the plot
            xanchor="left",
            x=0.0,  # Left-aligned, next to buttons
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            font=dict(size=12),
            tracegroupgap=5,  # Minimal gap between items
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        title=dict(text="", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        autosize=False,
    )

    if fit_annotation_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=fit_annotation_text,
            showarrow=False,
            font=dict(size=15, color="#111827"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(17,24,39,0.25)",
            borderwidth=1,
            borderpad=6,
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        type="log",
        dtick=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
        type="log",
        dtick=1,
    )

    extra_style = f""".dataset-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}"""

    pre_html = '<div class="dataset-loading" id="dataset-loading">Loading dataset landscape...</div>\n'

    extra_html = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const loading = document.getElementById('dataset-loading');
    const plot = document.getElementById('dataset-bubble');

    function showPlot() {
        if (loading) {
            loading.style.display = 'none';
        }
        if (plot) {
            plot.style.display = 'block';
            // Force Plotly to resize to fit the container
            if (typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(plot);
            }
        }
    }

    function hookPlotlyClick(attempts) {
        if (!plot || typeof plot.on !== 'function') {
            if (attempts < 40) {
                window.setTimeout(function() { hookPlotlyClick(attempts + 1); }, 60);
            }
            return;
        }
        plot.on('plotly_click', function(evt) {
            const point = evt && evt.points && evt.points[0];
            const url = point && point.customdata && point.customdata[9];
            if (url) {
                window.open(url, '_blank', 'noopener');
            }
        });
        showPlot();
        // Additional resize after a short delay to ensure proper rendering
        window.setTimeout(function() {
            if (typeof Plotly !== 'undefined' && plot) {
                Plotly.Plots.resize(plot);
            }
        }, 100);
    }

    hookPlotlyClick(0);
    showPlot();
});
</script>
"""

    return build_and_export_html(
        fig,
        out_path,
        div_id="dataset-bubble",
        height=height,
        extra_style=extra_style,
        pre_html=pre_html,
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


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, header=0, skipinitialspace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the dataset bubble chart.")
    parser.add_argument("source", type=Path, help="Path to dataset summary CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_bubble.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--x-axis",
        choices=["records", "duration_h", "size_gb", "tasks", "subjects"],
        default="records",
        help="Field for the bubble chart x-axis",
    )
    args = parser.parse_args()

    df = _read_dataset(args.source)
    output_path = generate_dataset_bubble(df, args.output, x_var=args.x_axis)
    print(f"Bubble chart saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
