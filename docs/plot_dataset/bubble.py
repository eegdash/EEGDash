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


_BACKGROUND_CATS = {"Other", "Unknown", "other", "unknown"}


def _build_modality_traces(
    data,
    x_field,
    y_field,
    sizeref,
    custom_data,
    label_col,
    color_map,
    legendgroup_prefix,
    visible,
):
    """Build scatter traces, drawing neutral categories first (behind)."""
    present = [l for l in color_map if l in data[label_col].unique()]
    bg = [m for m in present if m in _BACKGROUND_CATS]
    fg = [m for m in present if m not in _BACKGROUND_CATS]
    ordered = bg + fg

    traces = []
    for modality in ordered:
        mask = data[label_col] == modality
        if not mask.any():
            continue
        subset = data[mask]
        is_bg = modality in _BACKGROUND_CATS
        if visible and is_bg:
            trace_visible = "legendonly"
        else:
            trace_visible = visible
        traces.append(
            go.Scatter(
                x=subset[x_field],
                y=subset[y_field],
                mode="markers",
                name=modality,
                marker=dict(
                    size=subset["bubble_size"],
                    color=color_map.get(modality, "#94a3b8"),
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=5,
                    line=dict(width=1, color="rgba(255,255,255,0.8)"),
                    opacity=0.35 if is_bg else 0.7,
                ),
                customdata=custom_data[mask.values],
                hovertemplate=None,
                visible=trace_visible,
                legendgroup=f"{legendgroup_prefix}-{modality}",
            )
        )
    return traces


def _build_hover_template(x_field: str, y_field: str) -> tuple[str, str]:
    x_map = {
        "duration_h": "Duration (x): %{x:.2f} h",
        "dur_per_subject": "Duration/subject (x): %{x:.1f} min",
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

    # Duration per subject (minutes) — key metric for spreading data
    data["dur_per_subject"] = np.where(
        (data["duration_h"] > 0) & (data["subjects"] > 0),
        data["duration_h"] * 60 / data["subjects"],
        np.nan,
    )

    x_field = (
        x_var
        if x_var
        in {"records", "duration_h", "size_gb", "tasks", "subjects", "dur_per_subject"}
        else "records"
    )
    axis_labels = {
        "records": "#Records",
        "duration_h": "Duration (hours)",
        "dur_per_subject": "Recording Duration / Subject (min)",
        "size_gb": "Size (GB)",
        "tasks": "#Tasks",
        "subjects": "#Subjects",
    }
    x_label = axis_labels[x_field]
    y_field = "subjects" if x_field != "subjects" else "records"
    y_label = axis_labels[y_field]
    x_hover, y_hover = _build_hover_template(x_field, y_field)

    required_columns = {x_field, y_field}
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

    # Bubble size = log(1 + duration_per_subject_min) — depth of recording per person
    data["bubble_size"] = np.log1p(data["dur_per_subject"].fillna(0).clip(lower=0))
    data["bubble_size"] = data["bubble_size"].clip(lower=0.5)  # min visible size
    size_max = data["bubble_size"].max()
    if not np.isfinite(size_max) or size_max <= 0:
        size_max = 1.0
    sizeref = (2.0 * size_max) / (34.0**2)

    sfreq_str = data["sfreq"].map(_format_int)
    nchans_str = data["nchans"].map(_format_int)
    size_str = data["size_bytes"].map(
        lambda bytes_: human_readable_size(safe_int(bytes_, 0))
    )

    # Duration strings for hover
    duration_str = data["duration_h"].map(
        lambda h: f"{h:.1f} h" if pd.notna(h) and h > 0 else "—"
    )
    dur_per_subj_str = data["dur_per_subject"].map(
        lambda m: f"{m:.1f} min" if pd.notna(m) and m > 0 else "—"
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
            duration_str.values,
            dur_per_subj_str.values,
        ]
    )

    # Build traces for experiment modality (default view)
    exp_traces = _build_modality_traces(
        data,
        x_field,
        y_field,
        sizeref,
        custom_data,
        "exp_modality_label",
        EXPERIMENTAL_MODALITY_COLORS,
        "exp",
        True,
    )

    # Build traces for recording modality (hidden by default)
    rec_traces = _build_modality_traces(
        data,
        x_field,
        y_field,
        sizeref,
        custom_data,
        "rec_modality_label",
        RECORDING_MODALITY_COLORS,
        "rec",
        False,
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
    fig.update_xaxes(
        title_text=x_label,
        type="log",
        title_font=dict(size=18),
        tickfont=dict(size=15),
    )
    fig.update_yaxes(
        title_text=y_label,
        type="log",
        title_font=dict(size=18),
        tickfont=dict(size=15),
    )

    # ---------- Reference line, OLS fit, and arrow (all robust in log space)
    numeric_x = pd.to_numeric(data[x_field], errors="coerce")
    numeric_y = pd.to_numeric(data[y_field], errors="coerce")
    mask = (
        np.isfinite(numeric_x)
        & np.isfinite(numeric_y)
        & (numeric_x > 0)
        & (numeric_y > 0)
    )

    if mask.sum() >= 2:
        log_x = np.log10(numeric_x[mask])
        log_y = np.log10(numeric_y[mask])

        # Draw 1:1 line only when both axes share units (e.g., records vs subjects)
        show_one_to_one = x_field in {"records", "duration_h", "size_gb", "tasks"}
        # 1:1 line across the data range
        lx_min = max(log_x.min(), log_y.min())
        lx_max = min(log_x.max(), log_y.max())
        if show_one_to_one and lx_min < lx_max:
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
                line=dict(color="#9ca3af", width=1.2, dash="dash"),
            )

        # (ellipse removed — density is self-evident)

        # Arrow label ~60% along the 1:1 segment for stable placement
        ref_labels = {
            "records": "One record per subject",
            "duration_h": "One hour per subject",
            "size_gb": "One GB per subject",
            "tasks": "One task per subject",
        }
        ref_text = ref_labels.get(x_field)
        if ref_text and lx_min < lx_max:
            t = 0.92  # position above the data cloud, on the clean line
            annot_log = (1 - t) * lx_min + t * lx_max
            annot_xy = np.log10(10**annot_log)
            fig.add_annotation(
                x=annot_xy,
                y=annot_xy,
                text=ref_text,
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=1.5,
                arrowcolor="#9ca3af",
                ax=-80,
                ay=-50,
                axref="pixel",
                ayref="pixel",
                font=dict(size=17, color="#6b7280"),
                align="left",
            )

    # NOTE: visibility arrays built later, after all traces (including size legend) are added

    # ---------- Hover and styling ----------
    x_hover, y_hover = _build_hover_template(x_field, y_field)
    hover_template = (
        "<b>%{customdata[0]}</b>"
        f"<br>{x_hover}"
        f"<br>{y_hover}"
        "<br>Subjects (total): %{customdata[1]:,}"
        "<br>Records (total): %{customdata[2]:,}"
        "<br>Duration (total): %{customdata[10]}"
        "<br>Duration/subject: %{customdata[11]}"
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
            y=1.02,
            yanchor="bottom",
            buttons=[
                dict(
                    label="Experiment Modality",
                    method="restyle",
                    args=["visible", []],  # Updated after all traces added
                ),
                dict(
                    label="Recording Modality",
                    method="restyle",
                    args=["visible", []],  # Updated after all traces added
                ),
            ],
            pad={"r": 8, "t": 8},
            showactive=True,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            font=dict(size=12, color="#374151"),
        )
    ]

    fig.update_layout(
        height=height,
        width=max_width,
        margin=dict(l=60, r=80, t=50, b=60),
        template="plotly_white",
        updatemenus=updatemenus,
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
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=16,
        ),
        title=dict(text="", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        autosize=False,
    )

    # (footnote removed — branding handles attribution)

    # ---------- Bubble size legend (reference circles via hidden traces) ----------
    ref_durations = [5, 30, 120, 480]  # minutes per subject
    ref_labels = ["5 min", "30 min", "2 h", "8 h"]
    ref_bubble_sizes = [np.log1p(d) for d in ref_durations]

    # Use a secondary x-axis in paper space for the legend bubbles
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=0, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add reference bubbles as real scatter points on a second axes pair
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
        ),
        yaxis2=dict(
            overlaying="y",
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
        ),
    )

    # Title
    fig.add_annotation(
        xref="x2",
        yref="y2",
        x=0.92,
        y=0.52,
        xanchor="center",
        yanchor="bottom",
        text="<b>Bubble size</b> = duration / subject",
        showarrow=False,
        font=dict(size=14, color="#374151"),
    )

    # Compute pixel sizes for legend bubbles (independent of chart sizeref)
    legend_sizeref = sizeref * 0.55
    legend_x_circle = 0.88
    legend_x_label = 0.93

    # Place reference bubbles vertically, largest at top (natural reading order)
    for i, (bsz, label) in enumerate(zip(ref_bubble_sizes, ref_labels)):
        cy = 0.22 + i * 0.07
        fig.add_trace(
            go.Scatter(
                x=[legend_x_circle],
                y=[cy],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker=dict(
                    size=[bsz],
                    sizemode="area",
                    sizeref=legend_sizeref,
                    sizemin=6,
                    color="rgba(148,163,184,0.45)",
                    line=dict(width=1, color="rgba(100,116,139,0.35)"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            xref="x2",
            yref="y2",
            x=legend_x_label,
            y=cy,
            xanchor="left",
            yanchor="middle",
            text=label,
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
        )

    # ---------- Build visibility arrays for toggle buttons ----------
    # Must be AFTER all traces are added (OLS line + size legend traces)
    n_total_traces = len(fig.data)
    n_extra_traces = n_total_traces - n_exp_traces - n_rec_traces

    # Match initial state: background categories as "legendonly"
    exp_vis = []
    for t in fig.data[:n_exp_traces]:
        if t.visible == "legendonly":
            exp_vis.append("legendonly")
        else:
            exp_vis.append(True)
    rec_vis = []
    for t in fig.data[n_exp_traces : n_exp_traces + n_rec_traces]:
        # Rec traces start hidden; when toggled on, bg ones should be legendonly
        if t.name in _BACKGROUND_CATS:
            rec_vis.append("legendonly")
        else:
            rec_vis.append(True)

    exp_visible = exp_vis + [False] * n_rec_traces + [True] * n_extra_traces
    rec_visible = [False] * n_exp_traces + rec_vis + [True] * n_extra_traces

    # Update button args with final visibility arrays
    fig.layout.updatemenus[0].buttons[0].args = ["visible", exp_visible]
    fig.layout.updatemenus[0].buttons[1].args = ["visible", rec_visible]

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        zeroline=False,
        type="log",
        dtick=1,
        selector=dict(overlaying=None),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        zeroline=False,
        type="log",
        dtick=1,
        selector=dict(overlaying=None),
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
