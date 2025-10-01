from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:  # Allow execution as a script or module
    from .colours import MODALITY_COLOR_MAP
    from .utils import get_dataset_url, human_readable_size, primary_modality, safe_int
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import MODALITY_COLOR_MAP  # type: ignore
    from utils import (  # type: ignore
        get_dataset_url,
        human_readable_size,
        primary_modality,
        safe_int,
    )

__all__ = ["generate_dataset_bubble"]


def _to_numeric_median_list(val) -> float | None:
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        pass

    s = str(val).strip().strip("[]")
    if not s:
        return None

    try:
        nums = [float(x) for x in s.split(",") if str(x).strip()]
        if not nums:
            return None
        return float(np.median(nums))
    except Exception:
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

    data["modality_label"] = data.get("modality of exp").apply(primary_modality)

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

    fig = px.scatter(
        data,
        x=x_field,
        y=y_field,
        size="size_gb",
        color="modality_label",
        hover_name="dataset",
        custom_data=[
            data["dataset"],
            data["subjects"],
            data["records"],
            data["tasks"],
            nchans_str,
            sfreq_str,
            data["size_bytes"].map(
                lambda bytes_: human_readable_size(safe_int(bytes_, 0))
            ),
            data["modality_label"],
            data["dataset_url"],
        ],
        size_max=40,
        labels={
            y_field: y_label,
            "modality_label": "Modality",
            x_field: x_label,
        },
        color_discrete_map=MODALITY_COLOR_MAP,
        title="",
        category_orders={
            "modality_label": [
                label
                for label in MODALITY_COLOR_MAP.keys()
                if label in data["modality_label"].unique()
            ]
        },
        log_x=True,
        log_y=True,
    )

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
                    line=dict(color="#111827", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            residuals = log_y - (slope * log_x + intercept)
            r_squared = 1 - np.sum(residuals**2) / ss_tot
            fit_annotation_text = f"log-log OLS fit R² = {r_squared:.3f}"

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
        "<br>Modality: %{customdata[7]}"
        "<br><i>Click bubble to open dataset page</i>"
        "<extra></extra>"
    )

    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "markers" not in mode:
            continue
        trace.marker.update(
            sizemin=6,
            sizemode="area",
            sizeref=sizeref,
            line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
            opacity=0.75,
        )
        trace.hovertemplate = hover_template

    fig.update_layout(
        height=height,
        width=max_width,
        margin=dict(l=60, r=40, t=80, b=60),
        template="plotly_white",
        legend=dict(
            title="Modality",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.99,
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        title=dict(text="", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        autosize=True,
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

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="dataset-bubble",
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

    styled_html = f"""
<style>
#dataset-bubble {{
    width: 100% !important;
    max-width: {max_width}px;
    height: {height}px !important;
    min-height: {height}px;
    margin: 0 auto;
}}
#dataset-bubble .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
.dataset-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}
</style>
<div class="dataset-loading" id="dataset-loading">Loading dataset landscape...</div>
{html_content}
<script>
document.addEventListener('DOMContentLoaded', function() {{
    const loading = document.getElementById('dataset-loading');
    const plot = document.getElementById('dataset-bubble');

    function showPlot() {{
        if (loading) {{
            loading.style.display = 'none';
        }}
        if (plot) {{
            plot.style.display = 'block';
        }}
    }}

    function hookPlotlyClick(attempts) {{
        if (!plot || typeof plot.on !== 'function') {{
            if (attempts < 40) {{
                window.setTimeout(function() {{ hookPlotlyClick(attempts + 1); }}, 60);
            }}
            return;
        }}
        plot.on('plotly_click', function(evt) {{
            const point = evt && evt.points && evt.points[0];
            const url = point && point.customdata && point.customdata[8];
            if (url) {{
                window.open(url, '_blank', 'noopener');
            }}
        }});
        showPlot();
    }}

    hookPlotlyClick(0);
    showPlot();
}});
</script>
"""

    out_path.write_text(styled_html, encoding="utf-8")
    return out_path


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, header=0, skipinitialspace=True)


def main() -> None:
    import argparse

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
