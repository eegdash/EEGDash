"""MOABB-style circle-packing bubble visualization for EEGDash.

This module creates a MOABB-inspired visualization using D3.js circle packing:
- Each **dataset** is a cluster of circles
- Each **circle** represents one subject
- Circle **size** = log(records per subject) - data volume per subject
- Circle **opacity** = number of sessions (fewer sessions = more opaque)
- Circle **color** = category (recording modality by default)
- Datasets are **grouped by modality** into distinct regional clusters

Performance optimizations:
- Limited bubbles per dataset (max 50) for faster rendering
- CSS-based hover states for GPU-accelerated transitions
- Grouped SVG elements by dataset for efficient selection
- Reduced force simulation iterations

Inspired by the MOABB plotting style and hierarchical data visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:  # Allow execution as a script or module
    from .colours import MODALITY_COLOR_MAP
    from .utils import (
        get_dataset_url,
        human_readable_size,
        primary_recording_modality,
        safe_int,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import MODALITY_COLOR_MAP  # type: ignore
    from utils import (  # type: ignore
        get_dataset_url,
        human_readable_size,
        primary_recording_modality,
        safe_int,
    )

__all__ = ["generate_moabb_bubble"]

# Maximum bubbles per dataset - reduced for performance
MAX_BUBBLES_PER_DATASET = 50

# Modality display order (for consistent regional positioning)
MODALITY_ORDER = ["EEG", "MEG", "iEEG", "fNIRS", "EMG", "Other"]

# Background colors for modality regions (semi-transparent)
MODALITY_BG_COLORS = {
    "EEG": "rgba(59, 130, 246, 0.08)",
    "MEG": "rgba(168, 85, 247, 0.08)",
    "iEEG": "rgba(6, 182, 212, 0.08)",
    "fNIRS": "rgba(249, 115, 22, 0.08)",
    "EMG": "rgba(16, 185, 129, 0.08)",
    "Other": "rgba(148, 163, 184, 0.08)",
}


def _get_bubble_size(records_per_subject: float, scale: float = 1.0) -> float:
    """Calculate bubble size from data volume per subject."""
    size = max(1.0, float(records_per_subject))
    return np.log(size + 1.0) * scale * 12


def _get_alpha(n_sessions: int) -> float:
    """Calculate bubble alpha based on number of sessions."""
    alphas = [0.9, 0.75, 0.6, 0.45, 0.35]
    idx = min(max(0, n_sessions - 1), len(alphas) - 1)
    return alphas[idx]


def _to_numeric_median_list(val: Any) -> float | None:
    """Compute median from nchans/sfreq data."""
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(val, list) and val and isinstance(val[0], dict) and "val" in val[0]:
        vals = []
        weights = []
        for item in val:
            v = item.get("val")
            if v is not None:
                try:
                    vals.append(float(v))
                    weights.append(item.get("count", 1))
                except (ValueError, TypeError):
                    continue
        if not vals:
            return None
        return float(np.median(np.repeat(vals, weights)))

    if isinstance(val, (list, np.ndarray, pd.Series)):
        if len(val) == 0:
            return None
        try:
            return float(np.nanmedian(val))
        except (ValueError, TypeError):
            pass

    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass

    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        pass

    return None


def _format_int(value: Any) -> str:
    """Format a value as integer string."""
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def generate_moabb_bubble(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    color_by: Literal["modality", "type", "pathology"] = "modality",
    scale: float = 1.0,
    width: int = 1400,
    height: int = 900,
) -> Path:
    """Generate optimized MOABB-style circle-packing bubble plot using D3.js."""
    data = df.copy()

    if data.empty or "dataset" not in data.columns:
        out_path = Path(out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            '<div class="dataset-loading">No data available.</div>', encoding="utf-8"
        )
        return out_path

    data["dataset"] = data["dataset"].astype(str)
    data = data[data["dataset"].str.lower() != "test"]

    # Ensure required columns
    for col, fallbacks in [
        ("n_subjects", ["subjects"]),
        ("n_sessions", ["sessions"]),
        ("n_records", ["total_files", "records"]),
    ]:
        if col not in data.columns:
            for fb in fallbacks:
                if fb in data.columns:
                    data[col] = data[fb]
                    break
            else:
                data[col] = 1

    # Get modality
    modality_col = None
    for col in ["record_modality", "recording_modality", "modality of exp", "modality"]:
        if col in data.columns:
            modality_col = col
            break
    data["category"] = (
        data[modality_col].apply(primary_recording_modality) if modality_col else "EEG"
    )

    # Clean numeric columns
    for col in ["n_subjects", "n_sessions", "n_records"]:
        data[col] = (
            pd.to_numeric(data[col], errors="coerce")
            .fillna(1)
            .astype(int)
            .clip(lower=1)
        )

    data = data[data["n_subjects"] >= 1].copy()
    data["records_per_subject"] = (data["n_records"] / data["n_subjects"]).clip(lower=1)
    data = data.sort_values("dataset").reset_index(drop=True)

    # Optional metadata
    if "nchans_set" in data.columns:
        data["nchans_median"] = data["nchans_set"].apply(_to_numeric_median_list)
    elif "nchans" in data.columns:
        data["nchans_median"] = pd.to_numeric(data["nchans"], errors="coerce")
    else:
        data["nchans_median"] = None

    if "sampling_freqs" in data.columns:
        data["sfreq_median"] = data["sampling_freqs"].apply(_to_numeric_median_list)
    elif "sfreq" in data.columns:
        data["sfreq_median"] = pd.to_numeric(data["sfreq"], errors="coerce")
    else:
        data["sfreq_median"] = None

    if "size_bytes" in data.columns:
        data["size_bytes_clean"] = pd.to_numeric(
            data["size_bytes"], errors="coerce"
        ).fillna(0)
    else:
        data["size_bytes_clean"] = 0

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data.empty:
        out_path.write_text(
            '<div class="dataset-loading">No data available.</div>', encoding="utf-8"
        )
        return out_path

    # Build hierarchy
    hierarchy = {"name": "root", "children": []}
    modality_groups: dict[str, list[dict]] = {}

    for _, row in data.iterrows():
        modality = row["category"]
        if modality not in modality_groups:
            modality_groups[modality] = []

        n_subjects = int(row["n_subjects"])
        n_sessions = int(row["n_sessions"])
        n_records = int(row["n_records"])
        records_per_subject = float(row["records_per_subject"])
        effective_subjects = min(n_subjects, MAX_BUBBLES_PER_DATASET)

        nchans = row.get("nchans_median")
        sfreq = row.get("sfreq_median")
        size_bytes = safe_int(row.get("size_bytes_clean", 0), 0)

        bubble_size = _get_bubble_size(records_per_subject, scale)
        alpha = _get_alpha(n_sessions)

        dataset_node = {
            "name": row["dataset"],
            "title": row.get("dataset_title", "") or "",
            "modality": modality,
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
            "n_records": n_records,
            "records_per_subject": round(records_per_subject, 1),
            "nchans": _format_int(nchans) if nchans else "—",
            "sfreq": _format_int(sfreq) if sfreq else "—",
            "size": human_readable_size(size_bytes) if size_bytes else "—",
            "url": get_dataset_url(row["dataset"]),
            "alpha": alpha,
            "color": MODALITY_COLOR_MAP.get(modality, "#94a3b8"),
            "children": [
                {"name": f"s{i}", "value": bubble_size}
                for i in range(effective_subjects)
            ],
        }
        modality_groups[modality].append(dataset_node)

    ordered_modalities = [m for m in MODALITY_ORDER if m in modality_groups]
    ordered_modalities += [m for m in modality_groups if m not in ordered_modalities]

    for modality in ordered_modalities:
        hierarchy["children"].append(
            {
                "name": modality,
                "color": MODALITY_COLOR_MAP.get(modality, "#94a3b8"),
                "bgColor": MODALITY_BG_COLORS.get(modality, "rgba(148,163,184,0.08)"),
                "children": modality_groups[modality],
            }
        )

    hierarchy_json = json.dumps(hierarchy)

    # Generate optimized HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOABB Bubble Chart</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Inter, system-ui, -apple-system, sans-serif; background: #fff; }}
        #moabb-bubble {{ width: 100%; height: {height}px; display: block; }}

        /* CSS-based hover states for GPU acceleration */
        .dataset-group {{ transition: opacity 0.12s ease-out; }}
        .dataset-group.dimmed {{ opacity: 0.15; }}
        .dataset-group.highlighted .bubble {{
            stroke-width: 2.5px;
            transform: scale(1.08);
            transform-origin: center;
        }}
        .bubble {{
            transition: transform 0.1s ease-out, stroke-width 0.1s ease-out;
            will-change: transform, opacity;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(255,255,255,0.98);
            border: 2px solid #3b82f6;
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 14px;
            line-height: 1.5;
            box-shadow: 0 6px 24px rgba(0,0,0,0.15);
            pointer-events: none;
            z-index: 1000;
            max-width: 340px;
            display: none;
        }}
        .tooltip-title {{ font-weight: 700; font-size: 16px; color: #1f2937; margin-bottom: 2px; }}
        .tooltip-subtitle {{ color: #6b7280; font-size: 13px; margin-bottom: 10px; }}
        .tooltip-row {{ display: flex; justify-content: space-between; margin: 4px 0; gap: 16px; }}
        .tooltip-label {{ color: #6b7280; font-size: 13px; }}
        .tooltip-value {{ font-weight: 600; color: #1f2937; font-size: 13px; }}
        .tooltip-hint {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #e5e7eb; color: #3b82f6; font-size: 12px; font-style: italic; }}

        .legend {{
            position: absolute; top: 12px; left: 12px;
            background: rgba(255,255,255,0.95);
            padding: 14px 18px; border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 14px; line-height: 1.7;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .legend-title {{ font-weight: 700; margin-bottom: 6px; color: #374151; font-size: 15px; }}
        .legend-item {{ color: #4b5563; font-size: 13px; }}

        .modality-legend {{
            position: absolute; top: 12px; right: 12px;
            background: rgba(255,255,255,0.95);
            padding: 14px 18px; border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .modality-legend-title {{ font-weight: 700; margin-bottom: 8px; color: #374151; font-size: 15px; }}
        .modality-legend-item {{ display: flex; align-items: center; margin: 5px 0; cursor: pointer; font-size: 13px; }}
        .modality-legend-item:hover {{ opacity: 0.7; }}
        .modality-swatch {{ width: 14px; height: 14px; border-radius: 50%; margin-right: 8px; }}

        .hint {{
            position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
            background: rgba(255,255,255,0.95); padding: 8px 16px; border-radius: 6px;
            font-size: 13px; color: #4b5563;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
    </style>
</head>
<body>
    <div id="container" style="position:relative;width:{width}px;height:{height}px;margin:0 auto;">
        <svg id="moabb-bubble"></svg>
        <div class="tooltip"></div>
        <div class="legend">
            <div class="legend-title">Legend</div>
            <div class="legend-item"><b>Circle size</b>: log(records/subject)</div>
            <div class="legend-item"><b>Opacity</b>: fewer sessions = more opaque</div>
            <div class="legend-item"><b>Each circle</b> = 1 subject</div>
        </div>
        <div class="modality-legend" id="modality-legend"></div>
        <div class="hint"><b>Hover</b> to highlight · <b>Scroll</b> to zoom · <b>Click</b> to open</div>
    </div>

<script>
const data = {hierarchy_json};
const width = {width}, height = {height};

const svg = d3.select("#moabb-bubble").attr("width", width).attr("height", height);
const g = svg.append("g");
const tooltip = d3.select(".tooltip");

// Build legend
const legendContainer = d3.select("#modality-legend");
legendContainer.append("div").attr("class", "modality-legend-title").text("Recording Modality");
data.children.forEach(m => {{
    const item = legendContainer.append("div").attr("class", "modality-legend-item");
    item.append("div").attr("class", "modality-swatch").style("background", m.color);
    item.append("span").text(`${{m.name}} (${{m.children.length}})`);
}});

// Pack each modality separately
const modalityPacks = [];
data.children.forEach(modality => {{
    const h = d3.hierarchy({{ name: modality.name, children: modality.children }})
        .sum(d => d.value || 0).sort((a, b) => b.value - a.value);
    const size = Math.sqrt(h.value) * 2.8 + 80;
    const pack = d3.pack().size([size, size]).padding(d => d.depth === 0 ? 15 : d.depth === 1 ? 6 : 1.5);
    modalityPacks.push({{ modality, packed: pack(h), size, radius: size / 2 }});
}});

// Position modalities with force simulation (reduced iterations)
const sim = d3.forceSimulation(modalityPacks)
    .force("x", d3.forceX(width / 2).strength(0.06))
    .force("y", d3.forceY(height / 2).strength(0.06))
    .force("collide", d3.forceCollide(d => d.radius + 25).strength(0.85))
    .stop();
for (let i = 0; i < 150; i++) sim.tick();

// Render
modalityPacks.forEach(mp => {{
    const mg = g.append("g").attr("data-modality", mp.modality.name)
        .attr("transform", `translate(${{mp.x - mp.size/2}},${{mp.y - mp.size/2}})`);

    // Background circle
    mg.append("circle").attr("cx", mp.size/2).attr("cy", mp.size/2).attr("r", mp.radius + 15)
        .attr("fill", mp.modality.bgColor).attr("stroke", mp.modality.color)
        .attr("stroke-width", 1.5).attr("stroke-dasharray", "4,4").style("pointer-events", "none");

    // Modality label
    mg.append("text").attr("x", mp.size/2).attr("y", -12).attr("text-anchor", "middle")
        .attr("font-size", "16px").attr("font-weight", "700").attr("fill", mp.modality.color)
        .text(mp.modality.name);

    const datasetNodes = mp.packed.descendants().filter(d => d.depth === 1);
    const subjectNodes = mp.packed.descendants().filter(d => d.depth === 2);

    // Group bubbles by dataset for efficient hover
    const datasetGroups = {{}};
    subjectNodes.forEach(node => {{
        const ds = node.parent.data;
        if (!datasetGroups[ds.name]) datasetGroups[ds.name] = {{ ds, nodes: [] }};
        datasetGroups[ds.name].nodes.push(node);
    }});

    Object.values(datasetGroups).forEach(({{ ds, nodes }}) => {{
        const dg = mg.append("g").attr("class", "dataset-group").attr("data-dataset", ds.name);

        nodes.forEach(node => {{
            dg.append("circle").attr("class", "bubble")
                .attr("cx", node.x).attr("cy", node.y).attr("r", Math.max(node.r, 2))
                .attr("fill", ds.color).attr("fill-opacity", ds.alpha)
                .attr("stroke", "rgba(255,255,255,0.6)").attr("stroke-width", 0.5);
        }});

        // Single event handler per dataset group
        dg.on("mouseenter", function(event) {{
            // Dim all, highlight this
            g.selectAll(".dataset-group").classed("dimmed", true).classed("highlighted", false);
            d3.select(this).classed("dimmed", false).classed("highlighted", true);

            tooltip.style("display", "block").style("border-color", ds.color)
                .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 10) + "px")
                .html(`<div class="tooltip-title">${{ds.name}}</div>
                    <div class="tooltip-subtitle">${{ds.title || ''}}</div>
                    <div class="tooltip-row"><span class="tooltip-label">📊 Subjects</span><span class="tooltip-value">${{ds.n_subjects.toLocaleString()}}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">🔄 Sessions</span><span class="tooltip-value">${{ds.n_sessions.toLocaleString()}}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">📁 Records</span><span class="tooltip-value">${{ds.n_records.toLocaleString()}}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">📡 Channels</span><span class="tooltip-value">${{ds.nchans}}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">⚡ Sampling</span><span class="tooltip-value">${{ds.sfreq}} Hz</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">💾 Size</span><span class="tooltip-value">${{ds.size}}</span></div>
                    <div class="tooltip-hint">Click to open dataset page →</div>`);
        }})
        .on("mousemove", event => tooltip.style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 10) + "px"))
        .on("mouseleave", () => {{
            g.selectAll(".dataset-group").classed("dimmed", false).classed("highlighted", false);
            tooltip.style("display", "none");
        }})
        .on("click", () => {{ if (ds.url) window.open(ds.url, '_blank', 'noopener'); }})
        .style("cursor", "pointer");
    }});

    // Dataset labels
    datasetNodes.filter(d => d.r > 25 && d.data.n_subjects > 3).forEach(node => {{
        const label = node.data.name.length > 10 ? node.data.name.substring(0, 10).toUpperCase() : node.data.name.toUpperCase();
        mg.append("text").attr("x", node.x).attr("y", node.y).attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle").attr("font-size", Math.max(9, Math.min(12, node.r / 3)) + "px")
            .attr("font-weight", "600").attr("fill", "#1f2937").style("pointer-events", "none")
            .style("text-shadow", "0 0 2px white, 0 0 2px white").text(label);
    }});
}});

// Zoom
const zoom = d3.zoom().scaleExtent([0.4, 3]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// Center view
const bounds = g.node().getBBox();
const scale = Math.min(0.88 * width / bounds.width, 0.88 * height / bounds.height, 1);
const tx = width / 2 - scale * (bounds.x + bounds.width / 2);
const ty = height / 2 - scale * (bounds.y + bounds.height / 2);
svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
</script>
</body>
</html>
"""

    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point for MOABB bubble chart generation."""
    parser = argparse.ArgumentParser(description="Generate MOABB bubble chart.")
    parser.add_argument("source", type=Path, help="Path to dataset summary CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("dataset_moabb_bubble.html")
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    df = pd.read_csv(args.source, index_col=False, header=0, skipinitialspace=True)
    output_path = generate_moabb_bubble(
        df, args.output, scale=args.scale, width=args.width, height=args.height
    )
    print(f"MOABB bubble chart saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
