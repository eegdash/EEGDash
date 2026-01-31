"""MOABB-style circle-packing bubble visualization for EEGDash.

This module creates a MOABB-inspired visualization using D3.js circle packing:
- Each **dataset** is a cluster of circles
- Each **circle** represents one subject
- Circle **size** = log(records per subject) - data volume per subject
- Circle **opacity** = number of sessions (fewer sessions = more opaque)
- Circle **color** = category (recording modality by default)
- Datasets are **grouped by modality** into distinct regional clusters

Features:
- **D3.js circle packing**: Uses d3.pack() for optimal, aesthetically pleasing layouts
- Hover-based interactivity: Dataset clusters highlight on hover, others dim
- Rich tooltips: Subjects, sessions, records, channels, sampling rate, size
- Click to open: Each dataset links to its detail page
- SVG-based rendering: Crisp circles at any zoom level

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

# Maximum bubbles per dataset to prevent performance issues with very large datasets
MAX_BUBBLES_PER_DATASET = 200

# Modality display order (for consistent regional positioning)
MODALITY_ORDER = ["EEG", "MEG", "iEEG", "fNIRS", "EMG", "Other"]

# Background colors for modality regions (semi-transparent)
MODALITY_BG_COLORS = {
    "EEG": "rgba(59, 130, 246, 0.08)",  # Blue tint
    "MEG": "rgba(168, 85, 247, 0.08)",  # Purple tint
    "iEEG": "rgba(6, 182, 212, 0.08)",  # Cyan tint
    "fNIRS": "rgba(249, 115, 22, 0.08)",  # Orange tint
    "EMG": "rgba(16, 185, 129, 0.08)",  # Green tint
    "Other": "rgba(148, 163, 184, 0.08)",  # Gray tint
}


def _get_bubble_size(records_per_subject: float, scale: float = 1.0) -> float:
    """Calculate bubble size from data volume per subject.

    Args:
        records_per_subject: Number of records per subject (proxy for trials)
        scale: Scaling factor

    Returns:
        Bubble size value for D3

    """
    size = max(1.0, float(records_per_subject))
    return np.log(size + 1.0) * scale * 10


def _get_alpha(n_sessions: int) -> float:
    """Calculate bubble alpha based on number of sessions.

    Args:
        n_sessions: Number of sessions

    Returns:
        Alpha value (0.3 to 0.9)

    """
    alphas = [0.9, 0.75, 0.6, 0.45, 0.35]
    idx = min(max(0, n_sessions - 1), len(alphas) - 1)
    return alphas[idx]


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
                    continue
        if not vals:
            return None
        return float(np.median(np.repeat(vals, weights)))

    # Handle literal collections
    if isinstance(val, (list, np.ndarray, pd.Series)):
        if len(val) == 0:
            return None
        try:
            return float(np.nanmedian(val))
        except (ValueError, TypeError):
            pass

    # Handle scalar NaNs
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        if np.asarray(pd.isna(val)).any():
            return None

    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        pass

    # Fall back to string parsing
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


def _format_int(value: Any) -> str:
    """Format a value as integer string, or empty string if invalid."""
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
    """Generate MOABB-style circle-packing bubble plot using D3.js.

    Each dataset becomes a cluster of circles (one per subject), arranged
    using D3's circle packing algorithm. Datasets are grouped by modality
    with color encoding the selected category.

    Args:
        df: DataFrame with dataset information
        out_html: Output HTML file path
        color_by: Column to use for coloring ("modality", "type", "pathology")
        scale: Scaling factor for bubble sizes
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        Path to the generated HTML file

    """
    data = df.copy()

    # Input validation
    if data.empty or "dataset" not in data.columns:
        out_path = Path(out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty_html = '<div class="dataset-loading">No dataset records available for MOABB bubble plot.</div>'
        out_path.write_text(empty_html, encoding="utf-8")
        return out_path

    # Filter out test datasets
    data["dataset"] = data["dataset"].astype(str)
    data = data[data["dataset"].str.lower() != "test"]

    # Ensure required columns exist
    if "n_subjects" not in data.columns:
        if "subjects" in data.columns:
            data["n_subjects"] = data["subjects"]
        else:
            data["n_subjects"] = 1

    if "n_sessions" not in data.columns:
        if "sessions" in data.columns:
            data["n_sessions"] = data["sessions"]
        else:
            data["n_sessions"] = 1

    if "n_records" not in data.columns:
        if "total_files" in data.columns:
            data["n_records"] = data["total_files"]
        elif "records" in data.columns:
            data["n_records"] = data["records"]
        else:
            data["n_records"] = 1

    # Get modality column
    modality_col = None
    for col in ["record_modality", "recording_modality", "modality of exp", "modality"]:
        if col in data.columns:
            modality_col = col
            break

    if modality_col:
        data["category"] = data[modality_col].apply(primary_recording_modality)
    else:
        data["category"] = "EEG"

    # Clean data
    data["n_subjects"] = pd.to_numeric(data["n_subjects"], errors="coerce").fillna(1)
    data["n_sessions"] = pd.to_numeric(data["n_sessions"], errors="coerce").fillna(1)
    data["n_records"] = pd.to_numeric(data["n_records"], errors="coerce").fillna(1)

    # Convert to int
    data["n_subjects"] = data["n_subjects"].astype(int).clip(lower=1)
    data["n_sessions"] = data["n_sessions"].astype(int).clip(lower=1)
    data["n_records"] = data["n_records"].astype(int).clip(lower=1)

    # Filter to datasets with at least 1 subject
    data = data[data["n_subjects"] >= 1].copy()

    # Derive records per subject for sizing
    data["records_per_subject"] = data["n_records"] / data["n_subjects"]
    data["records_per_subject"] = (
        data["records_per_subject"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=1.0)
    )

    data = data.sort_values("dataset").reset_index(drop=True)

    # Extract additional metadata for rich tooltips
    if "n_tasks" not in data.columns:
        if "tasks" in data.columns:
            data["n_tasks"] = pd.to_numeric(data["tasks"], errors="coerce").fillna(0)
        else:
            data["n_tasks"] = 0
    else:
        data["n_tasks"] = pd.to_numeric(data["n_tasks"], errors="coerce").fillna(0)

    # Compute nchans and sfreq medians
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

    # Size in bytes
    if "size_bytes" in data.columns:
        data["size_bytes_clean"] = pd.to_numeric(
            data["size_bytes"], errors="coerce"
        ).fillna(0)
    else:
        data["size_bytes_clean"] = 0

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data.empty:
        empty_html = '<div class="dataset-loading">No dataset records available for MOABB bubble plot.</div>'
        out_path.write_text(empty_html, encoding="utf-8")
        return out_path

    # =========================================================================
    # Build hierarchical data structure for D3
    # =========================================================================

    # Build hierarchy: root -> modality -> dataset -> subjects
    hierarchy = {"name": "root", "children": []}

    # Group by modality
    modality_groups: dict[str, list[dict]] = {}
    for _, row in data.iterrows():
        modality = row["category"]
        if modality not in modality_groups:
            modality_groups[modality] = []

        n_subjects = int(row["n_subjects"])
        n_sessions = int(row["n_sessions"])
        n_records = int(row["n_records"])
        records_per_subject = float(row["records_per_subject"])

        # Limit subjects per dataset
        effective_subjects = min(n_subjects, MAX_BUBBLES_PER_DATASET)

        # Extract metadata
        nchans = row.get("nchans_median")
        sfreq = row.get("sfreq_median")
        size_bytes = safe_int(row.get("size_bytes_clean", 0), 0)

        nchans_str = _format_int(nchans) if nchans else "—"
        sfreq_str = _format_int(sfreq) if sfreq else "—"
        size_str = human_readable_size(size_bytes) if size_bytes else "—"

        # Calculate bubble size based on records per subject
        bubble_size = _get_bubble_size(records_per_subject, scale)
        alpha = _get_alpha(n_sessions)

        # Create dataset node with subject children
        dataset_node = {
            "name": row["dataset"],
            "title": row.get("dataset_title", "") or "",
            "modality": modality,
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
            "n_records": n_records,
            "records_per_subject": round(records_per_subject, 1),
            "nchans": nchans_str,
            "sfreq": sfreq_str,
            "size": size_str,
            "url": get_dataset_url(row["dataset"]),
            "alpha": alpha,
            "color": MODALITY_COLOR_MAP.get(modality, "#94a3b8"),
            "children": [
                {"name": f"s{i}", "value": bubble_size}
                for i in range(effective_subjects)
            ],
        }

        modality_groups[modality].append(dataset_node)

    # Order modalities consistently
    ordered_modalities = [m for m in MODALITY_ORDER if m in modality_groups]
    ordered_modalities += [m for m in modality_groups if m not in ordered_modalities]

    for modality in ordered_modalities:
        modality_node = {
            "name": modality,
            "color": MODALITY_COLOR_MAP.get(modality, "#94a3b8"),
            "bgColor": MODALITY_BG_COLORS.get(modality, "rgba(148,163,184,0.08)"),
            "children": modality_groups[modality],
        }
        hierarchy["children"].append(modality_node)

    hierarchy_json = json.dumps(hierarchy)

    # =========================================================================
    # Generate HTML with D3.js visualization
    # =========================================================================

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOABB Bubble Chart</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Inter, system-ui, -apple-system, sans-serif;
            background: #ffffff;
        }}
        #moabb-bubble {{
            width: 100%;
            height: {height}px;
            display: block;
        }}
        .dataset-loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: {height}px;
            font-family: Inter, system-ui, sans-serif;
            color: #6b7280;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 13px;
            line-height: 1.5;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            pointer-events: none;
            z-index: 1000;
            max-width: 320px;
        }}
        .tooltip-title {{
            font-weight: 600;
            font-size: 14px;
            color: #1f2937;
            margin-bottom: 2px;
        }}
        .tooltip-subtitle {{
            color: #6b7280;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            margin: 3px 0;
        }}
        .tooltip-label {{
            color: #6b7280;
        }}
        .tooltip-value {{
            font-weight: 500;
            color: #1f2937;
        }}
        .tooltip-hint {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #e5e7eb;
            color: #3b82f6;
            font-size: 11px;
            font-style: italic;
        }}
        .legend {{
            position: absolute;
            top: 15px;
            left: 15px;
            background: rgba(255,255,255,0.95);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 12px;
            line-height: 1.6;
        }}
        .legend-title {{
            font-weight: 600;
            margin-bottom: 6px;
            color: #374151;
        }}
        .legend-item {{
            color: #6b7280;
        }}
        .modality-legend {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(255,255,255,0.95);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 12px;
        }}
        .modality-legend-title {{
            font-weight: 600;
            margin-bottom: 8px;
            color: #374151;
        }}
        .modality-legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
            cursor: pointer;
        }}
        .modality-legend-item:hover {{
            opacity: 0.8;
        }}
        .modality-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .hint {{
            position: absolute;
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255,255,255,0.9);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <div id="container" style="position: relative; width: {width}px; height: {height}px; margin: 0 auto;">
        <svg id="moabb-bubble"></svg>
        <div class="tooltip" style="display: none;"></div>
        <div class="legend">
            <div class="legend-title">Legend</div>
            <div class="legend-item"><b>Circle size</b>: log(records per subject)</div>
            <div class="legend-item"><b>Opacity</b>: fewer sessions = more opaque</div>
            <div class="legend-item"><b>Each circle</b> = 1 subject</div>
        </div>
        <div class="modality-legend" id="modality-legend"></div>
        <div class="hint">
            <b>Hover</b> to highlight dataset · <b>Scroll</b> to zoom · <b>Drag</b> to pan · <b>Click</b> to open
        </div>
    </div>

    <script>
        const data = {hierarchy_json};
        const width = {width};
        const height = {height};
        const margin = 60;

        // Create SVG
        const svg = d3.select("#moabb-bubble")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height]);

        // Create container for zoom
        const g = svg.append("g");

        // Tooltip
        const tooltip = d3.select(".tooltip");

        // Build modality legend
        const legendContainer = d3.select("#modality-legend");
        legendContainer.append("div")
            .attr("class", "modality-legend-title")
            .text("Recording Modality");

        data.children.forEach(modality => {{
            const item = legendContainer.append("div")
                .attr("class", "modality-legend-item")
                .on("click", () => {{
                    // Toggle modality visibility
                    const modalityGroup = g.select(`[data-modality="${{modality.name}}"]`);
                    const isHidden = modalityGroup.style("opacity") === "0.1";
                    modalityGroup.style("opacity", isHidden ? 1 : 0.1);
                }});

            item.append("div")
                .attr("class", "modality-swatch")
                .style("background", modality.color);

            item.append("span")
                .text(`${{modality.name}} (${{modality.children.length}})`);
        }});

        // Create pack layout - pack modalities separately then position them
        const modalityPacks = [];
        const packPadding = 15;

        data.children.forEach((modality, i) => {{
            // Create hierarchy for this modality
            const modalityHierarchy = d3.hierarchy({{
                name: modality.name,
                children: modality.children
            }})
            .sum(d => d.value || 0)
            .sort((a, b) => b.value - a.value);

            // Calculate size based on number of datasets
            const modalitySize = Math.sqrt(modalityHierarchy.value) * 2.5 + 100;

            // Pack this modality
            const pack = d3.pack()
                .size([modalitySize, modalitySize])
                .padding(d => d.depth === 0 ? 20 : d.depth === 1 ? 8 : 2);

            const packedModality = pack(modalityHierarchy);

            modalityPacks.push({{
                modality: modality,
                packed: packedModality,
                size: modalitySize,
                radius: modalitySize / 2
            }});
        }});

        // Position modalities in a force-directed layout
        const simulation = d3.forceSimulation(modalityPacks)
            .force("x", d3.forceX(width / 2).strength(0.05))
            .force("y", d3.forceY(height / 2).strength(0.05))
            .force("collide", d3.forceCollide(d => d.radius + 30).strength(0.8))
            .stop();

        // Run simulation
        for (let i = 0; i < 300; i++) simulation.tick();

        // Render each modality group
        modalityPacks.forEach(mp => {{
            const modalityGroup = g.append("g")
                .attr("data-modality", mp.modality.name)
                .attr("transform", `translate(${{mp.x - mp.size/2}}, ${{mp.y - mp.size/2}})`);

            // Draw modality background circle
            modalityGroup.append("circle")
                .attr("cx", mp.size / 2)
                .attr("cy", mp.size / 2)
                .attr("r", mp.radius + 20)
                .attr("fill", mp.modality.bgColor)
                .attr("stroke", mp.modality.color)
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4,4")
                .style("pointer-events", "none");

            // Add modality label
            modalityGroup.append("text")
                .attr("x", mp.size / 2)
                .attr("y", -10)
                .attr("text-anchor", "middle")
                .attr("font-size", "14px")
                .attr("font-weight", "600")
                .attr("fill", mp.modality.color)
                .text(mp.modality.name);

            // Process nodes - group by dataset
            const nodes = mp.packed.descendants().filter(d => d.depth > 0);

            // Group subjects by dataset
            const datasetNodes = nodes.filter(d => d.depth === 1);
            const subjectNodes = nodes.filter(d => d.depth === 2);

            // Draw subject circles (leaves)
            subjectNodes.forEach(node => {{
                const dataset = node.parent.data;
                const alpha = dataset.alpha || 0.7;

                modalityGroup.append("circle")
                    .attr("class", "subject-bubble")
                    .attr("data-dataset", dataset.name)
                    .attr("cx", node.x)
                    .attr("cy", node.y)
                    .attr("r", Math.max(node.r, 2))
                    .attr("fill", dataset.color)
                    .attr("fill-opacity", alpha)
                    .attr("stroke", "rgba(255,255,255,0.5)")
                    .attr("stroke-width", 0.5)
                    .style("cursor", "pointer")
                    .on("mouseover", function(event) {{
                        // Highlight this dataset
                        const dsName = dataset.name;
                        g.selectAll(".subject-bubble")
                            .style("opacity", function() {{
                                return d3.select(this).attr("data-dataset") === dsName ? 1 : 0.15;
                            }})
                            .attr("stroke-width", function() {{
                                return d3.select(this).attr("data-dataset") === dsName ? 2 : 0.5;
                            }})
                            .attr("stroke", function() {{
                                return d3.select(this).attr("data-dataset") === dsName
                                    ? dataset.color : "rgba(255,255,255,0.5)";
                            }});

                        // Show tooltip
                        tooltip.style("display", "block")
                            .html(`
                                <div class="tooltip-title">${{dataset.name}}</div>
                                <div class="tooltip-subtitle">${{dataset.title || ''}}</div>
                                <div class="tooltip-row"><span class="tooltip-label">📊 Subjects</span><span class="tooltip-value">${{dataset.n_subjects.toLocaleString()}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">🔄 Sessions</span><span class="tooltip-value">${{dataset.n_sessions.toLocaleString()}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">📁 Records</span><span class="tooltip-value">${{dataset.n_records.toLocaleString()}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">📈 Records/Subject</span><span class="tooltip-value">${{dataset.records_per_subject}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">📡 Channels</span><span class="tooltip-value">${{dataset.nchans}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">⚡ Sampling</span><span class="tooltip-value">${{dataset.sfreq}} Hz</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">💾 Size</span><span class="tooltip-value">${{dataset.size}}</span></div>
                                <div class="tooltip-row"><span class="tooltip-label">🎯 Modality</span><span class="tooltip-value">${{dataset.modality}}</span></div>
                                <div class="tooltip-hint">Click to open dataset page →</div>
                            `)
                            .style("left", (event.pageX + 15) + "px")
                            .style("top", (event.pageY - 10) + "px");
                    }})
                    .on("mousemove", function(event) {{
                        tooltip
                            .style("left", (event.pageX + 15) + "px")
                            .style("top", (event.pageY - 10) + "px");
                    }})
                    .on("mouseout", function() {{
                        // Reset all bubbles
                        g.selectAll(".subject-bubble")
                            .style("opacity", 1)
                            .attr("stroke-width", 0.5)
                            .attr("stroke", "rgba(255,255,255,0.5)");

                        tooltip.style("display", "none");
                    }})
                    .on("click", function() {{
                        if (dataset.url) {{
                            window.open(dataset.url, '_blank', 'noopener');
                        }}
                    }});
            }});

            // Add dataset labels for larger datasets
            datasetNodes
                .filter(d => d.r > 30 && d.data.n_subjects > 5)
                .forEach(node => {{
                    const dataset = node.data;
                    const labelText = dataset.name.length > 12
                        ? dataset.name.substring(0, 12).toUpperCase()
                        : dataset.name.toUpperCase();

                    modalityGroup.append("text")
                        .attr("x", node.x)
                        .attr("y", node.y)
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "middle")
                        .attr("font-size", Math.min(10, node.r / 3) + "px")
                        .attr("font-weight", "500")
                        .attr("fill", "#374151")
                        .style("pointer-events", "none")
                        .text(labelText);
                }});
        }});

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Center the visualization initially
        const bounds = g.node().getBBox();
        const dx = bounds.width;
        const dy = bounds.height;
        const x = bounds.x + dx / 2;
        const y = bounds.y + dy / 2;
        const scale = Math.min(0.9 * width / dx, 0.9 * height / dy, 1);
        const translate = [width / 2 - scale * x, height / 2 - scale * y];

        svg.call(zoom.transform, d3.zoomIdentity
            .translate(translate[0], translate[1])
            .scale(scale));
    </script>
</body>
</html>
"""

    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point for MOABB bubble chart generation."""
    parser = argparse.ArgumentParser(
        description="Generate MOABB-style bubble chart for EEGDash datasets."
    )
    parser.add_argument("source", type=Path, help="Path to dataset summary CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_moabb_bubble.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for bubble sizes",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Chart width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Chart height in pixels",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.source, index_col=False, header=0, skipinitialspace=True)
    output_path = generate_moabb_bubble(
        df, args.output, scale=args.scale, width=args.width, height=args.height
    )
    print(f"MOABB bubble chart saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
