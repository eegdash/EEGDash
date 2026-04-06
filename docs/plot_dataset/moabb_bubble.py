"""MOABB-style circle-packing bubble visualization for EEGDash.

This module creates a MOABB-inspired visualization using D3.js circle packing:
- Each **dataset** is a cluster of circles
- Each **circle** represents one subject
- Circle **size** = log(records per subject) - data volume per subject
- Circle **opacity** = number of sessions (fewer sessions = more opaque)
- Circle **color** = category (recording modality by default)
- Datasets are **grouped by modality** into distinct regional clusters

Performance optimizations:
- Canvas-based rendering for 100K+ circles without DOM overhead
- D3 quadtree for O(log n) hover hit-testing
- Simple row-based grid layout (no force simulation)
- D3 pack layout computed once, rendered as pixels

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

# Maximum bubbles per dataset — canvas rendering handles large counts easily
MAX_BUBBLES_PER_DATASET = 100

# Modality display order (for consistent regional positioning)
MODALITY_ORDER = ["EEG", "MEG", "iEEG", "fNIRS", "EMG", "Other"]

# Background colors for modality regions (semi-transparent)
MODALITY_BG_COLORS = {
    "EEG": "rgba(59, 130, 246, 0.06)",
    "MEG": "rgba(168, 85, 247, 0.06)",
    "iEEG": "rgba(6, 182, 212, 0.06)",
    "fNIRS": "rgba(249, 115, 22, 0.06)",
    "EMG": "rgba(16, 185, 129, 0.06)",
    "Other": "rgba(148, 163, 184, 0.06)",
}

# Top-N datasets to label per modality (by subject count)
TOP_LABELS_PER_MODALITY = 5


def _get_bubble_size(records_per_subject: float, scale: float = 1.0) -> float:
    """Calculate bubble size from data volume per subject."""
    size = max(1.0, float(records_per_subject))
    return np.log(size + 1.0) * scale * 12


def _get_alpha(n_sessions: int) -> float:
    """Calculate bubble alpha based on number of sessions."""
    alphas = [0.92, 0.82, 0.72, 0.60, 0.50]
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
    width: int = 1600,
    height: int = 1100,
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

    # Mark the top-N datasets per modality for labeling (by subject count)
    for modality, nodes in modality_groups.items():
        sorted_by_subjects = sorted(nodes, key=lambda d: d["n_subjects"], reverse=True)
        top_names = {d["name"] for d in sorted_by_subjects[:TOP_LABELS_PER_MODALITY]}
        for node in nodes:
            node["showLabel"] = node["name"] in top_names

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

    # Generate canvas-based HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOABB Bubble Chart</title>
    <script defer src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Inter, system-ui, -apple-system, sans-serif; background: #fff; }}
        #moabb-canvas {{ display: block; cursor: grab; }}
        #moabb-canvas:active {{ cursor: grabbing; }}

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
            padding: 16px 20px; border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 15px; line-height: 1.7;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .legend-title {{ font-weight: 700; margin-bottom: 6px; color: #374151; font-size: 17px; }}
        .legend-item {{ color: #4b5563; font-size: 14px; }}

        .modality-legend {{
            position: absolute; top: 210px; left: 12px;
            background: rgba(255,255,255,0.95);
            padding: 12px 16px; border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            max-height: calc(100% - 230px);
            overflow-y: auto;
        }}
        .modality-legend-title {{ font-weight: 700; margin-bottom: 8px; color: #374151; font-size: 16px; }}
        .modality-legend-item {{ display: flex; align-items: center; margin: 6px 0; cursor: pointer; font-size: 14px; transition: opacity 0.15s; }}
        .modality-legend-item:hover {{ opacity: 0.7; }}
        .modality-legend-item.hidden {{ opacity: 0.35; text-decoration: line-through; }}
        .modality-swatch {{ width: 16px; height: 16px; border-radius: 50%; margin-right: 10px; border: 2px solid transparent; }}
        .modality-legend-item.hidden .modality-swatch {{ background: #ccc !important; }}

        .controls {{
            position: absolute; bottom: 50px; right: 12px;
            display: flex; flex-direction: column; gap: 4px;
        }}
        .controls button {{
            width: 36px; height: 36px; border: 1px solid rgba(0,0,0,0.15);
            background: rgba(255,255,255,0.95); border-radius: 6px;
            font-size: 18px; cursor: pointer; color: #374151;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transition: background 0.15s, transform 0.1s;
        }}
        .controls button:hover {{ background: #f3f4f6; }}
        .controls button:active {{ transform: scale(0.95); }}

        .search-box {{
            position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
            display: flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.95); padding: 8px 14px;
            border-radius: 8px; border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .search-box input {{
            border: none; outline: none; font-size: 14px; width: 180px;
            background: transparent; color: #1f2937;
        }}
        .search-box input::placeholder {{ color: #9ca3af; }}
        .search-icon {{ color: #6b7280; font-size: 14px; }}
        .search-clear {{ cursor: pointer; color: #9ca3af; font-size: 16px; display: none; }}
        .search-clear:hover {{ color: #6b7280; }}
        .search-results {{ font-size: 12px; color: #6b7280; margin-left: 8px; }}

        .hint {{
            position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
            background: rgba(255,255,255,0.95); padding: 10px 20px; border-radius: 6px;
            font-size: 14px; color: #4b5563;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
    </style>
</head>
<body>
    <div id="container" style="position:relative;width:100%;max-width:{width}px;height:{height}px;margin:0 auto;overflow:hidden;">
        <canvas id="moabb-canvas" width="{width}" height="{height}"></canvas>
        <div class="tooltip" id="tooltip"></div>
        <div class="legend">
            <div class="legend-title">Legend</div>
            <div class="legend-item"><b>Circle size</b>: log(records/subject)</div>
            <div class="legend-item"><b>Opacity</b>: fewer sessions = more opaque</div>
            <div class="legend-item"><b>Each circle</b> = 1 subject</div>
            <div class="legend-item" id="total-datasets" style="margin-top:8px;padding-top:8px;border-top:1px solid #e5e7eb;font-weight:600;"></div>
        </div>
        <div class="modality-legend" id="modality-legend"></div>
        <div class="search-box">
            <span class="search-icon">&#128269;</span>
            <input type="text" id="search-input" placeholder="Search datasets...">
            <span class="search-clear" id="search-clear">&times;</span>
            <span class="search-results" id="search-results"></span>
        </div>
        <div class="controls">
            <button id="zoom-in" title="Zoom in">+</button>
            <button id="zoom-out" title="Zoom out">&minus;</button>
            <button id="zoom-reset" title="Reset view">&#8962;</button>
        </div>
        <div class="hint"><b>Hover</b> to highlight &middot; <b>Scroll/buttons</b> to zoom &middot; <b>Click</b> to open &middot; <b>Click legend</b> to filter</div>
    </div>

<script>
// Wait for D3 to load (deferred script)
function init() {{
if (typeof d3 === "undefined") {{ setTimeout(init, 50); return; }}

const data = {hierarchy_json};
const W = {width}, H = {height};
const dpr = window.devicePixelRatio || 1;
const canvas = document.getElementById("moabb-canvas");
const ctx = canvas.getContext("2d");

// HiDPI setup
canvas.width = W * dpr;
canvas.height = H * dpr;
canvas.style.width = W + "px";
canvas.style.height = H + "px";
ctx.scale(dpr, dpr);

const tooltip = document.getElementById("tooltip");

// ---- Compute totals ----
const totalDatasets = data.children.reduce((s, m) => s + m.children.length, 0);
const totalSubjects = data.children.reduce((s, m) =>
    s + m.children.reduce((a, ds) => a + ds.n_subjects, 0), 0);
document.getElementById("total-datasets").innerHTML =
    "<b>" + totalDatasets + "</b> datasets &middot; <b>" + totalSubjects.toLocaleString() + "</b> subjects";

// ---- Pack each modality with D3 ----
const modalityPacks = [];
data.children.forEach(modality => {{
    const h = d3.hierarchy({{ name: modality.name, children: modality.children }})
        .sum(d => d.value || 0).sort((a, b) => b.value - a.value);
    const size = Math.sqrt(h.value) * 2.8 + 80;
    const pack = d3.pack().size([size, size])
        .padding(d => d.depth === 0 ? 15 : d.depth === 1 ? 6 : 1.5);
    modalityPacks.push({{ modality: modality, packed: pack(h), size: size, radius: size / 2 }});
}});

// ---- Smart row-based grid layout ----
// Sort largest first, then wrap rows based on total content width
modalityPacks.sort((a, b) => b.size - a.size);
const gap = 35;
// Compute total width to determine a sensible row-break threshold
const totalW = modalityPacks.reduce((s, mp) => s + mp.size + gap, 0);
const rowBreak = Math.max(totalW * 0.55, modalityPacks[0].size + gap * 2);
let curX = gap, curY = gap, rowH = 0;
modalityPacks.forEach(mp => {{
    if (curX + mp.size > rowBreak && curX > gap) {{
        curX = gap;
        curY += rowH + gap + 30;
        rowH = 0;
    }}
    mp.ox = curX;
    mp.oy = curY + 30;
    curX += mp.size + gap;
    rowH = Math.max(rowH, mp.size);
}});

// ---- Flatten all circles into arrays for canvas rendering ----
const allCircles = [];   // {{x, y, r, color, alpha, dsIdx}}
const allDatasets = [];   // {{name, title, modality, color, ...metadata}}
const dsNameToIdx = {{}};
const labelCircles = [];  // {{x, y, r, label, modality}}

const hiddenModalities = new Set();
let highlightedDs = -1;
let searchMatches = null;  // null = no search active

modalityPacks.forEach(mp => {{
    const ox = mp.ox, oy = mp.oy;
    const mod = mp.modality;

    // Modality background region
    allCircles.push({{ x: ox + mp.size / 2, y: oy + mp.size / 2, r: mp.radius + 15,
        color: mod.bgColor, alpha: 1, dsIdx: -1, isBg: true,
        strokeColor: mod.color, strokeDash: true, modality: mod.name }});

    // Label position for modality title — large, bold, prominent
    labelCircles.push({{ x: ox + mp.size / 2, y: oy - 18, r: 0,
        label: mod.name, color: mod.color, isModality: true, modality: mod.name,
        count: mod.children.length }});

    const datasetNodes = mp.packed.descendants().filter(d => d.depth === 1);
    const subjectNodes = mp.packed.descendants().filter(d => d.depth === 2);

    // Build dataset index
    datasetNodes.forEach(node => {{
        const ds = node.data;
        if (dsNameToIdx[ds.name] === undefined) {{
            dsNameToIdx[ds.name] = allDatasets.length;
            allDatasets.push({{ name: ds.name, title: ds.title || "", modality: mod.name,
                color: ds.color, alpha: ds.alpha, url: ds.url,
                n_subjects: ds.n_subjects, n_sessions: ds.n_sessions,
                n_records: ds.n_records, nchans: ds.nchans, sfreq: ds.sfreq,
                size: ds.size, records_per_subject: ds.records_per_subject }});
        }}
        // Only label top-N datasets per modality (marked by Python)
        if (ds.showLabel && node.r > 12) {{
            const lbl = ds.name.toUpperCase();
            labelCircles.push({{ x: ox + node.x, y: oy + node.y + node.r + 11, r: node.r,
                label: lbl, color: "#374151", isModality: false, modality: mod.name,
                fontSize: Math.max(11, Math.min(14, node.r / 2.5)) }});
        }}
    }});

    subjectNodes.forEach(node => {{
        const ds = node.parent.data;
        const idx = dsNameToIdx[ds.name];
        allCircles.push({{ x: ox + node.x, y: oy + node.y, r: Math.max(node.r, 2),
            color: ds.color, alpha: ds.alpha, dsIdx: idx, isBg: false,
            modality: mod.name }});
    }});
}});

// ---- Quadtree for fast hover hit-testing ----
const quadtree = d3.quadtree()
    .x(d => d.x).y(d => d.y)
    .addAll(allCircles.filter(c => !c.isBg));

function findCircle(mx, my) {{
    let found = null;
    let bestDist = Infinity;
    quadtree.visit((node, x0, y0, x1, y1) => {{
        if (node.length) return false; // internal node, keep searching
        let d = node;
        do {{
            const dx = mx - d.data.x, dy = my - d.data.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < d.data.r + 2 && dist < bestDist) {{
                bestDist = dist;
                found = d.data;
            }}
        }} while ((d = d.next));
        return false;
    }});
    return found;
}}

// ---- Transform state (pan/zoom) ----
let tx = 0, ty = 0, scale = 1;

function computeInitialTransform() {{
    // Find bounding box of all content
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    allCircles.forEach(c => {{
        minX = Math.min(minX, c.x - c.r);
        minY = Math.min(minY, c.y - c.r - 20);
        maxX = Math.max(maxX, c.x + c.r);
        maxY = Math.max(maxY, c.y + c.r);
    }});
    const bw = maxX - minX, bh = maxY - minY;
    scale = Math.min(0.82 * W / bw, 0.82 * H / bh, 1);
    tx = W / 2 - scale * (minX + bw / 2);
    ty = H / 2 - scale * (minY + bh / 2);
}}
computeInitialTransform();
const initTx = tx, initTy = ty, initScale = scale;

// ---- Helper: parse rgba/hex color ----
function parseColor(c) {{
    if (c.startsWith("rgba")) {{
        const m = c.match(/rgba\\(([^)]+)\\)/);
        if (m) {{ const p = m[1].split(","); return {{ r: +p[0], g: +p[1], b: +p[2], a: +p[3] }}; }}
    }}
    if (c.startsWith("#") && c.length === 7) {{
        return {{ r: parseInt(c.slice(1,3),16), g: parseInt(c.slice(3,5),16),
                 b: parseInt(c.slice(5,7),16), a: 1 }};
    }}
    return {{ r: 148, g: 163, b: 184, a: 1 }};
}}

// ---- Render ----
function draw() {{
    ctx.save();
    ctx.clearRect(0, 0, W, H);
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    // Draw background regions first
    allCircles.forEach(c => {{
        if (!c.isBg) return;
        if (hiddenModalities.has(c.modality)) return;
        const col = parseColor(c.color);
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(" + col.r + "," + col.g + "," + col.b + "," + col.a + ")";
        ctx.fill();
        if (c.strokeColor) {{
            ctx.strokeStyle = c.strokeColor;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }}
    }});

    // Draw subject circles
    allCircles.forEach(c => {{
        if (c.isBg) return;
        if (hiddenModalities.has(c.modality)) return;
        const ds = allDatasets[c.dsIdx];

        let alpha = c.alpha;
        // Dimming logic
        if (highlightedDs >= 0 && c.dsIdx !== highlightedDs) alpha *= 0.15;
        if (searchMatches !== null && !searchMatches.has(c.dsIdx)) alpha *= 0.1;

        const col = parseColor(c.color);
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(" + col.r + "," + col.g + "," + col.b + "," + alpha + ")";
        ctx.fill();

        // Highlight stroke for hovered dataset
        if (highlightedDs >= 0 && c.dsIdx === highlightedDs) {{
            ctx.strokeStyle = "rgba(0,0,0,0.5)";
            ctx.lineWidth = 1.5 / scale;
            ctx.stroke();
        }} else if (searchMatches !== null && searchMatches.has(c.dsIdx)) {{
            ctx.strokeStyle = "#fbbf24";
            ctx.lineWidth = 2 / scale;
            ctx.stroke();
        }} else {{
            ctx.strokeStyle = "rgba(255,255,255,0.6)";
            ctx.lineWidth = 0.5 / scale;
            ctx.stroke();
        }}
    }});

    // Draw labels
    labelCircles.forEach(lc => {{
        if (hiddenModalities.has(lc.modality)) return;
        let alpha = 1;
        if (highlightedDs >= 0) alpha = 0.3;
        if (searchMatches !== null) alpha = 0.3;

        ctx.save();
        ctx.globalAlpha = alpha;
        if (lc.isModality) {{
            // Large bold modality title — primary visual anchor
            ctx.font = "800 24px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            // White halo for contrast
            ctx.strokeStyle = "rgba(255,255,255,0.9)";
            ctx.lineWidth = 5;
            ctx.lineJoin = "round";
            const titleText = lc.label + " (" + (lc.count || "") + ")";
            ctx.strokeText(titleText, lc.x, lc.y);
            ctx.fillStyle = lc.color;
            ctx.fillText(titleText, lc.x, lc.y);
        }} else {{
            const fs = lc.fontSize || 12;
            ctx.font = "600 " + fs + "px Inter, system-ui, sans-serif";
            ctx.fillStyle = lc.color;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            // White halo for readability
            ctx.strokeStyle = "rgba(255,255,255,0.85)";
            ctx.lineWidth = 3;
            ctx.lineJoin = "round";
            ctx.strokeText(lc.label, lc.x, lc.y);
            ctx.fillText(lc.label, lc.x, lc.y);
        }}
        ctx.restore();
    }});

    ctx.restore();
}}

draw();

// ---- Mouse interaction ----
function canvasToWorld(ex, ey) {{
    const rect = canvas.getBoundingClientRect();
    const cx = ex - rect.left, cy = ey - rect.top;
    return [(cx - tx) / scale, (cy - ty) / scale];
}}

let lastHoverDs = -1;
canvas.addEventListener("mousemove", function(e) {{
    const [wx, wy] = canvasToWorld(e.clientX, e.clientY);
    const hit = findCircle(wx, wy);
    const dsIdx = hit ? hit.dsIdx : -1;

    if (dsIdx !== lastHoverDs) {{
        lastHoverDs = dsIdx;
        highlightedDs = dsIdx;
        draw();

        if (dsIdx >= 0) {{
            const ds = allDatasets[dsIdx];
            canvas.style.cursor = "pointer";
            tooltip.style.display = "block";
            tooltip.style.borderColor = ds.color;
            tooltip.innerHTML =
                '<div class="tooltip-title">' + ds.name + '</div>' +
                '<div class="tooltip-subtitle">' + ds.title + '</div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Subjects</span><span class="tooltip-value">' + ds.n_subjects.toLocaleString() + '</span></div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Sessions</span><span class="tooltip-value">' + ds.n_sessions.toLocaleString() + '</span></div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Records</span><span class="tooltip-value">' + ds.n_records.toLocaleString() + '</span></div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Channels</span><span class="tooltip-value">' + ds.nchans + '</span></div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Sampling</span><span class="tooltip-value">' + ds.sfreq + ' Hz</span></div>' +
                '<div class="tooltip-row"><span class="tooltip-label">Size</span><span class="tooltip-value">' + ds.size + '</span></div>' +
                '<div class="tooltip-hint">Click to open dataset page &rarr;</div>';
        }} else {{
            canvas.style.cursor = "grab";
            tooltip.style.display = "none";
        }}
    }}

    if (tooltip.style.display === "block") {{
        tooltip.style.left = (e.clientX + 15) + "px";
        tooltip.style.top = (e.clientY - 10) + "px";
    }}
}});

canvas.addEventListener("mouseleave", function() {{
    highlightedDs = -1;
    lastHoverDs = -1;
    tooltip.style.display = "none";
    canvas.style.cursor = "grab";
    draw();
}});

canvas.addEventListener("click", function(e) {{
    const [wx, wy] = canvasToWorld(e.clientX, e.clientY);
    const hit = findCircle(wx, wy);
    if (hit && hit.dsIdx >= 0) {{
        const url = allDatasets[hit.dsIdx].url;
        if (url) window.open(url, "_blank", "noopener");
    }}
}});

// ---- Zoom with wheel ----
canvas.addEventListener("wheel", function(e) {{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.max(0.3, Math.min(4, scale * factor));
    tx = cx - (cx - tx) * (newScale / scale);
    ty = cy - (cy - ty) * (newScale / scale);
    scale = newScale;
    draw();
}}, {{ passive: false }});

// ---- Pan with drag ----
let dragging = false, dragX = 0, dragY = 0;
canvas.addEventListener("mousedown", function(e) {{
    if (e.button === 0) {{ dragging = true; dragX = e.clientX; dragY = e.clientY; }}
}});
window.addEventListener("mousemove", function(e) {{
    if (!dragging) return;
    tx += e.clientX - dragX;
    ty += e.clientY - dragY;
    dragX = e.clientX;
    dragY = e.clientY;
    draw();
}});
window.addEventListener("mouseup", function() {{ dragging = false; }});

// ---- Zoom controls ----
document.getElementById("zoom-in").addEventListener("click", function() {{
    const f = 1.4;
    tx = W/2 - (W/2 - tx) * f;
    ty = H/2 - (H/2 - ty) * f;
    scale *= f;
    draw();
}});
document.getElementById("zoom-out").addEventListener("click", function() {{
    const f = 1 / 1.4;
    tx = W/2 - (W/2 - tx) * f;
    ty = H/2 - (H/2 - ty) * f;
    scale *= f;
    draw();
}});
document.getElementById("zoom-reset").addEventListener("click", function() {{
    tx = initTx; ty = initTy; scale = initScale;
    draw();
}});

// ---- Modality legend with filtering ----
const legendContainer = document.getElementById("modality-legend");
const legendTitle = document.createElement("div");
legendTitle.className = "modality-legend-title";
legendTitle.textContent = "Recording Modality";
legendContainer.appendChild(legendTitle);

data.children.forEach(m => {{
    const item = document.createElement("div");
    item.className = "modality-legend-item";
    const swatch = document.createElement("div");
    swatch.className = "modality-swatch";
    swatch.style.background = m.color;
    item.appendChild(swatch);
    const span = document.createElement("span");
    span.textContent = m.name + " (" + m.children.length + ")";
    item.appendChild(span);

    item.addEventListener("click", function() {{
        if (hiddenModalities.has(m.name)) {{
            hiddenModalities.delete(m.name);
            item.classList.remove("hidden");
        }} else {{
            hiddenModalities.add(m.name);
            item.classList.add("hidden");
        }}
        draw();
    }});
    legendContainer.appendChild(item);
}});

// ---- Search ----
const searchInput = document.getElementById("search-input");
const searchClear = document.getElementById("search-clear");
const searchResultsEl = document.getElementById("search-results");

function performSearch(query) {{
    const q = query.toLowerCase().trim();
    searchClear.style.display = q ? "block" : "none";

    if (!q) {{
        searchMatches = null;
        searchResultsEl.textContent = "";
        draw();
        return;
    }}

    searchMatches = new Set();
    allDatasets.forEach((ds, i) => {{
        if (ds.name.toLowerCase().includes(q)) searchMatches.add(i);
    }});
    searchResultsEl.textContent = searchMatches.size > 0 ? searchMatches.size + " found" : "No matches";
    draw();
}}

searchInput.addEventListener("input", function(e) {{ performSearch(e.target.value); }});
searchClear.addEventListener("click", function() {{
    searchInput.value = "";
    performSearch("");
    searchInput.focus();
}});

}} // end init
document.addEventListener("DOMContentLoaded", init);
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
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    args = parser.parse_args()

    df = pd.read_csv(args.source, index_col=False, header=0, skipinitialspace=True)
    output_path = generate_moabb_bubble(
        df, args.output, scale=args.scale, width=args.width, height=args.height
    )
    print(f"MOABB bubble chart saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
