"""MOABB-style circle-packing bubble visualization for EEGDash.

This module creates a MOABB-inspired visualization where:
- Each **dataset** is a cluster of circles
- Each **circle** represents one subject
- Circle **size** = log(records per subject) - data volume per subject
- Circle **opacity** = number of sessions (fewer sessions = more opaque)
- Circle **color** = category (recording modality by default)
- Datasets are packed into a single cluster using force-directed collision detection

Features:
- Hover-based interactivity: Dataset clusters highlight on hover, others dim
- Rich tooltips: Subjects, sessions, records, channels, sampling rate, size
- Click to open: Each dataset links to its detail page
- Visual polish: Subtle glow effects and smooth transitions

Ported from MOABB (moabb/analysis/plotting.py lines 668-1004)
and (moabb/datasets/utils.py lines 383-491).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:  # Allow execution as a script or module
    from .colours import MODALITY_COLOR_MAP
    from .utils import (
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_recording_modality,
        safe_int,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import MODALITY_COLOR_MAP  # type: ignore
    from utils import (  # type: ignore
        build_and_export_html,
        get_dataset_url,
        human_readable_size,
        primary_recording_modality,
        safe_int,
    )

__all__ = ["generate_moabb_bubble"]

# Maximum bubbles per dataset to prevent performance issues with very large datasets
MAX_BUBBLES_PER_DATASET = 200


# =============================================================================
# Hexagonal grid and bubble coordinate utilities (from MOABB plotting.py)
# =============================================================================


def _get_hexa_grid(n: int, diameter: float, center: tuple[float, float]) -> tuple:
    """Generate hexagonal grid positions for n bubbles.

    Args:
        n: Number of positions to generate (will generate more than n)
        diameter: Diameter of each bubble
        center: (x, y) center of the grid

    Returns:
        Tuple of (x_coords, y_coords) arrays

    """
    # Use a fixed seed for reproducible layouts
    rng = np.random.default_rng(42)
    grid_size = int(np.ceil(np.sqrt(n))) + 1

    x = np.arange(grid_size) - grid_size // 2 + rng.random() * 0.1
    y = np.arange(grid_size) - grid_size // 2 + rng.random() * 0.1
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    # Hexagonal offset: shift alternating rows by 0.5
    return (
        np.concatenate([x, x + 0.5]) * diameter + center[0],
        np.concatenate([y, y + 0.5]) * diameter * np.sqrt(3) + center[1],
    )


def _get_bubble_coordinates(
    n: int, diameter: float, center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Get n bubble coordinates arranged in a hexagonal pattern, sorted by distance from center.

    Args:
        n: Number of bubbles
        diameter: Diameter of each bubble
        center: (x, y) center of the arrangement

    Returns:
        Tuple of (x_coords, y_coords) arrays of length n

    """
    x, y = _get_hexa_grid(n, diameter, center)
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    sort_idx = dist.argsort()
    x = x[sort_idx]
    y = y[sort_idx]
    return x[:n], y[:n]


# =============================================================================
# BubbleChart class for force-directed collision detection (from MOABB utils.py)
# =============================================================================


class _BubbleChart:
    """Force-directed bubble packing for dataset clusters.

    From https://matplotlib.org/stable/gallery/misc/packed_bubbles.html
    Ported from MOABB moabb/datasets/utils.py.
    """

    def __init__(self, area: np.ndarray, bubble_spacing: float = 0.0):
        """Setup for bubble collapse.

        Args:
            area: Array of bubble areas
            bubble_spacing: Minimal spacing between bubbles after collapsing

        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # Calculate initial grid layout for bubbles
        length = int(np.ceil(np.sqrt(len(self.bubbles))))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[: len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[: len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass weighted by bubble area."""
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> np.ndarray:
        """Calculate distance between bubble center and other bubble centers."""
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble: np.ndarray, bubbles: np.ndarray) -> np.ndarray:
        """Calculate distance between bubble outlines (negative = collision)."""
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble: np.ndarray, bubbles: np.ndarray) -> int:
        """Count how many bubbles this bubble collides with."""
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble: np.ndarray, bubbles: np.ndarray) -> np.ndarray:
        """Return indices of bubbles this bubble collides with."""
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations: int = 50) -> None:
        """Move bubbles toward the center of mass using force-directed algorithm.

        Args:
            n_iterations: Number of iterations to run

        """
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)

                # Direction vector from bubble to center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # Normalize direction vector
                norm = np.sqrt(dir_vec.dot(dir_vec))
                if norm == 0:
                    continue
                dir_vec = dir_vec / norm

                # Calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # Check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # Try to move around a colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # Direction to colliding bubble
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        norm = np.sqrt(dir_vec.dot(dir_vec))
                        if norm == 0:
                            continue
                        dir_vec = dir_vec / norm

                        # Calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])

                        # Test which direction to go
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def get_centers(self) -> np.ndarray:
        """Return array of bubble center coordinates."""
        return self.bubbles[:, :2]

    def get_radii(self) -> np.ndarray:
        """Return array of bubble radii."""
        return self.bubbles[:, 2]


# =============================================================================
# Bubble size and alpha calculations
# =============================================================================


def _get_bubble_diameter(
    records_per_subject: float,
    scale: float = 0.5,
) -> float:
    """Calculate bubble diameter from data volume per subject.

    Args:
        records_per_subject: Number of records per subject (proxy for trials)
        scale: Scaling factor

    Returns:
        Bubble diameter

    """
    size = max(1.0, float(records_per_subject))
    return np.log(size + 1.0) * scale


def _get_alpha(n_sessions: int, alphas: list[float] | None = None) -> float:
    """Calculate bubble alpha based on number of sessions.

    Args:
        n_sessions: Number of sessions
        alphas: List of alpha values for 1, 2, 3, 4, 5+ sessions

    Returns:
        Alpha value (0.2 to 0.8)

    """
    if alphas is None:
        alphas = [0.8, 0.65, 0.5, 0.35, 0.2]
    idx = min(max(0, n_sessions - 1), len(alphas) - 1)
    return alphas[idx]


def _get_dataset_area(
    n_subjects: int,
    records_per_subject: float,
    scale: float = 0.5,
    gap: float = 0.1,
) -> float:
    """Calculate total area needed for a dataset's bubbles.

    Args:
        n_subjects: Number of subjects (circles)
        records_per_subject: Number of records per subject
        scale: Scaling factor
        gap: Gap between bubbles

    Returns:
        Total area for the dataset cluster

    """
    diameter = _get_bubble_diameter(records_per_subject, scale) + gap
    # Hexagonal packing area approximation
    return n_subjects * 3 * 3**0.5 / 8 * diameter**2


# =============================================================================
# Dataset bubble data generation
# =============================================================================


def _dataset_bubble_data(
    dataset_name: str,
    n_subjects: int,
    n_sessions: int,
    records_per_subject: float,
    category: str,
    center: tuple[float, float],
    color: str,
    dataset_idx: int,
    scale: float = 0.5,
    gap: float = 0.1,
    alphas: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate bubble coordinates for one dataset.

    Args:
        dataset_name: Name of the dataset
        n_subjects: Number of subjects (circles to draw)
        n_sessions: Number of sessions (affects alpha)
        records_per_subject: Number of records per subject (affects size)
        category: Category label (e.g., modality)
        center: (x, y) center of the dataset cluster
        color: Hex color for the bubbles
        dataset_idx: Index of this dataset for JS cross-referencing
        scale: Scaling factor for bubble size
        gap: Gap between bubbles
        alphas: List of alpha values by session count

    Returns:
        List of dicts with {x, y, radius, color, alpha, dataset_name, category, dataset_idx}

    """
    if n_subjects < 1:
        return []

    # Limit bubbles per dataset for performance
    effective_subjects = min(n_subjects, MAX_BUBBLES_PER_DATASET)

    diameter = _get_bubble_diameter(records_per_subject, scale)
    alpha = _get_alpha(n_sessions, alphas)

    x, y = _get_bubble_coordinates(effective_subjects, diameter + gap, center)

    bubbles = []
    for xi, yi in zip(x, y):
        bubbles.append(
            {
                "x": float(xi),
                "y": float(yi),
                "radius": diameter / 2,
                "color": color,
                "alpha": alpha,
                "dataset_name": dataset_name,
                "category": category,
                "n_subjects": n_subjects,
                "n_sessions": n_sessions,
                "records_per_subject": records_per_subject,
                "dataset_idx": dataset_idx,
            }
        )
    return bubbles


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


# =============================================================================
# Main visualization function
# =============================================================================


def generate_moabb_bubble(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    color_by: Literal["modality", "type", "pathology"] = "modality",
    scale: float = 0.5,
    gap: float = 0.06,
    meta_gap: float = 0.8,
    width: int | None = None,
    height: int = 900,
) -> Path:
    """Generate MOABB-style circle-packing bubble plot for all datasets.

    Each dataset becomes a cluster of circles (one per subject), arranged
    in a hexagonal pattern. Datasets are packed into a single cluster, with
    color encoding the selected category.

    Args:
        df: DataFrame with dataset information
        out_html: Output HTML file path
        color_by: Column to use for coloring ("modality", "type", "pathology")
        scale: Scaling factor for bubble sizes
        gap: Gap between bubbles within clusters
        meta_gap: Gap between dataset clusters in the packed layout
        width: Chart width in pixels (defaults to height * 1.4)
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

    # Derive records per subject for sizing (avoid over-scaling large datasets)
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

    all_bubbles = []
    dataset_centers = []

    areas = []
    dataset_info = []
    for idx, (_, row) in enumerate(data.iterrows()):
        n_subjects = int(row["n_subjects"])
        n_sessions = int(row["n_sessions"])
        n_records = int(row["n_records"])
        records_per_subject = float(row["records_per_subject"])
        category = row["category"]
        area = _get_dataset_area(n_subjects, records_per_subject, scale, gap)
        areas.append(area)

        # Extract additional metadata
        n_tasks = safe_int(row.get("n_tasks", 0), 0)
        nchans = row.get("nchans_median")
        sfreq = row.get("sfreq_median")
        size_bytes = safe_int(row.get("size_bytes_clean", 0), 0)

        dataset_info.append(
            {
                "name": row["dataset"],
                "n_subjects": n_subjects,
                "n_sessions": n_sessions,
                "n_records": n_records,
                "records_per_subject": records_per_subject,
                "category": category,
                "title": row.get("dataset_title", ""),
                "n_tasks": n_tasks,
                "nchans": nchans,
                "sfreq": sfreq,
                "size_bytes": size_bytes,
                "idx": idx,
            }
        )

    bubble_chart = _BubbleChart(np.array(areas), bubble_spacing=meta_gap)
    bubble_chart.collapse(n_iterations=100)
    centers = bubble_chart.get_centers()
    centers = centers - centers.mean(axis=0)

    for i, info in enumerate(dataset_info):
        cx, cy = centers[i]
        area = areas[i]
        cluster_radius = float(np.sqrt(area / np.pi))
        color = MODALITY_COLOR_MAP.get(info["category"], "#94a3b8")
        bubbles = _dataset_bubble_data(
            dataset_name=info["name"],
            n_subjects=info["n_subjects"],
            n_sessions=info["n_sessions"],
            records_per_subject=info["records_per_subject"],
            category=info["category"],
            center=(cx, cy),
            color=color,
            dataset_idx=i,
            scale=scale,
            gap=gap,
        )
        all_bubbles.extend(bubbles)

        # Format metadata for hover display
        nchans_str = _format_int(info["nchans"]) if info["nchans"] else "—"
        sfreq_str = _format_int(info["sfreq"]) if info["sfreq"] else "—"
        size_str = (
            human_readable_size(info["size_bytes"]) if info["size_bytes"] else "—"
        )

        dataset_centers.append(
            {
                "name": info["name"],
                "x": float(cx),
                "y": float(cy),
                "category": info["category"],
                "n_subjects": info["n_subjects"],
                "n_sessions": info["n_sessions"],
                "n_records": info["n_records"],
                "records_per_subject": info["records_per_subject"],
                "area": area,
                "radius": cluster_radius,
                "url": get_dataset_url(info["name"]),
                "title": info["title"] or "",
                "n_tasks": info["n_tasks"],
                "nchans_str": nchans_str,
                "sfreq_str": sfreq_str,
                "size_str": size_str,
                "idx": i,
            }
        )

    layout_margin = dict(l=20, r=20, t=60, b=40)  # Extra bottom margin for hint text
    plot_width = width if width is not None else int(height * 1.4)
    plot_height = max(height - layout_margin["t"] - layout_margin["b"], 1)
    plot_width_inner = max(plot_width - layout_margin["l"] - layout_margin["r"], 1)
    target_ratio = plot_width_inner / plot_height

    x_vals = np.array([b["x"] for b in all_bubbles])
    y_vals = np.array([b["y"] for b in all_bubbles])
    r_vals = np.array([b["radius"] for b in all_bubbles])
    x_min = np.min(x_vals - r_vals)
    x_max = np.max(x_vals + r_vals)
    y_min = np.min(y_vals - r_vals)
    y_max = np.max(y_vals + r_vals)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    current_ratio = x_span / y_span

    x_scale = target_ratio / current_ratio
    if x_scale > 1.02:
        center_map = {d["name"]: d["x"] for d in dataset_centers}
        for d in dataset_centers:
            d["x"] *= x_scale
        for b in all_bubbles:
            cx = center_map.get(b["dataset_name"])
            if cx is None:
                continue
            b["x"] = (b["x"] - cx) + (cx * x_scale)

        x_vals = np.array([b["x"] for b in all_bubbles])
        x_min = np.min(x_vals - r_vals)
        x_max = np.max(x_vals + r_vals)
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_pad_right = x_span * 0.06
    x_range = [x_center - x_span / 2, x_center + x_span / 2 + x_pad_right]
    y_range = [y_center - y_span / 2, y_center + y_span / 2]
    size_scale = plot_height / y_span

    # Create figure
    fig = go.Figure()

    # Build a mapping of dataset name -> index for JS highlight functionality
    dataset_name_to_idx = {info["name"]: info["idx"] for info in dataset_info}

    # Use WebGL scatter traces instead of SVG shapes for performance
    # Group bubbles by dataset for hover highlighting
    bubbles_by_dataset = {}
    for bubble in all_bubbles:
        ds_name = bubble["dataset_name"]
        if ds_name not in bubbles_by_dataset:
            bubbles_by_dataset[ds_name] = []
        bubbles_by_dataset[ds_name].append(bubble)

    # Track trace indices for each dataset (for JS hover highlighting)
    dataset_trace_map = {}  # dataset_name -> list of trace indices
    trace_idx = 0

    # Add bubble traces per dataset (one trace per dataset for highlighting)
    # Sort by category first for legend ordering, then by dataset name
    datasets_sorted = sorted(
        bubbles_by_dataset.keys(),
        key=lambda d: (
            0
            if bubbles_by_dataset[d][0]["category"] == "EEG"
            else 1
            if bubbles_by_dataset[d][0]["category"] == "Other"
            else 2
            if bubbles_by_dataset[d][0]["category"] == "MEG"
            else 3
            if bubbles_by_dataset[d][0]["category"] == "iEEG"
            else 4
            if bubbles_by_dataset[d][0]["category"] == "fNIRS"
            else 5,
            d,
        ),
    )

    # Track which categories we've added to legend
    legend_added = set()

    for ds_name in datasets_sorted:
        ds_bubbles = bubbles_by_dataset[ds_name]
        cat = ds_bubbles[0]["category"]
        color = MODALITY_COLOR_MAP.get(cat, "#94a3b8")
        ds_idx = dataset_name_to_idx.get(ds_name, 0)

        # Convert radius to marker size (Plotly uses diameter in pixels)
        sizes = [max(b["radius"] * 2 * size_scale, 4) for b in ds_bubbles]
        alphas = [b["alpha"] for b in ds_bubbles]

        # Only show in legend once per category
        show_legend = cat not in legend_added and cat != "Other"
        if show_legend:
            legend_added.add(cat)

        fig.add_trace(
            go.Scattergl(
                x=[b["x"] for b in ds_bubbles],
                y=[b["y"] for b in ds_bubbles],
                mode="markers",
                name=cat,
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=alphas,
                    line=dict(width=0.5, color="rgba(0,0,0,0.1)"),
                ),
                customdata=[[ds_name, ds_idx]] * len(ds_bubbles),
                hoverinfo="skip",  # Hover handled by invisible center markers
                showlegend=show_legend,
                legendgroup=cat,
            )
        )

        # Track this trace for the dataset
        if ds_name not in dataset_trace_map:
            dataset_trace_map[ds_name] = []
        dataset_trace_map[ds_name].append(trace_idx)
        trace_idx += 1

    # Add hover traces for interactivity (radius-scaled invisible markers at dataset centers)
    # These provide the hover targets and rich tooltips
    categories = sorted(set(d["category"] for d in dataset_centers))
    for category in categories:
        cat_data = [d for d in dataset_centers if d["category"] == category]
        color = MODALITY_COLOR_MAP.get(category, "#94a3b8")

        # Scale hover marker size to match cluster extent (min 35px for small clusters)
        hover_sizes = [max(35, d["radius"] * 2 * size_scale * 0.9) for d in cat_data]

        fig.add_trace(
            go.Scatter(
                x=[d["x"] for d in cat_data],
                y=[d["y"] for d in cat_data],
                mode="markers",
                name=f"{category}_hover",
                marker=dict(
                    size=hover_sizes,
                    color=color,
                    opacity=0.01,  # Nearly invisible but captures hover
                ),
                # Rich customdata for hover and JS highlighting
                # [0]=name, [1]=subjects, [2]=sessions, [3]=records, [4]=records/subj,
                # [5]=url, [6]=idx, [7]=title, [8]=category, [9]=tasks,
                # [10]=nchans, [11]=sfreq, [12]=size
                customdata=[
                    [
                        d["name"],
                        d["n_subjects"],
                        d["n_sessions"],
                        d["n_records"],
                        d["records_per_subject"],
                        d["url"],
                        d["idx"],
                        d["title"],
                        d["category"],
                        d["n_tasks"],
                        d["nchans_str"],
                        d["sfreq_str"],
                        d["size_str"],
                    ]
                    for d in cat_data
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "<span style='color:#6b7280;font-size:0.9em'>%{customdata[7]}</span><br>"
                    "<br>"
                    "📊 <b>Subjects</b>: %{customdata[1]:,}<br>"
                    "🔄 <b>Sessions</b>: %{customdata[2]:,}<br>"
                    "📁 <b>Records</b>: %{customdata[3]:,}<br>"
                    "📈 <b>Records/Subject</b>: %{customdata[4]:.1f}<br>"
                    "<br>"
                    "📡 <b>Channels</b>: %{customdata[10]}<br>"
                    "⚡ <b>Sampling</b>: %{customdata[11]} Hz<br>"
                    "💾 <b>Size</b>: %{customdata[12]}<br>"
                    "🎯 <b>Modality</b>: %{customdata[8]}<br>"
                    "<br>"
                    "<i style='color:#3b82f6'>Click to open dataset page →</i>"
                    "<extra></extra>"
                ),
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor=color,
                    font=dict(size=12, color="#1f2937"),
                ),
            )
        )
        trace_idx += 1

    # Add dataset labels for the largest clusters with collision detection
    label_count = min(60, max(30, int(len(dataset_centers) * 0.08)))
    label_candidates = sorted(dataset_centers, key=lambda d: d["area"], reverse=True)[
        :label_count
    ]

    # Simple collision detection: track placed label positions
    placed_labels: list[tuple[float, float, float, float]] = []  # (x, y, width, height)
    label_char_width = 0.08 * (
        x_span / 10
    )  # Approximate character width in data coords
    label_height = 0.12 * (y_span / 10)  # Approximate label height in data coords

    for d in label_candidates:
        label_text = _short_label(d["name"])
        label_width = len(label_text) * label_char_width
        label_x = d["x"]
        label_y = d["y"] + max(0.4, d["radius"] * 0.15)

        # Check for collision with existing labels
        has_collision = False
        for px, py, pw, ph in placed_labels:
            # Check if bounding boxes overlap (with some padding)
            if (
                abs(label_x - px) < (label_width + pw) / 2 + label_char_width
                and abs(label_y - py) < (label_height + ph) / 2 + label_height * 0.3
            ):
                has_collision = True
                break

        if has_collision:
            continue  # Skip this label to avoid overlap

        # Place the label
        placed_labels.append((label_x, label_y, label_width, label_height))
        label_color = MODALITY_COLOR_MAP.get(d["category"], "#94a3b8")
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=label_text,
            showarrow=False,
            font=dict(size=8, color="#1f2937"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=label_color,
            borderwidth=1,
            borderpad=2,
        )

    fig.update_layout(
        height=height,
        width=plot_width,
        margin=layout_margin,
        template="plotly_white",
        legend=dict(
            title="Recording Modality",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            range=x_range,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=y_range,
        ),
        autosize=False,
        showlegend=True,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font=dict(
                family="Inter, system-ui, -apple-system, sans-serif",
                size=12,
                color="#1f2937",
            ),
            align="left",
        ),
        hovermode="closest",
    )

    # Add legend annotation
    _add_moabb_legend(fig, scale=scale)

    # Add interactive hint at the bottom
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.02,
        text="<b>Hover</b> to highlight dataset · <b>Scroll</b> to zoom · <b>Drag</b> to pan · <b>Click</b> to open",
        showarrow=False,
        font=dict(size=11, color="#6b7280"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4,
        xanchor="center",
        yanchor="top",
    )

    # Build dataset trace mapping JSON for JavaScript
    dataset_trace_json = json.dumps(dataset_trace_map)
    n_bubble_traces = trace_idx - len(categories)  # Exclude hover traces

    extra_style = f""".dataset-loading {{
    display: flex;
    justify-content: center;
    align-items: center;
    height: {height}px;
    font-family: Inter, system-ui, sans-serif;
    color: #6b7280;
}}
#moabb-bubble .hoverlayer .hovertext {{
    transition: opacity 0.15s ease;
}}"""

    pre_html = '<div class="dataset-loading" id="moabb-bubble-loading">Loading MOABB bubble plot...</div>\n'

    # JavaScript with hover highlighting functionality
    extra_html = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    const loading = document.getElementById('moabb-bubble-loading');
    const plot = document.getElementById('moabb-bubble');

    // Dataset name -> trace indices mapping (for highlight/dim)
    const datasetTraceMap = {dataset_trace_json};
    const nBubbleTraces = {n_bubble_traces};

    // Store original opacity and size for each trace
    let originalStyles = null;
    let isHighlighting = false;
    let hoverTimeout = null;

    function showPlot() {{
        if (loading) loading.style.display = 'none';
        if (plot) {{
            plot.style.display = 'block';
            if (typeof Plotly !== 'undefined') Plotly.Plots.resize(plot);
        }}
    }}

    function captureOriginalStyles() {{
        if (!plot || !plot.data || originalStyles) return;
        originalStyles = plot.data.slice(0, nBubbleTraces).map(function(trace) {{
            return {{
                opacity: trace.marker && trace.marker.opacity
                    ? (Array.isArray(trace.marker.opacity)
                        ? trace.marker.opacity.slice()
                        : trace.marker.opacity)
                    : 0.7,
                size: trace.marker && trace.marker.size
                    ? (Array.isArray(trace.marker.size)
                        ? trace.marker.size.slice()
                        : trace.marker.size)
                    : 10,
                lineWidth: trace.marker && trace.marker.line
                    ? trace.marker.line.width || 0.5
                    : 0.5,
                lineColor: trace.marker && trace.marker.line
                    ? trace.marker.line.color || 'rgba(0,0,0,0.1)'
                    : 'rgba(0,0,0,0.1)',
                color: trace.marker && trace.marker.color
                    ? trace.marker.color
                    : '#3b82f6'
            }};
        }});
    }}

    // Convert hex color to rgba for glow effect
    function hexToRgba(hex, alpha) {{
        if (!hex || hex.charAt(0) !== '#') return 'rgba(59, 130, 246, ' + alpha + ')';
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
    }}

    function highlightDataset(datasetName) {{
        if (!plot || !originalStyles || isHighlighting) return;
        isHighlighting = true;

        const highlightTraces = datasetTraceMap[datasetName] || [];
        const updates = {{}};

        // Build updates for all bubble traces
        for (let i = 0; i < nBubbleTraces; i++) {{
            const isHighlighted = highlightTraces.includes(i);
            const orig = originalStyles[i];

            if (!updates['marker.opacity']) updates['marker.opacity'] = [];
            if (!updates['marker.size']) updates['marker.size'] = [];
            if (!updates['marker.line.width']) updates['marker.line.width'] = [];
            if (!updates['marker.line.color']) updates['marker.line.color'] = [];

            if (isHighlighted) {{
                // Highlighted: scale up, increase opacity, add colored glow
                const origOpacity = Array.isArray(orig.opacity) ? orig.opacity : [orig.opacity];
                const origSize = Array.isArray(orig.size) ? orig.size : [orig.size];
                const glowColor = hexToRgba(orig.color, 0.7);

                updates['marker.opacity'].push(origOpacity.map(function(o) {{
                    return Math.min(1.0, (typeof o === 'number' ? o : 0.7) + 0.3);
                }}));
                updates['marker.size'].push(origSize.map(function(s) {{
                    return (typeof s === 'number' ? s : 10) * 1.15;
                }}));
                updates['marker.line.width'].push(3);
                updates['marker.line.color'].push(glowColor);
            }} else {{
                // Dimmed: reduce opacity significantly for clear contrast
                const origOpacity = Array.isArray(orig.opacity) ? orig.opacity : [orig.opacity];
                const origSize = Array.isArray(orig.size) ? orig.size : [orig.size];

                updates['marker.opacity'].push(origOpacity.map(function(o) {{
                    return (typeof o === 'number' ? o : 0.7) * 0.18;
                }}));
                updates['marker.size'].push(origSize);
                updates['marker.line.width'].push(orig.lineWidth);
                updates['marker.line.color'].push(orig.lineColor);
            }}
        }}

        // Apply updates to all bubble traces at once
        const traceIndices = Array.from({{length: nBubbleTraces}}, function(_, i) {{ return i; }});
        Plotly.restyle(plot, updates, traceIndices).then(function() {{
            isHighlighting = false;
        }});
    }}

    function restoreOriginalStyles() {{
        if (!plot || !originalStyles || isHighlighting) return;
        isHighlighting = true;

        const updates = {{}};
        for (let i = 0; i < nBubbleTraces; i++) {{
            const orig = originalStyles[i];

            if (!updates['marker.opacity']) updates['marker.opacity'] = [];
            if (!updates['marker.size']) updates['marker.size'] = [];
            if (!updates['marker.line.width']) updates['marker.line.width'] = [];
            if (!updates['marker.line.color']) updates['marker.line.color'] = [];

            updates['marker.opacity'].push(orig.opacity);
            updates['marker.size'].push(orig.size);
            updates['marker.line.width'].push(orig.lineWidth);
            updates['marker.line.color'].push(orig.lineColor);
        }}

        const traceIndices = Array.from({{length: nBubbleTraces}}, function(_, i) {{ return i; }});
        Plotly.restyle(plot, updates, traceIndices).then(function() {{
            isHighlighting = false;
        }});
    }}

    function hookPlotlyEvents(attempts) {{
        if (!plot || typeof plot.on !== 'function') {{
            if (attempts < 40) {{
                window.setTimeout(function() {{ hookPlotlyEvents(attempts + 1); }}, 60);
            }}
            return;
        }}

        // Capture original styles after plot is ready
        captureOriginalStyles();

        // Click handler for opening dataset pages
        plot.on('plotly_click', function(evt) {{
            const point = evt && evt.points && evt.points[0];
            const url = point && point.customdata && point.customdata[5];
            if (url) window.open(url, '_blank', 'noopener');
        }});

        // Hover handler for highlighting datasets
        plot.on('plotly_hover', function(evt) {{
            if (hoverTimeout) {{
                clearTimeout(hoverTimeout);
                hoverTimeout = null;
            }}
            const point = evt && evt.points && evt.points[0];
            const datasetName = point && point.customdata && point.customdata[0];
            if (datasetName && datasetTraceMap[datasetName]) {{
                highlightDataset(datasetName);
            }}
        }});

        // Unhover handler to restore original styles (with debounce)
        plot.on('plotly_unhover', function() {{
            if (hoverTimeout) clearTimeout(hoverTimeout);
            hoverTimeout = setTimeout(function() {{
                restoreOriginalStyles();
            }}, 50);
        }});

        showPlot();
        window.setTimeout(function() {{
            if (typeof Plotly !== 'undefined' && plot) {{
                Plotly.Plots.resize(plot);
                captureOriginalStyles();
            }}
        }}, 100);
    }}

    hookPlotlyEvents(0);
    showPlot();
}});
</script>
"""

    return build_and_export_html(
        fig,
        out_path,
        div_id="moabb-bubble",
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
                "filename": "moabb_bubble",
                "height": height,
                "width": plot_width,
                "scale": 2,
            },
        },
    )


def _short_label(name: str, limit: int = 12) -> str:
    """Return a trimmed dataset label for annotations."""
    text = str(name).upper()
    if len(text) <= limit:
        return text
    return text[:limit]


def _add_moabb_legend(
    fig: go.Figure,
    scale: float = 0.5,
    x_offset: float = 0.02,
    y_offset: float = 0.02,
) -> None:
    """Add MOABB-style legend showing size and opacity encoding.

    Args:
        fig: Plotly figure to add legend to
        scale: Scale factor used for bubbles
        x_offset: X position (paper coordinates)
        y_offset: Y position (paper coordinates)

    """
    # Size legend
    legend_text = (
        "<b>Circle size</b>: log(records per subject)<br>"
        "<b>Opacity</b>: fewer sessions = more opaque<br>"
        "<b>Each circle</b> = 1 subject"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x_offset,
        y=1 - y_offset,
        text=legend_text,
        showarrow=False,
        font=dict(size=12, color="#374151"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        borderpad=8,
        align="left",
        xanchor="left",
        yanchor="top",
    )


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
        default=0.5,
        help="Scaling factor for bubble sizes",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.source, index_col=False, header=0, skipinitialspace=True)
    output_path = generate_moabb_bubble(df, args.output, scale=args.scale)
    print(f"MOABB bubble chart saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
