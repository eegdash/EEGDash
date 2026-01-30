from __future__ import annotations

"""Utilities to generate the EEG Dash dataset treemap."""

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go

try:  # Allow import both as a package and as a script
    from .colours import (
        CANONICAL_MAP,
        MODALITY_COLOR_MAP,
        MODALITY_EMOJI,
        PATHOLOGY_COLOR_MAP,
        PATHOLOGY_PASTEL_OVERRIDES,
        hex_to_rgba,
    )
    from .utils import build_and_export_html
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import (  # type: ignore
        CANONICAL_MAP,
        MODALITY_COLOR_MAP,
        MODALITY_EMOJI,
        PATHOLOGY_COLOR_MAP,
        PATHOLOGY_PASTEL_OVERRIDES,
        hex_to_rgba,
    )
    from utils import build_and_export_html  # type: ignore

__all__ = ["generate_dataset_treemap"]

_CATEGORY_COLUMNS = (
    ("Type Subject", "population_type"),
    ("modality of exp", "experimental_modality"),
)

_DATASET_COLUMN = "dataset"
_DATASET_ALIAS = "dataset_name"
_SEPARATORS = ("/", "|", ";", ",")
_DEFAULT_COLOR = "#94a3b8"


def _tokenise_cell(value: object, column_key: str) -> list[str]:
    """Split multi-valued cells, normalise, and keep Unknown buckets."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        tokens = []
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            tokens = []
        else:
            normalised = text
            for sep in _SEPARATORS:
                normalised = normalised.replace(sep, ",")
            tokens = [tok.strip() for tok in normalised.split(",") if tok.strip()]

    if not tokens:
        return ["Unknown"]

    canonical = CANONICAL_MAP.get(column_key, {})
    resolved: list[str] = []
    for token in tokens:
        lowered = token.lower()
        resolved.append(canonical.get(lowered, token))
    final = [tok if tok else "Unknown" for tok in resolved]
    return final or ["Unknown"]


def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {orig for orig, _ in _CATEGORY_COLUMNS} | {
        _DATASET_COLUMN,
        "n_records",
        "n_subjects",
        "duration_hours_total",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rename_map = {
        "n_records": "records",
        "n_subjects": "subjects",
        "duration_hours_total": "duration_hours",
        _DATASET_COLUMN: _DATASET_ALIAS,
    }
    for original, alias in _CATEGORY_COLUMNS:
        rename_map[original] = alias

    renamed = df.rename(columns=rename_map)
    columns_to_keep = [_DATASET_ALIAS] + [alias for _, alias in _CATEGORY_COLUMNS]
    cleaned = renamed.loc[:, columns_to_keep].copy()
    numeric = renamed[["records", "subjects", "duration_hours"]]

    cleaned[_DATASET_ALIAS] = (
        cleaned[_DATASET_ALIAS]
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
        .fillna("Unknown")
    )

    cleaned = cleaned.join(numeric)

    for original, alias in _CATEGORY_COLUMNS:
        cleaned[alias] = cleaned[alias].map(lambda v: _tokenise_cell(v, original))
        cleaned = cleaned.explode(alias).reset_index(drop=True)
        cleaned[alias] = cleaned[alias].fillna("Unknown")

    cleaned["records"] = pd.to_numeric(cleaned["records"], errors="coerce").fillna(0)
    cleaned["subjects"] = pd.to_numeric(cleaned["subjects"], errors="coerce").fillna(0)
    cleaned["duration_hours"] = pd.to_numeric(
        cleaned["duration_hours"], errors="coerce"
    )
    cleaned.loc[cleaned["duration_hours"] < 0, "duration_hours"] = pd.NA

    hours = cleaned["duration_hours"]
    fallback_mask = hours.isna() | (hours <= 0)
    cleaned["hours_from_records"] = 0.0
    cleaned.loc[fallback_mask, "hours_from_records"] = cleaned.loc[
        fallback_mask, "records"
    ]

    cleaned["hours"] = hours.fillna(0)
    cleaned.loc[fallback_mask, "hours"] = cleaned.loc[fallback_mask, "records"]
    cleaned["hours"] = cleaned["hours"].fillna(0).clip(lower=0)

    cleaned["records"] = cleaned["records"].clip(lower=0)
    cleaned["subjects"] = cleaned["subjects"].clip(lower=0)
    cleaned["hours_from_records"] = cleaned["hours_from_records"].clip(lower=0)

    return cleaned[
        [
            "population_type",
            "experimental_modality",
            "dataset_name",
            "hours",
            "records",
            "subjects",
            "hours_from_records",
        ]
    ]


def _abbreviate(value: float | int) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "0"

    if not math.isfinite(num):
        return "0"
    if num == 0:
        return "0"

    if abs(num) < 1000:
        rounded = round(num / 10.0) * 10.0
        if rounded == 0 and num > 0:
            rounded = 10.0
        return f"{int(rounded):,}"

    thresholds = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ]
    for divisor, suffix in thresholds:
        if abs(num) >= divisor:
            scaled = num / divisor
            scaled = round(scaled, 1)
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{num:.0f}"


def _lighten_hex(hex_color: str, factor: float = 0.55) -> str:
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return _DEFAULT_COLOR
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return _DEFAULT_COLOR
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return _DEFAULT_COLOR
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _pathology_colors(name: str) -> tuple[str, str, str]:
    """Return (fill_rgba, legend_hex, group_key)."""
    base_hex = PATHOLOGY_PASTEL_OVERRIDES.get(name)
    if not base_hex:
        fallback = PATHOLOGY_COLOR_MAP.get(name)
        if fallback:
            base_hex = _lighten_hex(fallback, 0.6)
        else:
            base_hex = PATHOLOGY_PASTEL_OVERRIDES.get("Clinical", _DEFAULT_COLOR)

    fill = hex_to_rgba(base_hex, alpha=0.65)
    if name == "Healthy":
        group = "healthy"
    elif name == "Unknown":
        group = "unknown"
    else:
        group = "clinical"
    return fill, base_hex, group


def _filter_zero_nodes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mask = (df["subjects"] > 0) | (df[column] == "Unknown")
    return df.loc[mask].copy()


def _format_label(
    name: str,
    subjects: float | int,
    hours: float | int,
    records: float | int,
    hours_from_records: float | int,
    *,
    font_px: int = 13,
) -> str:
    subjects_value = float(subjects) if pd.notna(subjects) else 0.0
    hours_value = float(hours) if pd.notna(hours) else 0.0
    records_value = float(records) if pd.notna(records) else 0.0
    fallback_value = float(hours_from_records) if pd.notna(hours_from_records) else 0.0

    subjects_text = _abbreviate(subjects_value)
    if hours_value > 0:
        secondary_text = f"{hours_value:.0f} h"
    elif fallback_value > 0:
        secondary_text = f"{_abbreviate(records_value)} rec"
    else:
        secondary_text = "records unavailable"
    return (
        f"{name}<br><span style='font-size:{font_px}px;'>{subjects_text} subj"
        f" | {secondary_text}</span>"
    )


def _build_nodes(
    dataset_level: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    dataset_level = dataset_level.sort_values(
        ["population_type", "experimental_modality", "dataset_name"]
    ).reset_index(drop=True)

    level2 = dataset_level.groupby(
        ["population_type", "experimental_modality"], dropna=False, as_index=False
    ).agg(
        hours=("hours", "sum"),
        records=("records", "sum"),
        subjects=("subjects", "sum"),
        hours_from_records=("hours_from_records", "sum"),
    )
    level2 = _filter_zero_nodes(level2, "experimental_modality")

    level1 = level2.groupby(["population_type"], dropna=False, as_index=False).agg(
        hours=("hours", "sum"),
        records=("records", "sum"),
        subjects=("subjects", "sum"),
        hours_from_records=("hours_from_records", "sum"),
    )
    level1 = _filter_zero_nodes(level1, "population_type")

    nodes: list[dict[str, object]] = []
    legend_entries: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    modality_meta: dict[str, dict[str, str]] = {}
    modality_priority = {
        name: idx for idx, name in enumerate(MODALITY_COLOR_MAP.keys())
    }

    total_subjects = level1["subjects"].sum()
    total_hours = level1["hours"].sum()
    total_records = level1["records"].sum()
    total_from_records = level1["hours_from_records"].sum()

    root_label = _format_label(
        "EEG Dash Datasets",
        total_subjects,
        total_hours,
        total_records,
        total_from_records,
        font_px=18,
    )
    nodes.append(
        {
            "id": "EEG Dash datasets",
            "parent": "",
            "name": "EEG Dash datasets",
            "text": root_label,
            "value": float(total_subjects),
            "color": "white",
            "hover": root_label,
        }
    )

    for _, row in level1.iterrows():
        name = row["population_type"] or "Unknown"
        node_id = name
        label = _format_label(
            name,
            row["subjects"],
            row["hours"],
            row["records"],
            row["hours_from_records"],
            font_px=16,
        )
        fill_color, _, group = _pathology_colors(name)
        seen_groups.add(group)
        nodes.append(
            {
                "id": node_id,
                "parent": "EEG Dash datasets",
                "name": name,
                "text": label,
                "value": float(row["subjects"]),
                "color": fill_color,
                "hover": label,
            }
        )

    for _, row in level2.iterrows():
        modality = row["experimental_modality"] or "Unknown"
        parent = row["population_type"] or "Unknown"
        node_id = f"{parent} / {modality}"
        emoji_symbol = MODALITY_EMOJI.get(modality)
        modality_label = modality
        if emoji_symbol and row["subjects"] >= 120:
            modality_label = f"{emoji_symbol} {modality}"
        label = _format_label(
            modality_label,
            row["subjects"],
            row["hours"],
            row["records"],
            row["hours_from_records"],
            font_px=16,
        )
        color = MODALITY_COLOR_MAP.get(modality, _DEFAULT_COLOR)
        legend_label = (
            f"{(emoji_symbol + ' ') if emoji_symbol else ''}{modality}".strip()
        )
        if modality not in modality_meta:
            order = modality_priority.get(modality, len(modality_priority))
            modality_meta[modality] = {
                "name": legend_label,
                "color": color,
                "group": "level2",
                "order": 100 + order,
                "legendgroup": "modalities",
            }
        nodes.append(
            {
                "id": node_id,
                "parent": parent,
                "name": modality_label,
                "text": label,
                "value": float(row["subjects"]),
                "color": color,
                "hover": label,
            }
        )

    dataset_level = _filter_zero_nodes(dataset_level, "dataset_name")
    for _, row in dataset_level.iterrows():
        dataset_name = row["dataset_name"] or "Unknown"
        modality = row["experimental_modality"] or "Unknown"
        parent = f"{row['population_type']} / {modality}"
        node_id = f"{parent} / {dataset_name}"
        label = _format_label(
            dataset_name,
            row["subjects"],
            row["hours"],
            row["records"],
            row["hours_from_records"],
            font_px=16,
        )
        if dataset_name == "Unknown":
            color = _DEFAULT_COLOR
        else:
            color = MODALITY_COLOR_MAP.get(modality, _DEFAULT_COLOR)
        nodes.append(
            {
                "id": node_id,
                "parent": parent,
                "name": dataset_name,
                "text": label,
                "value": float(row["subjects"]),
                "color": color,
                "hover": label,
            }
        )

    group_config = {
        "healthy": {
            "name": "Healthy",
            "color": PATHOLOGY_PASTEL_OVERRIDES.get("Healthy", "#bbf7d0"),
            "order": 0,
        },
        "clinical": {
            "name": "Clinical",
            "color": PATHOLOGY_PASTEL_OVERRIDES.get("Clinical", "#f8d0d0"),
            "order": 1,
        },
        "unknown": {
            "name": "To be Categorised",
            "color": PATHOLOGY_PASTEL_OVERRIDES.get("Unknown", "#d0d7df"),
            "order": 2,
        },
    }

    for key, cfg in group_config.items():
        if key in seen_groups:
            legend_entries.append(
                {
                    "name": cfg["name"],
                    "color": cfg["color"],
                    "group": "level1",
                    "order": cfg["order"],
                    "legendgroup": "populations",
                }
            )

    legend_entries.extend(
        sorted(modality_meta.values(), key=lambda item: item["order"])
    )

    return nodes, legend_entries


def _build_figure(
    nodes: Iterable[dict[str, object]],
    legend_entries: Iterable[dict[str, str]],
) -> go.Figure:
    node_list = list(nodes)
    if not node_list:
        raise ValueError("No data available to render the treemap.")

    legend_list = list(legend_entries)
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for entry in legend_list:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        deduped.append(entry)

    deduped.sort(key=lambda item: item.get("order", 999))

    fig = go.Figure(
        go.Treemap(
            ids=[node["id"] for node in node_list],
            labels=[node["name"] for node in node_list],
            parents=[node["parent"] for node in node_list],
            values=[node["value"] for node in node_list],
            text=[node["text"] for node in node_list],
            customdata=[[node["hover"]] for node in node_list],
            branchvalues="total",
            marker=dict(
                colors=[node["color"] for node in node_list],
                line=dict(color="white", width=1),
                pad=dict(t=15, r=15, b=15, l=15),
            ),
            textinfo="text",
            hovertemplate="%{customdata[0]}<extra></extra>",
            pathbar=dict(
                visible=True, edgeshape="/", thickness=34, textfont=dict(size=14)
            ),
            textfont=dict(size=24),
            insidetextfont=dict(size=24),
            # Increase pad to create more visual separation between tiles,
            # especially the top-level (population_type) nodes.
            tiling=dict(pad=8, packing="squarify"),
            # Slightly more transparent root to avoid harsh borders
            root=dict(color="rgba(255,255,255,0.98)"),
        )
    )

    # Add legend swatches. increase marker size and use a thin white border so
    # legend squares visually separate from adjacent tiles when exported.
    for entry in deduped:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    size=14,
                    symbol="square",
                    color=entry["color"],
                    line=dict(color="white", width=1.5),
                ),
                name=entry["name"],
                showlegend=True,
                hoverinfo="skip",
                xaxis="x2",
                yaxis="y2",
                legendgroup=entry.get("legendgroup"),
            )
        )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            itemsizing="constant",
            traceorder="normal",
            itemclick=False,
            itemdoubleclick=False,
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        legend_traceorder="normal",
    )

    fig.update_layout(
        xaxis2=dict(visible=False),
        yaxis2=dict(visible=False),
    )

    return fig


def generate_dataset_treemap(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    width: int = 1260,
) -> Path:
    """Generate the dataset treemap and return the output path."""
    cleaned = _preprocess_dataframe(df)
    aggregated = cleaned.groupby(
        ["population_type", "experimental_modality", "dataset_name"],
        dropna=False,
        as_index=False,
    ).agg(
        hours=("hours", "sum"),
        records=("records", "sum"),
        subjects=("subjects", "sum"),
        hours_from_records=("hours_from_records", "sum"),
    )

    aggregated = _filter_zero_nodes(aggregated, "dataset_name")
    nodes, legend_entries = _build_nodes(aggregated)
    fig = _build_figure(nodes, legend_entries)
    # Tune text sizes and margins so the increased padding doesn't cause
    # labels to overflow. Keeping uniformtext minsize slightly lower ensures
    # smaller tiles don't get crowded, while the higher top margin combined
    # with the raised legend keeps it clear of the pathbar.
    fig.update_layout(
        uniformtext=dict(minsize=20, mode="hide"),
        margin=dict(t=132, l=28, r=28, b=36),
        hoverlabel=dict(font=dict(size=14), align="left"),
        height=880,
        width=width,
        autosize=False,
    )

    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-treemap-plot",
        height=880,
    )
