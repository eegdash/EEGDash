from __future__ import annotations

"""Utilities to generate the EEG Dash dataset treemap."""

from pathlib import Path
from typing import Iterable

import math
import pandas as pd
import plotly.graph_objects as go

try:  # Allow import both as a package and as a script
    from .colours import CANONICAL_MAP, MODALITY_COLOR_MAP, PATHOLOGY_COLOR_MAP
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import CANONICAL_MAP, MODALITY_COLOR_MAP, PATHOLOGY_COLOR_MAP  # type: ignore

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

    thresholds = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ]
    for divisor, suffix in thresholds:
        if abs(num) >= divisor:
            scaled = num / divisor
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{num:.0f}"


def _filter_zero_nodes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    mask = (df["hours"] > 0) | (df[column] == "Unknown")
    return df.loc[mask].copy()


def _format_label(
    name: str,
    hours: float | int,
    records: float | int,
    hours_from_records: float | int,
) -> str:
    area_value = float(hours) if pd.notna(hours) else 0.0
    records_value = float(records) if pd.notna(records) else 0.0
    fallback_value = float(hours_from_records) if pd.notna(hours_from_records) else 0.0

    unit = " record" if math.isclose(area_value, fallback_value, rel_tol=1e-6) else ""
    area_text = f"{area_value:.0f}"
    records_text = _abbreviate(records_value)
    return (
        f"{name}<br><span style='font-size:11px;'>{area_text}{unit}"
        f" | {records_text} rec</span>"
    )


def _build_nodes(dataset_level: pd.DataFrame) -> list[dict[str, object]]:
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

    total_hours = level1["hours"].sum()
    total_records = level1["records"].sum()
    total_from_records = level1["hours_from_records"].sum()

    root_label = _format_label(
        "EEG Dash Datasets",
        total_hours,
        total_records,
        total_from_records,
    )
    nodes.append(
        {
            "id": "EEG Dash datasets",
            "parent": "",
            "name": "EEG Dash datasets",
            "text": root_label,
            "value": float(total_hours),
            "color": "white",
            "hover": root_label,
        }
    )

    for _, row in level1.iterrows():
        name = row["population_type"] or "Unknown"
        node_id = name
        label = _format_label(
            name,
            row["hours"],
            row["records"],
            row["hours_from_records"],
        )
        color = PATHOLOGY_COLOR_MAP.get(name, _DEFAULT_COLOR)
        nodes.append(
            {
                "id": node_id,
                "parent": "EEG Dash datasets",
                "name": name,
                "text": label,
                "value": float(row["hours"]),
                "color": color,
                "hover": label,
            }
        )

    for _, row in level2.iterrows():
        modality = row["experimental_modality"] or "Unknown"
        parent = row["population_type"] or "Unknown"
        node_id = f"{parent} / {modality}"
        label = _format_label(
            modality,
            row["hours"],
            row["records"],
            row["hours_from_records"],
        )
        color = MODALITY_COLOR_MAP.get(modality, _DEFAULT_COLOR)
        nodes.append(
            {
                "id": node_id,
                "parent": parent,
                "name": modality,
                "text": label,
                "value": float(row["hours"]),
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
            row["hours"],
            row["records"],
            row["hours_from_records"],
        )
        color = MODALITY_COLOR_MAP.get(modality, _DEFAULT_COLOR)
        nodes.append(
            {
                "id": node_id,
                "parent": parent,
                "name": dataset_name,
                "text": label,
                "value": float(row["hours"]),
                "color": color,
                "hover": label,
            }
        )

    return nodes


def _build_figure(nodes: Iterable[dict[str, object]]) -> go.Figure:
    node_list = list(nodes)
    if not node_list:
        raise ValueError("No data available to render the treemap.")

    return go.Figure(
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
                line=dict(color="white", width=2),
            ),
            textinfo="text",
            hovertemplate="%{customdata[0]}<extra></extra>",
            pathbar=dict(visible=True, edgeshape="/"),
        )
    )


def generate_dataset_treemap(
    df: pd.DataFrame,
    out_html: str | Path,
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
    nodes = _build_nodes(aggregated)
    fig = _build_figure(nodes)
    fig.update_layout(
        uniformtext=dict(minsize=10, mode="hide"),
        margin=dict(t=20, l=10, r=10, b=10),
    )

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path
