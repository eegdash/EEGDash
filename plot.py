from __future__ import annotations

"""Generate a Sankey diagram from the EEG-Dash dataset summary.

The script loads ``eegdash/dataset/dataset_summary.csv`` (by default) and builds
an interactive Plotly Sankey diagram connecting three categorical columns. This
mirrors how the documentation summarises datasets across subject type, modality,
and experiment type, but can be reused with any trio of categorical columns via
CLI arguments.
"""

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

from eegdash.sankey_helpers import (
    CANONICAL_MAP,
    COLUMN_COLOR_MAPS,
    hex_to_rgba,
)

DEFAULT_COLUMNS = ["Type Subject", "modality of exp", "type of exp"]


def _load_dataframe(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        index_col=False,
        header=0,
        skipinitialspace=True,
    )
    missing = [col for col in columns if col not in df.columns]
    if missing:
        msg = f"Columns not found in dataframe: {missing}"
        raise ValueError(msg)

    cleaned = df.copy()

    # Process each column for cleaning and normalization
    for col in columns:
        # 1. Fill original NaN values with the string 'Unknown'
        cleaned[col] = cleaned[col].fillna("Unknown")

        # 2. Split multi-valued cells
        cleaned[col] = cleaned[col].astype(str).str.split("/|;|,")
        cleaned = cleaned.explode(col)

        # 3. Clean up whitespace and any empty strings created by splitting
        cleaned[col] = cleaned[col].str.strip()
        cleaned[col] = cleaned[col].replace(["", "nan"], "Unknown")

        # 4. Apply canonical mapping to standardize terms
        if col in CANONICAL_MAP:
            mapping = CANONICAL_MAP[col]
            # Use .str.lower() for case-insensitive mapping
            cleaned[col] = cleaned[col].str.lower().map(mapping).fillna(cleaned[col])

    # 5. Apply special rule for 'Type Subject' after all other processing
    if "Type Subject" in columns:
        # Identify values that are NOT 'Healthy' or 'Unknown'
        is_healthy = cleaned["Type Subject"] == "Healthy"
        is_unknown = cleaned["Type Subject"] == "Unknown"
        # Set all other values to 'Clinical'
        cleaned.loc[~is_healthy & ~is_unknown, "Type Subject"] = "Clinical"

    return cleaned[columns]


def _build_sankey_data(df: pd.DataFrame, columns: Sequence[str]):
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[str, str], int] = {}

    for col in columns:
        color_map = COLUMN_COLOR_MAPS.get(col, {})

        # Sort unique values to ensure "Unknown" appears at the bottom
        all_unique = df[col].unique()
        # Separate "Unknown" and sort the rest alphabetically
        known_values = sorted([v for v in all_unique if v != "Unknown"])
        unique_values = known_values
        # Add "Unknown" to the end if it exists
        if "Unknown" in all_unique:
            unique_values.append("Unknown")

        for val in unique_values:
            if (col, val) not in node_index:
                node_index[(col, val)] = len(node_labels)
                node_labels.append(val)
                node_colors.append(color_map.get(val, "#94a3b8"))

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for idx in range(len(columns) - 1):
        col_from, col_to = columns[idx], columns[idx + 1]

        # Use the color from the source node for the link
        source_color_map = COLUMN_COLOR_MAPS.get(col_from, {})

        # Group by source and target columns and count occurrences
        grouped = df.groupby([col_from, col_to]).size().reset_index(name="count")

        for _, row in grouped.iterrows():
            source_val, target_val, count = row[col_from], row[col_to], row["count"]

            source_node_idx = node_index.get((col_from, source_val))
            target_node_idx = node_index.get((col_to, target_val))

            if source_node_idx is not None and target_node_idx is not None:
                sources.append(source_node_idx)
                targets.append(target_node_idx)
                values.append(count)

                # Assign color to the link based on the source node
                source_color = source_color_map.get(source_val, "#94a3b8")
                link_colors.append(hex_to_rgba(source_color))

    return node_labels, node_colors, sources, targets, values, link_colors


def build_sankey(df: pd.DataFrame, columns: Sequence[str]) -> go.Figure:
    (
        labels,
        colors,
        sources,
        targets,
        values,
        link_colors,
    ) = _build_sankey_data(df, columns)

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            label=labels,
            color=colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )

    fig = go.Figure(sankey)

    fig.update_layout(
        font=dict(size=12),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sankey diagram from the dataset summary CSV."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("eegdash/dataset/dataset_summary.csv"),
        help="Path to the dataset summary CSV file.",
    )
    parser.add_argument(
        "--columns",
        nargs=3,
        metavar=("FIRST", "SECOND", "THIRD"),
        default=DEFAULT_COLUMNS,
        help="Three categorical columns to connect in the Sankey plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_summary_sankey.html"),
        help="Output HTML file for the interactive Sankey diagram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Dataset summary CSV not found at {args.source}")

    columns = list(args.columns)
    df = _load_dataframe(args.source, columns)
    fig = build_sankey(df, columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(args.output),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )
    print(f"Sankey diagram saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
