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

try:  # Support execution as a script or as a package module
    from .colours import CANONICAL_MAP, COLUMN_COLOR_MAPS, hex_to_rgba
except ImportError:  # pragma: no cover - fallback for direct script execution
    from colours import CANONICAL_MAP, COLUMN_COLOR_MAPS, hex_to_rgba

DEFAULT_COLUMNS = ["Type Subject", "modality of exp", "type of exp"]
__all__ = ["generate_dataset_sankey", "build_sankey"]


def _prepare_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    all_columns = list(columns)
    if "n_subjects" not in all_columns:
        all_columns.append("n_subjects")

    missing = [col for col in all_columns if col not in df.columns]
    if missing:
        msg = f"Columns not found in dataframe: {missing}"
        raise ValueError(msg)

    cleaned = df.copy()

    # Fill missing n_subjects with 1 (to count as at least one dataset)
    # and ensure the column is numeric integer type.
    cleaned["n_subjects"] = (
        pd.to_numeric(cleaned["n_subjects"], errors="coerce").fillna(1).astype(int)
    )

    # Process each column for cleaning and normalization
    for col in columns:
        # 1. Fill original NaN values with the string 'Unknown'
        cleaned[col] = cleaned[col].fillna("Unknown")

        # 2. Split multi-valued cells
        cleaned[col] = cleaned[col].astype(str).str.split(r"/|;|,", regex=True)
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
        # The user wants to preserve original labels but color them as 'Clinical'.
        # The relabeling to 'Clinical' is now removed. The coloring logic will handle this.
        pass

    return cleaned[all_columns]


def _load_dataframe(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        index_col=False,
        header=0,
        skipinitialspace=True,
    )
    return _prepare_dataframe(df, columns)


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

                # Get color from map - each condition has its own color
                node_color = color_map.get(val, "#94a3b8")
                node_colors.append(node_color)

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    link_hover_labels: list[str] = []

    for idx in range(len(columns) - 1):
        col_from, col_to = columns[idx], columns[idx + 1]

        # Use the color from the source node for the link
        source_color_map = COLUMN_COLOR_MAPS.get(col_from, {})

        # Group by source and target, getting both sum of subjects and count of datasets
        grouped = (
            df.groupby([col_from, col_to])
            .agg(
                subject_sum=("n_subjects", "sum"),
                dataset_count=("n_subjects", "size"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            source_val, target_val, subject_sum, dataset_count = (
                row[col_from],
                row[col_to],
                row["subject_sum"],
                row["dataset_count"],
            )

            source_node_idx = node_index.get((col_from, source_val))
            target_node_idx = node_index.get((col_to, target_val))

            if source_node_idx is not None and target_node_idx is not None:
                sources.append(source_node_idx)
                targets.append(target_node_idx)
                values.append(subject_sum)  # Weight links by sum of subjects
                link_hover_labels.append(
                    f"{source_val} → {target_val}:<br>"
                    f"{subject_sum} subjects in {dataset_count} datasets"
                )

                # Assign color to the link based on the source node
                source_color = source_color_map.get(source_val, "#94a3b8")
                link_colors.append(hex_to_rgba(source_color))

    # Add counts (subjects and datasets) and percentages to the first column labels
    first_col_name = columns[0]
    first_col_stats = df.groupby(first_col_name).agg(
        subject_sum=("n_subjects", "sum"),
        dataset_count=("n_subjects", "size"),
    )
    total_subjects = first_col_stats["subject_sum"].sum()

    for i, label in enumerate(node_labels):
        col, val = next((k for k, v in node_index.items() if v == i), (None, None))
        if col == first_col_name and val in first_col_stats.index:
            stats = first_col_stats.loc[val]
            subject_sum = stats["subject_sum"]
            dataset_count = stats["dataset_count"]
            percentage = (
                (subject_sum / total_subjects) * 100 if total_subjects > 0 else 0
            )
            node_labels[i] = (
                f"{label}<br>({subject_sum} subjects, {dataset_count} datasets, {percentage:.1f}%)"
            )

    return (
        node_labels,
        node_colors,
        sources,
        targets,
        values,
        link_colors,
        link_hover_labels,
    )


def build_sankey(df: pd.DataFrame, columns: Sequence[str]) -> go.Figure:
    (
        labels,
        colors,
        sources,
        targets,
        values,
        link_colors,
        link_hover_labels,
    ) = _build_sankey_data(df, columns)

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=18,
            label=labels,
            color=colors,
            align="left",  # Align all labels to the left of the node bars
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover_labels,
        ),
    )

    fig = go.Figure(sankey)

    fig.update_layout(
        font=dict(size=14),
        height=900,
        width=None,
        autosize=True,
        margin=dict(t=40, b=40, l=40, r=40),
        annotations=[
            dict(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Population Type",
                showarrow=False,
                font=dict(size=16, color="black"),
            ),
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Experimental Modality",
                showarrow=False,
                font=dict(size=16, color="black"),
            ),
            dict(
                x=1,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Cognitive Domain",
                showarrow=False,
                font=dict(size=16, color="black"),
            ),
            dict(
                x=0,
                y=-0.15,  # Position the note below the chart
                xref="paper",
                yref="paper",
                text='<b>Note on "Unknown" category:</b> This large portion represents datasets that are still pending categorization.',
                showarrow=False,
                align="left",
                xanchor="left",
                font=dict(size=12, color="dimgray"),
            ),
        ],
    )
    return fig


def generate_dataset_sankey(
    df: pd.DataFrame,
    out_html: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """Generate the dataset Sankey diagram and write it to *out_html*."""
    selected_columns = list(columns) if columns is not None else list(DEFAULT_COLUMNS)
    prepared = _prepare_dataframe(df, selected_columns)
    fig = build_sankey(prepared, selected_columns)

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="dataset-sankey",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    styled_html = f"""
<div id="dataset-sankey-wrapper" style="width: 100%; height: 100%;">
    {html_content}
</div>
<script>
    window.addEventListener('load', function() {{
        window.dispatchEvent(new Event('resize'));
    }});
</script>
"""
    out_path.write_text(styled_html, encoding="utf-8")
    return out_path


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
