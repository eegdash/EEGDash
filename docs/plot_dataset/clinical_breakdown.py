"""Utilities to generate the stacked clinical breakdown chart."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    from .colours import PATHOLOGY_PASTEL_OVERRIDES
    from .utils import (
        build_and_export_html,
        detect_modality_column,
        normalize_modality_string,
    )
except ImportError:
    from colours import PATHOLOGY_PASTEL_OVERRIDES  # type: ignore
    from utils import (  # type: ignore
        build_and_export_html,
        detect_modality_column,
        normalize_modality_string,
    )


def generate_clinical_stacked_bar(df: pd.DataFrame, out_html: str | Path) -> Path:
    """Generate stacked bar chart of Subject Type by Modality."""
    # Preprocess
    df = df.copy()

    # Ensure subjects column
    subj_col = "n_subjects" if "n_subjects" in df.columns else "subjects"
    if subj_col in df.columns:
        df["n_subjects_clean"] = pd.to_numeric(df[subj_col], errors="coerce").fillna(0)
    else:
        df["n_subjects_clean"] = 0

    # Ensure modality column exists (handling dataframe variations)
    # Clinical breakdown uses a slightly different order, prioritizing experimental_modality
    mod_col = detect_modality_column(
        df,
        candidates=(
            "experimental_modality",
            "recording_modality",
            "record_modality",
            "record modality",
            "modality of exp",
            "modality",
        ),
    )
    if mod_col:
        df["modality"] = df[mod_col].apply(lambda x: str(x) if x else "Other")

    if "population_type" not in df.columns:
        if "Type Subject" in df.columns:
            df["population_type"] = (
                df["Type Subject"].fillna("Unknown").replace("", "Unknown")
            )
        elif "clinical" in df.columns:
            df["population_type"] = df.apply(
                lambda r: r.get("clinical", {}).get("purpose", "Healthy")
                if r.get("clinical", {}).get("is_clinical")
                else "Healthy",
                axis=1,
            )
        else:
            df["population_type"] = "Unknown"

    # Clean up population_type: handle nan strings and empty values
    df["population_type"] = df["population_type"].apply(
        lambda x: "Unknown"
        if (pd.isna(x) or str(x).lower() in ("nan", "none", ""))
        else str(x)
    )

    # Normalize modality and filter out invalid values
    df["Modality"] = df["modality"].apply(normalize_modality_string)
    df = df[df["Modality"] != "Unknown"]  # Remove rows with unknown modality

    # Group by Modality and Population
    summary = (
        df.groupby(["Modality", "population_type"])
        .agg(Count=("population_type", "size"), Subjects=("n_subjects_clean", "sum"))
        .reset_index()
    )

    # We need to create a figure that supports toggling.
    # For a stacked bar, traces are usually separated by the stack group (here population_type).
    # To toggle Y values (Count vs Subjects), we can use updatemenus to restyle the 'y' attribute of the traces.

    # Unique populations serve as the traces
    populations = sorted(summary["population_type"].unique())

    fig = go.Figure()

    for pop in populations:
        data = summary[summary["population_type"] == pop]
        # We need to ensure alignment of x-axis categories if data is sparse?
        # Plotly handles this if we pass x/y correctly.

        # Color handling
        color = PATHOLOGY_PASTEL_OVERRIDES.get(pop, "#999999")

        fig.add_trace(
            go.Bar(
                name=pop,
                x=data["Modality"],
                y=data["Count"],  # Default to count
                marker_color=color,
                customdata=data[
                    "Subjects"
                ],  # Store subjects in customdata for safe keeping? or just re-assign
                # We will establish two sets of Y values: one for counts, one for subjects
            )
        )

    # Build the arrays for the updatemenu
    # We need to update 'y' for ALL traces (one per population).
    # Each button arg gives a list of new values for the attribute.

    # Get full list of modalities (union) to ensure sorting consistency is tricky with raw lists.
    # Better: pivot table to ensure alignment.
    pivot_count = summary.pivot(
        index="Modality", columns="population_type", values="Count"
    ).fillna(0)
    pivot_subj = summary.pivot(
        index="Modality", columns="population_type", values="Subjects"
    ).fillna(0)

    # Re-build traces using the pivot to ensure X alignment
    fig = go.Figure()
    x_cats = pivot_count.index.tolist()

    for pop in populations:
        if pop in pivot_count.columns:
            y_counts = pivot_count[pop].tolist()
            # y_subjs = pivot_subj[pop].tolist()
            color = PATHOLOGY_PASTEL_OVERRIDES.get(pop, "#999999")

            fig.add_trace(
                go.Bar(
                    name=pop,
                    x=x_cats,
                    y=y_counts,
                    marker_color=color,
                    # Store the alternative view logic?
                    # Actually, updatemenus 'args' for 'y' expects a list of arrays, one per trace.
                )
            )

    # Construct the update arrays
    # There are N traces (where N = len(populations))
    # We need a list of N arrays for 'y'
    y_arrays_count = []
    y_arrays_subj = []

    for pop in populations:
        if pop in pivot_count.columns:
            y_arrays_count.append(pivot_count[pop].tolist())
            y_arrays_subj.append(pivot_subj[pop].tolist())

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        font=dict(size=14),
        yaxis_title="Number of Studies",
        xaxis_title="Electrophysiology Modality",
        height=650,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                {"y": y_arrays_count},
                                {"yaxis.title.text": "Number of Studies"},
                            ],
                            label="Studies",
                            method="restyle",
                        ),
                        dict(
                            args=[
                                {"y": y_arrays_subj},
                                {"yaxis.title.text": "Number of Subjects"},
                            ],
                            label="Subjects",
                            method="restyle",
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
        ],
        margin=dict(t=100, l=60, r=40, b=80),
        autosize=True,
    )

    return build_and_export_html(
        fig,
        out_html,
        div_id="dataset-clinical-plot",
        height=650,
    )
