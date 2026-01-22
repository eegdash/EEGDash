"""Utilities to generate the stacked clinical breakdown chart."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    from .colours import PATHOLOGY_PASTEL_OVERRIDES
except ImportError:
    from colours import PATHOLOGY_PASTEL_OVERRIDES


def _normalize_modality(mod):
    if not isinstance(mod, str) or pd.isna(mod):
        return "Unknown"
    l = mod.lower().strip()
    if l in ("nan", "none", ""):
        return "Unknown"

    # Priority checks - consistent with growth.py
    if "ieeg" in l or "intracranial" in l:
        return "iEEG"
    if "meg" in l:
        return "MEG"
    if "fnirs" in l:
        return "fNIRS"
    if "emg" in l:
        return "EMG"
    if "fmri" in l:
        return "fMRI"
    if "mri" in l:
        return "MRI"
    if "eeg" in l:
        return "EEG"
    if "ecg" in l:
        return "ECG"
    if "behavior" in l:
        return "Behavior"

    # Fallback: clean up the string
    cleaned = (
        mod.replace("['", "").replace("']", "").replace('["', "").replace('"]', "")
    )
    return cleaned.title() if cleaned else "Unknown"


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

    # Ensure columns exist (handling dataframe variations)
    # ... Same logic as before ...
    if "experimental_modality" in df.columns:
        df["modality"] = df["experimental_modality"]
    elif "recording_modality" in df.columns:
        df["modality"] = df["recording_modality"].apply(
            lambda x: str(x) if x else "Other"
        )
    elif "record_modality" in df.columns:
        df["modality"] = df["record_modality"]
    elif "record modality" in df.columns:
        df["modality"] = df["record modality"]
    elif "modality of exp" in df.columns:
        df["modality"] = df["modality of exp"]

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
    df["Modality"] = df["modality"].apply(_normalize_modality)
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

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        div_id="dataset-clinical-plot",
    )

    styled_html = f"""
<style>
#dataset-clinical-plot {{
    width: 100% !important;
    height: 650px !important;
    min-height: 650px;
    margin: 0 auto;
}}
#dataset-clinical-plot .plotly-graph-div {{
    width: 100% !important;
    height: 100% !important;
}}
</style>
{html_content}
"""
    out_path.write_text(styled_html, encoding="utf-8")
    return out_path
