"""Utilities to generate the dataset growth plot."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from .colours import MODALITY_COLOR_MAP
except ImportError:
    from colours import MODALITY_COLOR_MAP


def generate_dataset_growth(df: pd.DataFrame, out_html: str | Path) -> Path:
    """Generate cumulative growth plot."""
    df = df.copy()

    # Extract creation date
    date_col = None
    for col in ["dataset_created_at", "created"]:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        # Create empty plot with annotation
        fig = px.line(
            title="Cumulative Number of Dataset Publications (Data Unavailable)"
        )
        fig.add_annotation(text="Timestamp data missing", showarrow=False)
        p = Path(out_html)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(p, include_plotlyjs="cdn", full_html=True)
        return p

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # Ensure subjects column is numeric
    subj_col = "n_subjects" if "n_subjects" in df.columns else "subjects"
    if subj_col in df.columns:
        df["n_subjects_clean"] = pd.to_numeric(df[subj_col], errors="coerce").fillna(0)
    else:
        df["n_subjects_clean"] = 0

    # Normalize Modality
    def normalize_mod(val):
        if not isinstance(val, str):
            return "Unknown"
        val_lower = val.lower()

        # Consistent mapping with main dashboard
        if "ieeg" in val_lower or "intracranial" in val_lower:
            return "iEEG"
        if "meg" in val_lower:
            return "MEG"
        if "fnirs" in val_lower:
            return "fNIRS"
        if "emg" in val_lower:
            return "EMG"
        if "fmri" in val_lower or "functional magnetic resonance" in val_lower:
            return "fMRI"
        if "mri" in val_lower:
            return "MRI"
        if "eeg" in val_lower:
            return "EEG"
        if "ecg" in val_lower:
            return "ECG"
        if "behavior" in val_lower:
            return "Behavior"

        # Cleanup
        cleaned = (
            val.replace("['", "").replace("']", "").replace('["', "").replace('"]', "")
        )
        return cleaned.title() if cleaned else "Unknown"

    mod_col = None
    for candidate in [
        "recording_modality",
        "record_modality",
        "experimental_modality",
        "modality of exp",
        "modality",
    ]:
        if candidate in df.columns:
            mod_col = candidate
            break

    if mod_col:
        df["Modality"] = df[mod_col].apply(normalize_mod)
    else:
        df["Modality"] = "Unknown"

    parts = []
    for mod, group in df.groupby("Modality"):
        group = group.sort_values("date")
        group["cumulative_datasets"] = range(1, len(group) + 1)
        group["cumulative_subjects"] = group["n_subjects_clean"].cumsum()
        parts.append(group)

    if not parts:
        fig = px.line(title="Cumulative Number of Dataset Publications (No Data)")
        p = Path(out_html)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(p, include_plotlyjs="cdn", full_html=True)
        return p

    df_plot = pd.concat(parts)

    # We use Graph Objects directly to support better updatemenu control or update px figure?
    # Px is easier to generate the traces. We can generate two sets of traces?
    # Or just use updatemenus to switch y-data.

    # Let's use plotly express to create the initial figure (datasets)
    fig = px.line(
        df_plot,
        x="date",
        y="cumulative_datasets",
        color="Modality",
        title="Cumulative Growth",
        labels={
            "date": "Publication Date",
            "cumulative_datasets": "Number of Datasets",
        },
    )

    fig.update_layout(template="plotly_white", font=dict(size=14))
    fig.update_traces(line=dict(width=3))

    # To implement the button, we need to supply the 'cumulative_subjects' data to the traces.
    # However, px splits traces by color (modality).
    # We need to construct the update dictionary carefully.
    # It's safer to build with graph objects if we want precision, or just stick to switching the whole data source (via transforms which are deprecated/complex)
    # Easiest way: re-assign y-values in the JS update.
    # But we need to know WHICH trace corresponds to WHICH modality to grab the right y-values.

    # ALTERNATIVE: Create two sets of traces, one visible, one hidden, and toggle visibility.
    # This is standard for dropdowns.

    fig = go.Figure()

    modalities = sorted(df_plot["Modality"].unique())
    # Trace group 1: Datasets
    for mod in modalities:
        grp = df_plot[df_plot["Modality"] == mod]
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["cumulative_datasets"],
                mode="lines",
                name=mod,
                legendgroup=mod,
                visible=True,
                line=dict(width=3, color=MODALITY_COLOR_MAP.get(mod, "#999999")),
            )
        )

    # Trace group 2: Subjects
    for mod in modalities:
        grp = df_plot[df_plot["Modality"] == mod]
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["cumulative_subjects"],
                mode="lines",
                name=mod,
                legendgroup=mod,
                visible=False,  # Hidden by default
                line=dict(
                    width=3, dash="dot", color=MODALITY_COLOR_MAP.get(mod, "#999999")
                ),  # Optional style diff
                showlegend=False,  # Share legend item? Or create duplicates?
                # Standard pattern: hide legend duplicates
            )
        )

    # Update Menus
    n_mods = len(modalities)
    # View 1: Top n_mods traces visible, bottom n_mods hidden
    view_datasets = [True] * n_mods + [False] * n_mods
    # View 2: Top n_mods hidden, bottom n_mods visible
    view_subjects = [False] * n_mods + [True] * n_mods

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                {"visible": view_datasets},
                                {"yaxis.title.text": "Number of Datasets"},
                            ],
                            label="Datasets",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"visible": view_subjects},
                                {"yaxis.title.text": "Number of Subjects"},
                            ],
                            label="Subjects",
                            method="update",
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
        ]
    )

    # Responsive layout
    fig.update_layout(
        yaxis_title="Number of Datasets",
        xaxis_title="Publication Date",
        template="plotly_white",
        font=dict(size=14),
        legend=dict(x=1.02, y=1),
        margin=dict(t=100, l=40, r=40, b=40),
        height=600,
        autosize=True,
    )

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        div_id="dataset-growth-plot",
    )

    styled_html = f"""
<div id="dataset-growth-wrapper" style="width: 100%; height: 100%;">
    {html_content}
</div>
<script>
    // Force resize on load to ensure fit
    window.addEventListener('load', function() {{
        window.dispatchEvent(new Event('resize'));
    }});
</script>
"""
    out_path.write_text(styled_html, encoding="utf-8")
    return out_path
