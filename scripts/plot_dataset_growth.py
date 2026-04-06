"""Generate a cumulative growth plot of datasets by modality."""

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px


def main():
    parser = argparse.ArgumentParser(description="Plot dataset growth.")
    parser.add_argument(
        "--input", type=Path, default=Path("consolidated/openneuro_datasets.json")
    )
    parser.add_argument("--output", type=Path, default=Path("dataset_growth.html"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return

    with open(args.input) as f:
        datasets = json.load(f)

    # Flatten for modalities
    records = []

    target_modalities = {"eeg": "EEG", "meg": "MEG", "ieeg": "iEEG"}

    for ds in datasets:
        # Check timestamps dict first
        created_at = None
        if "timestamps" in ds and isinstance(ds["timestamps"], dict):
            created_at = ds["timestamps"].get("created") or ds["timestamps"].get(
                "dataset_created_at"
            )

        # Fallback to top level
        if not created_at:
            created_at = ds.get("dataset_created_at") or ds.get("created")

        if not created_at:
            continue

        # Get modalities
        mods = ds.get("experimental_modalities", [])
        if not mods and "recording_modality" in ds:
            mods = [ds["recording_modality"]]

        # Normalize and filter
        found_mods = set()
        for m in mods:
            m_norm = m.lower()
            if m_norm in target_modalities:
                found_mods.add(target_modalities[m_norm])

        for m in found_mods:
            records.append(
                {"date": created_at, "modality": m, "dataset_id": ds.get("dataset_id")}
            )

    print(f"Found {len(records)} records.")
    if not records:
        print("No records found. Checking sample data:")
        if datasets:
            print(f"Sample keys: {datasets[0].keys()}")
            print(f"Sample created_at: {datasets[0].get('dataset_created_at')}")
            print(f"Sample mods: {datasets[0].get('experimental_modalities')}")
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Cumulative count
    df["count"] = 1
    df["cumulative"] = df.groupby("modality")["count"].cumsum()

    # Plot
    fig = px.line(
        df,
        x="date",
        y="cumulative",
        color="modality",
        title="Cumulative Number of Dataset Publications",
        labels={
            "date": "Dataset Publication Date",
            "cumulative": "Number of Dataset Publications",
            "modality": "Modality",
        },
        color_discrete_map={
            "EEG": "#636EFA",  # Plotly blue
            "MEG": "#EF553B",  # Plotly red
            "iEEG": "#00CC96",  # Plotly green
        },
    )

    # Style
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=14),
        title_font=dict(size=24),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            title=None,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
    )

    fig.update_traces(line=dict(width=3))

    # Save
    fig.write_html(args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
