"""================================
Clinical Dataset Summary
================================

This example demonstrates how to summarize and visualize the distribution of
clinical vs. healthy datasets across different recording modalities.
"""

# %%
# Loading the data
# ----------------
#
# We use the :class:`eegdash.EEGDash` client to fetch all available datasets.
# For this example, we will also load local curation data if fields are missing,
# to demonstrate the visualization capabilities with the latest categorized metadata.

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from eegdash import EEGDash

# Initialize client (public read access)
client = EEGDash()

# Try fetching from API
try:
    datasets = client.find_datasets(limit=1000)
    if isinstance(datasets, dict) and "data" in datasets:
        datasets = datasets["data"]
except Exception:
    datasets = []

print(f"Fetched {len(datasets)} datasets from API.")

# Fallback/Augment with local JSON if API returns few results (e.g. dev environment)
json_path = Path("consolidated/openneuro_datasets.json")
if len(datasets) < 10 and json_path.exists():
    print(f"API returned few results. augmenting with {json_path}...")
    with open(json_path) as f:
        local_datasets = json.load(f)
    # Convert to DF and combine? simpler to just use local if API is empty
    datasets = local_datasets
    print(f"Using {len(datasets)} datasets from local JSON.")

# Convert to DataFrame
df = pd.DataFrame(datasets)

# Ensure dataset_id key consistency (JSON uses dataset_id, API might vary)
if "dataset_id" not in df.columns and "dataset" in df.columns:
    df["dataset_id"] = df["dataset"]

# %%
# Augmenting with Local Metadata (For Demonstration)
# --------------------------------------------------
# We will use the local CSV as the primary source if available to ensure we verify
# the categorization results, merging with API data for any missing fields.

csv_path = Path("scripts/metadata_curation.csv")
if csv_path.exists():
    print(f"Loading local curation from {csv_path}...")
    curation_df = pd.read_csv(csv_path)

    # Merge with right join to prioritize categorized datasets
    # API/JSON data provides the 'recording_modality' which is missing from CSV
    df_merged = pd.merge(
        df, curation_df, on="dataset_id", how="right", suffixes=("_api", "_csv")
    )

    # Resolve columns
    for col in ["is_clinical", "clinical_purpose", "paradigm_modality"]:
        if f"{col}_csv" in df_merged.columns:
            df_merged[col] = df_merged[f"{col}_csv"].combine_first(
                df_merged.get(f"{col}_api")
            )

    # Defaults
    if "is_clinical" not in df_merged.columns:
        df_merged["is_clinical"] = False

    # Backfill fields if missing in CSV but present in API struct
    if "clinical" in df_merged.columns:

        def safe_extract(row, key, default):
            if isinstance(row, dict):
                return row.get(key, default)
            return default

        df_merged["is_clinical"] = df_merged["is_clinical"].fillna(
            df_merged["clinical"].apply(lambda x: safe_extract(x, "is_clinical", False))
        )

else:
    df_merged = df.copy()
    # Mock extraction if no CSV
    df_merged["is_clinical"] = False
    df_merged["clinical_purpose"] = "Healthy"
    if "paradigm" in df_merged.columns:
        df_merged["paradigm_modality"] = df_merged["paradigm"].apply(
            lambda x: x.get("modality") if isinstance(x, dict) else "Other"
        )

# %%
# Data Cleaning
# -------------


# Normalize Modality (from recording_modality, NOT paradigm_modality)
# recording_modality is usually a list e.g. ["eeg"]
def normalize_recording_modality(val):
    if isinstance(val, list):
        # Flatten: prioritize iEEG > MEG > EEG
        val_str = " ".join([str(v).lower() for v in val])
    elif isinstance(val, str):
        val_str = val.lower()
    else:
        return "Other"

    if "ieeg" in val_str or "intracranial" in val_str:
        return "iEEG"
    if "meg" in val_str:
        return "MEG"
    if "eeg" in val_str:
        return "EEG"
    return "Other"


# Use recording_modality if available, else paradigm (imperfect proxy)
if "recording_modality" in df_merged.columns:
    df_merged["Modality"] = df_merged["recording_modality"].apply(
        normalize_recording_modality
    )
else:
    # Fallback if recording_modality lost in merge (e.g. if CSV had IDs not in JSON)
    df_merged["Modality"] = "Unknown"


# Normalize Subject Type (Clinical Purpose)
def normalize_purpose(row):
    # Check is_clinical bool
    is_clin = row.get("is_clinical")
    if is_clin is False or str(is_clin).lower() == "false":
        return "Healthy"

    purpose = row.get("clinical_purpose")
    if (
        not isinstance(purpose, str)
        or not purpose.strip()
        or purpose.lower() in ["unspecified clinical", "nan", "none"]
    ):
        # If is_clinical is True but purpose unspecified -> Unknown Clinical
        if is_clin:
            return "Unspecified Clinical"
        return "Healthy"

    return purpose.title()


df_merged["Subject Type"] = df_merged.apply(normalize_purpose, axis=1)

# Filter for main modalities
plot_df = df_merged[df_merged["Modality"].isin(["EEG", "MEG", "iEEG"])]

print(f"Plotting {len(plot_df)} studies.")
print("Subject Types:", plot_df["Subject Type"].unique())

# %%
# Plotting
# --------

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create stacked bar chart using histogram
if not plot_df.empty:
    ax = sns.histplot(
        data=plot_df,
        x="Modality",
        hue="Subject Type",
        multiple="stack",
        shrink=0.8,
        palette="tab20",
        edgecolor="white",
    )

    plt.title("Number of Studies by Modality and Subject Type")
    plt.ylabel("Number of Studies")
    plt.xlabel("Electrophysiology Modality")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("clinical_breakdown.png")  # Save for verification
    plt.show()
else:
    print("No data to plot.")
