import json
import math
import os
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

# API Configuration
API_BASE_URL = "https://data.eegdash.org/api"
DEFAULT_DATABASE = "eegdash"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_SUMMARY_PATH = PROJECT_ROOT / "eegdash" / "dataset" / "dataset_summary.csv"


def human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(os.fstat(0).st_size).bit_length() // 10 if size_bytes > 0 else 0
    # The above line is wrong for generic size conversion, let's use a standard one
    # Re-implementing based on prepare_summary_tables.py logic or standard logic
    if size_bytes == 0:
        return "0 B"
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def fetch_datasets_from_api(
    database: str = DEFAULT_DATABASE, limit: int = 1000
) -> list[dict]:
    """Fetch all datasets from the EEGDash API."""
    all_datasets = []
    skip = 0

    while True:
        url = f"{API_BASE_URL}/{database}/datasets/summary?limit={limit}&skip={skip}"
        print(f"Fetching: {url}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching from API: {e}")
            raise

        if not data.get("success"):
            raise ValueError(f"API returned error: {data}")

        datasets = data.get("data", [])
        all_datasets.extend(datasets)

        if len(datasets) < limit:
            break

        skip += limit

    print(f"Fetched {len(all_datasets)} datasets total")
    return all_datasets


def fetch_global_record_stats(database: str, dataset_ids: set[str] = None):
    """Fetch nchans and sfreq aggregated stats for all datasets."""
    print("Fetching global record statistics (aggregated endpoint)...")
    url = f"{API_BASE_URL}/{database}/datasets/stats/records"

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            resp_json = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching global stats: {e}")
        return {}

    data = resp_json.get("data", {})
    print(f"Fetched stats for {len(data)} datasets")
    return data


def parse_freqs(value) -> str:
    """Parse frequencies/channels list and return mode with * if variable."""
    if not value:
        return ""

    counts = Counter()

    # 1. Handle API aggregation format (list of dicts)
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], dict)
        and "val" in value[0]
    ):
        for item in value:
            val = item.get("val")
            count = item.get("count", 1)
            if val is not None:
                try:
                    # Normalize to int for display (e.g. 64.0 -> 64)
                    val_int = int(float(val))
                    counts[val_int] += count
                except (ValueError, TypeError):
                    pass

    else:
        # Fallback / manual handling if needed, but API usually sends list of dicts
        # For simplicity reusing the logic from prepare_summary_tables if needed
        pass

    if not counts:
        return ""

    most_common_val, _ = counts.most_common(1)[0]

    # Check if there is variation
    if len(counts) == 1:
        return f"{most_common_val}"
    else:
        return f"{most_common_val}*"


def main():
    print(f"Loading existing CSV from {DATASET_SUMMARY_PATH}")
    if DATASET_SUMMARY_PATH.exists():
        df_old = pd.read_csv(DATASET_SUMMARY_PATH)
        # Ensure we don't have Unnamed columns if possible, but they might be index
        if "Unnamed: 0" in df_old.columns:
            # If it looks like an index, we might want to drop it and let pandas recreate it?
            # actually preserve it or drop it, usually it's just index.
            # Let's drop it for processing and maybe add it back or let to_csv handle valid index
            pass
    else:
        print(
            "Warning: Existing CSV not found. Specific manual columns will be missing."
        )
        df_old = pd.DataFrame(columns=["dataset"])

    # Fetch data
    api_datasets = fetch_datasets_from_api()
    global_stats = fetch_global_record_stats(DEFAULT_DATABASE)

    new_rows = []
    for ds in api_datasets:
        ds_id = ds.get("dataset_id", "").strip()
        if not ds_id:
            continue

        ds_stats = global_stats.get(ds_id, {})
        nchans_list = ds_stats.get("nchans_counts", [])
        sfreq_list = ds_stats.get("sfreq_counts", [])

        # Construct row with API data
        # Note: Mapping keys to match existing CSV columns
        row = {
            "dataset": ds_id,
            "n_records": ds.get("total_files", 0) or 0,
            "n_subjects": ds.get("demographics", {}).get("subjects_count", 0) or 0,
            "n_tasks": len(ds.get("tasks", [])) or 0,
            "nchans_set": parse_freqs(nchans_list),
            "sampling_freqs": parse_freqs(sfreq_list),
            # "duration_hours_total": 0.0, # Not readily available in summary?
            "size": human_readable_size(ds.get("size_bytes") or 0),
            "size_bytes": ds.get("size_bytes") or 0,
            "source": ds.get("source", ""),
            "DatasetID": ds_id,  # Seems redundant but present in CSV
            "Type Subject": ds.get("study_domain", ""),
            "modality of exp": ", ".join(ds.get("modalities", []) or []),
            "type of exp": ds.get("study_design", ""),
            "record_modality": ", ".join(ds.get("recording_modality", []) or [])
            if isinstance(ds.get("recording_modality"), list)
            else ds.get("recording_modality", ""),
        }
        new_rows.append(row)

    df_new = pd.DataFrame(new_rows)
    print(f"Constructed DataFrame from API with {len(df_new)} rows")

    # Merge strategy:
    # 1. Start with df_old
    # 2. Update with df_new where df_new has values
    # 3. Append new datasets from df_new

    # We want to preserve columns that are in df_old but NOT in df_new (e.g. '10-20 system')
    # We also want to update columns that are in both.

    # Set index for easier update
    if not df_old.empty:
        df_old.set_index("dataset", inplace=True)
    df_new.set_index("dataset", inplace=True)

    # Align columns
    # New columns found in API but not in old ?
    for col in df_new.columns:
        if col not in df_old.columns:
            df_old[col] = None

    # Update logic
    # We loop through new data and update old data
    updated_count = 0
    new_count = 0

    for idx, row in df_new.iterrows():
        if idx in df_old.index:
            # Update existing
            for col, val in row.items():
                if val is not None and val != "" and val != []:
                    # Optional: Logic to only overwrite if old value is empty?
                    # OR overwrite always because API is truth?
                    # User said: "save the existent information within the columns in the case of empty information within the API."
                    # So if API is NOT empty, we overwrite.
                    df_old.at[idx, col] = val
            updated_count += 1
        else:
            # Add new row
            # We need to make sure we respect the columns structure of df_old
            df_old.loc[idx] = row
            new_count += 1

    print(f"Updated {updated_count} datasets, Added {new_count} new datasets.")

    # Reset index to make 'dataset' a column again
    df_old.reset_index(inplace=True)

    # Ensure column order matches original ideal or specific order?
    # Let's try to keep original order if possible.
    original_columns = [
        "dataset",
        "n_records",
        "n_subjects",
        "n_tasks",
        "nchans_set",
        "sampling_freqs",
        "duration_hours_total",
        "size",
        "size_bytes",
        "source",
        "s3_item_count",
        "DatasetID",
        "Type Subject",
        "10-20 system",
        "modality of exp",
        "type of exp",
        "record_modality",
    ]

    # Add any extra columns that might have appeared and are important
    current_cols = df_old.columns.tolist()
    final_cols = []

    # First the standard ones
    for col in original_columns:
        if col in current_cols:
            final_cols.append(col)

    # Then any others (at the end)
    for col in current_cols:
        if col not in final_cols and col not in ["Unnamed: 0"]:  # Drop old index col
            final_cols.append(col)

    df_final = df_old[final_cols]

    print(f"Saving to {DATASET_SUMMARY_PATH}")
    df_final.to_csv(DATASET_SUMMARY_PATH, index=True)  # Check if we want index=False?
    # The original file had Unnamed: 0 which is usually the saved index.
    # If I save with index=True, I get an unnamed column (the index).
    # Let's check the original file again.
    # Unnamed: 0,dataset,...
    # Yes, it looks like it was saved with index=True.

    print("Done.")


if __name__ == "__main__":
    main()
