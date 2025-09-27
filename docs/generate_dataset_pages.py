#!/usr/bin/env python3
"""Generate individual documentation pages for each EEGDash dataset.

This script creates individual RST files for each dataset with comprehensive
information including metadata, usage examples, and dataset statistics.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the parent directory to the path to import eegdash modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from eegdash.dataset.registry import _markdown_table


def create_dataset_page_template(dataset_id: str, row_series: pd.Series) -> str:
    """Create an RST page template for a specific dataset."""
    # Extract key metadata
    n_subjects = row_series.get("n_subjects", "Unknown")
    n_records = row_series.get("n_records", "Unknown")
    n_tasks = row_series.get("n_tasks", "Unknown")
    modality = row_series.get("modality of exp", "")
    exp_type = row_series.get("type of exp", "")
    subject_type = row_series.get("Type Subject", "")
    duration = row_series.get("duration_hours_total", "Unknown")
    size = row_series.get("size", "Unknown")

    # Create description
    description_parts = []
    if modality and str(modality).strip():
        description_parts.append(f"**{modality}**")
    if exp_type and str(exp_type).strip():
        description_parts.append(f"{exp_type}")
    if subject_type and str(subject_type).strip():
        description_parts.append(f"{subject_type} subjects")

    description = (
        " | ".join(description_parts)
        if description_parts
        else "EEG dataset from OpenNeuro"
    )

    # Generate the metadata table
    table_content = _markdown_table(row_series)

    # Create the RST content
    rst_content = f'''.. _{dataset_id.lower()}:

{dataset_id.upper()}
{"=" * len(dataset_id)}

OpenNeuro Dataset {dataset_id}
------------------------------

{description}

This dataset contains **{n_subjects} subjects** with **{n_records} recordings** across **{n_tasks} tasks**.
Total duration: **{duration} hours**. Dataset size: **{size}**.

Dataset Overview
----------------

{table_content}

Usage Examples
--------------

Basic usage:

.. code-block:: python

    from eegdash.dataset import {dataset_id.upper()}

    # Initialize the dataset
    dataset = {dataset_id.upper()}(cache_dir="./data")

    # Check dataset size
    print(f"Number of recordings: {{len(dataset)}}")

    # Access first recording
    if len(dataset) > 0:
        recording = dataset[0]
        print(f"Recording description: {{recording.description}}")

Loading EEG Data:

.. code-block:: python

    # Load raw EEG data
    if len(dataset) > 0:
        recording = dataset[0]
        raw = recording.load()

        # Inspect the data
        print(f"Sampling rate: {{raw.info['sfreq']}} Hz")
        print(f"Number of channels: {{len(raw.ch_names)}}")
        print(f"Duration: {{raw.times[-1]:.1f}} seconds")
        print(f"Channel names: {{raw.ch_names[:5]}}...")  # First 5 channels

Advanced Filtering:

.. code-block:: python

    # Filter by specific criteria (if applicable)
    filtered_dataset = {dataset_id.upper()}(
        cache_dir="./data",
        query={{"task": "RestingState"}}  # Example filter
    )

    # Combine with other datasets
    from eegdash import EEGDashDataset

    # Load multiple datasets
    combined = EEGDashDataset(
        cache_dir="./data",
        dataset=["{dataset_id}", "ds002718"],  # Multiple datasets
        subject=["001", "002"]  # Specific subjects
    )

Dataset Information
-------------------

**Dataset ID**: {dataset_id}

**OpenNeuro URL**: https://openneuro.org/datasets/{dataset_id}

**NeMAR URL**: https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}

**Key Statistics**:

- **Subjects**: {n_subjects}
- **Recordings**: {n_records}
- **Tasks**: {n_tasks}
- **Duration**: {duration} hours
- **Size**: {size}
- **Modality**: {modality or "EEG"}
- **Experiment Type**: {exp_type or "Not specified"}
- **Subject Type**: {subject_type or "Not specified"}

Related Documentation
---------------------

- :class:`eegdash.api.EEGDashDataset` - Main dataset class
- :doc:`../api_core` - Core API reference
- :doc:`../overview` - EEGDash overview

See Also
--------

- `OpenNeuro dataset page <https://openneuro.org/datasets/{dataset_id}>`_
- `NeMAR data explorer <https://nemar.org/dataexplorer/detail?dataset_id={dataset_id}>`_
- :doc:`dataset_index` - Browse all available datasets
'''

    return rst_content


def generate_dataset_index_page(df: pd.DataFrame) -> str:
    """Generate an index page listing all datasets."""
    # Group datasets by modality for better organization
    modalities = df.groupby("modality of exp").size().sort_values(ascending=False)

    rst_content = """.. _dataset_index:

Dataset Index
=============

EEGDash provides access to **255 EEG datasets** from OpenNeuro. Each dataset has its own dedicated documentation page with detailed metadata, usage examples, and statistics.

Quick Statistics
----------------

- **Total Datasets**: 255
- **Total Subjects**: {total_subjects:,}
- **Total Recordings**: {total_records:,}
- **Total Duration**: {total_duration:.1f} hours
- **Total Size**: {total_size:.1f} GB

Browse by Modality
------------------

""".format(
        total_subjects=df["n_subjects"].sum(),
        total_records=df["n_records"].sum(),
        total_duration=df["duration_hours_total"].sum(),
        total_size=df["size_bytes"].sum() / (1024**3),  # Convert to GB
    )

    # Add modality sections
    for modality, count in modalities.head(10).items():
        if pd.isna(modality) or modality == "":
            modality = "Other"

        rst_content += f"""
{modality} ({count} datasets)
{"^" * (len(modality) + len(f" ({count} datasets)"))}

"""

        # List datasets for this modality
        modality_datasets = (
            df[df["modality of exp"] == modality]
            if modality != "Other"
            else df[df["modality of exp"].isna() | (df["modality of exp"] == "")]
        )

        # Show ALL datasets for this modality (no truncation)
        for _, row in modality_datasets.iterrows():
            dataset_id = row["dataset"]
            n_subjects = row["n_subjects"]
            n_records = row["n_records"]
            exp_type = row.get("type of exp", "")

            rst_content += f"- :doc:`{dataset_id} <datasets/{dataset_id}>` - {n_subjects} subjects, {n_records} recordings"
            if exp_type and pd.notna(exp_type):
                rst_content += f" ({exp_type})"
            rst_content += "\n"

        rst_content += "\n"

    # Add alphabetical index
    rst_content += """
Complete Alphabetical Index
---------------------------

.. toctree::
   :maxdepth: 1
   :glob:

   datasets/*

All Datasets (Alphabetical)
---------------------------

"""

    # Add alphabetical list
    for _, row in df.sort_values("dataset").iterrows():
        dataset_id = row["dataset"]
        n_subjects = row["n_subjects"]
        n_records = row["n_records"]
        size = row["size"]

        rst_content += f"- :doc:`{dataset_id} <datasets/{dataset_id}>` - {n_subjects} subjects, {n_records} recordings, {size}\n"

    return rst_content


def main():
    """Generate all dataset documentation pages."""
    # Load dataset metadata
    csv_path = (
        Path(__file__).parent.parent / "eegdash" / "dataset" / "dataset_summary.csv"
    )
    df = pd.read_csv(csv_path, comment="#", skip_blank_lines=True)

    print(f"Generating documentation for {len(df)} datasets...")

    # Create output directories
    output_dir = Path(__file__).parent / "source" / "api" / "datasets"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate individual dataset pages
    for _, row in df.iterrows():
        dataset_id = row["dataset"]
        print(f"  Generating {dataset_id}...")

        # Create RST content
        rst_content = create_dataset_page_template(dataset_id, row)

        # Write to file
        output_file = output_dir / f"{dataset_id}.rst"
        with open(output_file, "w") as f:
            f.write(rst_content)

    # Generate index page
    print("Generating dataset index page...")
    index_content = generate_dataset_index_page(df)
    index_file = Path(__file__).parent / "source" / "api" / "api_dataset.rst"
    with open(index_file, "w") as f:
        f.write(index_content)

    print(f"✅ Generated {len(df)} dataset pages + index page")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Index page: {index_file}")


if __name__ == "__main__":
    main()
