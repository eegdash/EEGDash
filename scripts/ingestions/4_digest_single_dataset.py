r"""Digest a single OpenNeuro dataset and generate JSON for MongoDB ingestion.

This script processes a single BIDS dataset to extract metadata and create
a two-tier JSON structure optimized for the new EEGDash API Gateway architecture:

- Core metadata: Essential fields needed to load the dataset (always loaded)
- Enriched metadata: Additional information loaded on-demand for performance

The generated JSON files are ready for ingestion into MongoDB via the API Gateway
at https://data.eegdash.org using the admin endpoints.

Architecture:
    MongoDB (via API Gateway) <- JSON files <- This script <- BIDS dataset

Output files:
    - {dataset_id}_core.json: Core metadata for efficient querying
    - {dataset_id}_enriched.json: Extended metadata loaded on-demand
    - {dataset_id}_full_manifest.json: Complete metadata for reference
    - {dataset_id}_summary.json: Processing summary and statistics
Usage:
    python digest_single_dataset.py ds002718 --dataset-dir test_diggestion/ds002718
Upload to MongoDB:
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def split_core_and_enriched_metadata(record: dict[str, Any]) -> tuple[dict, dict]:
    """Split a record into core and enriched metadata.

    Core metadata includes fields strictly necessary to:
    - Identify the recording
    - Locate the data file
    - Load the dataset

    Enriched metadata includes:
    - Participant information
    - EEG technical details
    - BIDS dependencies
    - Additional JSON metadata

    Parameters
    ----------
    record : dict
        The full metadata record from EEGDash

    Returns
    -------
    core : dict
        Core metadata fields (always loaded)
    enriched : dict
        Enriched metadata fields (loaded on-demand)

    """
    # Core fields: minimal set to identify and load the recording
    core_fields = {
        "data_name",  # Unique identifier
        "dataset",  # Dataset ID (e.g., ds002718)
        "bidspath",  # Path within BIDS structure
        "subject",  # Subject identifier
        "task",  # Task name
        "session",  # Session identifier (optional)
        "run",  # Run number (optional)
        "modality",  # Data modality (eeg, meg, etc.)
        "sampling_frequency",  # Sampling rate (needed for basic validation)
        "nchans",  # Number of channels
        "ntimes",  # Number of time points
    }

    core = {k: record.get(k) for k in core_fields if k in record}

    # Everything else goes into enriched metadata
    enriched = {k: v for k, v in record.items() if k not in core_fields}

    return core, enriched


def digest_dataset(
    dataset_id: str,
    dataset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Process a single dataset and generate JSON metadata.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "ds002718")
    dataset_dir : Path
        Path to the local BIDS dataset directory
    output_dir : Path
        Directory where JSON output will be saved

    Returns
    -------
    dict
        Summary of the digestion process

    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"Processing dataset: {dataset_id}")
    print(f"  Source: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Extract metadata directly from BIDS dataset (no DB connection needed)
    from eegdash.bids_eeg_metadata import load_eeg_attrs_from_bids_file
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    try:
        # Use allow_symlinks=True for metadata extraction without loading raw data
        # This allows processing git-annex repositories where files are symlinks
        bids_dataset = EEGBIDSDataset(
            data_dir=str(dataset_dir),
            dataset=dataset_id,
            allow_symlinks=True,  # Enable metadata extraction from symlinked files
        )
        print(f"✓ Loaded BIDS dataset: {len(bids_dataset.get_files())} files found")
        print("  Mode: Metadata extraction (symlinks allowed)")
    except Exception as exc:
        print(f"✗ Error creating BIDS dataset: {exc}")
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": str(exc),
        }

    # Extract metadata for each file
    records = []
    errors = []

    print("\nExtracting metadata from files...")
    for idx, bids_file in enumerate(bids_dataset.get_files(), 1):
        file_name = Path(bids_file).name
        print(
            f"  [{idx:3d}/{len(bids_dataset.get_files()):3d}] {file_name[:50]:<50}",
            end=" ",
        )

        try:
            eeg_attrs = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
            records.append(eeg_attrs)
            print("✓")
        except Exception as exc:
            print(f"✗ {str(exc)[:50]}")
            errors.append({"file": str(bids_file), "error": str(exc)})

    manifest = {
        "dataset": dataset_id,
        "source": str(dataset_dir.resolve()),
        "record_count": len(records),
        "records": records,
    }
    if errors:
        manifest["errors"] = errors

    print(f"\n✓ Extracted metadata for {manifest['record_count']} recordings")
    if manifest.get("errors"):
        print(f"  ⚠ {len(manifest['errors'])} errors encountered")

    # Split into core and enriched metadata
    core_records = []
    enriched_records = []

    for record in manifest["records"]:
        core, enriched = split_core_and_enriched_metadata(record)

        # Store data_name in enriched for linking
        if "data_name" in core:
            enriched["data_name"] = core["data_name"]

        core_records.append(core)
        enriched_records.append(enriched)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full manifest (for reference/debugging)
    full_manifest_path = output_dir / f"{dataset_id}_full_manifest.json"
    with full_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_serializer)
    print(f"\n✓ Saved full manifest: {full_manifest_path}")

    # 2. Core metadata (for efficient loading)
    core_manifest = {
        "dataset": dataset_id,
        "record_count": len(core_records),
        "records": core_records,
    }
    core_path = output_dir / f"{dataset_id}_core.json"
    with core_path.open("w", encoding="utf-8") as f:
        json.dump(core_manifest, f, indent=2, default=_json_serializer)
    print(f"✓ Saved core metadata: {core_path}")

    # 3. Enriched metadata (for on-demand loading)
    enriched_manifest = {
        "dataset": dataset_id,
        "record_count": len(enriched_records),
        "records": enriched_records,
    }
    enriched_path = output_dir / f"{dataset_id}_enriched.json"
    with enriched_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_manifest, f, indent=2, default=_json_serializer)
    print(f"✓ Saved enriched metadata: {enriched_path}")

    # Create summary
    summary = {
        "status": "success",
        "dataset_id": dataset_id,
        "record_count": manifest["record_count"],
        "error_count": len(manifest.get("errors", [])),
        "outputs": {
            "full_manifest": str(full_manifest_path),
            "core_metadata": str(core_path),
            "enriched_metadata": str(enriched_path),
        },
        "upload_instructions": {
            "description": "Upload to MongoDB via API Gateway",
            "endpoint": "https://data.eegdash.org/admin/{database}/records/bulk",
            "auth_header": "Authorization: Bearer AdminWrite2025SecureToken",
            "example_curl": f"curl -X POST https://data.eegdash.org/admin/eegdashstaging/records/bulk -H 'Authorization: Bearer AdminWrite2025SecureToken' -H 'Content-Type: application/json' -d @{core_path}",
        },
    }

    # Save summary
    summary_path = output_dir / f"{dataset_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    return summary


def _json_serializer(obj):
    """Handle non-serializable objects for JSON export."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return sorted(list(obj))
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="Digest a single OpenNeuro dataset to extract metadata."
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="Dataset identifier (e.g., ds002718)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the local BIDS dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("digestion_output"),
        help="Directory for output JSON files (default: digestion_output)",
    )

    args = parser.parse_args()

    try:
        summary = digest_dataset(
            dataset_id=args.dataset_id,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 60)
        print("Digestion completed successfully!")
        print("=" * 60)
        print(f"Dataset: {summary['dataset_id']}")
        print(f"Records processed: {summary['record_count']}")
        if summary["error_count"] > 0:
            print(f"Errors: {summary['error_count']}")
        print("\nOutputs:")
        for name, path in summary["outputs"].items():
            print(f"  {name}: {path}")
        print("\n" + "=" * 60)
        print("Next Steps: Upload to MongoDB")
        print("=" * 60)
        print("\nTo upload the core metadata to MongoDB:")
        print(f"\n{summary['upload_instructions']['example_curl']}")
        print("\nReplace 'eegdashstaging' with 'eegdash' for production database.")

        return 0

    except Exception as e:
        print(f"\n✗ Digestion failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
