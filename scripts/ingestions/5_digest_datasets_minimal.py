r"""Digest cloned BIDS datasets and generate minimal MongoDB records.

This script processes locally cloned BIDS datasets to extract MINIMAL metadata
needed for the MongoDB database. It focuses on the core attributes required
for the EEGDash API without loading actual EEG data or extracting unnecessary metadata.

Architecture:
    Cloned BIDS datasets -> This script -> Minimal JSON records -> MongoDB (via API)

Key Differences from Full Digestion:
- Only extracts CORE attributes (10 fields defined in eegdash.const)
- No participant metadata extraction (participant_tsv, eeg_json)
- No BIDS dependencies tracking
- Faster processing (metadata-only, no data loading)
- Smaller JSON files (minimal records)

Core Attributes Extracted (from eegdash.const.config["attributes"]):
1. data_name - Unique identifier (dataset_filename)
2. dataset - Dataset ID (e.g., ds002718)
3. bidspath - S3 path for download
4. subject - Subject identifier
5. task - Task name
6. session - Session identifier (optional)
7. run - Run number (optional)
8. modality - Data modality (eeg, meg, ieeg)
9. sampling_frequency - Sampling rate in Hz
10. nchans - Number of channels
11. ntimes - Number of time points
# Note: NEMAR datasets (nm*) automatically use s3://nemar/ regardless of --s3-base

Usage:
    # Process all cloned datasets (OpenNeuro + NEMAR)
    python 5_digest_datasets_minimal.py \\
        --cloned-dir data/cloned_all \\
        --output-dir digestion_output \\
        --s3-base s3://openneuro.org

    # Process specific datasets
    python 5_digest_datasets_minimal.py \\
        --cloned-dir data/cloned_all \\
        --output-dir digestion_output \\
        --datasets ds002718 ds005506

    # Process with parallel processing
    python 5_digest_datasets_minimal.py \\
        --cloned-dir data/cloned_all \\
        --output-dir digestion_output \\
        --workers 4

Output Structure:
    digestion_output/
    ├── ds002718/
    │   ├── ds002718_minimal.json        # Minimal records for MongoDB
    │   └── ds002718_summary.json        # Processing summary
    ├── ds005506/
    │   ├── ds005506_minimal.json
    │   └── ds005506_summary.json
    └── batch_summary.json               # Overall batch results

Upload to MongoDB:
    # Upload single dataset
    curl -X POST https://data.eegdash.org/admin/eegdashstaging/records/bulk \\
         -H "Authorization: Bearer AdminWrite2025SecureToken" \\
         -H "Content-Type: application/json" \\
         -d @digestion_output/ds002718/ds002718_minimal.json

    # Bulk upload all datasets (Python)
    python scripts/upload_to_mongodb.py \\
        --input-dir digestion_output \\
        --database eegdashstaging
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_minimal_metadata(
    dataset_id: str, dataset_dir: Path, s3_base: str
) -> tuple[list[dict], list[dict]]:
    """Extract minimal metadata from a BIDS dataset.

    Only extracts the core 11 fields required for MongoDB without loading
    participant info, BIDS dependencies, or actual EEG data.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "ds002718")
    dataset_dir : Path
        Path to the local BIDS dataset directory
    s3_base : str
        S3 base URL (e.g., "s3://openneuro.org")

    Returns
    -------
    records : list of dict
        List of minimal metadata records
    errors : list of dict
        List of errors encountered during extraction

    """
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    try:
        # Use allow_symlinks=True for metadata extraction without loading raw data
        bids_dataset = EEGBIDSDataset(
            data_dir=str(dataset_dir),
            dataset=dataset_id,
            allow_symlinks=True,  # Enable metadata extraction from git-annex symlinks
        )
    except Exception as exc:
        return [], [
            {"dataset": dataset_id, "error": f"Failed to load BIDS dataset: {exc}"}
        ]

    records = []
    errors = []

    files = bids_dataset.get_files()

    for bids_file in files:
        _ = Path(bids_file).name

        try:
            # Extract minimal attributes directly without loading full metadata
            record = extract_minimal_record(
                bids_dataset, bids_file, dataset_id, s3_base
            )
            records.append(record)

        except Exception as exc:
            errors.append(
                {"dataset": dataset_id, "file": str(bids_file), "error": str(exc)}
            )

    return records, errors


def extract_minimal_record(
    bids_dataset, bids_file: str, dataset_id: str, s3_base: str
) -> dict[str, Any]:
    """Extract minimal metadata for a single BIDS file.

    Only extracts the 11 core attributes defined in eegdash.const.config["attributes"].

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object
    bids_file : str
        Path to the BIDS file
    dataset_id : str
        Dataset identifier
    s3_base : str
        S3 base URL

    Returns
    -------
    dict
        Minimal metadata record with only core attributes

    """
    file_name = Path(bids_file).name

    # Construct S3 path based on dataset type
    # NEMAR datasets (nm*) use s3://nemar/nm000XXX/...
    # OpenNeuro datasets (ds*) use s3://openneuro.org/ds00XXXX/...
    openneuro_path = dataset_id + bids_file.split(dataset_id)[1]

    if dataset_id.startswith("nm"):
        # NEMAR datasets use s3://nemar/{dataset_id}/...
        s3_path = f"s3://nemar/{openneuro_path}"
    else:
        # OpenNeuro datasets use the provided s3_base
        s3_path = f"{s3_base.rstrip('/')}/{openneuro_path}"

    # Extract only the 11 core attributes
    record = {
        # Required fields
        "data_name": f"{dataset_id}_{file_name}",
        "dataset": dataset_id,
        "bidspath": s3_path,
        # BIDS entity fields
        "subject": bids_dataset.get_bids_file_attribute("subject", bids_file),
        "task": bids_dataset.get_bids_file_attribute("task", bids_file),
        "session": bids_dataset.get_bids_file_attribute("session", bids_file),
        "run": bids_dataset.get_bids_file_attribute("run", bids_file),
        "modality": bids_dataset.get_bids_file_attribute("modality", bids_file),
        # Technical metadata
        "sampling_frequency": bids_dataset.get_bids_file_attribute("sfreq", bids_file),
        "nchans": bids_dataset.get_bids_file_attribute("nchans", bids_file),
        "ntimes": bids_dataset.get_bids_file_attribute("ntimes", bids_file),
    }

    return record


def digest_single_dataset(
    dataset_id: str, cloned_dir: Path, output_dir: Path, s3_base: str
) -> dict[str, Any]:
    """Process a single dataset and generate minimal JSON.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    cloned_dir : Path
        Directory containing cloned datasets
    output_dir : Path
        Directory for output JSON files
    s3_base : str
        S3 base URL

    Returns
    -------
    dict
        Summary of the digestion process

    """
    dataset_dir = cloned_dir / dataset_id

    if not dataset_dir.exists():
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Dataset directory not found: {dataset_dir}",
            "record_count": 0,
            "error_count": 0,
        }

    print(f"[{dataset_id}] Processing...")

    try:
        # Extract minimal metadata
        records, errors = extract_minimal_metadata(dataset_id, dataset_dir, s3_base)

        # Create output directory
        dataset_output_dir = output_dir / dataset_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Save minimal JSON for MongoDB
        minimal_json = {
            "dataset": dataset_id,
            "record_count": len(records),
            "records": records,
        }

        minimal_path = dataset_output_dir / f"{dataset_id}_minimal.json"
        with minimal_path.open("w", encoding="utf-8") as f:
            json.dump(minimal_json, f, indent=2, default=_json_serializer)

        # Save summary
        summary = {
            "status": "success",
            "dataset_id": dataset_id,
            "record_count": len(records),
            "error_count": len(errors),
            "errors": errors if errors else None,
            "output_file": str(minimal_path),
            "processed_at": datetime.now().isoformat(),
        }

        summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[{dataset_id}] ✓ {len(records)} records, {len(errors)} errors")

        return summary

    except Exception as exc:
        error_summary = {
            "status": "error",
            "dataset_id": dataset_id,
            "error": str(exc),
            "record_count": 0,
            "error_count": 1,
            "processed_at": datetime.now().isoformat(),
        }

        print(f"[{dataset_id}] ✗ {str(exc)}")

        return error_summary


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
        description="Digest cloned BIDS datasets to extract minimal MongoDB metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cloned-dir",
        type=Path,
        required=True,
        help="Directory containing cloned BIDS datasets",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("digestion_output"),
        help="Directory for output JSON files (default: digestion_output)",
    )

    parser.add_argument(
        "--s3-base",
        type=str,
        default="s3://openneuro.org",
        help="S3 base URL (default: s3://openneuro.org)",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific dataset IDs to process (default: all in cloned-dir)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )

    args = parser.parse_args()

    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = args.datasets
    else:
        # Auto-detect from cloned directory
        datasets_to_process = [
            d.name
            for d in args.cloned_dir.iterdir()
            if d.is_dir() and (d.name.startswith("ds") or d.name.startswith("nm"))
        ]

    print("=" * 80)
    print("MINIMAL DATASET DIGESTION")
    print("=" * 80)
    print(f"Cloned directory: {args.cloned_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"S3 base URL: {args.s3_base}")
    print(f"Datasets to process: {len(datasets_to_process)}")
    print(f"Parallel workers: {args.workers}")
    print("=" * 80)
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process datasets
    start_time = datetime.now()
    summaries = []

    if args.workers == 1:
        # Sequential processing
        for dataset_id in datasets_to_process:
            summary = digest_single_dataset(
                dataset_id, args.cloned_dir, args.output_dir, args.s3_base
            )
            summaries.append(summary)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    digest_single_dataset,
                    dataset_id,
                    args.cloned_dir,
                    args.output_dir,
                    args.s3_base,
                ): dataset_id
                for dataset_id in datasets_to_process
            }

            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Calculate statistics
    successful = [s for s in summaries if s["status"] == "success"]
    failed = [s for s in summaries if s["status"] == "error"]
    total_records = sum(s["record_count"] for s in successful)
    total_errors = sum(s["error_count"] for s in summaries)

    # Save batch summary
    batch_summary = {
        "processed_at": end_time.isoformat(),
        "duration_seconds": duration,
        "total_datasets": len(datasets_to_process),
        "successful_datasets": len(successful),
        "failed_datasets": len(failed),
        "total_records": total_records,
        "total_errors": total_errors,
        "datasets": summaries,
    }

    batch_summary_path = args.output_dir / "batch_summary.json"
    with batch_summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    # Print results
    print()
    print("=" * 80)
    print("DIGESTION COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration:.1f} seconds")
    print(f"Datasets processed: {len(datasets_to_process)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"Total records extracted: {total_records}")
    print(f"Total errors: {total_errors}")
    print(f"\nBatch summary: {batch_summary_path}")
    print("=" * 80)

    if failed:
        print("\nFailed datasets:")
        for s in failed:
            print(f"  ✗ {s['dataset_id']}: {s.get('error', 'Unknown error')}")

    print("\n" + "=" * 80)
    print("NEXT STEPS: Upload to MongoDB")
    print("=" * 80)
    print("\nOption 1: Upload single dataset")
    print(
        "  curl -X POST http://137.110.244.65:3000/admin/eegdashstaging/records/bulk \\"
    )
    print("       -H 'Authorization: Bearer AdminWrite2025SecureToken' \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d @digestion_output/ds002718/ds002718_minimal.json")
    print("\nOption 2: Bulk upload all datasets (requires Python script)")
    print("  See: scripts/upload_to_mongodb.py")
    print("=" * 80)

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
