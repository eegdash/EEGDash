#!/usr/bin/env python3
"""Inject digested records into MongoDB via API Gateway.

Upload core metadata from digested datasets into MongoDB.

Usage:
    # Inject all digested datasets to staging
    python 4_inject.py --input digestion_output --database eegdashstaging

    # Inject to production
    python 4_inject.py --input digestion_output --database eegdash

    # Inject specific datasets
    python 4_inject.py --input digestion_output --database eegdashstaging --datasets ds002718 ds005506

    # Dry run (validate without uploading)
    python 4_inject.py --input digestion_output --database eegdashstaging --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Default API configuration
DEFAULT_API_URL = "https://data.eegdash.org"


def find_record_files(input_dir: Path, datasets: list[str] | None = None) -> list[Path]:
    """Find all record JSON files in the digestion output.

    Looks for:
    - {dataset_id}_records.json (from minimal mode, new schema)
    - {dataset_id}_core.json (from full mode)
    - {dataset_id}_minimal.json (legacy)
    """
    record_files = []

    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name

        # Filter by specific datasets if provided
        if datasets and dataset_id not in datasets:
            continue

        # Look for record files in order of preference
        records_path = dataset_dir / f"{dataset_id}_records.json"  # New schema
        core_path = dataset_dir / f"{dataset_id}_core.json"  # Full mode
        minimal_path = dataset_dir / f"{dataset_id}_minimal.json"  # Legacy

        if records_path.exists():
            record_files.append(records_path)
        elif core_path.exists():
            record_files.append(core_path)
        elif minimal_path.exists():
            record_files.append(minimal_path)

    return record_files


def load_records(json_path: Path) -> list[dict]:
    """Load records from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "records" in data:
        return data["records"]
    else:
        raise ValueError(f"Unexpected format in {json_path}")


def inject_records(
    records: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
) -> dict:
    """Upload records to MongoDB via API Gateway.

    Returns
    -------
    dict
        Result with inserted_count
    """
    from eegdash.http_api_client import HTTPAPICollection

    collection = HTTPAPICollection(
        base_url=f"{api_url}/admin/{database}/records",
        admin_token=admin_token,
    )

    result = collection.insert_many(records)

    return {
        "inserted_count": result.inserted_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inject digested records into MongoDB via API Gateway."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("digestion_output"),
        help="Directory containing digested datasets (default: digestion_output/)",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        choices=["eegdashstaging", "eegdash"],
        help="Target MongoDB database (eegdashstaging or eegdash)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"EEGDash API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Admin token (default: from EEGDASH_ADMIN_TOKEN env var)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset IDs to inject (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files without uploading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Maximum records per API request (default: 1000)",
    )

    args = parser.parse_args()

    # Get admin token
    admin_token = args.token or os.environ.get("EEGDASH_ADMIN_TOKEN")
    if not admin_token and not args.dry_run:
        print("Error: Admin token required. Set EEGDASH_ADMIN_TOKEN or use --token", file=sys.stderr)
        return 1

    # Find record files
    record_files = find_record_files(args.input, args.datasets)
    print(f"Found {len(record_files)} datasets to inject")

    if not record_files:
        print("No record files found.")
        return 0

    # Process files
    stats = {"datasets": 0, "records": 0, "errors": 0}
    errors = []

    for json_path in tqdm(record_files, desc="Injecting"):
        dataset_id = json_path.parent.name

        try:
            records = load_records(json_path)

            if args.dry_run:
                print(f"  [DRY RUN] {dataset_id}: {len(records)} records")
                stats["datasets"] += 1
                stats["records"] += len(records)
            else:
                # Upload in batches
                for i in range(0, len(records), args.batch_size):
                    batch = records[i : i + args.batch_size]
                    result = inject_records(batch, args.api_url, args.database, admin_token)
                    stats["records"] += result["inserted_count"]

                stats["datasets"] += 1

        except Exception as e:
            stats["errors"] += 1
            errors.append({"dataset": dataset_id, "error": str(e)})
            print(f"  Error processing {dataset_id}: {e}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print("INJECTION SUMMARY")
    print("=" * 60)
    print(f"  Database:   {args.database}")
    print(f"  Datasets:   {stats['datasets']}")
    print(f"  Records:    {stats['records']}")
    print(f"  Errors:     {stats['errors']}")

    if args.dry_run:
        print("\n  [DRY RUN - no data uploaded]")

    print("=" * 60)

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err['dataset']}: {err['error']}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
