#!/usr/bin/env python3
"""Inject digested datasets and records into MongoDB via API Gateway.

Upload Dataset and Record documents from digested datasets into separate MongoDB collections.

Usage:
    # Inject all digested datasets to staging
    python 4_inject.py --input digestion_output --database eegdashstaging

    # Inject to production
    python 4_inject.py --input digestion_output --database eegdash

    # Inject specific datasets
    python 4_inject.py --input digestion_output --database eegdashstaging --datasets ds002718 ds005506

    # Dry run (validate without uploading)
    python 4_inject.py --input digestion_output --database eegdashstaging --dry-run

    # Inject only datasets (skip records)
    python 4_inject.py --input digestion_output --database eegdashstaging --only-datasets

    # Inject only records (skip datasets)
    python 4_inject.py --input digestion_output --database eegdashstaging --only-records
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Default API configuration
DEFAULT_API_URL = "https://data.eegdash.org"


def find_digested_datasets(input_dir: Path, datasets: list[str] | None = None) -> list[Path]:
    """Find all dataset directories in the digestion output.

    Returns
    -------
    list[Path]
        List of dataset directories containing _dataset.json and _records.json files
    """
    dataset_dirs = []

    for dataset_dir in sorted(input_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name

        # Skip special files/dirs
        if dataset_id.startswith(".") or dataset_id.startswith("_"):
            continue

        # Filter by specific datasets if provided
        if datasets and dataset_id not in datasets:
            continue

        # Check for either dataset or records file (new schema)
        dataset_file = dataset_dir / f"{dataset_id}_dataset.json"
        records_file = dataset_dir / f"{dataset_id}_records.json"

        if dataset_file.exists() or records_file.exists():
            dataset_dirs.append(dataset_dir)

    return dataset_dirs


def load_dataset(dataset_dir: Path) -> dict | None:
    """Load a Dataset document from a directory."""
    dataset_id = dataset_dir.name
    dataset_file = dataset_dir / f"{dataset_id}_dataset.json"

    if not dataset_file.exists():
        return None

    with open(dataset_file) as f:
        return json.load(f)


def load_records(dataset_dir: Path) -> list[dict]:
    """Load Records from a directory.

    Supports both new schema (_records.json) and legacy formats.
    """
    dataset_id = dataset_dir.name

    # Try new schema first
    records_file = dataset_dir / f"{dataset_id}_records.json"
    if records_file.exists():
        with open(records_file) as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                return data["records"]
            elif isinstance(data, list):
                return data

    # Try legacy formats
    for legacy_name in [f"{dataset_id}_core.json", f"{dataset_id}_minimal.json"]:
        legacy_file = dataset_dir / legacy_name
        if legacy_file.exists():
            with open(legacy_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "records" in data:
                    return data["records"]

    return []


def inject_datasets(
    datasets: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
) -> dict:
    """Upload Dataset documents to MongoDB via API Gateway.

    Returns
    -------
    dict
        Result with inserted_count
    """
    from eegdash.http_api_client import HTTPAPICollection

    collection = HTTPAPICollection(
        base_url=f"{api_url}/admin/{database}/datasets",
        admin_token=admin_token,
    )

    result = collection.insert_many(datasets)

    return {
        "inserted_count": result.inserted_count,
    }


def inject_records(
    records: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
) -> dict:
    """Upload Record documents to MongoDB via API Gateway.

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
        description="Inject digested datasets and records into MongoDB via API Gateway."
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
    parser.add_argument(
        "--only-datasets",
        action="store_true",
        help="Only inject Dataset documents (skip Records)",
    )
    parser.add_argument(
        "--only-records",
        action="store_true",
        help="Only inject Record documents (skip Datasets)",
    )

    args = parser.parse_args()

    # Validate args
    if args.only_datasets and args.only_records:
        print("Error: Cannot use both --only-datasets and --only-records", file=sys.stderr)
        return 1

    # Get admin token
    admin_token = args.token or os.environ.get("EEGDASH_ADMIN_TOKEN")
    if not admin_token and not args.dry_run:
        print("Error: Admin token required. Set EEGDASH_ADMIN_TOKEN or use --token", file=sys.stderr)
        return 1

    # Find dataset directories
    dataset_dirs = find_digested_datasets(args.input, args.datasets)
    print(f"Found {len(dataset_dirs)} datasets to inject")

    if not dataset_dirs:
        print("No digested datasets found.")
        return 0

    # Collect all documents
    all_datasets = []
    all_records = []
    errors = []

    print("\nLoading documents...")
    for dataset_dir in tqdm(dataset_dirs, desc="Loading"):
        dataset_id = dataset_dir.name

        try:
            # Load dataset document
            if not args.only_records:
                dataset = load_dataset(dataset_dir)
                if dataset:
                    all_datasets.append(dataset)

            # Load record documents
            if not args.only_datasets:
                records = load_records(dataset_dir)
                all_records.extend(records)

        except Exception as e:
            errors.append({"dataset": dataset_id, "error": str(e)})
            print(f"  Error loading {dataset_id}: {e}", file=sys.stderr)

    print(f"\nLoaded {len(all_datasets)} datasets and {len(all_records)} records")

    # Stats tracking
    stats = {
        "datasets_injected": 0,
        "records_injected": 0,
        "errors": len(errors),
    }

    if args.dry_run:
        print("\n[DRY RUN] Would inject:")
        print(f"  - {len(all_datasets)} datasets to {args.database}.datasets")
        print(f"  - {len(all_records)} records to {args.database}.records")
        stats["datasets_injected"] = len(all_datasets)
        stats["records_injected"] = len(all_records)

    else:
        # Inject datasets
        if all_datasets and not args.only_records:
            print(f"\nInjecting {len(all_datasets)} datasets...")
            try:
                for i in range(0, len(all_datasets), args.batch_size):
                    batch = all_datasets[i : i + args.batch_size]
                    result = inject_datasets(batch, args.api_url, args.database, admin_token)
                    stats["datasets_injected"] += result["inserted_count"]
                    print(f"  Batch {i // args.batch_size + 1}: {result['inserted_count']} datasets")
            except Exception as e:
                stats["errors"] += 1
                errors.append({"dataset": "datasets_collection", "error": str(e)})
                print(f"  Error injecting datasets: {e}", file=sys.stderr)

        # Inject records
        if all_records and not args.only_datasets:
            print(f"\nInjecting {len(all_records)} records...")
            try:
                for i in range(0, len(all_records), args.batch_size):
                    batch = all_records[i : i + args.batch_size]
                    result = inject_records(batch, args.api_url, args.database, admin_token)
                    stats["records_injected"] += result["inserted_count"]
                    if (i // args.batch_size) % 10 == 0:
                        print(f"  Progress: {i + len(batch)} / {len(all_records)} records")
            except Exception as e:
                stats["errors"] += 1
                errors.append({"dataset": "records_collection", "error": str(e)})
                print(f"  Error injecting records: {e}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print("INJECTION SUMMARY")
    print("=" * 60)
    print(f"  Database:   {args.database}")
    print(f"  Datasets:   {stats['datasets_injected']}")
    print(f"  Records:    {stats['records_injected']}")
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
