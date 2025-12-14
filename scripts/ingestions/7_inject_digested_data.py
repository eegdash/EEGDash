#!/usr/bin/env python3
"""Inject digested EEGDash data into MongoDB via API Gateway.

This script uploads core metadata from digested datasets into MongoDB
using the existing HTTPAPICollection.insert_many() method.

Usage:
    python 7_inject_digested_data.py \
        --api-url https://data.eegdash.org \
        --database eegdashstaging \
        --digestion-dir digestion_output
"""

import argparse
import json
import sys
from pathlib import Path
from time import time

from tqdm import tqdm

# Use the existing HTTP API client
from eegdash.http_api_client import HTTPAPICollection


def inject_dataset(
    collection: HTTPAPICollection,
    dataset_id: str,
    core_json_path: Path,
) -> dict:
    """Upload core metadata for a single dataset to MongoDB.

    Parameters
    ----------
    collection : HTTPAPICollection
        The API collection to insert into
    dataset_id : str
        Dataset ID (e.g., "ds003768")
    core_json_path : Path
        Path to the ds*_core.json file

    Returns
    -------
    dict
        Result including inserted_count

    """
    with open(core_json_path) as f:
        records = json.load(f)

    result = collection.insert_many(records)

    return {
        "dataset_id": dataset_id,
        "inserted_count": result.inserted_count,
    }


def inject_all_datasets(
    api_url: str,
    database: str,
    admin_token: str,
    digestion_dir: Path,
) -> dict:
    """Inject all digested datasets into MongoDB.

    Parameters
    ----------
    api_url : str
        EEGDash API URL
    database : str
        Target MongoDB database name
    admin_token : str
        Admin authentication token
    digestion_dir : Path
        Directory containing digested dataset files

    Returns
    -------
    dict
        Summary of injection results

    """
    # Create the collection interface using existing class
    collection = HTTPAPICollection(
        api_url=api_url,
        database=database,
        collection="records",
        auth_token=admin_token,
        is_admin=True,
    )

    # Find all core.json files
    core_files = sorted(digestion_dir.glob("*_core.json"))

    if not core_files:
        raise ValueError(f"No core JSON files found in {digestion_dir}")

    print(f"Found {len(core_files)} datasets to inject")
    print(f"Target: {api_url}/admin/{database}")
    print()

    results = {
        "total": len(core_files),
        "success": 0,
        "failed": 0,
        "total_records": 0,
        "start_time": time(),
        "datasets": [],
        "errors": [],
    }

    print(f"Starting injection of {len(core_files)} datasets...")
    print("=" * 70)

    # Process datasets sequentially (API stability)
    with tqdm(total=len(core_files), desc="Injecting datasets") as pbar:
        for core_file in core_files:
            dataset_id = core_file.stem.replace("_core", "")
            try:
                result = inject_dataset(collection, dataset_id, core_file)
                results["success"] += 1
                inserted = result.get("inserted_count", 0)
                results["total_records"] += inserted
                results["datasets"].append(
                    {
                        "dataset": dataset_id,
                        "status": "success",
                        "records_inserted": inserted,
                    }
                )
                pbar.set_description(f"{dataset_id}: ✓ {inserted} records")

            except Exception as e:
                results["failed"] += 1
                error_msg = str(e)
                results["errors"].append(
                    {
                        "dataset": dataset_id,
                        "error": error_msg,
                    }
                )
                results["datasets"].append(
                    {
                        "dataset": dataset_id,
                        "status": "failed",
                        "error": error_msg[:100],
                    }
                )
                pbar.set_description(f"{dataset_id}: ✗ {error_msg[:30]}")

            pbar.update(1)

    results["end_time"] = time()
    results["duration_seconds"] = results["end_time"] - results["start_time"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Inject digested EEGDash data into MongoDB."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="https://data.eegdash.org",
        help="EEGDash API URL",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="eegdashstaging",
        help="Target MongoDB database name",
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        default="AdminWrite2025SecureToken",
        help="Admin authentication token",
    )
    parser.add_argument(
        "--digestion-dir",
        type=Path,
        default=Path("digestion_output"),
        help="Directory with digested datasets",
    )

    args = parser.parse_args()

    try:
        results = inject_all_datasets(
            api_url=args.api_url,
            database=args.database,
            admin_token=args.admin_token,
            digestion_dir=args.digestion_dir,
        )

        # Print summary
        print()
        print("=" * 70)
        print("INJECTION SUMMARY")
        print("=" * 70)
        print(f"Total datasets:     {results['total']}")
        print(f"Successful:         {results['success']}")
        print(f"Failed:             {results['failed']}")
        print(f"Total records:      {results['total_records']}")
        print(f"Duration:           {results['duration_seconds']:.1f}s")
        if results["success"] > 0:
            print(
                f"Avg records/dataset: {results['total_records'] / results['success']:.0f}"
            )
        print()

        # Show failed datasets
        if results["failed"] > 0:
            print("Failed datasets:")
            for d in results["errors"][:10]:
                print(f"  - {d['dataset']}: {d['error'][:60]}")
            if len(results["errors"]) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")
            print()

        # Save full results
        summary_path = args.digestion_dir / "INJECTION_SUMMARY.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full summary saved to: {summary_path}")
        print()
        print(f"Data injected into: {args.api_url}/admin/{args.database}")
        print(f"Query: curl {args.api_url}/{args.database}/records")

        return 0 if results["failed"] == 0 else 1

    except Exception as e:
        print(f"✗ Injection failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
