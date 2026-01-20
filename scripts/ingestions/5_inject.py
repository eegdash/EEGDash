#!/usr/bin/env python3
"""Inject digested datasets and records into MongoDB via API Gateway.

Upload Dataset and Record documents from digested datasets into separate MongoDB collections.

IMPORTANT: Validation is run automatically before injection to ensure data quality.
Use --skip-validation to bypass this check (not recommended).

Usage:
    # Inject all digested datasets to development
    python 5_inject.py --input digestion_output --database eegdash_dev

    # Inject to production
    python 5_inject.py --input digestion_output --database eegdash

    # Inject specific datasets
    python 5_inject.py --input digestion_output --database eegdash_dev --datasets ds002718 ds005506

    # Dry run (validate without uploading)
    python 5_inject.py --input digestion_output --database eegdash_dev --dry-run

    # Inject only datasets (skip records)
    python 5_inject.py --input digestion_output --database eegdash_dev --only-datasets

    # Inject only records (skip datasets)
    python 5_inject.py --input digestion_output --database eegdash_dev --only-records

    # Force injection even if unchanged
    python 5_inject.py --input digestion_output --database eegdash_dev --force

    # Skip validation (not recommended)
    python 5_inject.py --input digestion_output --database eegdash_dev --skip-validation
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from _fingerprint import fingerprint_from_records
from _http import (
    HTTPStatusError,
    RequestError,
    get_client,
    make_retry_client,
    request_json,
)
from _validate import validate_digestion_output
from tqdm import tqdm

# Datasets to explicitly ignore during ingestion
EXCLUDED_DATASETS = {
    "test",
    "ds003380",
    "ABUDUKADI",
    "ABUDUKADI_2",
    "ABUDUKADI_3",
    "ABUDUKADI_4",
    "AILIJIANG",
    "AILIJIANG_3",
    "AILIJIANG_4",
    "AILIJIANG_5",
    "AILIJIANG_7",
    "AILIJIANG_8",
    "BAIHETI",
    "BAIHETI_2",
    "BAIHETI_3",
    "BIAN_3",
    "BIN_27",
    "BLIX",
    "BOJIN",
    "BOUSSAGOL",
    "AISHENG",
    "ACHOLA",
    "ANASHKIN",
    "ANJUM",
    "BARBIERI",
    "BIN_8",
    "BIN_9",
    "BING_4",
    "BING_8",
    "BOWEN_4",
    "AZIZAH",
    "BAO",
    "BAO-YOU",
    "BAO_2",
    "BENABBOU",
    "BING",
    "BOXIN",
}

# Default API configuration
DEFAULT_API_URL = "https://data.eegdash.org"


def fetch_existing_dataset(
    api_url: str,
    database: str,
    dataset_id: str,
):
    """Fetch existing dataset metadata from the API (if present)."""
    url = f"{api_url}/api/{database}/datasets/{dataset_id}"
    data, response = request_json("get", url, timeout=30, client=get_client())
    if response is None:
        return None
    if response.status_code == 404:
        return None
    if response.status_code != 200 or data is None:
        return None
    return data.get("data", {})


def _make_session(auth_token: str):
    """Create session with retry strategy and auth."""
    return make_retry_client(auth_token)


def find_digested_datasets(
    input_dir: Path, datasets: list[str] | None = None
) -> list[Path]:
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

        # Skip special files/dirs or excluded datasets
        if (
            dataset_id.startswith(".")
            or dataset_id.startswith("_")
            or dataset_id in EXCLUDED_DATASETS
        ):
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
    Flattens entities to top-level fields for compatibility with EEGDash API.
    """
    dataset_id = dataset_dir.name

    # Try new schema first
    records_file = dataset_dir / f"{dataset_id}_records.json"
    if records_file.exists():
        with open(records_file) as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                records = data["records"]
            elif isinstance(data, list):
                records = data
            else:
                records = []
            return [_flatten_entities(r) for r in records]

    # Try legacy formats
    for legacy_name in [f"{dataset_id}_core.json", f"{dataset_id}_minimal.json"]:
        legacy_file = dataset_dir / legacy_name
        if legacy_file.exists():
            with open(legacy_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "records" in data:
                    records = data["records"]
                else:
                    records = []
                return [_flatten_entities(r) for r in records]

    return []


def _flatten_entities(record: dict) -> dict:
    """Flatten entities dict to top-level fields for EEGDash API compatibility.

    The EEGDash API expects subject, task, session, run at the top level,
    not nested in an entities dict.
    """
    result = record.copy()

    # Extract entities to top level if present
    entities = result.pop("entities", {})
    if entities:
        for key in ("subject", "task", "session", "run"):
            if key in entities and key not in result:
                result[key] = entities[key]

    # Keep entities_mne as-is for reference
    return result


def inject_datasets(
    datasets: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
    batch_size: int = 100,
    client=None,
) -> dict:
    """Upload Dataset documents to MongoDB via API Gateway.

    Returns
    -------
    dict
        Result with inserted_count

    """
    session = client or _make_session(admin_token)
    url = f"{api_url}/admin/{database}/datasets/bulk"

    inserted_count = 0
    errors = []

    def sanitize_for_json(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(i) for i in obj]
        return obj

    for i in range(0, len(datasets), batch_size):
        batch = datasets[i : i + batch_size]
        try:
            result, _response = request_json(
                "post",
                url,
                json_body=sanitize_for_json(batch),
                timeout=60,
                raise_for_status=True,
                raise_for_request=True,
                client=session,
            )
            inserted_count += (result or {}).get("inserted_count", len(batch))
        except (RequestError, HTTPStatusError) as e:
            errors.append(f"Batch {i // batch_size}: {e}")

    return {
        "inserted_count": inserted_count,
        "errors": errors,
    }


def inject_records(
    records: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
    batch_size: int = 1000,
    client=None,
) -> dict:
    """Upload Record documents to MongoDB via API Gateway.

    Returns
    -------
    dict
        Result with inserted_count

    """
    session = client or _make_session(admin_token)
    # Use the new upsert endpoint
    url = f"{api_url}/admin/{database}/records/upsert"

    inserted_count = 0
    updated_count = 0
    errors = []

    # Batch insert records
    import math

    def sanitize_for_json(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(i) for i in obj]
        return obj

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def inject_batch(batch_idx, batch):
        try:
            result, _ = request_json(
                "post",
                url,
                json_body=sanitize_for_json(batch),
                timeout=60,
                raise_for_status=True,
                raise_for_request=True,
                client=session,
            )
            return {
                "inserted": (result or {}).get("inserted_count", 0),
                "updated": (result or {}).get("updated_count", 0),
                "error": None,
            }
        except (RequestError, HTTPStatusError) as e:
            return {"inserted": 0, "updated": 0, "error": f"Batch {batch_idx}: {e}"}

    # Prepare batches
    batches = []
    for i in range(0, len(records), batch_size):
        batches.append((i // batch_size, records[i : i + batch_size]))

    # Parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(inject_batch, idx, batch): idx for idx, batch in batches
        }
        for future in tqdm(
            as_completed(futures), total=len(batches), desc="Injecting batches"
        ):
            res = future.result()
            if res["error"]:
                errors.append(res["error"])
            else:
                inserted_count += res["inserted"]
                updated_count += res["updated"]

    return {
        "inserted_count": inserted_count,
        "updated_count": updated_count,
        "errors": errors,
    }


def _ensure_fingerprint(dataset_id: str, dataset: dict | None, records: list[dict]):
    """Ensure ingestion_fingerprint is set on dataset or derived from records."""
    if dataset is None:
        dataset = {"dataset_id": dataset_id}
    if dataset.get("ingestion_fingerprint"):
        return dataset
    if records:
        dataset["ingestion_fingerprint"] = fingerprint_from_records(
            dataset_id,
            dataset.get("source", "unknown"),
            records,
        )
    return dataset


def filter_changed_datasets(
    dataset_ids: list[str],
    datasets_by_id: dict[str, dict],
    records_by_id: dict[str, list[dict]],
    api_url: str,
    database: str,
):
    """Return dataset IDs that are new or updated, plus skipped IDs."""
    changed_ids: list[str] = []
    skipped_ids: list[str] = []

    for dataset_id in dataset_ids:
        dataset = datasets_by_id.get(dataset_id)
        records = records_by_id.get(dataset_id, [])
        dataset = _ensure_fingerprint(dataset_id, dataset, records)
        datasets_by_id[dataset_id] = dataset

        existing = fetch_existing_dataset(api_url, database, dataset_id)
        existing_fp = (existing or {}).get("ingestion_fingerprint")
        current_fp = dataset.get("ingestion_fingerprint")

        if not existing:
            changed_ids.append(dataset_id)
            continue
        if existing_fp and current_fp and existing_fp == current_fp:
            skipped_ids.append(dataset_id)
            continue
        changed_ids.append(dataset_id)

    return changed_ids, skipped_ids


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
        choices=[
            "eegdash",
            "eegdash_dev",
            "eegdash_archive",
            "eegdash_staging",
            "eegdash_v1",
        ],
        help=(
            "Target MongoDB database (eegdash=production, "
            "eegdash_dev=development, eegdash_archive=old data, "
            "eegdash_staging=staging, eegdash_v1=legacy)"
        ),
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Inject even if ingestion_fingerprint matches existing dataset",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation before injection (not recommended)",
    )
    parser.add_argument(
        "--data-quality-threshold",
        type=float,
        default=10.0,
        help="Max percentage of records with missing nchans/sampling_frequency before warning (default: 10%%)",
    )

    args = parser.parse_args()

    # Validate args
    if args.only_datasets and args.only_records:
        print(
            "Error: Cannot use both --only-datasets and --only-records", file=sys.stderr
        )
        return 1

    # Get admin token (validated later if injection is needed)
    admin_token = args.token or os.environ.get("EEGDASH_ADMIN_TOKEN")

    # Run validation first (unless explicitly skipped)
    if not args.skip_validation:
        print("Running validation...")
        validation_result = validate_digestion_output(args.input, verbose=False)
        print(validation_result.summary())

        # Check for critical errors
        if not validation_result.is_valid():
            print(
                "\nValidation FAILED - fix errors before injection or use --skip-validation",
                file=sys.stderr,
            )
            return 1

        # Check data quality threshold
        total_records = validation_result.stats["records_checked"]
        missing_nchans = validation_result.stats["missing_nchans"]
        missing_sampling_frequency = validation_result.stats[
            "missing_sampling_frequency"
        ]

        if total_records > 0:
            nchans_pct = missing_nchans / total_records * 100
            sampling_frequency_pct = missing_sampling_frequency / total_records * 100

            if nchans_pct > args.data_quality_threshold:
                print(
                    f"\nWARNING: {nchans_pct:.1f}% of records missing nchans "
                    f"(threshold: {args.data_quality_threshold}%)",
                    file=sys.stderr,
                )
                if not args.dry_run:
                    print(
                        "Use --skip-validation to proceed anyway, or fix the data first.",
                        file=sys.stderr,
                    )
                    return 1

            if sampling_frequency_pct > args.data_quality_threshold:
                print(
                    f"\nWARNING: {sampling_frequency_pct:.1f}% of records missing sampling_frequency "
                    f"(threshold: {args.data_quality_threshold}%)",
                    file=sys.stderr,
                )
                if not args.dry_run:
                    print(
                        "Use --skip-validation to proceed anyway, or fix the data first.",
                        file=sys.stderr,
                    )
                    return 1

        print("\nValidation PASSED - proceeding with injection\n")

    # Find dataset directories
    dataset_dirs = find_digested_datasets(args.input, args.datasets)
    print(f"Found {len(dataset_dirs)} datasets to inject")

    if not dataset_dirs:
        print("No digested datasets found.")
        return 0

    # Collect all documents
    all_datasets = []
    all_records = []
    dataset_docs: dict[str, dict] = {}
    errors = []

    print("\nLoading documents...")
    for dataset_dir in tqdm(dataset_dirs, desc="Loading"):
        dataset_id = dataset_dir.name

        try:
            # Load record documents
            records = load_records(dataset_dir)

            # Skip if empty (unless force-including empty datasets)
            if not records and not args.only_datasets:
                # print(f"  Skipping empty dataset {dataset_id}", file=sys.stderr)
                continue

            if not args.only_datasets:
                all_records.extend(records)

            # Load dataset document (only if records found or forced)
            dataset = load_dataset(dataset_dir)
            if dataset:
                dataset_docs[dataset_id] = dataset
                if not args.only_records:
                    all_datasets.append(dataset)

        except Exception as e:
            errors.append({"dataset": dataset_id, "error": str(e)})
            print(f"  Error loading {dataset_id}: {e}", file=sys.stderr)

    print(f"\nLoaded {len(all_datasets)} datasets and {len(all_records)} records")

    datasets_by_id = {ds_id: ds for ds_id, ds in dataset_docs.items() if ds_id and ds}
    records_by_id: dict[str, list[dict]] = {}
    for record in all_records:
        dataset_id = record.get("dataset")
        if not dataset_id:
            continue
        records_by_id.setdefault(dataset_id, []).append(record)

    dataset_ids = sorted(set(datasets_by_id) | set(records_by_id))

    for dataset_id in dataset_ids:
        dataset = datasets_by_id.get(dataset_id)
        records = records_by_id.get(dataset_id, [])
        datasets_by_id[dataset_id] = _ensure_fingerprint(dataset_id, dataset, records)

    skipped_ids: list[str] = []
    if not args.force:
        changed_ids, skipped_ids = filter_changed_datasets(
            dataset_ids,
            datasets_by_id,
            records_by_id,
            args.api_url,
            args.database,
        )
        changed_set = set(changed_ids)
        all_datasets = [
            datasets_by_id[ds_id] for ds_id in changed_ids if ds_id in datasets_by_id
        ]
        all_records = [r for r in all_records if r.get("dataset") in changed_set]
        print(
            f"Filtered to {len(changed_ids)} changed/new datasets "
            f"(skipped {len(skipped_ids)} unchanged)"
        )

        if not changed_ids:
            print("No updated datasets detected. Skipping injection.")
            return 0

    # Stats tracking
    stats = {
        "datasets_injected": 0,
        "records_injected": 0,
        "errors": len(errors),
        "datasets_skipped": len(skipped_ids),
    }

    if args.dry_run:
        print("\n[DRY RUN] Would inject:")
        print(f"  - {len(all_datasets)} datasets to {args.database}.datasets")
        print(f"  - {len(all_records)} records to {args.database}.records")
        stats["datasets_injected"] = len(all_datasets)
        stats["records_injected"] = len(all_records)

    else:
        if not admin_token:
            print(
                "Error: Admin token required. Set EEGDASH_ADMIN_TOKEN or use --token",
                file=sys.stderr,
            )
            return 1
        # Inject datasets
        if all_datasets and not args.only_records:
            print(f"\nInjecting {len(all_datasets)} datasets...")
            with _make_session(admin_token) as client:
                # Use smaller batch size for datasets to avoid timeouts
                ds_batch_size = 20
                for i in range(0, len(all_datasets), ds_batch_size):
                    try:
                        batch = all_datasets[i : i + ds_batch_size]
                        result = inject_datasets(
                            batch,
                            args.api_url,
                            args.database,
                            admin_token,
                            client=client,
                        )
                        stats["datasets_injected"] += result.get("inserted_count", 0)
                        print(
                            f"  Batch {i // ds_batch_size + 1}: {result.get('inserted_count', 0)} datasets"
                        )
                    except Exception as e:
                        stats["errors"] += 1
                        errors.append({"dataset": "datasets_batch", "error": str(e)})
                        print(
                            f"  Error injecting dataset batch {i // ds_batch_size + 1}: {e}",
                            file=sys.stderr,
                        )

        # Inject records
        if all_records and not args.only_datasets:
            print(f"\nInjecting {len(all_records)} records...")
            try:
                with _make_session(admin_token) as client:
                    result = inject_records(
                        all_records,
                        args.api_url,
                        args.database,
                        admin_token,
                        batch_size=args.batch_size,
                        client=client,
                    )
                    stats["records_injected"] += result.get("inserted_count", 0)
                    stats["records_updated"] = stats.get(
                        "records_updated", 0
                    ) + result.get("updated_count", 0)

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
    print(f"  Records Ins:{stats['records_injected']}")
    print(f"  Records Upd:{stats.get('records_updated', 0)}")
    print(f"  Skipped:    {stats['datasets_skipped']}")
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
