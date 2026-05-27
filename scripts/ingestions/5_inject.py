#!/usr/bin/env python3
"""Inject digested datasets, records, and montages into MongoDB via API Gateway.

Upload Dataset, Record, and Montage documents from digested datasets into
separate MongoDB collections. Montages are deduplicated by ``hash`` across
datasets (two datasets that share a cap only upload the layout once).

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

    # Inject only datasets (skip records and montages)
    python 5_inject.py --input digestion_output --database eegdash_dev --only-datasets

    # Inject only records (skip datasets and montages)
    python 5_inject.py --input digestion_output --database eegdash_dev --only-records

    # Inject only montages (skip datasets and records)
    python 5_inject.py --input digestion_output --database eegdash_dev --only-montages

    # Skip the montage-registry leg (datasets + records only)
    python 5_inject.py --input digestion_output --database eegdash_dev --skip-montages

    # Force injection even if unchanged
    python 5_inject.py --input digestion_output --database eegdash_dev --force

    # Skip validation (not recommended)
    python 5_inject.py --input digestion_output --database eegdash_dev --skip-validation
"""

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm

from _http import (
    HTTPStatusError,
    RequestError,
    make_authed_client,
    request_json,
)
from _inject_plan import (
    EXCLUDED_DATASETS,
    _ensure_fingerprint,
    _flatten_entities,
    build_injection_plan,
    fetch_existing_dataset,
    filter_changed_datasets,
    find_digested_datasets,
    load_dataset,
    load_montages,
    load_records,
)

# Default API configuration
DEFAULT_API_URL = "https://data.eegdash.org"

__all__ = [
    "EXCLUDED_DATASETS",
    "_ensure_fingerprint",
    "_flatten_entities",
    "fetch_existing_dataset",
    "filter_changed_datasets",
    "find_digested_datasets",
    "load_dataset",
    "load_montages",
    "load_records",
]


def _default_workers() -> int:
    """Pick a worker pool size from the CPU count, capped at 8.

    Returns
    -------
    int
        ``min(8, max(2, (os.cpu_count() or 2) * 2))``.

    Notes
    -----
    Replaces the hard-coded ``max_workers=8``. CI workers run on anything
    from 2 vCPUs (GitHub free tier) to 16+ vCPUs (self-hosted runners); a
    single magic constant is wrong for both extremes. The cap at 8
    matches the previous behaviour for hosts with >= 4 vCPUs so this
    is a non-regression on the original CI; smaller hosts now get a
    sensible smaller pool.
    """
    return min(8, max(2, (os.cpu_count() or 2) * 2))


def _sanitize_for_json(obj):
    """Sanitize object for JSON serialization, handling NaN/Inf floats."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    return obj


def _bulk_upsert_batch(
    batch_idx: int,
    batch: list,
    url: str,
    session,
    *,
    timeout: float,
) -> dict:
    """Worker for parallel bulk upserts (records or montages)."""
    try:
        result, _ = request_json(
            "post",
            url,
            json_body=_sanitize_for_json(batch),
            timeout=timeout,
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


def _make_session(auth_token: str):
    """Create an authed HTTP client. Retries inject at the request site."""
    return make_authed_client(auth_token)


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

    for i in range(0, len(datasets), batch_size):
        batch = datasets[i : i + batch_size]
        try:
            result, _response = request_json(
                "post",
                url,
                json_body=_sanitize_for_json(batch),
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

    # Prepare batches
    batches = []
    for i in range(0, len(records), batch_size):
        batches.append((i // batch_size, records[i : i + batch_size]))

    # Parallel execution. Each batch has a 60s per-request timeout via the
    # inner _bulk_upsert_batch; as_completed gets a generous wall-clock
    # safety margin so a *hung* worker (TCP keepalive never trips) doesn't
    # deadlock the main thread indefinitely.
    per_batch_timeout_s = 60.0
    wall_clock_budget_s = max(120.0, len(batches) * per_batch_timeout_s + 30.0)
    with ThreadPoolExecutor(max_workers=_default_workers()) as executor:
        futures = {
            executor.submit(
                _bulk_upsert_batch,
                idx,
                batch,
                url,
                session,
                timeout=per_batch_timeout_s,
            ): idx
            for idx, batch in batches
        }
        for future in tqdm(
            as_completed(futures, timeout=wall_clock_budget_s),
            total=len(batches),
            desc="Injecting batches",
        ):
            res = future.result(timeout=per_batch_timeout_s)
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


def inject_montages(
    montages: list[dict],
    api_url: str,
    database: str,
    admin_token: str,
    batch_size: int = 100,
    client=None,
) -> dict:
    """Upload Montage documents to the registry via the bulk upsert endpoint.

    Each doc must carry a ``hash`` key — the endpoint idempotently upserts
    by hash, keeping ``first_seen`` / ``representative_dataset`` /
    ``representative_subject`` set only on the first insert (``$setOnInsert``).
    Payloads are capped at 500 per the server's validator; ``batch_size``
    defaults to 100 to keep request bodies under a few MB.
    """
    session = client or _make_session(admin_token)
    url = f"{api_url}/admin/{database}/montages/bulk"

    inserted_count = 0
    updated_count = 0
    errors: list[str] = []

    batches = []
    for i in range(0, len(montages), batch_size):
        batches.append((i // batch_size, montages[i : i + batch_size]))

    # Same timeout discipline as inject_records.
    per_batch_timeout_s = 120.0
    wall_clock_budget_s = max(180.0, len(batches) * per_batch_timeout_s + 30.0)
    with ThreadPoolExecutor(max_workers=_default_workers()) as executor:
        futures = {
            executor.submit(
                _bulk_upsert_batch,
                idx,
                batch,
                url,
                session,
                timeout=per_batch_timeout_s,
            ): idx
            for idx, batch in batches
        }
        for future in tqdm(
            as_completed(futures, timeout=wall_clock_budget_s),
            total=len(batches),
            desc="Injecting montages",
        ):
            res = future.result(timeout=per_batch_timeout_s)
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


def main():
    # CLI + env var parsing + validation, all in one place (C6.5).
    # The 95 lines of argparse boilerplate + 25 lines of mutually-
    # exclusive-flag checks that used to live here are now declarative
    # fields + @model_validator hooks on InjectConfig. Tests construct
    # InjectConfig directly; this function stays the only argv-bound
    # entry point.
    from pydantic import ValidationError

    from _inject_config import load_inject_config_from_argv

    try:
        args = load_inject_config_from_argv()
    except ValidationError as exc:
        # Render validation errors as a clean per-field list rather than
        # pydantic's verbose default. Preserves the "exit 1 on bad
        # config" contract the previous argparse layer had.
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    admin_token = args.token  # already env-fallback'd via AliasChoices

    # Run validation first (unless explicitly skipped)
    if not args.skip_validation:
        # Lazy import: pulls eegdash.schemas transitively, which we don't
        # want to require on inject-only hosts.
        from _validate import validate_digestion_output

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

    want_datasets = not args.only_records and not args.only_montages
    want_records = not args.only_datasets and not args.only_montages
    want_montages = (
        not args.only_datasets and not args.only_records and not args.skip_montages
    ) or args.only_montages

    print("\nLoading documents...")
    plan = build_injection_plan(
        dataset_dirs,
        want_datasets=want_datasets,
        want_records=want_records,
        want_montages=want_montages,
        force=args.force,
        only_montages=args.only_montages,
        api_url=args.api_url,
        database=args.database,
        progress=lambda dirs: tqdm(dirs, desc="Loading"),
    )
    all_datasets = plan.datasets
    all_records = plan.records
    all_montages = plan.montages
    errors = list(plan.errors)

    print(
        f"\nLoaded {len(all_datasets)} datasets, {len(all_records)} records, "
        f"{len(all_montages)} unique montages "
        f"({plan.duplicate_montage_sightings} cross-dataset duplicates collapsed)"
    )

    if not args.force and not args.only_montages:
        print(
            f"Filtered to {len(plan.changed_ids)} changed/new datasets "
            f"(skipped {len(plan.skipped_ids)} unchanged)"
        )

        if not plan.changed_ids and not all_montages:
            print("No updated datasets detected. Skipping injection.")
            return 0

    # Stats tracking
    stats = {
        "datasets_injected": 0,
        "records_injected": 0,
        "records_updated": 0,
        "montages_injected": 0,
        "montages_updated": 0,
        "errors": len(errors),
        "datasets_skipped": len(plan.skipped_ids),
    }

    if args.dry_run:
        print("\n[DRY RUN] Would inject:")
        print(f"  - {len(all_datasets)} datasets to {args.database}.datasets")
        print(f"  - {len(all_records)} records to {args.database}.records")
        print(f"  - {len(all_montages)} montages to {args.database}.montages")
        stats["datasets_injected"] = len(all_datasets)
        stats["records_injected"] = len(all_records)
        stats["montages_injected"] = len(all_montages)

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
                    except (
                        httpx.RequestError,
                        httpx.HTTPStatusError,
                        ValueError,
                        KeyError,
                    ) as e:
                        # Network failure or malformed batch. Record the
                        # error; do not abort the entire injection.
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
                    stats["records_updated"] += result.get("updated_count", 0)

            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
                KeyError,
            ) as e:
                stats["errors"] += 1
                errors.append({"dataset": "records_collection", "error": str(e)})
                print(f"  Error injecting records: {e}", file=sys.stderr)

        # Inject montages — deduplicated by hash across all datasets.
        if all_montages and want_montages:
            print(f"\nInjecting {len(all_montages)} unique montages...")
            try:
                with _make_session(admin_token) as client:
                    result = inject_montages(
                        all_montages,
                        args.api_url,
                        args.database,
                        admin_token,
                        client=client,
                    )
                    stats["montages_injected"] = result.get("inserted_count", 0)
                    stats["montages_updated"] = result.get("updated_count", 0)
                    for err in result.get("errors", []):
                        errors.append({"dataset": "montages_collection", "error": err})
                        stats["errors"] += 1
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                ValueError,
                KeyError,
            ) as e:
                stats["errors"] += 1
                errors.append({"dataset": "montages_collection", "error": str(e)})
                print(f"  Error injecting montages: {e}", file=sys.stderr)

        # Compute stats for affected datasets if requested
        if args.compute_stats and not args.only_datasets and not args.only_montages:
            # Get unique dataset IDs from injected records
            affected_datasets = sorted(
                set(r.get("dataset") for r in all_records if r.get("dataset"))
            )
            if affected_datasets:
                print(f"\nComputing stats for {len(affected_datasets)} datasets...")
                try:
                    datasets_param = ",".join(affected_datasets)
                    url = f"{args.api_url}/admin/{args.database}/datasets/compute-stats?datasets={datasets_param}"
                    with _make_session(admin_token) as client:
                        result, _response = request_json(
                            "post",
                            url,
                            timeout=120,
                            raise_for_status=True,
                            raise_for_request=True,
                            client=client,
                        )
                        stats["stats_computed"] = (result or {}).get(
                            "datasets_updated", 0
                        )
                        print(
                            f"  Stats computed for {stats['stats_computed']} datasets"
                        )
                except (
                    httpx.RequestError,
                    httpx.HTTPStatusError,
                    ValueError,
                    KeyError,
                ) as e:
                    # Stats-compute step is optional — log a warning and
                    # continue. The records / datasets are already written.
                    print(f"  Warning: Failed to compute stats: {e}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print("INJECTION SUMMARY")
    print("=" * 60)
    print(f"  Database:   {args.database}")
    print(f"  Datasets:   {stats['datasets_injected']}")
    print(f"  Records Ins:{stats['records_injected']}")
    print(f"  Records Upd:{stats.get('records_updated', 0)}")
    print(f"  Montages Ins:{stats['montages_injected']}")
    print(f"  Montages Upd:{stats.get('montages_updated', 0)}")
    if stats.get("stats_computed"):
        print(f"  Stats Comp: {stats['stats_computed']}")
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
