#!/usr/bin/env python3
"""Patch NEMAR records whose ``storage.base`` still points at OpenNeuro.

Companion to ``patch_nemar_source.py``. That script fixes the
``Dataset.source`` field for nm-prefixed datasets that were mislabeled by
the pre-PR-#327 ``detect_source()`` bug. Records, however, also encode a
``storage.base`` (e.g. ``s3://openneuro.org/nm000237``) — and they were
left untouched by the dataset-document patch.

This script flips ``storage.base`` from ``s3://openneuro.org/<id>`` to
``s3://nemar/<id>`` (the value the digest pipeline writes today, see
``STORAGE_CONFIGS`` in ``3_digest.py``) for every record under each
nm-prefixed dataset. Without this fix, ``EEGDashRaw._raw_uri`` resolves to
the OpenNeuro mirror and downloads 404 with::

    DataIntegrityError: Primary data file not found on S3:
    s3://openneuro.org/nm000237/sub-13/.../sub-13_..._eeg.bdf

Usage
-----
    # Dry-run (default)
    python scripts/ingestions/patch_nemar_records_storage.py

    # Apply against prod
    EEGDASH_API_TOKEN=... python scripts/ingestions/patch_nemar_records_storage.py --apply

    # Restrict to a specific list of dataset IDs
    python scripts/ingestions/patch_nemar_records_storage.py --ids nm000237 --apply

    # Target a non-prod DB
    python scripts/ingestions/patch_nemar_records_storage.py --database eegdash_staging --apply

Environment variables
---------------------
    EEGDASH_API_URL    API base URL (default: https://data.eegdash.org)
    EEGDASH_API_TOKEN  Admin token; required when ``--apply`` is set.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from eegdash.http_api_client import get_client

NM_PATTERN = re.compile(r"^nm\d+$")
WRONG_BASE_PREFIX = "s3://openneuro.org/"
RIGHT_BASE_PREFIX = "s3://nemar/"


def discover_nm_dataset_ids(client, page_size: int = 1000) -> list[str]:
    """Return every nm-prefixed dataset_id known to the API."""
    url = f"{client.api_url}/api/{client.database}/datasets"
    ids: list[str] = []
    skip = 0
    while True:
        resp = client._session.get(
            url, params={"limit": page_size, "skip": skip}, timeout=60
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        ids.extend(
            d["dataset_id"]
            for d in data
            if NM_PATTERN.match(str(d.get("dataset_id", "")))
        )
        if len(data) < page_size:
            break
        skip += page_size
    return ids


def patch_dataset_records(client, dataset_id: str, apply: bool) -> tuple[int, int]:
    """Flip ``storage.base`` for one dataset's records.

    Returns ``(matched, modified)``. In dry-run mode ``modified`` is the
    count of records that *would* change.
    """
    wrong_base = f"{WRONG_BASE_PREFIX}{dataset_id}"
    right_base = f"{RIGHT_BASE_PREFIX}{dataset_id}"

    query = {"dataset": dataset_id, "storage.base": wrong_base}

    if not apply:
        matched = client.count_documents(query)
        return matched, matched

    return client.update_many(
        query,
        {"storage.base": right_base, "storage.backend": "s3"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database", default="eegdash")
    parser.add_argument("--api-url", default=None)
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Explicit list of nm* dataset IDs (default: auto-discover all nm*)",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.apply and not os.getenv("EEGDASH_API_TOKEN"):
        print(
            "error: --apply requires EEGDASH_API_TOKEN in the environment",
            file=sys.stderr,
        )
        return 2

    client = get_client(api_url=args.api_url, database=args.database)
    print(f"Database: {args.database}")
    print(f"API URL:  {client.api_url}")
    print(f"Mode:     {'APPLY' if args.apply else 'DRY-RUN'}\n")

    if args.ids:
        ids = [i for i in args.ids if NM_PATTERN.match(i)]
        skipped = [i for i in args.ids if not NM_PATTERN.match(i)]
        for s in skipped:
            print(f"skip (not nm-prefixed): {s}")
    else:
        print("Discovering nm* datasets...")
        ids = discover_nm_dataset_ids(client)
        print(f"  {len(ids)} nm* datasets found\n")

    total_matched = 0
    total_modified = 0
    affected: list[tuple[str, int, int]] = []

    for dsid in ids:
        try:
            matched, modified = patch_dataset_records(client, dsid, apply=args.apply)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {dsid}: {e}", file=sys.stderr)
            continue

        if matched:
            affected.append((dsid, matched, modified))
            total_matched += matched
            total_modified += modified
            verb = "would update" if not args.apply else "updated"
            print(f"{dsid}: {verb} {modified}/{matched} records")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Datasets affected: {len(affected)}")
    print(f"Records matched:   {total_matched}")
    print(f"Records modified:  {total_modified}")

    if not args.apply and total_matched:
        print("\nRe-run with --apply (and EEGDASH_API_TOKEN) to commit changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
