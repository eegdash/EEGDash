#!/usr/bin/env python3
"""Patch NEMAR datasets that were mislabeled as ``source="openneuro"``.

A bug in ``detect_source()`` (fixed in PR #327) caused nm-prefixed NEMAR
datasets to fall through to a ``return "openneuro"`` default in the
digestion stage, so records and dataset documents reached MongoDB with
the wrong ``source`` field. This script corrects already-ingested data.

Strategy
--------
1. Discover nm-prefixed datasets whose ``source`` is not ``"nemar"`` (by
   default by listing ``source=openneuro`` via the public API and
   client-side filtering, or from an explicit ``--ids`` list).
2. PATCH each dataset document's ``source`` field via
   ``/admin/{db}/datasets/{id}``.

The ``records`` collection is intentionally not touched: records carry
only a ``dataset`` foreign key back to ``Dataset.dataset_id`` and have
no ``source`` field of their own, so correcting the dataset document is
sufficient.

Usage
-----
    # Dry-run (default) — show what would change
    python scripts/ingestions/patch_nemar_source.py

    # Apply against prod
    EEGDASH_API_TOKEN=... python scripts/ingestions/patch_nemar_source.py --apply

    # Restrict to a specific list of IDs
    python scripts/ingestions/patch_nemar_source.py --ids nm000176 nm000195 --apply

    # Target a non-prod DB
    python scripts/ingestions/patch_nemar_source.py --database eegdash_staging --apply

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
from typing import Any

from eegdash.http_api_client import get_client

NM_PATTERN = re.compile(r"^nm\d+$")


def discover_mislabeled(client, page_size: int = 1000) -> list[dict[str, Any]]:
    """List all nm-prefixed datasets whose source is not ``nemar``.

    The public ``/api/{db}/datasets`` endpoint only supports filtering by
    ``source`` and ``modality``, so we pull ``source=openneuro`` and
    filter for nm-prefixed ids client-side.
    """
    url = f"{client.api_url}/api/{client.database}/datasets"
    mislabeled: list[dict[str, Any]] = []
    skip = 0
    while True:
        resp = client._session.get(
            url,
            params={"limit": page_size, "skip": skip, "source": "openneuro"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        mislabeled.extend(
            d for d in data if NM_PATTERN.match(str(d.get("dataset_id", "")))
        )
        if len(data) < page_size:
            break
        skip += page_size
    return mislabeled


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database", default="eegdash", help="Database name (default: eegdash)"
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL (defaults to $EEGDASH_API_URL or https://data.eegdash.org)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Explicit list of dataset IDs to patch (default: auto-discover)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default: dry-run, no writes)",
    )
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
    print(f"Mode:     {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    # Resolve the target list
    if args.ids:
        bad_ids: list[str] = []
        for dsid in args.ids:
            if not NM_PATTERN.match(dsid):
                print(f"skip (not nm-prefixed): {dsid}")
                continue
            ds = client.get_dataset(dsid)
            if ds is None:
                print(f"skip (not found): {dsid}")
                continue
            if ds.get("source") == "nemar":
                print(f"skip (already nemar): {dsid}")
                continue
            bad_ids.append(dsid)
        mislabeled_ids = bad_ids
    else:
        print("Discovering mislabeled nm* datasets (source=openneuro)...")
        mislabeled_docs = discover_mislabeled(client)
        mislabeled_ids = [d["dataset_id"] for d in mislabeled_docs]

    if not mislabeled_ids:
        print("Nothing to patch. Exiting.")
        return 0

    print(f"\nFound {len(mislabeled_ids)} nm* datasets with wrong source:")
    for dsid in mislabeled_ids:
        print(f"  - {dsid}")

    if not args.apply:
        print(
            "\n[DRY-RUN] Would PATCH:"
            f"\n  - {len(mislabeled_ids)} dataset documents (source -> 'nemar')"
            "\n\nRe-run with --apply to commit changes."
        )
        return 0

    # Patch dataset documents one by one
    dataset_updated = 0
    dataset_failed: list[tuple[str, str]] = []
    for dsid in mislabeled_ids:
        try:
            modified = client.update_dataset(dsid, {"source": "nemar"})
            if modified:
                dataset_updated += 1
                print(f"Updated dataset {dsid}")
            else:
                print(f"Unchanged {dsid} (already nemar?)")
        except Exception as e:  # noqa: BLE001 — surface any HTTP/transport error
            dataset_failed.append((dsid, str(e)))
            print(f"ERROR dataset {dsid}: {e}", file=sys.stderr)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Datasets updated: {dataset_updated}/{len(mislabeled_ids)}")
    if dataset_failed:
        print(f"Datasets failed:  {len(dataset_failed)}")
        for dsid, err in dataset_failed:
            print(f"  - {dsid}: {err}")

    return 0 if not dataset_failed else 1


if __name__ == "__main__":
    sys.exit(main())
