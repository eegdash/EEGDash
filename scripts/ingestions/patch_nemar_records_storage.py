#!/usr/bin/env python3
"""Patch NEMAR records whose ``storage`` is misrouted or stale.

Companion to ``patch_nemar_source.py``. That script fixes the
``Dataset.source`` field for nm-prefixed datasets that were mislabeled by
the pre-PR-#327 ``detect_source()`` bug. Records, however, also encode a
``storage.base`` (e.g. ``s3://openneuro.org/nm000237``) and a
``storage.backend`` — and the prior fixes left both fields stale for some
cohorts of records.

This script applies two independent corrections:

1. **Base** — flip ``storage.base`` from ``s3://openneuro.org/<id>`` to
   ``s3://nemar/<id>`` for any record that still points at OpenNeuro
   despite belonging to an ``nm*`` dataset. Without this, the runtime
   would resolve ``_raw_uri`` to the OpenNeuro mirror and 404.
2. **Backend** — flip ``storage.backend`` from ``"s3"`` to ``"nemar"`` for
   any NEMAR record. The runtime now uses ``"nemar"`` as a non-fetchable
   marker so it can fail loud with actionable guidance (git-annex / nemar
   CLI / NEMAR API) instead of opaquely failing on a closed S3 bucket.
   Records carrying ``backend="s3"`` would attempt anonymous S3 fetches
   that always fail because NEMAR's ``s3:ListBucket`` /
   ``s3:GetObject`` are closed by design.

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
WRONG_BACKEND = "s3"
RIGHT_BACKEND = "nemar"


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
    """Apply both the base and backend corrections to one dataset's records.

    Two cohorts are healed:

    * **base + backend** — records where ``storage.base`` still points at
      ``s3://openneuro.org/<id>``. Both ``base`` and ``backend`` get
      rewritten to the canonical NEMAR values.
    * **backend only** — records that already have the canonical
      ``s3://nemar/<id>`` base but stale ``storage.backend = "s3"``. Only
      ``backend`` is flipped to ``"nemar"``.

    Returns ``(matched, modified)``. In dry-run mode ``modified`` is the
    count of records that *would* change across both phases.
    """
    wrong_base = f"{WRONG_BASE_PREFIX}{dataset_id}"
    right_base = f"{RIGHT_BASE_PREFIX}{dataset_id}"

    base_query = {"dataset": dataset_id, "storage.base": wrong_base}
    backend_query = {
        "dataset": dataset_id,
        "storage.base": right_base,
        "storage.backend": WRONG_BACKEND,
    }

    if not apply:
        base_matched = client.count_documents(base_query)
        backend_matched = client.count_documents(backend_query)
        total = base_matched + backend_matched
        return total, total

    base_matched, base_modified = client.update_many(
        base_query,
        {"storage.base": right_base, "storage.backend": RIGHT_BACKEND},
    )
    backend_matched, backend_modified = client.update_many(
        backend_query,
        {"storage.backend": RIGHT_BACKEND},
    )
    return base_matched + backend_matched, base_modified + backend_modified


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
