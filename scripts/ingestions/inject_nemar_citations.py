#!/usr/bin/env python3
"""Inject NEMAR citation counts into EEGDash datasets.

This script downloads citation data from the NEMAR citations repository
and updates the EEGDash database with citation counts for each dataset.

Usage:
    python scripts/ingestions/inject_nemar_citations.py --dry-run
    python scripts/ingestions/inject_nemar_citations.py
    python scripts/ingestions/inject_nemar_citations.py --database eegdash_dev

Environment variables:
    EEGDASH_API_URL: API base URL (default: https://data.eegdash.org)
    EEGDASH_API_TOKEN: Admin token for write operations
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from eegdash.http_api_client import EEGDashAPIClient

# Default URL for NEMAR citations CSV
CSV_URL = "https://raw.githubusercontent.com/sccn/nemar-citations/main/citations/citations_02012025.csv"


def fetch_citation_csv(url: str = CSV_URL) -> dict[str, int]:
    """Download and parse citation counts CSV from GitHub.

    Parameters
    ----------
    url : str
        URL to the CSV file.

    Returns
    -------
    dict[str, int]
        Mapping of dataset_id -> citation_count.

    """
    print(f"Fetching citations from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    citations: dict[str, int] = {}
    reader = csv.DictReader(io.StringIO(response.text))

    for row in reader:
        dataset_id = row.get("dataset_id", "").strip()
        count_str = row.get("number_of_citations", "0")

        if not dataset_id:
            continue

        try:
            count = int(float(count_str))
        except (ValueError, TypeError):
            count = 0

        citations[dataset_id] = count

    print(f"Parsed {len(citations)} datasets from CSV")
    return citations


def inject_citations(
    client: EEGDashAPIClient,
    citations: dict[str, int],
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Update datasets with citation counts.

    Parameters
    ----------
    client : EEGDashAPIClient
        API client instance.
    citations : dict[str, int]
        Mapping of dataset_id -> citation_count.
    dry_run : bool
        If True, don't actually update, just report what would happen.

    Returns
    -------
    tuple[int, int, int]
        (updated_count, skipped_count, not_found_count)

    """
    updated = 0
    skipped = 0
    not_found = 0

    for dataset_id, count in citations.items():
        # Check if dataset exists
        dataset = client.get_dataset(dataset_id)

        if dataset is None:
            not_found += 1
            continue

        # Check if update is needed
        current_count = dataset.get("nemar_citation_count")
        if current_count == count:
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY-RUN] Would update {dataset_id}: {current_count} -> {count}")
            updated += 1
        else:
            try:
                modified = client.update_dataset(
                    dataset_id, {"nemar_citation_count": count}
                )
                if modified:
                    print(f"Updated {dataset_id}: {current_count} -> {count}")
                    updated += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error updating {dataset_id}: {e}")

    return updated, skipped, not_found


def main():
    parser = argparse.ArgumentParser(
        description="Inject NEMAR citation counts into EEGDash datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update, just show what would happen",
    )
    parser.add_argument(
        "--database",
        default="eegdash",
        help="Database name (default: eegdash)",
    )
    parser.add_argument(
        "--csv-url",
        default=CSV_URL,
        help="URL to the NEMAR citations CSV",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL (defaults to EEGDASH_API_URL env or https://data.eegdash.org)",
    )
    args = parser.parse_args()

    # Import here to avoid import errors if dependencies not installed
    from eegdash.http_api_client import get_client

    # Fetch citations
    try:
        citations = fetch_citation_csv(args.csv_url)
    except requests.RequestException as e:
        print(f"Error fetching CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not citations:
        print("No citations found in CSV", file=sys.stderr)
        sys.exit(1)

    # Get API client
    client = get_client(api_url=args.api_url, database=args.database)

    print(f"\nDatabase: {args.database}")
    print(f"Dry run: {args.dry_run}")
    print(f"Total datasets in CSV: {len(citations)}")
    print()

    # Inject citations
    updated, skipped, not_found = inject_citations(
        client, citations, dry_run=args.dry_run
    )

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Updated:   {updated}")
    print(f"Skipped:   {skipped} (already up to date)")
    print(f"Not found: {not_found} (not in EEGDash)")
    print(f"Total:     {updated + skipped + not_found}")

    if args.dry_run:
        print("\n[DRY-RUN] No changes were made")


if __name__ == "__main__":
    main()
