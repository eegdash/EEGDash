#!/usr/bin/env python3
"""Helper utilities for EEGDash API operations.

This module provides convenient functions for common API operations
with proper authentication and error handling.

Usage:
    # As a module
    from api_helper import EEGDashAPI

    api = EEGDashAPI()  # Uses EEGDASH_ADMIN_TOKEN from env
    api.check_auth()
    api.compute_stats(["ds001234", "ds005678"])

    # As CLI
    python api_helper.py check-auth
    python api_helper.py compute-stats ds001234 ds005678
    python api_helper.py data-quality
    python api_helper.py delete-dataset ds001234
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


class EEGDashAPI:
    """Helper class for EEGDash API operations."""

    def __init__(
        self,
        base_url: str | None = None,
        database: str = "eegdash",
        token: str | None = None,
    ):
        """Initialize the API helper.

        Parameters
        ----------
        base_url : str, optional
            API base URL (default: from EEGDASH_API_URL or https://data.eegdash.org)
        database : str
            Database name (default: "eegdash")
        token : str, optional
            Admin token (default: from EEGDASH_ADMIN_TOKEN env var)

        """
        self.base_url = base_url or os.getenv(
            "EEGDASH_API_URL", "https://data.eegdash.org"
        )
        self.database = database
        self.token = token or os.getenv("EEGDASH_ADMIN_TOKEN")
        self._session = requests.Session()

    def _get_headers(self, auth: bool = False) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if auth:
            if not self.token:
                raise ValueError(
                    "Admin token required. Set EEGDASH_ADMIN_TOKEN environment variable "
                    "or pass token to constructor."
                )
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        auth: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(auth=auth)

        try:
            response = self._session.request(
                method, url, headers=headers, timeout=60, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {e.response.text[:500] if e.response else 'No response'}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise

    # =========================================================================
    # Public API (no auth required)
    # =========================================================================

    def list_datasets(self, limit: int = 100) -> list[dict]:
        """List all datasets."""
        result = self._request("GET", f"/api/{self.database}/datasets?limit={limit}")
        return result.get("data", [])

    def get_dataset(self, dataset_id: str) -> dict | None:
        """Get a single dataset by ID."""
        try:
            result = self._request("GET", f"/api/{self.database}/datasets/{dataset_id}")
            return result.get("data")
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                return None
            raise

    def list_records(self, dataset: str | None = None, limit: int = 100) -> list[dict]:
        """List records, optionally filtered by dataset."""
        params = f"limit={limit}"
        if dataset:
            params += f"&dataset={dataset}"
        result = self._request("GET", f"/api/{self.database}/records?{params}")
        return result.get("data", [])

    def get_stats(self) -> dict[str, dict]:
        """Get aggregated stats for all datasets."""
        result = self._request("GET", f"/api/{self.database}/datasets/stats/records")
        return result.get("data", {})

    # =========================================================================
    # Admin API (auth required)
    # =========================================================================

    def check_auth(self) -> bool:
        """Check if authentication is working."""
        if not self.token:
            print("ERROR: No admin token configured")
            print("  Set EEGDASH_ADMIN_TOKEN environment variable")
            print("  Example: export EEGDASH_ADMIN_TOKEN='your-token'")
            return False

        try:
            # Try to access an admin endpoint
            self._request(
                "POST",
                f"/admin/{self.database}/datasets/compute-stats?datasets=__test__",
                auth=True,
            )
            print("SUCCESS: Authentication working")
            print(f"  API URL: {self.base_url}")
            print(f"  Database: {self.database}")
            print(f"  Token: {self.token[:10]}...{self.token[-4:]}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 403:
                print("ERROR: Authentication failed (403 Forbidden)")
                print("  Token may be invalid or expired")
            else:
                print(f"ERROR: {e}")
            return False

    def compute_stats(self, datasets: list[str] | None = None) -> dict:
        """Recompute statistics for datasets.

        Parameters
        ----------
        datasets : list[str], optional
            List of dataset IDs to process. If None, processes all datasets.

        Returns
        -------
        dict
            API response with processed/updated counts

        """
        endpoint = f"/admin/{self.database}/datasets/compute-stats"
        if datasets:
            endpoint += f"?datasets={','.join(datasets)}"

        return self._request("POST", endpoint, auth=True)

    def delete_dataset(self, dataset_id: str, include_records: bool = True) -> bool:
        """Delete a dataset and optionally its records.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to delete
        include_records : bool
            If True, also delete all records for this dataset

        Returns
        -------
        bool
            True if deletion was successful

        """
        try:
            if include_records:
                # Delete records first
                self._request(
                    "DELETE",
                    f"/admin/{self.database}/records?dataset={dataset_id}",
                    auth=True,
                )
                print(f"  Deleted records for {dataset_id}")

            # Delete dataset
            self._request(
                "DELETE",
                f"/admin/{self.database}/datasets/{dataset_id}",
                auth=True,
            )
            print(f"  Deleted dataset {dataset_id}")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                print(f"  Dataset {dataset_id} not found")
                return False
            raise

    # =========================================================================
    # Data Quality Checks
    # =========================================================================

    def check_data_quality(self) -> dict[str, Any]:
        """Check data quality and report missing metadata."""
        stats = self.get_stats()
        # Stats already contains all datasets, use those keys
        datasets = [{"dataset_id": ds_id} for ds_id in stats.keys()]

        total = len(datasets)
        missing_nchans = []
        missing_sfreq = []

        for ds in datasets:
            ds_id = ds.get("dataset_id", "")
            ds_stats = stats.get(ds_id, {})

            if not ds_stats.get("nchans_counts"):
                missing_nchans.append(ds_id)
            if not ds_stats.get("sfreq_counts"):
                missing_sfreq.append(ds_id)

        report = {
            "total_datasets": total,
            "missing_nchans": len(missing_nchans),
            "missing_sfreq": len(missing_sfreq),
            "missing_nchans_pct": round(100 * len(missing_nchans) / total, 1)
            if total
            else 0,
            "missing_sfreq_pct": round(100 * len(missing_sfreq) / total, 1)
            if total
            else 0,
            "datasets_missing_nchans": missing_nchans[:20],  # First 20
            "datasets_missing_sfreq": missing_sfreq[:20],
        }

        print("\n=== Data Quality Report ===")
        print(f"Total datasets: {total}")
        print(
            f"Missing nchans: {report['missing_nchans']} ({report['missing_nchans_pct']}%)"
        )
        print(
            f"Missing sfreq:  {report['missing_sfreq']} ({report['missing_sfreq_pct']}%)"
        )

        if missing_nchans:
            print("\nDatasets missing nchans (first 20):")
            for ds_id in missing_nchans[:20]:
                print(f"  - {ds_id}")

        return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EEGDash API Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_helper.py check-auth
  python api_helper.py compute-stats ds001234 ds005678
  python api_helper.py data-quality
  python api_helper.py delete-dataset test_dataset

Environment Variables:
  EEGDASH_ADMIN_TOKEN  Admin token for write operations (required for admin commands)
  EEGDASH_API_URL      API base URL (default: https://data.eegdash.org)
        """,
    )

    parser.add_argument(
        "--database",
        default="eegdash",
        help="Database name (default: eegdash)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="API base URL (default: from env or https://data.eegdash.org)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check-auth
    subparsers.add_parser("check-auth", help="Check if authentication is working")

    # compute-stats
    stats_parser = subparsers.add_parser(
        "compute-stats", help="Recompute statistics for datasets"
    )
    stats_parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset IDs to process (all if not specified)",
    )

    # data-quality
    subparsers.add_parser(
        "data-quality", help="Check data quality and missing metadata"
    )

    # delete-dataset
    delete_parser = subparsers.add_parser(
        "delete-dataset", help="Delete a dataset and its records"
    )
    delete_parser.add_argument("dataset_id", help="Dataset ID to delete")
    delete_parser.add_argument(
        "--keep-records",
        action="store_true",
        help="Keep records (only delete dataset document)",
    )

    # list-datasets
    list_parser = subparsers.add_parser("list-datasets", help="List datasets")
    list_parser.add_argument(
        "--limit", type=int, default=100, help="Max datasets to return"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    api = EEGDashAPI(base_url=args.api_url, database=args.database)

    try:
        if args.command == "check-auth":
            success = api.check_auth()
            return 0 if success else 1

        elif args.command == "compute-stats":
            datasets = args.datasets if args.datasets else None
            result = api.compute_stats(datasets)
            print(json.dumps(result, indent=2))
            return 0

        elif args.command == "data-quality":
            api.check_data_quality()
            return 0

        elif args.command == "delete-dataset":
            success = api.delete_dataset(
                args.dataset_id, include_records=not args.keep_records
            )
            return 0 if success else 1

        elif args.command == "list-datasets":
            datasets = api.list_datasets(limit=args.limit)
            for ds in datasets:
                print(
                    f"  {ds.get('dataset_id', 'unknown')}: {ds.get('name', 'No name')[:50]}"
                )
            print(f"\nTotal: {len(datasets)} datasets")
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
