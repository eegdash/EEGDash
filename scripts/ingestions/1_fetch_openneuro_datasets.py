"""Fetch OpenNeuro dataset IDs with metadata using requests library."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import requests

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

DATASETS_QUERY = """
    query ($modality: String!, $first: Int!, $after: String) {
      datasets(modality: $modality, first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            created
          }
        }
      }
    }
    """


def build_batch_modified_query(dataset_ids: list[str]) -> str:
    """Build a batch query to fetch modified dates for multiple datasets."""
    query_parts = []
    for i, dataset_id in enumerate(dataset_ids):
        # Create an alias for each dataset to avoid conflicts
        query_parts.append(
            f"""
    ds{i}: dataset(id: "{dataset_id}") {{
      id
      latestSnapshot {{
        created
      }}
    }}"""
        )

    return f"""
query {{
{chr(10).join(query_parts)}
}}
"""


def fetch_batch_modified(
    dataset_ids: list[str], timeout: float = 30.0
) -> dict[str, str]:
    """Fetch modified dates for a batch of datasets in a single query."""
    result = {}

    if not dataset_ids:
        return result

    try:
        query = build_batch_modified_query(dataset_ids)
        payload = {"query": query}
        response = requests.post(GRAPHQL_URL, json=payload, timeout=timeout)
        data = response.json()

        if "errors" in data or "data" not in data:
            return result

        response_data = data["data"]

        # Extract modified dates from batch response
        for i, dataset_id in enumerate(dataset_ids):
            alias = f"ds{i}"
            if alias in response_data:
                dataset = response_data[alias]
                if dataset:
                    latest_snapshot = dataset.get("latestSnapshot")
                    if latest_snapshot:
                        created = latest_snapshot.get("created")
                        if created:
                            result[dataset_id] = created
    except Exception as e:
        print(f"  Error fetching batch: {str(e)[:100]}")

    return result


def fetch_datasets(
    page_size: int = 100,
    timeout: float = 30.0,
    max_consecutive_errors: int = 5,
) -> Iterator[dict]:
    """Fetch all OpenNeuro datasets with id and modality using requests."""
    # Use smaller page size for iEEG due to known API issue at offset 50
    modality_configs = {
        "eeg": {"page_size": page_size, "max_errors": 3},
        "ieeg": {"page_size": 10, "max_errors": 5},  # Smaller for iEEG
        "meg": {"page_size": page_size, "max_errors": 3},
    }

    for modality in ["eeg", "ieeg", "meg"]:
        config = modality_configs[modality]
        cursor = None
        consecutive_errors = 0
        current_page_size = config["page_size"]
        print(f"Fetching {modality} datasets (page size: {current_page_size})...")

        while True:
            try:
                payload = {
                    "query": DATASETS_QUERY,
                    "variables": {
                        "modality": modality,
                        "first": current_page_size,
                        "after": cursor,
                    },
                }

                response = requests.post(GRAPHQL_URL, json=payload, timeout=timeout)
                response.raise_for_status()

                result = response.json()

                # Check for GraphQL errors
                if "errors" in result:
                    raise Exception(f"GraphQL Error: {result['errors'][0]['message']}")

                # Reset error counter on successful fetch
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)

                # Check if it's a server-side error
                if "Not Found" in error_msg or "INTERNAL_SERVER_ERROR" in error_msg:
                    print(
                        f"  [{modality}] Server error (attempt {consecutive_errors}/{config['max_errors']})"
                    )

                    if consecutive_errors < config["max_errors"]:
                        print("    Skipping this batch and continuing...")
                        continue

                    # If we've tried multiple times, move to next modality
                    print(
                        f"  [{modality}] Reached max consecutive errors, moving to next modality"
                    )
                    break
                else:
                    # For other errors, log and stop
                    print(f"  [{modality}] Error: {error_msg[:100]}")
                    break

            data = result.get("data")
            if not data:
                print(f"  [{modality}] No data in response")
                break

            page = data.get("datasets")
            if not page:
                print(f"  [{modality}] No page data")
                break

            edges = page.get("edges", [])
            if not edges:
                print(f"  [{modality}] No edges in response")
                break

            # Process each dataset in the page
            for edge in edges:
                node = edge.get("node")
                if not node:
                    continue

                dataset_id = node.get("id")
                created = node.get("created")

                # Get modified date (will be fetched separately if needed)
                modified = created  # Default to created date

                yield {
                    "dataset_id": dataset_id,
                    "modality": modality,
                    "created": created,
                    "modified": modified,
                }

            # Check if there are more pages
            page_info = page.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                print(f"  [{modality}] Completed")
                break
            cursor = page_info.get("endCursor")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all OpenNeuro datasets with metadata."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/openneuro_datasets.json"),
        help="Output JSON file (default: consolidated/openneuro_datasets.json).",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--fetch-modified",
        action="store_true",
        help="Fetch actual modified dates for each dataset (slower but complete)",
    )
    args = parser.parse_args()

    # Fetch main dataset list
    datasets = list(
        fetch_datasets(
            page_size=args.page_size,
            timeout=args.timeout,
        )
    )

    # Optionally fetch actual modified dates
    if args.fetch_modified:
        print(f"\nFetching actual modified dates for {len(datasets)} datasets...")
        batch_size = 20  # Batch 20 datasets per query

        for batch_start in range(0, len(datasets), batch_size):
            batch_end = min(batch_start + batch_size, len(datasets))
            batch = datasets[batch_start:batch_end]
            dataset_ids = [d["dataset_id"] for d in batch]

            # Fetch modified dates for this batch
            modified_dates = fetch_batch_modified(dataset_ids, timeout=args.timeout)

            # Update datasets with fetched modified dates
            for dataset in batch:
                dataset_id = dataset["dataset_id"]
                if dataset_id in modified_dates:
                    dataset["modified"] = modified_dates[dataset_id]

            if batch_end % 100 == 0 or batch_end == len(datasets):
                print(
                    f"  Fetched modified dates for {batch_end}/{len(datasets)} datasets"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(datasets, fh, indent=2)

    print(f"Saved {len(datasets)} dataset entries to {args.output}")


if __name__ == "__main__":
    main()
