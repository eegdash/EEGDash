"""Fetch OpenNeuro dataset IDs with metadata using gql."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

DATASETS_QUERY = gql(
    """
    query ($modality: String!, $first: Int!, $after: String) {
      datasets(modality: $modality, first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            created
            latestSnapshot {
              created
            }
          }
        }
      }
    }
    """
)


def fetch_datasets(
    page_size: int = 100,
    timeout: float = 30.0,
    retries: int = 3,
    max_consecutive_errors: int = 3,
) -> Iterator[dict]:
    """Fetch all OpenNeuro datasets with id and modality."""
    transport = RequestsHTTPTransport(
        url=GRAPHQL_URL,
        timeout=timeout,
        retries=retries,
        verify=True,
    )
    client = Client(transport=transport, fetch_schema_from_transport=False)

    modalities = ["eeg", "ieeg", "meg"]

    for modality in modalities:
        cursor: str | None = None
        consecutive_errors = 0
        error_cursors = set()  # Track cursors that caused errors

        while True:
            try:
                result = client.execute(
                    DATASETS_QUERY,
                    variable_values={
                        "modality": modality,
                        "first": page_size,
                        "after": cursor,
                    },
                )
                # Reset error counter on successful fetch
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)

                # Check if it's a server-side 404 error for specific datasets
                if "'Not Found'" in error_msg and cursor not in error_cursors:
                    print(
                        f"  Warning: Skipping {modality} page at cursor {cursor} due to server error (attempt {consecutive_errors}/{max_consecutive_errors})"
                    )
                    error_cursors.add(cursor)

                    # Try to skip this problematic cursor by moving forward
                    if consecutive_errors < max_consecutive_errors:
                        # Try smaller page size to isolate the problematic dataset
                        if page_size > 10:
                            page_size = max(10, page_size // 2)
                            print(
                                f"  Reducing page size to {page_size} to skip problematic dataset"
                            )
                            continue

                    # If we've tried multiple times, skip to next modality
                    print(
                        f"  Skipping remaining {modality} datasets after {consecutive_errors} consecutive errors"
                    )
                    break
                else:
                    # For other errors, log and retry with exponential backoff
                    print(f"  Error fetching {modality} datasets: {error_msg[:200]}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(
                            f"  Too many consecutive errors ({consecutive_errors}), moving to next modality"
                        )
                        break
                    continue

            page = result.get("datasets")
            if not page:
                break

            edges = page.get("edges", [])
            if not edges:
                break

            # Process each dataset in the page
            for edge in edges:
                node = edge.get("node")
                if not node:
                    continue

                # Get creation and last update dates
                created = node.get("created")
                latest_snapshot = node.get("latestSnapshot")
                modified = latest_snapshot.get("created") if latest_snapshot else None

                yield {
                    "dataset_id": node.get("id"),
                    "modality": modality,
                    "created": created,
                    "modified": modified,
                }

            # Check if there are more pages
            page_info = page.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
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
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    # Fetch and save
    datasets = list(
        fetch_datasets(
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(datasets, fh, indent=2)

    print(f"Saved {len(datasets)} dataset entries to {args.output}")


if __name__ == "__main__":
    main()
