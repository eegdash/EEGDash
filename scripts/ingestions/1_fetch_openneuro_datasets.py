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
    retries: int = 5,
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
            except Exception as e:
                # If we hit an error on a specific dataset, skip to next page
                print(
                    f"  Warning: Error fetching {modality} datasets at cursor {cursor}: {e}"
                )
                if page_size > 10:
                    page_size = max(10, page_size // 2)  # Reduce page size
                    continue
                break

            page = result["datasets"]
            if not page:
                break

            for edge in page["edges"]:
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

            if not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]


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
