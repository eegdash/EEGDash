"""Fetch OpenNeuro dataset IDs for one or more modalities using gql."""

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

DATASETS_QUERY = gql(
    """
    query ($first: Int!, $after: String) {
      datasets(first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            modalities
            modified
          }
        }
      }
    }
    """
)


def _iter_dataset_info(
    session: Client,
    *,
    page_size: int,
) -> Iterable[dict]:
    """Iterate over all datasets with id, modalities, and modified date."""
    cursor: str | None = None
    while True:
        variables = {
            "first": page_size,
            "after": cursor,
        }
        result = session.execute(DATASETS_QUERY, variable_values=variables)
        page = result["datasets"]
        if not page:
            return
        for edge in page["edges"]:
            node = edge["node"]
            dataset_id = node["id"]
            modalities = node.get("modalities", [])
            modified = node.get("modified")

            # Yield one entry per modality
            for modality in modalities:
                yield {
                    "dataset_id": dataset_id,
                    "modality": modality,
                    "last_update": modified,
                }

        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]


def _build_client(token: str | None, timeout: float, retries: int) -> Client:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    transport = RequestsHTTPTransport(
        url=GRAPHQL_URL,
        timeout=timeout,
        retries=retries,
        verify=True,
        headers=headers,
    )
    return Client(transport=transport, fetch_schema_from_transport=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch all OpenNeuro datasets with metadata (id, modality, last update date)."
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="GraphQL page size.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="GraphQL request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of automatic retries for 5xx/429 responses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/openneuro_datasets.json"),
        help="Path to save results as JSON (default: consolidated/openneuro_datasets.json).",
    )
    return parser.parse_args()


def _write_results(results: list[dict], output: Path) -> None:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def main() -> None:
    args = parse_args()

    token = os.getenv("OPENNEURO_TOKEN")

    client = _build_client(token, args.timeout, args.retries)

    with client as session:
        datasets = list(_iter_dataset_info(session, page_size=args.page_size))
        print(f"Fetched {len(datasets)} dataset entries")
        for entry in datasets[:5]:  # Show first 5 as preview
            print(
                f"  {entry['dataset_id']} ({entry['modality']}) - Updated: {entry['last_update']}"
            )
        if len(datasets) > 5:
            print(f"  ... and {len(datasets) - 5} more")

    _write_results(datasets, args.output)
    print(f"\nSaved {len(datasets)} datasets to {args.output}")


if __name__ == "__main__":
    main()
