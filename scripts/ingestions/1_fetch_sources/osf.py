"""Fetch EEG BIDS projects from Open Science Framework (OSF).

This script searches OSF for projects containing both "EEG" and "BIDS" keywords
using the OSF API v2. It retrieves comprehensive metadata including project IDs,
descriptions, contributors, files, and DOIs.

Output: consolidated/osf_projects.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def search_osf(
    query: str,
    size: int = 100,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Search OSF for projects matching the query.

    Args:
        query: Search query string
        size: Maximum number of results to fetch
        page_size: Number of results per page (max 100)

    Returns:
        List of OSF project dictionaries

    """
    base_url = "https://api.osf.io/v2/search/"

    # Build query parameters
    params = {
        "q": query,
        "page[size]": min(page_size, 100),  # OSF max is 100
    }

    print(f"Searching OSF with query: {query}")
    print(f"Max results to fetch: {size}")
    print(f"Results per page: {params['page[size]']}")

    all_records = []
    page = 1
    current_url = base_url

    while True:
        print(f"\nFetching page {page}...", end=" ", flush=True)

        try:
            response = requests.get(current_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"\nError fetching page {page}: {e}", file=sys.stderr)
            break

        data = response.json()
        records = data.get("data", [])

        if not records:
            print("No more results")
            break

        # Get total count from meta
        links = data.get("links", {})
        meta = links.get("meta", {})
        total = meta.get("total", 0)

        print(f"Got {len(records)} records (total available: {total})")
        all_records.extend(records)

        # Check if we've reached the requested size or all available records
        if len(all_records) >= size:
            print(f"Reached limit ({len(all_records)} records)")
            break

        # Check if there are more pages
        next_url = links.get("next")

        if not next_url:
            print("No more pages available")
            break

        # Update for next page
        current_url = next_url
        params = {}  # Next URL already has all params
        page += 1

        # Be nice to the API
        time.sleep(0.5)

    print(f"\nTotal records fetched: {len(all_records)}")
    return all_records


def extract_project_info(record: dict) -> dict[str, Any]:
    """Extract relevant information from an OSF project record.

    Args:
        record: Raw OSF project record dictionary

    Returns:
        Cleaned project dictionary

    """
    project_id = record.get("id", "")
    record_type = record.get("type", "")

    attributes = record.get("attributes", {})
    relationships = record.get("relationships", {})
    links = record.get("links", {})

    # Extract basic metadata
    title = attributes.get("title", "")
    description = attributes.get("description", "")
    category = attributes.get("category", "")
    tags = attributes.get("tags", [])

    # Extract dates
    date_created = attributes.get("date_created", "")
    date_modified = attributes.get("date_modified", "")

    # Extract flags
    is_public = attributes.get("public", False)
    is_registration = attributes.get("registration", False)
    is_preprint = attributes.get("preprint", False)
    is_fork = attributes.get("fork", False)
    wiki_enabled = attributes.get("wiki_enabled", False)

    # Extract license info
    node_license = attributes.get("node_license", {})
    license_year = node_license.get("year", "")
    copyright_holders = node_license.get("copyright_holders", [])

    # Extract DOI if available
    doi = None
    identifiers = (
        relationships.get("identifiers", {}).get("links", {}).get("related", {})
    )
    if identifiers:
        doi_href = identifiers.get("href", "")
        if doi_href:
            # We'd need to fetch this separately, but for now we'll note it exists
            doi = "available"  # Placeholder

    # Extract URLs
    html_url = links.get("html", "")
    self_url = links.get("self", "")

    # Build project dictionary
    project = {
        "project_id": project_id,
        "type": record_type,
        "title": title,
        "description": description,
        "category": category,
        "tags": tags,
        "date_created": date_created,
        "date_modified": date_modified,
        "is_public": is_public,
        "is_registration": is_registration,
        "is_preprint": is_preprint,
        "is_fork": is_fork,
        "wiki_enabled": wiki_enabled,
        "license_year": license_year,
        "copyright_holders": copyright_holders,
        "url": html_url,
        "api_url": self_url,
        "source": "osf",
    }

    # Add DOI if available
    if doi:
        project["doi"] = doi

    return project


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEG BIDS projects from Open Science Framework."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="EEG BIDS",
        help='Search query (default: "EEG BIDS").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/osf_projects.json"),
        help="Output JSON file path (default: consolidated/osf_projects.json).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Maximum number of results to fetch (default: 1000).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of results per page (max 100, default: 100).",
    )

    args = parser.parse_args()

    # Search OSF
    records = search_osf(
        query=args.query,
        size=args.size,
        page_size=args.page_size,
    )

    if not records:
        print("No records found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract project information
    print("\nExtracting project information...")
    projects = []
    for record in records:
        try:
            project = extract_project_info(record)
            projects.append(project)
        except Exception as e:
            print(
                f"Warning: Error extracting info for record {record.get('id')}: {e}",
                file=sys.stderr,
            )
            continue

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with args.output.open("w") as fh:
        json.dump(projects, fh, indent=2, sort_keys=True)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(projects)} projects to {args.output}")
    print(f"{'=' * 60}")
    print("\nProject Statistics:")

    # Count by category
    categories = {}
    for proj in projects:
        cat = proj.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    # Count by type
    types = {}
    for proj in projects:
        ptype = proj.get("type", "unknown")
        types[ptype] = types.get(ptype, 0) + 1

    print("\nBy Type:")
    for ptype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ptype}: {count}")

    # Public vs private
    public_count = sum(1 for proj in projects if proj.get("is_public", False))
    print(f"\nPublic projects: {public_count}/{len(projects)}")

    # Registrations
    registration_count = sum(
        1 for proj in projects if proj.get("is_registration", False)
    )
    print(f"Registrations: {registration_count}/{len(projects)}")

    # Preprints
    preprint_count = sum(1 for proj in projects if proj.get("is_preprint", False))
    print(f"Preprints: {preprint_count}/{len(projects)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
