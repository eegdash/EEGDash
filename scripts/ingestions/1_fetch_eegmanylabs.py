"""Fetch EEGManyLabs datasets from G-Node GIN organization.

This script scrapes the EEGManyLabs organization page on GIN (G-Node Infrastructure)
to retrieve information about all available datasets. GIN is a git-based repository
hosting service for neuroscience data.

Output: consolidated/eegmanylabs_datasets.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


def fetch_eegmanylabs_repos(
    organization: str = "EEGManyLabs",
    base_url: str = "https://gin.g-node.org",
) -> list[dict[str, Any]]:
    """Fetch all repositories from the EEGManyLabs organization.

    Args:
        organization: GIN organization name
        base_url: Base URL for GIN

    Returns:
        List of repository dictionaries with metadata

    """
    org_url = f"{base_url}/{organization}"

    print(f"Fetching repositories from {org_url}")

    try:
        response = requests.get(org_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching organization page: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all repository items
    repo_items = soup.find_all("div", class_="item")

    if not repo_items:
        print(f"Warning: No repositories found for {organization}", file=sys.stderr)
        return []

    datasets = []

    for item in repo_items:
        try:
            # Get repository link
            repo_link = item.find("a", class_="name")
            if not repo_link:
                continue

            repo_path = repo_link.get("href", "")
            if not repo_path.startswith(f"/{organization}/"):
                continue

            repo_name = repo_path.split("/")[-1]

            # Get description
            desc_elem = item.find("p", class_="description")
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Get metadata (stars, forks, updated time)
            meta_div = item.find("div", class_="meta")
            stars = 0
            forks = 0
            updated = ""

            if meta_div:
                # Extract stars
                star_elem = meta_div.find("a", href=re.compile(r"/stars$"))
                if star_elem:
                    stars = int(star_elem.get_text(strip=True))

                # Extract forks
                fork_elem = meta_div.find("a", href=re.compile(r"/forks$"))
                if fork_elem:
                    forks = int(fork_elem.get_text(strip=True))

                # Extract update time
                time_elem = meta_div.find("span", class_="time")
                if time_elem:
                    updated = time_elem.get("title", time_elem.get_text(strip=True))

            # Construct clone URLs
            clone_url = f"{base_url}/{organization}/{repo_name}.git"
            ssh_url = f"git@gin.g-node.org:{organization}/{repo_name}.git"

            dataset = {
                "dataset_id": repo_name,
                "full_name": f"{organization}/{repo_name}",
                "description": description,
                "url": f"{base_url}/{organization}/{repo_name}",
                "clone_url": clone_url,
                "ssh_url": ssh_url,
                "stars": stars,
                "forks": forks,
                "updated": updated,
                "source": "gin",
                "organization": organization,
            }

            datasets.append(dataset)

        except Exception as e:
            print(f"Warning: Error parsing repository item: {e}", file=sys.stderr)
            continue

    print(f"Found {len(datasets)} repositories")

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEGManyLabs datasets from GIN organization."
    )
    parser.add_argument(
        "--organization",
        type=str,
        default="EEGManyLabs",
        help="GIN organization name (default: EEGManyLabs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/eegmanylabs_datasets.json"),
        help="Output JSON file path (default: consolidated/eegmanylabs_datasets.json).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://gin.g-node.org",
        help="Base URL for GIN (default: https://gin.g-node.org).",
    )

    args = parser.parse_args()

    # Fetch repositories
    datasets = fetch_eegmanylabs_repos(
        organization=args.organization,
        base_url=args.base_url,
    )

    if not datasets:
        print("No datasets fetched. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2, sort_keys=True)

    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
