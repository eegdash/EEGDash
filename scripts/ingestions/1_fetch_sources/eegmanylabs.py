"""Fetch EEGManyLabs datasets from G-Node GIN organization.

This script scrapes the EEGManyLabs organization page on GIN (G-Node Infrastructure)
to retrieve information about all available datasets. GIN is a git-based repository
hosting service for neuroscience data.

Note: GIN API does not support listing org repos, so we use web scraping + individual API calls.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Add parent paths for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eegdash.records import Dataset, create_dataset

GIN_BASE_URL = "https://gin.g-node.org"
GIN_API_URL = f"{GIN_BASE_URL}/api/v1"


def fetch_repo_details(org: str, repo: str, timeout: float = 10.0) -> dict | None:
    """Fetch repository details via GIN API."""
    url = f"{GIN_API_URL}/repos/{org}/{repo}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def parse_repo_name(name: str) -> dict:
    """Parse EEGManyLabs repo name to extract metadata.
    
    Format: EEGManyLabs_Replication_AuthorYear_[Raw|Processed]
    """
    result = {
        "study_type": None,
        "original_study": None,
        "data_type": None,
    }
    
    parts = name.split("_")
    if len(parts) >= 3:
        if parts[1] == "Replication":
            result["study_type"] = "replication"
        
        # Extract original study (e.g., "ClarkHillyard1996")
        if len(parts) >= 3:
            result["original_study"] = parts[2]
        
        # Check for Raw/Processed suffix
        if parts[-1] in ["Raw", "Processed"]:
            result["data_type"] = parts[-1].lower()
    
    return result


def fetch_eegmanylabs_repos(
    organization: str = "EEGManyLabs",
    timeout: float = 30.0,
) -> list[Dataset]:
    """Fetch all repositories from the EEGManyLabs organization.

    Args:
        organization: GIN organization name
        timeout: Request timeout in seconds

    Returns:
        List of Dataset documents

    """
    org_url = f"{GIN_BASE_URL}/{organization}"

    print(f"Fetching repositories from {org_url}")

    try:
        response = requests.get(org_url, timeout=timeout)
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
    total_processed = 0

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
            total_processed += 1

            # Get description from page
            desc_elem = item.find("p", class_="description")
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Fetch detailed info via API
            repo_details = fetch_repo_details(organization, repo_name)
            
            # Parse repo name for study metadata
            name_meta = parse_repo_name(repo_name)
            
            # Build dataset name
            name = description or repo_name
            if name_meta["original_study"]:
                name = f"EEGManyLabs: {name_meta['original_study']} Replication"
                if name_meta["data_type"]:
                    name += f" ({name_meta['data_type'].capitalize()})"
            
            # Determine if this is processed data
            data_processed = name_meta["data_type"] == "processed" if name_meta["data_type"] else None
            
            # Build study domain from name metadata
            study_domain = None
            if name_meta["study_type"] == "replication":
                study_domain = f"Replication: {name_meta['original_study']}"

            # Create Dataset document
            dataset = create_dataset(
                dataset_id=repo_name,
                name=name,
                source="gin",
                recording_modality="eeg",
                modalities=["eeg"],
                data_processed=data_processed,
                study_domain=study_domain,
                study_design="replication" if name_meta["study_type"] == "replication" else None,
                species="Human",
                dataset_modified_at=repo_details.get("updated_at") if repo_details else None,
            )

            datasets.append(dataset)
            
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} repositories...")

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
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )

    args = parser.parse_args()

    print(f"Fetching EEGManyLabs datasets from GIN: {args.organization}")

    # Fetch repositories
    datasets = fetch_eegmanylabs_repos(
        organization=args.organization,
        timeout=args.timeout,
    )

    if not datasets:
        print("No datasets fetched. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2, sort_keys=True)

    print(f"\nSaved {len(datasets)} dataset entries to {args.output}")

    # Print summary
    if datasets:
        print("\nSummary:")
        print(f"  Total datasets: {len(datasets)}")
        
        # Count by data type
        raw_count = sum(1 for d in datasets if "Raw" in d.get("dataset_id", ""))
        processed_count = sum(1 for d in datasets if "Processed" in d.get("dataset_id", ""))
        print(f"  Raw datasets: {raw_count}")
        print(f"  Processed datasets: {processed_count}")


if __name__ == "__main__":
    main()
