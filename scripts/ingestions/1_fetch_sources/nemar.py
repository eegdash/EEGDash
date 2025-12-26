"""Fetch NEMAR datasets from GitHub organization with BIDS metadata.

NEMAR datasets are hosted on GitHub under the nemardatasets organization.
This script fetches repository info and tries to extract BIDS metadata
from dataset_description.json and participants.tsv files.
"""

import argparse
import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import requests

# Add parent paths for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eegdash.records import Dataset, create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically

GITHUB_API_URL = "https://api.github.com"
GITHUB_RAW_URL = "https://raw.githubusercontent.com"


def fetch_bids_description(org: str, repo: str, branch: str, timeout: float = 10.0) -> dict | None:
    """Fetch dataset_description.json from a repository."""
    url = f"{GITHUB_RAW_URL}/{org}/{repo}/{branch}/dataset_description.json"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def fetch_participants_tsv(org: str, repo: str, branch: str, timeout: float = 10.0) -> list[dict] | None:
    """Fetch and parse participants.tsv from a repository."""
    url = f"{GITHUB_RAW_URL}/{org}/{repo}/{branch}/participants.tsv"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                return None
            headers = lines[0].split("\t")
            participants = []
            for line in lines[1:]:
                values = line.split("\t")
                participant = dict(zip(headers, values))
                participants.append(participant)
            return participants
    except Exception:
        pass
    return None


def extract_ages_from_participants(participants: list[dict] | None) -> list[int]:
    """Extract ages from participants data."""
    if not participants:
        return []
    
    ages = []
    for p in participants:
        # Try common age column names
        age_val = p.get("age") or p.get("Age") or p.get("AGE")
        if age_val:
            try:
                age = int(float(age_val))
                if 0 < age < 150:  # Sanity check
                    ages.append(age)
            except (ValueError, TypeError):
                pass
    return ages


def fetch_repositories(
    organization: str = "nemardatasets",
    page_size: int = 100,
    timeout: float = 30.0,
    retries: int = 5,
    fetch_bids: bool = True,
) -> Iterator[Dataset]:
    """Fetch all repositories from a GitHub organization.

    Args:
        organization: GitHub organization name
        page_size: Number of repositories per page (max 100)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        fetch_bids: Whether to fetch BIDS metadata from repos

    Yields:
        Dataset documents

    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }

    page = 1
    total_fetched = 0

    while True:
        url = f"{GITHUB_API_URL}/orgs/{organization}/repos"
        params = {
            "per_page": min(page_size, 100),
            "page": page,
            "type": "all",
            "sort": "created",
            "direction": "asc",
        }

        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

                # Check rate limit
                if response.status_code == 403 and "rate limit" in response.text.lower():
                    print("  Warning: GitHub API rate limit exceeded")
                    remaining = response.headers.get("X-RateLimit-Remaining", "?")
                    reset = response.headers.get("X-RateLimit-Reset", "?")
                    print(f"  Remaining: {remaining}, Reset: {reset}")
                    return

                response.raise_for_status()
                repos = response.json()

                # If empty list, we've fetched all repositories
                if not repos:
                    return

                for repo in repos:
                    repo_name = repo.get("name")

                    # Skip special GitHub repositories
                    if repo_name in [".github", ".gitignore", "README"]:
                        continue
                    
                    # NEMAR datasets start with "nm" prefix
                    if not repo_name.startswith("nm"):
                        continue

                    total_fetched += 1
                    
                    # Fetch BIDS metadata
                    bids_desc = None
                    participants = None
                    if fetch_bids:
                        branch = repo.get("default_branch", "main")
                        bids_desc = fetch_bids_description(organization, repo_name, branch)
                        participants = fetch_participants_tsv(organization, repo_name, branch)
                    
                    # Extract metadata from BIDS description
                    authors = []
                    funding = []
                    license_str = None
                    bids_version = None
                    dataset_doi = None
                    name = repo.get("description") or repo_name
                    
                    if bids_desc:
                        name = bids_desc.get("Name") or name
                        authors = bids_desc.get("Authors") or []
                        funding = bids_desc.get("Funding") or []
                        license_str = bids_desc.get("License")
                        bids_version = bids_desc.get("BIDSVersion")
                        dataset_doi = bids_desc.get("DatasetDOI")
                    
                    # Extract ages from participants
                    ages = extract_ages_from_participants(participants)
                    subjects_count = len(participants) if participants else 0
                    
                    # Create Dataset document
                    yield create_dataset(
                        dataset_id=repo_name,
                        name=name,
                        source="nemar",
                        recording_modality="eeg",  # NEMAR is EEG-focused
                        modalities=["eeg"],
                        bids_version=bids_version,
                        license=license_str,
                        authors=authors if isinstance(authors, list) else [authors] if authors else [],
                        funding=funding if isinstance(funding, list) else [funding] if funding else [],
                        dataset_doi=dataset_doi,
                        subjects_count=subjects_count,
                        ages=ages,
                        species="Human",  # NEMAR datasets are human
                        dataset_modified_at=repo.get("pushed_at"),
                    )
                    
                    if total_fetched % 20 == 0:
                        print(f"  Processed {total_fetched} repositories...")

                # Check if there are more pages
                link_header = response.headers.get("Link", "")
                if 'rel="next"' not in link_header:
                    return

                page += 1
                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(f"  Warning: Error fetching page {page} (attempt {attempt}/{retries}): {e}")
                if attempt >= retries:
                    print(f"  Skipping to next page after {retries} failed attempts")
                    page += 1
                    break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NEMAR datasets from GitHub organization with BIDS metadata."
    )
    parser.add_argument(
        "--organization",
        type=str,
        default="nemardatasets",
        help="GitHub organization name (default: nemardatasets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/nemar_datasets.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Repositories per page (max 100, default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retry attempts (default: 5)",
    )
    parser.add_argument(
        "--skip-bids",
        action="store_true",
        help="Skip fetching BIDS metadata (faster but less info)",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output, default: omitted for determinism)",
    )
    args = parser.parse_args()

    print(f"Fetching NEMAR datasets from: {args.organization}")
    print(f"Fetching BIDS metadata: {not args.skip_bids}")

    datasets = list(
        fetch_repositories(
            organization=args.organization,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
            fetch_bids=not args.skip_bids,
        )
    )

    # Add digested_at timestamp if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    print(f"\nSaved {len(datasets)} dataset entries to {args.output}")

    # Print summary
    if datasets:
        print("\nSummary:")
        print(f"  Total datasets: {len(datasets)}")
        
        # Count with BIDS info
        with_bids = sum(1 for d in datasets if d.get("bids_version"))
        print(f"  With BIDS version: {with_bids}")
        
        # Count with participants
        with_subjects = sum(1 for d in datasets if d.get("demographics", {}).get("subjects_count", 0) > 0)
        print(f"  With subject count: {with_subjects}")
        
        # Total subjects
        total_subjects = sum(d.get("demographics", {}).get("subjects_count", 0) for d in datasets)
        print(f"  Total subjects: {total_subjects}")
        
        # With ages
        with_ages = sum(1 for d in datasets if d.get("demographics", {}).get("ages"))
        print(f"  With age info: {with_ages}")


if __name__ == "__main__":
    main()
