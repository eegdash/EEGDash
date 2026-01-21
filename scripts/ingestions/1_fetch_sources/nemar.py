"""Fetch NEMAR datasets from NEMAR CLI or GitHub organization with BIDS metadata.

NEMAR datasets are hosted on GitHub under the nemardatasets organization.
This script fetches dataset info via the NEMAR CLI (preferred) or GitHub API
(fallback), and tries to extract BIDS metadata from dataset_description.json
and participants.tsv files.

The NEMAR CLI (`nemar-cli`) is installed via Bun and returns full metadata JSON,
making it the preferred method for fetching dataset information.
"""

import argparse
import sys
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from _github import (
    fetch_first_repo_file_text,
    fetch_repo_file_json,
    fetch_repo_file_text,
    iter_org_repos,
)
from _nemar import (
    is_nemar_cli_available,
    iter_nemar_datasets,
    map_cli_dataset_to_schema,
)
from _serialize import save_datasets_deterministically, setup_paths

setup_paths()
from eegdash.schemas import Dataset, create_dataset


def fetch_bids_description(
    org: str, repo: str, branch: str, timeout: float = 10.0
) -> dict | None:
    """Fetch dataset_description.json from a repository."""
    return fetch_repo_file_json(
        org,
        repo,
        "dataset_description.json",
        ref=branch,
        timeout=timeout,
    )


def fetch_participants_tsv(
    org: str, repo: str, branch: str, timeout: float = 10.0
) -> list[dict] | None:
    """Fetch and parse participants.tsv from a repository."""
    text = fetch_repo_file_text(
        org,
        repo,
        "participants.tsv",
        ref=branch,
        timeout=timeout,
    )
    if not text:
        return None
    try:
        df = pd.read_csv(StringIO(text), sep="\t", dtype="string")
    except Exception:
        return None
    return df.to_dict(orient="records") if not df.empty else None


def extract_ages_from_participants(participants: list[dict] | None) -> list[int]:
    """Extract ages from participants data."""
    if not participants:
        return []

    df = pd.DataFrame(participants)
    if df.empty:
        return []

    age_cols = [c for c in df.columns if str(c).strip().lower() == "age"]
    if not age_cols:
        return []

    ages = pd.to_numeric(df[age_cols[0]], errors="coerce")
    ages = ages[(ages > 0) & (ages < 150)].dropna()
    return [int(a) for a in ages.astype(float).tolist()]


def fetch_readme(org: str, repo: str, branch: str, timeout: float = 10.0) -> str | None:
    """Fetch README from a repository."""
    readme_names = ["README.md", "README", "README.txt", "readme.md", "readme"]
    return fetch_first_repo_file_text(
        org,
        repo,
        readme_names,
        ref=branch,
        timeout=timeout,
    )


def fetch_repositories_via_cli(
    timeout: float = 120.0,
    fetch_bids: bool = True,
    limit: int | None = None,
) -> Iterator[Dataset]:
    """Fetch all datasets via NEMAR CLI.

    Args:
        timeout: Request timeout in seconds
        fetch_bids: Whether to fetch additional BIDS metadata from repos
        limit: Maximum number of datasets to fetch

    Yields:
        Dataset documents

    """
    total_fetched = 0

    for cli_data in iter_nemar_datasets(timeout=timeout):
        if limit and total_fetched >= limit:
            break

        # Map CLI fields to our schema
        mapped = map_cli_dataset_to_schema(cli_data)

        dataset_id = mapped.get("dataset_id") or ""
        if not dataset_id:
            continue

        # Skip non-NEMAR datasets (should start with "nm")
        if not dataset_id.startswith("nm"):
            continue

        total_fetched += 1

        # Get fields from CLI mapping
        name = mapped.get("name") or dataset_id
        authors = mapped.get("authors") or []
        license_str = mapped.get("license")
        bids_version = mapped.get("bids_version")
        dataset_doi = mapped.get("dataset_doi")
        subjects_count = mapped.get("subjects_count") or 0
        source_url = mapped.get("source_url")
        funding = mapped.get("funding") or []
        readme = mapped.get("readme")
        ages = mapped.get("ages") or []

        # Optionally fetch additional BIDS metadata from GitHub
        # (for fields not available in CLI output)
        if fetch_bids and (not readme or not ages):
            participants = fetch_participants_tsv("nemardatasets", dataset_id, "main")
            if not readme:
                readme = fetch_readme("nemardatasets", dataset_id, "main")
            if not ages and participants:
                ages = extract_ages_from_participants(participants)
            if not subjects_count and participants:
                subjects_count = len(participants)

        # Create Dataset document
        yield create_dataset(
            dataset_id=dataset_id,
            name=name,
            source="nemar",
            readme=readme,
            recording_modality="eeg",  # NEMAR is EEG-focused
            experimental_modalities=["eeg"],
            bids_version=bids_version,
            license=license_str,
            authors=authors
            if isinstance(authors, list)
            else [authors]
            if authors
            else [],
            funding=funding
            if isinstance(funding, list)
            else [funding]
            if funding
            else [],
            dataset_doi=dataset_doi,
            subjects_count=subjects_count,
            ages=ages,
            species="Human",  # NEMAR datasets are human
            source_url=source_url,
            dataset_created_at=mapped.get("created_at"),
            dataset_modified_at=mapped.get("updated_at"),
        )

        if total_fetched % 20 == 0:
            print(f"  Processed {total_fetched} datasets via CLI...")


def fetch_repositories_via_github(
    organization: str = "nemardatasets",
    page_size: int = 100,
    timeout: float = 30.0,
    retries: int = 5,
    fetch_bids: bool = True,
    limit: int | None = None,
) -> Iterator[Dataset]:
    """Fetch all repositories from a GitHub organization.

    Args:
        organization: GitHub organization name
        page_size: Number of repositories per page (max 100)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        fetch_bids: Whether to fetch BIDS metadata from repos
        limit: Maximum number of datasets to fetch

    Yields:
        Dataset documents

    """
    total_fetched = 0

    for repo in iter_org_repos(
        organization,
        per_page=page_size,
        timeout=timeout,
        retries=retries,
    ):
        if limit and total_fetched >= limit:
            break

        repo_name = str(repo.get("name") or "")
        if not repo_name:
            continue

        # Skip special GitHub repositories
        if repo_name in {".github", ".gitignore", "README"}:
            continue

        # NEMAR datasets start with "nm" prefix
        if not repo_name.startswith("nm"):
            continue

        total_fetched += 1

        # Fetch BIDS metadata
        bids_desc = None
        participants = None
        readme = None
        branch = str(repo.get("default_branch") or "main")
        if fetch_bids:
            bids_desc = fetch_bids_description(organization, repo_name, branch)
            participants = fetch_participants_tsv(organization, repo_name, branch)
            readme = fetch_readme(organization, repo_name, branch)

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

        # Build NEMAR GitHub URL
        nemar_url = (
            repo.get("html_url") or f"https://github.com/nemardatasets/{repo_name}"
        )

        # Create Dataset document
        yield create_dataset(
            dataset_id=repo_name,
            name=name,
            source="nemar",
            readme=readme,
            recording_modality="eeg",  # NEMAR is EEG-focused
            experimental_modalities=["eeg"],
            bids_version=bids_version,
            license=license_str,
            authors=authors
            if isinstance(authors, list)
            else [authors]
            if authors
            else [],
            funding=funding
            if isinstance(funding, list)
            else [funding]
            if funding
            else [],
            dataset_doi=dataset_doi,
            subjects_count=subjects_count,
            ages=ages,
            species="Human",  # NEMAR datasets are human
            source_url=nemar_url,
            dataset_modified_at=repo.get("pushed_at"),
        )

        if total_fetched % 20 == 0:
            print(f"  Processed {total_fetched} repositories via GitHub...")


def fetch_repositories(
    organization: str = "nemardatasets",
    page_size: int = 100,
    timeout: float = 30.0,
    retries: int = 5,
    fetch_bids: bool = True,
    limit: int | None = None,
    use_cli: bool | None = None,
) -> Iterator[Dataset]:
    """Fetch all NEMAR datasets using CLI (preferred) or GitHub API (fallback).

    Args:
        organization: GitHub organization name (for GitHub fallback)
        page_size: Number of repositories per page (max 100, GitHub only)
        timeout: Request timeout in seconds
        retries: Number of retry attempts (GitHub only)
        fetch_bids: Whether to fetch BIDS metadata from repos
        limit: Maximum number of datasets to fetch
        use_cli: Force CLI (True), force GitHub (False), or auto-detect (None)

    Yields:
        Dataset documents

    """
    # Determine which method to use
    if use_cli is None:
        # Auto-detect: try CLI first if available
        use_cli = is_nemar_cli_available()
        if use_cli:
            print("NEMAR CLI detected, using CLI method...")
        else:
            print("NEMAR CLI not available, using GitHub API fallback...")
    elif use_cli:
        if not is_nemar_cli_available():
            print("Warning: --use-cli specified but NEMAR CLI not found in PATH")
            print("Falling back to GitHub API...")
            use_cli = False
        else:
            print("Using NEMAR CLI method (forced)...")
    else:
        print("Using GitHub API method (forced)...")

    if use_cli:
        # Try CLI method
        datasets_found = False
        try:
            for dataset in fetch_repositories_via_cli(
                timeout=timeout,
                fetch_bids=fetch_bids,
                limit=limit,
            ):
                datasets_found = True
                yield dataset
        except Exception as e:
            print(f"CLI method failed: {e}")
            datasets_found = False

        # If CLI didn't yield any datasets, fall back to GitHub
        if not datasets_found:
            print("CLI returned no datasets, falling back to GitHub API...")
            yield from fetch_repositories_via_github(
                organization=organization,
                page_size=page_size,
                timeout=timeout,
                retries=retries,
                fetch_bids=fetch_bids,
                limit=limit,
            )
    else:
        # Use GitHub API
        yield from fetch_repositories_via_github(
            organization=organization,
            page_size=page_size,
            timeout=timeout,
            retries=retries,
            fetch_bids=fetch_bids,
            limit=limit,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NEMAR datasets via NEMAR CLI or GitHub API with BIDS metadata."
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
        help="Repositories per page (max 100, default: 100, GitHub only)",
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
        help="Number of retry attempts (default: 5, GitHub only)",
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of datasets to fetch (default: all)",
    )

    # Method selection: CLI vs GitHub
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--use-cli",
        action="store_true",
        help="Force using NEMAR CLI (falls back to GitHub if unavailable)",
    )
    method_group.add_argument(
        "--use-github",
        action="store_true",
        help="Force using GitHub API instead of NEMAR CLI",
    )

    args = parser.parse_args()

    # Determine CLI usage
    use_cli: bool | None = None
    if args.use_cli:
        use_cli = True
    elif args.use_github:
        use_cli = False
    # else: None = auto-detect

    print(f"Fetching NEMAR datasets from: {args.organization}")
    print(f"Fetching BIDS metadata: {not args.skip_bids}")

    datasets = list(
        fetch_repositories(
            organization=args.organization,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
            fetch_bids=not args.skip_bids,
            limit=args.limit,
            use_cli=use_cli,
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
        with_subjects = sum(
            1
            for d in datasets
            if d.get("demographics", {}).get("subjects_count", 0) > 0
        )
        print(f"  With subject count: {with_subjects}")

        # Total subjects
        total_subjects = sum(
            d.get("demographics", {}).get("subjects_count", 0) for d in datasets
        )
        print(f"  Total subjects: {total_subjects}")

        # With ages
        with_ages = sum(1 for d in datasets if d.get("demographics", {}).get("ages"))
        print(f"  With age info: {with_ages}")


if __name__ == "__main__":
    main()
