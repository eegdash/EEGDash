"""Fetch EEGManyLabs datasets from G-Node GIN organization.

This script scrapes the EEGManyLabs organization page on GIN (G-Node Infrastructure)
to retrieve information about all available datasets. GIN is a git-based repository
hosting service for neuroscience data.

Extracts rich metadata from:
- GIN API (repo stats)
- dataset_description.json (BIDS metadata, authors, funding)
- participants.tsv (subject demographics)
- README.md (paper DOIs, links)

Note: GIN API does not support listing org repos, so we use web scraping + individual API calls.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import generate_dataset_id, save_datasets_deterministically, setup_paths

setup_paths()
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


def fetch_raw_file(
    org: str, repo: str, filepath: str, branch: str = "main", timeout: float = 10.0
) -> str | None:
    """Fetch raw file content from GIN repository."""
    url = f"{GIN_BASE_URL}/{org}/{repo}/raw/{branch}/{filepath}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200 and not response.text.startswith("<!DOCTYPE"):
            return response.text
    except Exception:
        pass
    # Try master branch as fallback
    if branch == "main":
        return fetch_raw_file(org, repo, filepath, branch="master", timeout=timeout)
    return None


def fetch_bids_description(org: str, repo: str, timeout: float = 10.0) -> dict | None:
    """Fetch and parse dataset_description.json from repository."""
    content = fetch_raw_file(org, repo, "dataset_description.json", timeout=timeout)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    return None


def fetch_participants_tsv(org: str, repo: str, timeout: float = 10.0) -> dict | None:
    """Fetch and parse participants.tsv to extract demographics."""
    content = fetch_raw_file(org, repo, "participants.tsv", timeout=timeout)
    if not content:
        return None

    lines = content.strip().split("\n")
    if len(lines) < 2:
        return None

    # Parse header
    headers = lines[0].split("\t")

    # Parse data rows
    participants = []
    for line in lines[1:]:
        if line.strip():
            values = line.split("\t")
            row = dict(zip(headers, values))
            participants.append(row)

    if not participants:
        return None

    # Extract statistics
    result = {"n_subjects": len(participants)}

    # Age statistics
    ages = []
    for p in participants:
        age_str = p.get("age", "")
        try:
            age = float(age_str)
            if age > 0 and age < 120:  # Filter invalid ages
                ages.append(age)
        except (ValueError, TypeError):
            pass

    if ages:
        result["age_min"] = min(ages)
        result["age_max"] = max(ages)
        result["age_mean"] = sum(ages) / len(ages)

    # Sex distribution
    sex_counts = {}
    for p in participants:
        sex = p.get("sex", "").lower()
        if sex:
            sex_counts[sex] = sex_counts.get(sex, 0) + 1
    if sex_counts:
        result["sex_distribution"] = sex_counts

    # Handedness distribution
    hand_counts = {}
    for p in participants:
        hand = p.get("handedness", "").lower()
        if hand:
            hand_counts[hand] = hand_counts.get(hand, 0) + 1
    if hand_counts:
        result["handedness_distribution"] = hand_counts

    # Unique labs
    labs = set()
    for p in participants:
        lab = p.get("lab", "")
        if lab:
            labs.add(lab)
    if labs:
        result["labs"] = sorted(labs)
        result["n_labs"] = len(labs)

    return result


def fetch_readme(org: str, repo: str, timeout: float = 10.0) -> dict | None:
    """Fetch README.md and extract paper DOIs and links."""
    content = fetch_raw_file(org, repo, "README.md", timeout=timeout)
    if not content:
        return None

    result = {}

    # Extract OSF links
    osf_pattern = r"https?://osf\.io/([^\s\)/]+)"
    osf_links = re.findall(osf_pattern, content)

    # Extract specific DOIs from common patterns
    if "Replicated publication" in content:
        match = re.search(
            r"Replicated publication[:\*\s]+https?://doi\.org/([^\s\)]+)", content
        )
        if match:
            result["original_paper_doi"] = match.group(1)

    if "Replication paper" in content:
        match = re.search(
            r"Replication paper[:\*\s]+https?://[^\s\)]+/([^\s\)]+)", content
        )
        if match:
            result["replication_paper_url"] = match.group(0).split("**")[-1].strip()

    if "EEGManyLabs project" in content:
        match = re.search(
            r"EEGManyLabs project[:\*\s]+https?://doi\.org/([^\s\)]+)", content
        )
        if match:
            result["project_doi"] = match.group(1)

    if osf_links:
        result["osf_id"] = osf_links[0]
        result["osf_url"] = f"https://osf.io/{osf_links[0]}/"

    return result if result else None


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

            # Skip preprocessed datasets - we only want raw data
            if "Processed" in repo_name or "preprocessed" in repo_name.lower():
                print(f"  Skipping {repo_name} (preprocessed data)")
                continue

            total_processed += 1
            print(f"  Processing {repo_name}...")

            # Get description from page
            desc_elem = item.find("p", class_="description")
            page_description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Fetch detailed info via API
            repo_details = fetch_repo_details(organization, repo_name, timeout=timeout)

            # Fetch BIDS metadata
            bids_desc = fetch_bids_description(organization, repo_name, timeout=timeout)

            # Fetch participant demographics
            participants = fetch_participants_tsv(
                organization, repo_name, timeout=timeout
            )

            # Skip empty repositories (no BIDS metadata and no participants)
            if not bids_desc and not participants:
                print("    -> Skipping (empty or incomplete repository)")
                continue

            if participants:
                print(f"    -> Found {participants.get('n_subjects')} subjects")

            # Fetch README for paper DOIs
            readme_meta = fetch_readme(organization, repo_name, timeout=timeout)

            # Parse repo name for study metadata
            name_meta = parse_repo_name(repo_name)

            # Build dataset name from BIDS or fallback
            if bids_desc and bids_desc.get("Name"):
                name = bids_desc["Name"]
            else:
                name = page_description or repo_name
                if name_meta["original_study"]:
                    name = f"EEGManyLabs: {name_meta['original_study']} Replication"
                    if name_meta["data_type"]:
                        name += f" ({name_meta['data_type'].capitalize()})"

            # Determine if this is processed data
            data_processed = (
                name_meta["data_type"] == "processed"
                if name_meta["data_type"]
                else None
            )

            # Build study domain
            study_domain = None
            if name_meta["study_type"] == "replication":
                study_domain = f"Replication: {name_meta['original_study']}"

            # Extract license from BIDS
            license_info = None
            if bids_desc and bids_desc.get("License"):
                license_info = bids_desc["License"]

            # Extract authors from BIDS
            authors = None
            if bids_desc and bids_desc.get("Authors"):
                authors = bids_desc["Authors"]

            # Extract funding
            funding = None
            if bids_desc and bids_desc.get("Funding"):
                funding = bids_desc["Funding"]

            # Build associated paper DOI
            associated_paper = None
            if readme_meta and readme_meta.get("original_paper_doi"):
                associated_paper = readme_meta["original_paper_doi"]

            # Create Dataset document using all schema fields
            n_subjects = participants.get("n_subjects") if participants else None

            # Collect ages for age_min/max calculation
            ages_list = None
            if participants and participants.get("age_min") is not None:
                ages_list = [int(participants["age_min"]), int(participants["age_max"])]

            # Get age mean
            age_mean = None
            if participants and participants.get("age_mean"):
                age_mean = round(participants["age_mean"], 1)

            # Get sex/handedness distributions
            sex_dist = participants.get("sex_distribution") if participants else None
            hand_dist = (
                participants.get("handedness_distribution") if participants else None
            )

            # Get contributing labs
            labs = participants.get("labs") if participants else None

            # Build external links
            source_url = f"{GIN_BASE_URL}/{organization}/{repo_name}"
            osf_url = readme_meta.get("osf_url") if readme_meta else None
            paper_url = (
                readme_meta.get("replication_paper_url") if readme_meta else None
            )

            # Get repository stats
            stars = repo_details.get("stars_count") if repo_details else None
            forks = repo_details.get("forks_count") if repo_details else None
            watchers = repo_details.get("watchers_count") if repo_details else None

            # Generate dataset_id from replication authors (SurnameYEAR format)
            # Add data type suffix for raw/processed distinction
            data_type_suffix = (
                f"_{name_meta['data_type']}" if name_meta.get("data_type") else ""
            )
            updated_at = repo_details.get("updated_at") if repo_details else None

            base_id = generate_dataset_id(
                source="gin",
                authors=authors,
                date=updated_at,
                fallback_id=repo_name,
            )
            dataset_id = f"{base_id}{data_type_suffix}"

            # Store original repo name for reference
            gin_repo_name = repo_name

            dataset = create_dataset(
                dataset_id=dataset_id,
                name=name,
                source="gin",
                recording_modality="eeg",
                modalities=["eeg"],
                bids_version=bids_desc.get("BIDSVersion") if bids_desc else None,
                license=license_info,
                authors=authors,
                funding=funding,
                associated_paper_doi=associated_paper,
                data_processed=data_processed,
                study_domain=study_domain,
                study_design="replication"
                if name_meta["study_type"] == "replication"
                else None,
                species="Human",
                subjects_count=n_subjects,
                ages=ages_list,
                age_mean=age_mean,
                sex_distribution=sex_dist,
                handedness_distribution=hand_dist,
                contributing_labs=labs,
                source_url=source_url,
                osf_url=osf_url,
                paper_url=paper_url,
                stars=stars,
                forks=forks,
                watchers=watchers,
                dataset_modified_at=repo_details.get("updated_at")
                if repo_details
                else None,
            )

            # Store GIN repo name for reference
            dataset["gin_repo"] = gin_repo_name

            # Add project DOI if available (not yet in schema)
            if readme_meta and readme_meta.get("project_doi"):
                dataset["project_doi"] = readme_meta["project_doi"]

            datasets.append(dataset)

        except Exception as e:
            print(
                f"Warning: Error parsing repository {repo_name}: {e}", file=sys.stderr
            )
            continue

    print(f"\nFound {len(datasets)} repositories")

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
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output, default: omitted for determinism)",
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
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total datasets: {len(datasets)}")

        # Count by data type
        raw_count = sum(1 for d in datasets if "Raw" in d.get("dataset_id", ""))
        processed_count = sum(
            1 for d in datasets if "Processed" in d.get("dataset_id", "")
        )
        print(f"  Raw datasets: {raw_count}")
        print(f"  Processed datasets: {processed_count}")

        # Subject counts
        total_subjects = sum(
            d.get("demographics", {}).get("subjects_count", 0) or 0 for d in datasets
        )
        print(f"\nTotal subjects across all datasets: {total_subjects}")

        # BIDS metadata coverage
        with_bids = sum(1 for d in datasets if d.get("bids_version"))
        print(f"\nBIDS metadata available: {with_bids}/{len(datasets)}")

        # Authors
        all_authors = set()
        for d in datasets:
            if d.get("authors"):
                all_authors.update(d["authors"])
        print(f"Unique contributing authors: {len(all_authors)}")

        # Contributing labs
        n_labs_total = sum(d.get("n_contributing_labs", 0) or 0 for d in datasets)
        if n_labs_total > 0:
            print(
                f"Contributing labs (max per dataset): {max(d.get('n_contributing_labs', 0) or 0 for d in datasets)}"
            )

        # External links
        with_osf = sum(
            1 for d in datasets if d.get("external_links", {}).get("osf_url")
        )
        with_doi = sum(1 for d in datasets if d.get("associated_paper_doi"))
        print(f"\nDatasets with OSF link: {with_osf}")
        print(f"Datasets with paper DOI: {with_doi}")

        print("\n" + "-" * 60)
        print("Per-dataset details:")
        print("-" * 60)
        for d in datasets:
            print(f"\n  {d['dataset_id']}:")
            print(f"    Name: {d['name']}")
            demo = d.get("demographics", {})
            if demo.get("subjects_count"):
                print(f"    Subjects: {demo['subjects_count']}")
            if demo.get("age_min") is not None:
                age_mean = demo.get("age_mean", "?")
                print(
                    f"    Age range: {demo['age_min']}-{demo['age_max']} (mean: {age_mean})"
                )
            if d.get("bids_version"):
                print(f"    BIDS: v{d['bids_version']}")
            if d.get("license"):
                print(f"    License: {d['license']}")
            if d.get("n_contributing_labs"):
                print(f"    Labs: {d['n_contributing_labs']}")


if __name__ == "__main__":
    main()
