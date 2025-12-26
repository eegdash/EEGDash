"""Fetch neural recording datasets from Zenodo.

This script uses the Zenodo REST API to search for EEG, MEG, and other neural recording datasets.
It searches for specific recording modalities and the BIDS standard.

BIDS validation is performed by checking for:
- Required BIDS files (dataset_description.json)
- Optional BIDS files (participants.tsv, README, etc.)
- BIDS-like subject folder patterns (sub-XX or sub-XX.zip)
- BIDS dataset zip files (containing BIDS data)

Output: consolidated/zenodo_datasets.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from eegdash.records import create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import generate_dataset_id, save_datasets_deterministically

# Zenodo REST API endpoint
ZENODO_BASE_URL = "https://zenodo.org/api/records"

# Zenodo API key for authentication (doubles rate limit: 60→100 requests/min)
# Set via environment variable: export ZENODO_API_KEY="your_key_here"
ZENODO_API_KEY = os.environ.get("ZENODO_API_KEY", "")

# Neural recording modality keywords for searching
MODALITY_SEARCHES = {
    "eeg": "eeg",  # Lowercase to avoid rate limiting
    "meg": "meg",
    "emg": "emg",
    "fnirs": "fnirs",
    "lfp": "lfp",
    "ieeg": "ieeg",
}

# BIDS indicator files
BIDS_REQUIRED_FILES = ["dataset_description.json"]
BIDS_OPTIONAL_FILES = [
    "participants.tsv",
    "participants.json",
    "readme",
    "readme.md",
    "readme.txt",
    "changes",
]

# Patterns for BIDS subject folders/zips
BIDS_SUBJECT_PATTERN = re.compile(r"^sub-[a-zA-Z0-9]+(?:\.zip)?$", re.IGNORECASE)

# Patterns for BIDS dataset zips (contains BIDS data inside)
BIDS_DATASET_ZIP_PATTERN = re.compile(
    r"(?:bids|dataset).*\.zip$|.*_bids\.zip$", re.IGNORECASE
)


def validate_bids_structure(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate BIDS structure from file list.

    Checks for:
    - Required BIDS files (dataset_description.json)
    - Optional BIDS files (participants.tsv, README, etc.)
    - Subject-level files/zips (sub-XX or sub-XX.zip)
    - BIDS dataset zip files

    Args:
        files: List of file dictionaries from Zenodo API

    Returns:
        Dictionary with validation results:
        - is_bids: Whether dataset appears to be BIDS compliant
        - bids_files_found: List of BIDS files found
        - subject_count: Number of subject folders/zips detected
        - has_subject_zips: Whether subjects are in ZIP format
        - has_bids_zip: Whether there's a BIDS dataset zip

    """
    file_names = [f.get("key", "").lower() for f in files]
    file_names_original = [f.get("key", "") for f in files]

    # Check for required BIDS files
    found_required = []
    for bf in BIDS_REQUIRED_FILES:
        if bf.lower() in file_names:
            found_required.append(bf)

    # Check for optional BIDS files
    found_optional = []
    for bf in BIDS_OPTIONAL_FILES:
        if bf.lower() in file_names:
            found_optional.append(bf)

    # Check for subject folders/zips
    subject_files = []
    for fn in file_names_original:
        if BIDS_SUBJECT_PATTERN.match(fn):
            subject_files.append(fn)

    # Check for BIDS dataset zips
    bids_zips = []
    for fn in file_names_original:
        if BIDS_DATASET_ZIP_PATTERN.search(fn):
            bids_zips.append(fn)

    # Determine if it's BIDS
    # Accept if:
    # - has dataset_description.json OR
    # - has ≥2 subject folders/zips with BIDS naming OR
    # - has a BIDS dataset zip
    is_bids = len(found_required) > 0 or len(subject_files) >= 2 or len(bids_zips) > 0

    # Check if subjects are zipped
    has_subject_zips = any(fn.lower().endswith(".zip") for fn in subject_files)

    return {
        "is_bids": is_bids,
        "bids_files_found": found_required + found_optional,
        "subject_count": len(subject_files),
        "has_subject_zips": has_subject_zips,
        "has_bids_zip": len(bids_zips) > 0,
        "bids_zip_files": bids_zips[:5],  # Store first 5 for reference
    }


def fetch_zenodo_datasets(
    search_terms: list[str] | None = None,
    max_results: int = 500,
) -> list[dict[str, Any]]:
    """Fetch datasets from Zenodo REST API with intelligent rate limiting.

    Searches across all datasets and filters neural recording modalities locally.
    Uses progressive delays based on API response headers to avoid rate limiting.

    Authenticated requests get 100 req/min limit (vs 60 req/min for guest users).

    Args:
        search_terms: List of modality search terms (optional, for focused searches)
        max_results: Maximum total datasets to fetch across all searches

    Returns:
        List of unique dataset records from Zenodo

    """
    print(f"\n{'=' * 70}")
    print("Fetching datasets from Zenodo REST API")
    print(f"{'=' * 70}")
    print(f"Max results: {max_results}")
    if ZENODO_API_KEY:
        print("✓ Using authenticated requests (100 req/min limit)")
    else:
        print("⚠ Guest requests only (60 req/min limit)")
        print("  Tip: Set ZENODO_API_KEY env var for better rate limits")
    print(f"{'=' * 70}\n")

    all_records = {}  # Use dict for deduplication
    headers = {"Accept": "application/json"}

    # Add authentication if API key is available
    if ZENODO_API_KEY:
        headers["Authorization"] = f"Bearer {ZENODO_API_KEY}"

    # If search terms provided, search them; otherwise do broad search
    if search_terms is None:
        search_terms = ["neural recording", "EEG", "brain"]

    for search_term in search_terms:
        print(f"\nSearching for: {search_term}")
        print("-" * 70)

        page = 1
        page_size = 50
        total_for_term = 0
        consecutive_429s = 0
        max_consecutive_429s = 3

        while len(all_records) < max_results and total_for_term < (
            max_results // len(search_terms) + 100
        ):
            params = [
                ("q", search_term),
                ("page", str(page)),
                ("size", str(page_size)),
                ("sort", "-mostrecent"),
            ]

            print(f"  Page {page:3d}: ", end="", flush=True)

            try:
                response = requests.get(
                    ZENODO_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                )

                # Check status code
                if response.status_code == 429:
                    # Rate limited - use exponential backoff
                    consecutive_429s += 1
                    if consecutive_429s > max_consecutive_429s:
                        print("Too many rate limits, moving to next search term")
                        break

                    # Get reset time from headers if available
                    reset_after = response.headers.get("Retry-After")
                    if reset_after:
                        try:
                            wait_time = int(reset_after)
                        except:
                            wait_time = min(30, 2**consecutive_429s)
                    else:
                        wait_time = min(30, 2**consecutive_429s)

                    print(f"Rate limited (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                consecutive_429s = 0

                if response.status_code >= 500:
                    print(f"Server error ({response.status_code})")
                    time.sleep(10)
                    continue

                if response.status_code != 200:
                    print(f"Error {response.status_code}")
                    break

                data = response.json()
                hits = data.get("hits", {}).get("hits", [])

                if not hits:
                    print("No more results")
                    break

                # Add records to dict (deduplicates by ID)
                for record in hits:
                    record_id = record.get("id")
                    if record_id:
                        all_records[record_id] = record

                total_for_term += len(hits)
                unique_total = len(all_records)
                print(
                    f"✓ {len(hits):3d} records | Unique: {unique_total:5d}/{max_results}"
                )

                # Check if we have enough total
                if len(all_records) >= max_results:
                    break

                page += 1

                # Smart delay based on rate limit headers (respect API preferences)
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset_time = response.headers.get("X-RateLimit-Reset")

                if remaining:
                    try:
                        remaining_int = int(remaining)
                        if remaining_int <= 0:
                            # Out of requests, wait for reset
                            if reset_time:
                                try:
                                    reset_int = int(reset_time)
                                    import time as time_module

                                    current = time_module.time()
                                    wait = max(1, reset_int - current)
                                    print(
                                        f"    Rate limit exhausted, waiting {wait:.1f}s..."
                                    )
                                    time.sleep(wait)
                                except:
                                    time.sleep(60)
                            else:
                                time.sleep(60)
                        elif remaining_int < 5:
                            time.sleep(5)
                        elif remaining_int < 15:
                            time.sleep(2)
                        else:
                            time.sleep(0.5)
                    except (ValueError, TypeError):
                        time.sleep(1)
                else:
                    # No rate limit header, use conservative delay
                    time.sleep(1)

            except requests.Timeout:
                print("Timeout, skipping to next page")
                page += 1
                time.sleep(5)
            except requests.RequestException as e:
                print(f"Request error: {type(e).__name__}")
                time.sleep(5)
            except (json.JSONDecodeError, KeyError):
                print("JSON/data parse error")
                time.sleep(5)

    print(f"\n{'=' * 70}")
    print(f"Total unique records fetched: {len(all_records)}")
    print(f"{'=' * 70}\n")

    return list(all_records.values())


def extract_dataset_info(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract dataset information from Zenodo record.

    Args:
        record: Zenodo REST API record

    Returns:
        Dataset information in schema format, or None if invalid

    """
    try:
        record_id = str(record.get("id", ""))
        metadata = record.get("metadata", {})

        # Basic info
        title = metadata.get("title", "Zenodo Dataset")
        description = metadata.get("description", "")
        doi = record.get("doi") or metadata.get("doi", "")

        # Authors
        creators = []
        for creator in metadata.get("creators", []):
            if isinstance(creator, dict):
                # Handle both old and new API formats
                name = creator.get("name")
                if not name:
                    person = creator.get("person_or_org", {})
                    name = person.get("name")
                if name:
                    creators.append(name)

        # Check for neural recording keywords
        combined_text = ((title or "") + " " + (description or "")).lower()

        # Detect modalities
        modalities = []
        if any(x in combined_text for x in ["eeg", "electroencephalogr"]):
            modalities.append("eeg")
        if any(x in combined_text for x in ["meg", "magnetoencephalogr"]):
            modalities.append("meg")
        if any(x in combined_text for x in ["ieeg", "ecog", "intracranial"]):
            modalities.append("ieeg")
        if any(x in combined_text for x in ["fnirs", "nirs"]):
            modalities.append("fnirs")
        if any(x in combined_text for x in ["emg"]):
            modalities.append("emg")
        if any(x in combined_text for x in ["lfp", "local field potential"]):
            modalities.append("lfp")

        # Skip if no neural recording modality detected
        if not modalities:
            return None

        primary_modality = modalities[0]

        # Create dataset record
        source_url = record.get("links", {}).get("self_html", "")

        # Get files for BIDS validation and size calculation
        files = record.get("files", [])

        # Calculate total size
        total_size_bytes = sum(f.get("size", 0) for f in files)

        # Validate BIDS structure
        bids_validation = validate_bids_structure(files)

        # Get publication date for SurnameYEAR ID
        pub_date = metadata.get("publication_date") or record.get("created")

        # Generate SurnameYEAR dataset_id
        dataset_id = generate_dataset_id(
            source="zenodo",
            authors=creators,
            date=pub_date,
            fallback_id=record_id,
        )

        dataset = create_dataset(
            dataset_id=dataset_id,
            name=title,
            source="zenodo",
            recording_modality=primary_modality,
            modalities=modalities,
            authors=creators if creators else None,
            source_url=source_url,
            dataset_doi=doi if doi else None,
            total_files=len(files),
            size_bytes=total_size_bytes if total_size_bytes > 0 else None,
        )

        # Store original Zenodo ID for reference
        dataset["zenodo_id"] = record_id

        # Add BIDS validation results
        dataset["bids_validated"] = bids_validation["is_bids"]
        if bids_validation["bids_files_found"]:
            dataset["bids_files_found"] = bids_validation["bids_files_found"]
        if bids_validation["subject_count"] > 0:
            dataset["bids_subject_count"] = bids_validation["subject_count"]
            dataset["bids_has_subject_zips"] = bids_validation["has_subject_zips"]
        if bids_validation["has_bids_zip"]:
            dataset["bids_has_dataset_zip"] = True
            dataset["bids_zip_files"] = bids_validation["bids_zip_files"]

        return dataset

    except Exception as e:
        print(
            f"    Error extracting record {record.get('id', '?')}: {e}", file=sys.stderr
        )
        return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch neural recording datasets from Zenodo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/zenodo_datasets.json"),
        help="Output JSON file (default: consolidated/zenodo_datasets.json)",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=500,
        help="Maximum total results to fetch (default: 500)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Custom search queries (default: neural recording, EEG, brain)",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output)",
    )

    args = parser.parse_args()

    # Fetch records from Zenodo
    records = fetch_zenodo_datasets(
        search_terms=args.queries,
        max_results=args.max_per_query,
    )

    if not records:
        print("No datasets found from Zenodo", file=sys.stderr)
        sys.exit(1)

    # Extract dataset info
    print(f"Processing {len(records)} records...")
    datasets = []
    for i, record in enumerate(records, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(records)}...")

        dataset = extract_dataset_info(record)
        if dataset:
            datasets.append(dataset)

    if not datasets:
        print("No valid neural recording datasets found", file=sys.stderr)
        sys.exit(1)

    # Add digested_at if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save
    save_datasets_deterministically(datasets, args.output)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 70}")

    # BIDS validation summary
    bids_validated = sum(1 for ds in datasets if ds.get("bids_validated"))
    bids_with_subjects = sum(
        1 for ds in datasets if ds.get("bids_subject_count", 0) > 0
    )
    bids_with_zips = sum(1 for ds in datasets if ds.get("bids_has_subject_zips"))
    bids_with_dataset_zip = sum(1 for ds in datasets if ds.get("bids_has_dataset_zip"))

    print("\nBIDS Validation:")
    print(f"  Confirmed BIDS: {bids_validated}/{len(datasets)}")
    print(f"  With subject folders/zips: {bids_with_subjects}/{len(datasets)}")
    print(f"  Using subject-level ZIPs: {bids_with_zips}/{len(datasets)}")
    print(f"  With BIDS dataset ZIP: {bids_with_dataset_zip}/{len(datasets)}")

    # Statistics
    modalities = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities[mod] = modalities.get(mod, 0) + 1

    if modalities:
        print("\nModalities detected:")
        for mod in sorted(modalities.keys()):
            print(f"  {mod.upper():10s}: {modalities[mod]:4d}")

    # File statistics
    datasets_with_files = sum(1 for ds in datasets if ds.get("total_files", 0) > 0)
    print(f"\nDatasets with files: {datasets_with_files}/{len(datasets)}")

    # Total size
    total_size_bytes = sum(ds.get("size_bytes", 0) or 0 for ds in datasets)
    total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
    print(f"Total Size: {total_size_gb} GB")

    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
