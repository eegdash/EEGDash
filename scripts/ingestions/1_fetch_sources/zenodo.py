"""Fetch neural recording datasets from Zenodo.

This script uses the Zenodo REST API to search for EEG, MEG, fNIRS, and other
neural recording BIDS datasets.

Key features:
- Loads ZENODO_API_KEY from .env.zenodo file or environment variable
- Authenticated requests get 100 req/min (vs 60 req/min for guests)
- Filters for: open access + dataset type
- Focuses on EEG, MEG, fNIRS modalities (excludes iEEG, spiking, LFP)
- Uses simple keyword searches for reliability

BIDS validation is performed by checking for:
- Required BIDS files (dataset_description.json)
- Optional BIDS files (participants.tsv, README, etc.)
- BIDS-like subject folder patterns (sub-XX or sub-XX.zip)
- BIDS dataset zip files (containing BIDS data)

Output: consolidated/zenodo_full.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bids import (
    BIDS_DATASET_ZIP_PATTERN,
    validate_bids_structure_from_files,
)
from _http import HTTPStatusError, RequestError, TimeoutException, request_response
from _keywords import (
    ZENODO_EXCLUDED_MODALITIES as EXCLUDED_MODALITIES,
)
from _keywords import (
    ZENODO_SIMPLE_SEARCHES as SIMPLE_SEARCHES,
)
from _keywords import (
    ZENODO_TARGET_MODALITIES as TARGET_MODALITIES,
)
from _serialize import (
    extract_subjects_count,
    generate_dataset_id,
    save_datasets_deterministically,
    setup_paths,
)
from dotenv import load_dotenv

setup_paths()
from eegdash.schemas import create_dataset

# Zenodo REST API endpoint
ZENODO_BASE_URL = "https://zenodo.org/api/records"

# Load API key from .env.zenodo, .env or environment
_env_files = [
    Path(__file__).parent.parent.parent.parent / ".env.zenodo",
    Path(__file__).parent.parent.parent.parent / ".env",
]
for env_file in _env_files:
    if env_file.exists():
        load_dotenv(env_file)

# Zenodo API key for authentication (100 req/min vs 60 req/min for guests)
ZENODO_API_KEY = os.environ.get("ZENODO_API_KEY", "")


def fetch_zenodo_datasets(
    search_terms: list[str] | None = None,
    max_results: int = 1000,
    require_bids: bool = True,
) -> list[dict[str, Any]]:
    """Fetch datasets from Zenodo REST API with simple keyword searches.

    Uses simple keyword queries with URL parameter filters for:
    - resource_type=dataset (only datasets, not publications/software)
    - access_status=open (only open access)

    Authenticated requests get 100 req/min limit (vs 60 req/min for guests).

    Args:
        search_terms: List of search queries (default: SIMPLE_SEARCHES)
        max_results: Maximum total datasets to fetch across all searches
        require_bids: If True, only return datasets with BIDS indicators

    Returns:
        List of unique dataset records from Zenodo

    """
    print(f"\n{'=' * 70}")
    print("Fetching datasets from Zenodo REST API")
    print(f"{'=' * 70}")
    print(f"Max results: {max_results}")
    print(f"Require BIDS: {require_bids}")
    print(f"Target modalities: {', '.join(TARGET_MODALITIES)}")
    print(f"Excluded modalities: {', '.join(EXCLUDED_MODALITIES)}")

    if ZENODO_API_KEY:
        print(f"✓ Using authenticated requests (API key: {ZENODO_API_KEY[:10]}...)")
    else:
        print("⚠ Guest requests only (60 req/min limit)")
        print("  Tip: Create .env.zenodo file with ZENODO_API_KEY=your_key")
        print(
            "  Get key at: https://zenodo.org/account/settings/applications/tokens/new/"
        )
    print(f"{'=' * 70}\n")

    all_records = {}  # Use dict for deduplication by record ID
    headers = {"Accept": "application/json"}

    # Add authentication if API key is available
    if ZENODO_API_KEY:
        headers["Authorization"] = f"Bearer {ZENODO_API_KEY}"

    # Use simple searches by default (more reliable than complex Elasticsearch)
    if search_terms is None:
        search_terms = SIMPLE_SEARCHES

    for search_term in search_terms:
        print(f"\n{'=' * 60}")
        print(f"Search: {search_term}")
        print(f"{'=' * 60}")

        page = 1
        page_size = 100  # Max allowed by Zenodo
        total_for_term = 0
        consecutive_429s = 0
        max_consecutive_429s = 5

        while len(all_records) < max_results:
            # Build query - filters go as separate URL parameters, not in q
            # This matches the web search: q=EEG BIDS&f=access_status:open&f=resource_type:dataset
            params = {
                "q": search_term,
                "page": page,
                "size": page_size,
                "sort": "bestmatch",  # Most relevant first
                "access_status": "open",  # Only open access
                "resource_type": "dataset",  # Only datasets
            }

            print(f"  Page {page:3d}: ", end="", flush=True)

            try:
                response = request_response(
                    "get",
                    ZENODO_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                    raise_for_request=True,
                )

                # Check status code
                if response and response.status_code == 429:
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

                if response and response.status_code >= 500:
                    print(f"Server error ({response.status_code})")
                    time.sleep(10)
                    continue

                if not response or response.status_code != 200:
                    status = response.status_code if response else "no response"
                    print(f"Error {status}")
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

            except TimeoutException:
                print("Timeout, skipping to next page")
                page += 1
                time.sleep(5)
            except (RequestError, HTTPStatusError) as e:
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
        Dataset information in schema format, or None if invalid/excluded

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

        # Check for excluded modalities FIRST - skip these datasets entirely
        for excluded in EXCLUDED_MODALITIES:
            if excluded in combined_text:
                return None  # Skip this dataset

        # Detect target modalities only
        modalities = []
        if any(
            x in combined_text
            for x in ["eeg", "electroencephalogr", "electroencephalogram"]
        ):
            modalities.append("eeg")
        if any(x in combined_text for x in ["meg", "magnetoencephalogr"]):
            modalities.append("meg")
        if any(x in combined_text for x in ["fnirs", "nirs", "near-infrared"]):
            modalities.append("fnirs")

        # Skip if no target modality detected
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
        bids_validation = validate_bids_structure_from_files(
            files,
            name_key="key",
            dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
            dataset_zip_matcher="search",
        )

        # Get publication date for SurnameYEAR ID
        pub_date = metadata.get("publication_date") or record.get("created")

        # Generate SurnameYEAR dataset_id
        dataset_id = generate_dataset_id(
            source="zenodo",
            authors=creators,
            date=pub_date,
            fallback_id=record_id,
        )

        # Extract subject count from description using shared utility
        subjects_count = extract_subjects_count(description)

        dataset = create_dataset(
            dataset_id=dataset_id,
            name=title,
            source="zenodo",
            recording_modality=primary_modality,
            datatypes=modalities,
            authors=creators if creators else None,
            source_url=source_url,
            dataset_doi=doi if doi else None,
            total_files=len(files),
            size_bytes=total_size_bytes if total_size_bytes > 0 else None,
            subjects_count=subjects_count if subjects_count > 0 else None,
        )

        # Store original Zenodo ID for reference
        dataset["zenodo_id"] = record_id

        # Store demographics for downstream use
        dataset["demographics"] = {
            "subjects_count": subjects_count,
            "ages": [],
        }

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
        default=Path("consolidated/zenodo_full.json"),
        help="Output JSON file (default: consolidated/zenodo_full.json)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum total results to fetch (default: 1000)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Custom search queries (default: optimized EEG/MEG/fNIRS + BIDS)",
    )
    parser.add_argument(
        "--no-bids-filter",
        action="store_true",
        help="Don't filter out non-BIDS datasets (include all)",
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
        max_results=args.max_results,
        require_bids=not args.no_bids_filter,
    )

    if not records:
        print("No datasets found from Zenodo", file=sys.stderr)
        sys.exit(1)

    # Extract dataset info (filter by modality and exclude unwanted)
    print(f"\nProcessing {len(records)} records...")
    print(f"Filtering for modalities: {TARGET_MODALITIES}")
    print(f"Excluding: {EXCLUDED_MODALITIES}")

    datasets = []
    excluded_count = 0
    no_modality_count = 0

    for i, record in enumerate(records, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(records)}...")

        dataset = extract_dataset_info(record)
        if dataset:
            datasets.append(dataset)
        else:
            # Track why it was excluded
            metadata = record.get("metadata", {})
            combined_text = (
                (metadata.get("title", "") or "")
                + " "
                + (metadata.get("description", "") or "")
            ).lower()

            if any(excl in combined_text for excl in EXCLUDED_MODALITIES):
                excluded_count += 1
            else:
                no_modality_count += 1

    print(f"\n  ✓ Kept: {len(datasets)}")
    print(f"  ✗ Excluded (iEEG/LFP/etc): {excluded_count}")
    print(f"  ✗ No target modality: {no_modality_count}")

    if not datasets:
        print("No valid EEG/MEG/fNIRS datasets found", file=sys.stderr)
        sys.exit(1)

    # Optionally filter for BIDS-only
    if not args.no_bids_filter:
        bids_datasets = [ds for ds in datasets if ds.get("bids_validated")]
        if bids_datasets:
            print(
                f"\nFiltering for BIDS datasets: {len(bids_datasets)}/{len(datasets)}"
            )
            datasets = bids_datasets
        else:
            print("\nWarning: No BIDS-validated datasets found, keeping all")

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
