"""Fetch neural recording datasets from Zenodo.

This script uses the Zenodo REST API to search for EEG, MEG, and other neural recording datasets.
It searches for specific recording modalities and the BIDS standard.

Output: consolidated/zenodo_datasets.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from eegdash.records import create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically

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

        dataset = create_dataset(
            dataset_id=f"zenodo_{record_id}",
            name=title,
            source="zenodo",
            recording_modality=primary_modality,
            modalities=modalities,
            authors=creators if creators else None,
            source_url=source_url,
            dataset_doi=doi if doi else None,
        )

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

    # Statistics
    modalities = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities[mod] = modalities.get(mod, 0) + 1

    if modalities:
        print("\nModalities detected:")
        for mod in sorted(modalities.keys()):
            print(f"  {mod.upper():10s}: {modalities[mod]:4d}")

    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
