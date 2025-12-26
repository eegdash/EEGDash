"""Fetch neural recording datasets from Zenodo.

This script uses the Zenodo REST API to search for EEG-related datasets.
It handles rate limiting by using aggressive delays between requests.

Output: consolidated/zenodo_datasets.json
"""

import argparse
import json
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


def fetch_zenodo_datasets(
    max_results: int = 500,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch datasets from Zenodo REST API.
    
    Args:
        max_results: Maximum datasets to fetch
        access_token: Optional Zenodo API token
        
    Returns:
        List of dataset records from Zenodo
    """
    print(f"\n{'=' * 70}")
    print("Fetching datasets from Zenodo REST API")
    print(f"{'=' * 70}")
    print(f"Max results: {max_results}")
    print(f"{'=' * 70}\n")
    
    all_records = {}  # Use dict for deduplication
    headers = {"Accept": "application/json"}
    
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    
    # Search for all datasets (no specific query to avoid syntax errors)
    # Filter by type instead
    page = 1
    page_size = 50
    total_fetched = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while total_fetched < max_results:
        params = {
            "page": page,
            "size": page_size,
            "sort": "-mostrecent",
        }
        
        print(f"Page {page:3d}: ", end="", flush=True)
        
        try:
            response = requests.get(
                ZENODO_BASE_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            
            # Check status code
            if response.status_code == 429:
                # Rate limited
                print("Rate limited, waiting 60s...")
                time.sleep(60)
                consecutive_errors = 0
                continue
            
            if response.status_code >= 500:
                # Server error
                print(f"Server error ({response.status_code}), waiting 30s...")
                time.sleep(30)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many server errors, stopping.")
                    break
                continue
            
            if response.status_code != 200:
                print(f"Error {response.status_code}")
                break
            
            consecutive_errors = 0
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
            
            total_fetched = len(all_records)
            print(f"✓ {len(hits):3d} records | Total unique: {total_fetched:5d}/{max_results}")
            
            # Check if we have enough
            if total_fetched >= max_results:
                break
            
            page += 1
            
            # Aggressive delay to avoid rate limiting
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            
            if remaining != "?":
                remaining_int = int(remaining)
                if remaining_int < 5:
                    # Very close to limit
                    print("  Approaching rate limit, waiting 60s...", end="", flush=True)
                    time.sleep(60)
                    print(" Done.")
                elif remaining_int < 15:
                    # Getting close
                    time.sleep(10)
                else:
                    # Normal delay
                    time.sleep(3)
            else:
                # Unknown rate limit, use conservative delay
                time.sleep(5)
            
        except requests.Timeout:
            print("Timeout, waiting 30s...")
            consecutive_errors += 1
            time.sleep(30)
            if consecutive_errors >= max_consecutive_errors:
                print("Too many timeouts, stopping.")
                break
        except requests.RequestException as e:
            print(f"Error: {type(e).__name__}")
            consecutive_errors += 1
            time.sleep(15)
            if consecutive_errors >= max_consecutive_errors:
                print("Too many errors, stopping.")
                break
        except json.JSONDecodeError:
            print("JSON decode error")
            consecutive_errors += 1
            time.sleep(15)
            if consecutive_errors >= max_consecutive_errors:
                break
    
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
        doi = record.get("pids", {}).get("doi", {}).get("identifier", "")
        
        # Authors
        creators = []
        for creator in metadata.get("creators", []):
            if isinstance(creator, dict):
                person = creator.get("person_or_org", {})
                name = person.get("name", "")
                if name:
                    creators.append(name)
        
        # Check for neural recording keywords
        combined_text = (
            (title or "") + " " + (description or "")
        ).lower()
        
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
        print(f"    Error extracting record {record.get('id', '?')}: {e}", file=sys.stderr)
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
        "--max-results",
        type=int,
        default=500,
        help="Maximum datasets to fetch (default: 500)",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Zenodo API token (optional, for higher rate limits)",
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
        max_results=args.max_results,
        access_token=args.api_token,
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
