"""Fetch neural recording BIDS datasets from data.ru.nl (Radboud RDM repository).

This script searches the Radboud University Research Data Management repository
(data.ru.nl) for BIDS-compliant datasets across neural recording modalities using
the REST API endpoint. It retrieves dataset information and outputs in the EEGDash
Dataset schema format.

Data.ru.nl is the Research Data Management system of Radboud University, hosting
research data from Radboud University and affiliated institutions with support
for BIDS datasets.

The script uses the public REST API at /api/search/collections/published to discover
datasets matching specified modalities (EEG, MEG, EMG, fNIRS, LFP, iEEG).

Output: consolidated/datarn_datasets.json (Dataset schema format)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _http import HTTPStatusError, RequestError, build_headers, request_json
from _keywords import DATARN_MODALITY_SEARCHES as MODALITY_SEARCHES
from _serialize import (
    extract_subjects_count,
    generate_dataset_id,
    save_datasets_deterministically,
    setup_paths,
)

setup_paths()
from eegdash.records import create_dataset

# Base URL and API endpoint
DATARN_BASE_URL = "https://data.ru.nl"
DATARN_API_URL = f"{DATARN_BASE_URL}/api/search/collections/published"

# Default access levels
DEFAULT_ACCESS_LEVELS = ["OPEN_ACCESS", "REGISTERED_ACCESS", "RESTRICTED_ACCESS"]


def fetch_datasets_from_api(
    query: str,
    access_levels: list[str] | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch datasets from data.ru.nl REST API.

    Args:
        query: Search query term (e.g., "EEG", "MEG", "BIDS")
        access_levels: List of access levels to include
        timeout: Request timeout in seconds

    Returns:
        List of dataset records from API

    """
    if access_levels is None:
        access_levels = DEFAULT_ACCESS_LEVELS

    params = {
        "access": ",".join(access_levels),
        "q": query,
    }

    try:
        headers = build_headers()

        print(f"Querying API with: q={query}, access={','.join(access_levels)}")
        data, _response = request_json(
            "get",
            DATARN_API_URL,
            params=params,
            headers=headers,
            timeout=timeout,
            raise_for_status=True,
            raise_for_request=True,
        )
        if data is None:
            raise ValueError("Empty response")

        # API returns documents in a nested structure with "documents" and "document" keys
        datasets = []
        if isinstance(data, dict):
            # Handle the actual data.ru.nl API format: {"documents": [{"document": {...}, "id": "..."}]}
            documents = data.get("documents", [])
            for doc_wrapper in documents:
                if isinstance(doc_wrapper, dict):
                    # Extract the document and ID
                    doc = doc_wrapper.get("document", doc_wrapper)
                    doc_id = doc_wrapper.get("id", doc.get("id", ""))

                    # Enrich document with ID
                    if "persistentId" not in doc and doc_id:
                        doc["persistentId"] = doc_id
                    datasets.append(doc)

        print(f"Retrieved {len(datasets)} records from API for query '{query}'")
        return datasets

    except (RequestError, HTTPStatusError, ValueError) as e:
        print(f"Error fetching datasets for query '{query}': {e}", file=sys.stderr)
        return []


def extract_dataset_info(
    api_record: dict[str, Any],
    modality: str,
) -> dict[str, Any] | None:
    """Extract dataset information from API record.

    Args:
        api_record: Record from data.ru.nl API
        modality: Detected modality for this record

    Returns:
        Dataset schema document or None if extraction fails

    """
    try:
        # Extract basic info from API record
        # data.ru.nl API fields: title, description, authors[], doi, collectionIdentifier, relativeUrl, etc.
        original_id = (
            api_record.get("persistentId")
            or api_record.get("collectionIdentifier")
            or api_record.get("id", "datarn_unknown")
        )

        name = api_record.get("title", original_id)

        # Build URL from relative path if available
        relative_url = api_record.get("relativeUrl")
        if relative_url:
            url = f"{DATARN_BASE_URL}{relative_url}"
        else:
            url = f"{DATARN_BASE_URL}/dataset.xhtml?persistentId={original_id}"

        # Extract metadata
        description = api_record.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)
        description = description[:500] if description else ""

        # Extract authors
        authors = []
        if "authors" in api_record and api_record["authors"]:
            auth = api_record["authors"]
            if isinstance(auth, list):
                authors = auth[:10]  # Limit to first 10 authors
            elif isinstance(auth, str):
                authors = [auth]

        # If no authors, try contributorsDisplayNameList or managersDisplayNameList
        if not authors:
            authors = api_record.get("contributorsDisplayNameList", [])
        if not authors:
            authors = api_record.get("managersDisplayNameList", [])

        # Extract DOI
        doi = api_record.get("doi")

        # Detect BIDS from metadata
        is_bids = False
        search_text = f"{name} {description}".lower()
        if "bids" in search_text:
            is_bids = True

        # Get publication date
        pub_date = api_record.get("releaseDate") or api_record.get("publicationDate")

        # Generate SurnameYEAR dataset_id
        dataset_id = generate_dataset_id(
            source="datarn",
            authors=authors,
            date=pub_date,
            fallback_id=original_id,
        )

        # Extract subject count from description using shared utility
        subjects_count = extract_subjects_count(description)

        # Create Dataset document
        dataset = create_dataset(
            dataset_id=dataset_id,
            name=name,
            source="datarn",
            recording_modality=modality,
            authors=authors,
            dataset_doi=doi,
            source_url=url,
            subjects_count=subjects_count if subjects_count > 0 else None,
        )

        # Store original data.ru.nl ID for reference
        dataset["datarn_id"] = original_id

        # Store demographics for downstream use (for manifest->digester)
        if subjects_count > 0:
            dataset["demographics"] = {
                "subjects_count": subjects_count,
                "ages": [],
            }

        # Add BIDS info if detected
        if is_bids and "metadata" in dataset:
            dataset["metadata"]["bids_compatible"] = True

        return dataset

    except Exception as e:
        original_id = api_record.get("id", api_record.get("persistentId", "unknown"))
        print(f"Error extracting dataset {original_id}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from data.ru.nl (Radboud RDM).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/datarn_datasets.json"),
        help="Output JSON file path (default: consolidated/datarn_datasets.json).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Fetching neural recording datasets from data.ru.nl (Radboud RDM)")
    print("=" * 70)

    # Collect all datasets from all modality searches
    all_records = []

    for modality_key, modality_search_term in MODALITY_SEARCHES.items():
        print(f"\nSearching for {modality_search_term} datasets...")
        records = fetch_datasets_from_api(
            query=modality_search_term,
            timeout=args.timeout,
        )
        all_records.extend(records)
        time.sleep(0.5)  # Rate limiting

    # Also search for BIDS keyword
    print("\nSearching for BIDS datasets...")
    bids_records = fetch_datasets_from_api(
        query="BIDS",
        timeout=args.timeout,
    )
    all_records.extend(bids_records)

    # Deduplicate by persistent ID
    seen_ids = set()
    unique_records = []
    for record in all_records:
        persistent_id = record.get("persistentId") or record.get("id")
        if persistent_id and persistent_id not in seen_ids:
            seen_ids.add(persistent_id)
            unique_records.append(record)

    print(f"\nTotal unique records after deduplication: {len(unique_records)}")

    if not unique_records:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset info
    print(f"\nProcessing {len(unique_records)} records...")
    datasets = []
    for i, record in enumerate(unique_records, 1):
        # Try to detect modality from record
        name_desc = f"{record.get('name', '')} {record.get('description', '')}".lower()
        detected_modality = None

        for mod_key, mod_name in MODALITY_SEARCHES.items():
            if mod_name.lower() in name_desc or mod_key in name_desc:
                detected_modality = mod_key
                break

        if not detected_modality:
            detected_modality = "unknown"

        dataset = extract_dataset_info(record, detected_modality)
        if dataset:
            datasets.append(dataset)

        # Rate limiting
        if i % 10 == 0:
            print(f"  Processed {i}/{len(unique_records)} records...")
            time.sleep(0.5)

    if not datasets:
        print("No valid datasets extracted. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Add digested_at timestamp if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 70}")

    # Statistics
    modalities_found = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities_found[mod] = modalities_found.get(mod, 0) + 1

    if modalities_found:
        print("\nDatasets by modality:")
        for mod in sorted(modalities_found.keys()):
            count = modalities_found[mod]
            print(f"  {mod.upper():12s}: {count:4d}")

    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
