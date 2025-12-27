"""Fetch neural recording BIDS datasets from SciDB (Science Data Bank).

This script searches SciDB for BIDS-compliant datasets across multiple modalities
(EEG, MEG, iEEG, fNIRS) using the SciDB query service API. Each modality is searched
separately and results are deduplicated by dataset ID to ensure uniqueness.

Output: consolidated/scidb_datasets.json (Dataset schema format)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import generate_dataset_id, save_datasets_deterministically, setup_paths

setup_paths()
from eegdash.records import create_dataset

# BIDS indicator files that must be present at root level
BIDS_REQUIRED_FILES = ["dataset_description.json"]
BIDS_OPTIONAL_FILES = [
    "participants.tsv",
    "participants.json",
    "README",
    "README.md",
    "CHANGES",
]

# File tree API endpoint
FILETREE_API_URL = (
    "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
)


def clean_html_tags(text: str) -> str:
    """Remove HTML tags from text (e.g., highlighting spans from search results)."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", text)


def check_bids_structure(
    dataset_id: str,
    version: str = "V1",
    timeout: float = 15.0,
) -> tuple[bool, list[str], str | None]:
    """Check if a SciDB dataset has BIDS structure by examining its file tree.

    Args:
        dataset_id: The SciDB dataSetId (not the record id)
        version: Dataset version (default: V1)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_bids, found_bids_files, bids_version)

    """
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Accept": "application/json, text/plain, */*",
    }

    body = {
        "dataSetId": dataset_id,
        "version": version,
        "path": f"/{version}",
        "lastIndex": 0,
        "pageSize": 200,
    }

    try:
        response = requests.post(
            FILETREE_API_URL,
            json=body,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 20000:
            return False, [], None

        files = data.get("data", [])
        file_names = [f.get("fileName", "").lower() for f in files]

        # Check for required BIDS files
        found_required = [f for f in BIDS_REQUIRED_FILES if f.lower() in file_names]
        found_optional = [f for f in BIDS_OPTIONAL_FILES if f.lower() in file_names]

        is_bids = len(found_required) > 0
        found_files = found_required + found_optional

        # TODO: Could fetch dataset_description.json to get BIDS version
        bids_version = None

        return is_bids, found_files, bids_version

    except Exception:
        return False, [], None


# Modality keywords for searching SciDB - comprehensive keyword coverage
MODALITY_KEYWORDS = {
    "eeg": [
        "eeg",
        "electroencephalography",
        "electroencephalogram",
        "scalp eeg",
        "scalp-eeg",
    ],
    "meg": ["meg", "magnetoencephalography", "magnetoencephalogram"],
    "emg": ["emg", "electromyography", "electromyogram"],
    "fnirs": [
        "fnirs",
        "fNIRS",
        "nirs",
        "near-infrared spectroscopy",
        "near infrared spectroscopy",
        "functional near-infrared",
    ],
    "lfp": [
        "lfp",
        "local field potential",
        "local field potentials",
        "field potential",
        "field potentials",
    ],
    "spike": [
        "single unit",
        "single-unit",
        "multi-unit",
        "multiunit",
        "spike",
        "spike train",
        "neuronal firing",
        "unit activity",
        "single unit activity",
        "multi-unit activity",
    ],
    "mea": [
        "mea",
        "microelectrode array",
        "microelectrode arrays",
        "utah array",
        "neuropixels",
        "depth electrode",
    ],
    "ieeg": [
        "ieeg",
        "intracranial eeg",
        "intracranial electroencephalography",
        "intracranial electroencephalogram",
        "seeg",
        "stereoelectroencephalography",
        "ecog",
        "electrocorticography",
        "corticography",
        "subdural electrode",
        "subdural grid",
        "subdural strip",
    ],
    "bids": ["bids", "brain imaging data structure", "brain imaging data structures"],
}


def search_scidb_by_query(
    query: str,
    max_results: int = 100,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Search SciDB for datasets matching a single query.

    Args:
        query: Search query string
        max_results: Maximum number of results to fetch per query
        page_size: Number of results per page (max 100)

    Returns:
        List of SciDB dataset dictionaries

    """
    base_url = "https://www.scidb.cn/api/sdb-query-service/query"
    params = {"queryCode": "", "q": query}

    all_datasets = []
    page = 1
    actual_page_size = min(page_size, 100)

    while True:
        body = {
            "fileType": ["001"],  # 001 = dataset type
            "dataSetStatus": ["PUBLIC"],
            "copyrightCode": [],
            "publishDate": [],
            "ordernum": "6",  # Relevance sort
            "rorId": [],
            "ror": "",
            "taxonomyEn": [],
            "journalNameEn": [],
            "page": page,
            "size": actual_page_size,
        }

        try:
            response = requests.post(
                base_url,
                params=params,
                json=body,
                headers={
                    "Content-Type": "application/json;charset=utf-8",
                    "Accept": "application/json, text/plain, */*",
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}", file=sys.stderr)
            break

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}", file=sys.stderr)
            break

        # Check API response code
        code = data.get("code", 0)
        if code != 20000:
            message = data.get("message", "Unknown error")
            print(f"  API error (code {code}): {message}", file=sys.stderr)
            break

        # Extract records from nested data structure
        datasets = data.get("data", {}).get("data", [])
        if not datasets:
            break

        all_datasets.extend(datasets)

        # Check if we've reached max_results or end of results
        if len(datasets) < actual_page_size or len(all_datasets) >= max_results:
            break

        page += 1

    return all_datasets[:max_results]


def fetch_scidb_datasets(
    max_results_per_modality: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch SciDB datasets across multiple modalities with BIDS requirement.

    Args:
        max_results_per_modality: Max results per modality search

    Returns:
        Dictionary mapping modality -> list of datasets

    """
    results_by_modality = {}

    for modality, keywords in MODALITY_KEYWORDS.items():
        print(f"\nSearching {modality.upper()} datasets with BIDS requirement...")
        modality_datasets = []

        for keyword in keywords:
            query = f"{keyword} BIDS"
            print(f"  Query: '{query}'...", end=" ", flush=True)
            datasets = search_scidb_by_query(
                query, max_results=max_results_per_modality // len(keywords)
            )
            modality_datasets.extend(datasets)
            print(f"found {len(datasets)}")

        results_by_modality[modality] = modality_datasets

    return results_by_modality


def deduplicate_datasets(
    results_by_modality: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Deduplicate datasets across modalities by ID.

    Args:
        results_by_modality: Datasets grouped by modality

    Returns:
        Deduplicated list of datasets

    """
    seen_ids = set()
    unique_datasets = []
    modality_map = {}

    for modality, datasets in results_by_modality.items():
        for dataset in datasets:
            dataset_id = dataset.get("id")
            if dataset_id and dataset_id not in seen_ids:
                seen_ids.add(dataset_id)
                unique_datasets.append(dataset)
                modality_map[dataset_id] = modality

    return unique_datasets, modality_map


def process_scidb_dataset(
    record: dict[str, Any],
    modality: str,
) -> dict[str, Any] | None:
    """Convert SciDB record to Dataset schema format.

    Args:
        record: SciDB dataset record
        modality: Detected modality (eeg, meg, ieeg, fnirs)

    Returns:
        Dataset schema dictionary or None if invalid

    """
    try:
        original_id = str(record.get("id", ""))
        if not original_id:
            return None

        # Extract metadata - clean HTML tags from search result highlighting
        title = clean_html_tags(record.get("titleEn", "") or record.get("titleZh", ""))
        description = clean_html_tags(
            record.get("introductionEn", "") or record.get("introductionZh", "")
        )

        # Get the dataSetId (different from id - needed for file tree API)
        data_set_id = record.get("dataSetId", "")
        version = record.get("version", "V1")

        # Extract authors - clean HTML tags
        authors = []
        for author_dict in record.get("author", []):
            name = clean_html_tags(
                author_dict.get("nameEn", "") or author_dict.get("nameZh", "")
            )
            if name:
                authors.append(name)

        # Get publication date
        pub_date = record.get("publishTime") or record.get("createTime")

        # Generate SurnameYEAR dataset_id
        dataset_id = generate_dataset_id(
            source="scidb",
            authors=authors,
            date=pub_date,
            fallback_id=original_id,
        )

        # Create dataset using create_dataset
        dataset = create_dataset(
            dataset_id=dataset_id,
            name=title or "SciDB Dataset",
            source="scidb",
            recording_modality=modality,
            modalities=[modality],
            license=record.get("copyRight", {}).get("code") or None,
            authors=authors or None,
            source_url=f"https://www.scidb.cn/en/detail?id={original_id}",
        )

        # Add SciDB-specific metadata
        dataset["scidb_id"] = original_id
        dataset["scidb_dataset_id"] = data_set_id  # For file tree API
        dataset["scidb_version"] = version

        # Extract DOI - it's directly in the record
        if doi := record.get("doi"):
            dataset["dataset_doi"] = doi
            # Add to identifiers if present
            if "identifiers" not in dataset:
                dataset["identifiers"] = {}
            dataset["identifiers"]["doi"] = doi

        if description:
            dataset["description"] = clean_html_tags(description[:1000])

        return dataset

    except Exception as e:
        print(f"  Error processing dataset {record.get('id')}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from SciDB (Science Data Bank).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/scidb_datasets.json"),
        help="Output JSON file path (default: consolidated/scidb_datasets.json).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum results per modality (default: 100).",
    )
    parser.add_argument(
        "--validate-bids",
        action="store_true",
        help="Validate BIDS structure by checking file tree (slower but more accurate).",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output, default: omitted for determinism)",
    )

    args = parser.parse_args()

    # Fetch across all modalities
    print("Fetching BIDS datasets from SciDB across multiple modalities...")
    results_by_modality = fetch_scidb_datasets(
        max_results_per_modality=args.max_results
    )

    # Deduplicate
    unique_datasets, modality_map = deduplicate_datasets(results_by_modality)

    if not unique_datasets:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Convert to Dataset schema
    print(f"\nProcessing {len(unique_datasets)} unique datasets...")
    datasets = []
    bids_validated_count = 0
    bids_confirmed_count = 0

    for i, record in enumerate(unique_datasets, 1):
        modality = modality_map[record.get("id")]
        dataset = process_scidb_dataset(record, modality)
        if dataset:
            # Optionally validate BIDS structure
            if args.validate_bids:
                data_set_id = record.get("dataSetId", "")
                version = record.get("version", "V1")
                if data_set_id:
                    is_bids, bids_files, bids_version = check_bids_structure(
                        data_set_id, version
                    )
                    bids_validated_count += 1
                    if is_bids:
                        bids_confirmed_count += 1
                        dataset["bids_validated"] = True
                        dataset["bids_files_found"] = bids_files
                        if bids_version:
                            dataset["bids_version"] = bids_version
                    else:
                        dataset["bids_validated"] = False

                    if i % 10 == 0:
                        print(f"  Validated {i}/{len(unique_datasets)} datasets...")

            datasets.append(dataset)

    if args.validate_bids:
        print(
            f"\nBIDS Validation: {bids_confirmed_count}/{bids_validated_count} confirmed as BIDS"
        )

    # Add digested_at timestamp if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")

    # Statistics
    modalities_found = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities_found[mod] = modalities_found.get(mod, 0) + 1

    print("\nDatasets by modality:")
    for mod in sorted(MODALITY_KEYWORDS.keys()):
        count = modalities_found.get(mod, 0)
        print(f"  {mod.upper()}: {count}")

    datasets_with_doi = sum(
        1 for ds in datasets if ds.get("identifiers", {}).get("doi")
    )
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
