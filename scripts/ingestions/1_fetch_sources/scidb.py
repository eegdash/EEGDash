"""Fetch neural recording BIDS datasets from SciDB (Science Data Bank).

This script searches SciDB for BIDS-compliant datasets across multiple modalities
(EEG, MEG, iEEG, fNIRS) using the SciDB query service API. Each modality is searched
separately and results are deduplicated by dataset ID to ensure uniqueness.

Output: consolidated/scidb_datasets.json (Dataset schema format)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from eegdash.records import create_dataset

# Modality keywords for searching SciDB
MODALITY_KEYWORDS = {
    "eeg": ["eeg", "electroencephalography"],
    "meg": ["meg", "magnetoencephalography"],
    "ieeg": ["ieeg", "intracranial eeg", "ecog", "electrocorticography", "seeg"],
    "fnirs": ["fnirs", "nirs", "fnirs"],
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
            datasets = search_scidb_by_query(query, max_results=max_results_per_modality // len(keywords))
            modality_datasets.extend(datasets)
            print(f"found {len(datasets)}")
        
        results_by_modality[modality] = modality_datasets
    
    return results_by_modality


def deduplicate_datasets(results_by_modality: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
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
        dataset_id = str(record.get("id", ""))
        if not dataset_id:
            return None

        # Extract metadata
        title = record.get("titleEn", "") or record.get("titleZh", "")
        description = record.get("introductionEn", "") or record.get("introductionZh", "")
        
        # Extract authors
        authors = []
        for author_dict in record.get("author", []):
            name = author_dict.get("nameEn", "") or author_dict.get("nameZh", "")
            if name:
                authors.append(name)

        # Create dataset using create_dataset
        dataset = create_dataset(
            dataset_id=f"scidb_{dataset_id}",
            name=title or "SciDB Dataset",
            source="scidb",
            recording_modality=modality,
            modalities=[modality],
            license=record.get("copyRight", {}).get("code") or None,
            authors=authors or None,
            source_url=f"https://www.scidb.cn/en/detail?id={dataset_id}",
        )

        # Add SciDB-specific metadata
        dataset["scidb_id"] = dataset_id
        if doi := record.get("doi"):
            dataset["dataset_doi"] = doi
        if description:
            dataset["description"] = description[:1000]

        return dataset

    except Exception as e:
        print(f"  Error processing dataset {record.get('id')}: {e}", file=sys.stderr)
        return None

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

    args = parser.parse_args()

    # Fetch across all modalities
    print("Fetching BIDS datasets from SciDB across multiple modalities...")
    results_by_modality = fetch_scidb_datasets(max_results_per_modality=args.max_results)

    # Deduplicate
    unique_datasets, modality_map = deduplicate_datasets(results_by_modality)
    
    if not unique_datasets:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Convert to Dataset schema
    print(f"\nProcessing {len(unique_datasets)} unique datasets...")
    datasets = []
    for record in unique_datasets:
        modality = modality_map[record.get("id")]
        dataset = process_scidb_dataset(record, modality)
        if dataset:
            datasets.append(dataset)

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2)

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
    
    datasets_with_doi = sum(1 for ds in datasets if ds.get("identifiers", {}).get("doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")
    
    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")
    
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
