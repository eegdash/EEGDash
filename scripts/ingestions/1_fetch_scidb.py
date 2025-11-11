"""Fetch EEG BIDS datasets from SciDB (Science Data Bank).

This script searches SciDB for public datasets containing both "EEG" and "BIDS" keywords
using the SciDB query service API. It retrieves comprehensive metadata including DOIs,
CSTR identifiers, descriptions, authors, and file information.

Output: consolidated/scidb_datasets.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def search_scidb(
    query: str,
    size: int = 100,
    page_size: int = 100,
    public_only: bool = True,
) -> list[dict[str, Any]]:
    """Search SciDB for datasets matching the query.

    Args:
        query: Search query string
        size: Maximum number of results to fetch
        page_size: Number of results per page (max 100)
        public_only: Filter for public datasets only (default: True)

    Returns:
        List of SciDB dataset dictionaries

    """
    base_url = "https://www.scidb.cn/api/sdb-query-service/query"

    # Build query parameters
    params = {
        "queryCode": "",
        "q": query,
    }

    print(f"Searching SciDB with query: {query}")
    print(f"Max results to fetch: {size}")
    print(f"Results per page: {min(page_size, 100)}")
    if public_only:
        print("Filtering: Public datasets only")

    all_datasets = []
    page = 1
    actual_page_size = min(page_size, 100)  # SciDB max appears to be 100

    while True:
        print(f"\nFetching page {page}...", end=" ", flush=True)

        # Build request body
        body = {
            "fileType": ["001"],  # 001 = dataset type
            "dataSetStatus": ["PUBLIC"] if public_only else [],
            "copyrightCode": [],
            "publishDate": [],
            "ordernum": "6",  # Sort order (6 appears to be relevance)
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
            print(f"\nError fetching page {page}: {e}", file=sys.stderr)
            break

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"\nError parsing JSON response: {e}", file=sys.stderr)
            break

        # Check API response code
        code = data.get("code", 0)
        if code != 20000:
            message = data.get("message", "Unknown error")
            print(f"\nAPI Error (code {code}): {message}", file=sys.stderr)
            break

        # Extract records from nested data structure: response.data.data[]
        datasets = data.get("data", {}).get("data", [])

        if not datasets:
            print("No more results")
            break

        print(f"Got {len(datasets)} records (total: {len(all_datasets)})")
        all_datasets.extend(datasets)

        # If we got fewer results than page_size, we've reached the end
        if len(datasets) < actual_page_size:
            print("Reached end of results")
            break

        # Check if we've reached the requested size (only if size > 0)
        if size > 0 and len(all_datasets) >= size:
            print(f"Reached target limit ({len(all_datasets)} datasets)")
            break

        page += 1

        # Be nice to the API
        time.sleep(0.5)

    print(f"\nTotal datasets fetched: {len(all_datasets)}")
    return all_datasets[:size]  # Trim to exact size


def extract_dataset_info(record: dict) -> dict[str, Any]:
    """Extract relevant information from a SciDB dataset record.

    Args:
        record: SciDB dataset record dictionary

    Returns:
        Cleaned dataset dictionary with normalized fields

    """
    # Extract IDs
    dataset_id = record.get("id", "")
    doi = record.get("doi", "")
    cstr = record.get("cstr", "")
    pid = record.get("pid", "")

    # Extract basic metadata - API uses titleEn/titleZh pattern
    title_en = record.get("titleEn", "")
    title_zh = record.get("titleZh", "")
    abstract_en = record.get("introductionEn", "")
    abstract_zh = record.get("introductionZh", "")

    # Extract keywords
    keywords_en = record.get("keywordEn", [])
    if isinstance(keywords_en, str):
        keywords_en = [keywords_en] if keywords_en else []
    keywords_zh = record.get("keywordZh", [])
    if isinstance(keywords_zh, str):
        keywords_zh = [keywords_zh] if keywords_zh else []

    # Extract authors with affiliations
    authors = []
    for author in record.get("author", []):
        author_info = {
            "name_en": author.get("nameEn", ""),
            "name_zh": author.get("nameZh", ""),
        }
        if author.get("email"):
            author_info["email"] = author["email"]

        # Handle affiliations (organizations)
        affiliations = []
        for org in author.get("organizations", []):
            aff_info = {
                "name_en": org.get("nameEn", ""),
                "name_zh": org.get("nameZh", ""),
            }
            # Extract ROR ID if available
            for orgid in org.get("orgids", []):
                if orgid.get("type") == "ROR":
                    aff_info["ror_id"] = orgid.get("value")
                    break
            affiliations.append(aff_info)
        if affiliations:
            author_info["affiliations"] = affiliations

        authors.append(author_info)

    # Extract dates
    publication_date = record.get("dataSetPublishDate", "")
    create_time = record.get("create_time", "")
    update_time = record.get("update_time", "")

    # Extract access and status
    data_set_status = record.get("dataSetStatus", "")

    # Extract license information
    license_info = record.get("copyRight", {})
    license_code = license_info.get("code", "")

    # Extract file information
    file_size_bytes = record.get("size", 0)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else 0

    # Extract metrics
    download_count = record.get("download", 0)
    visit_count = record.get("visit", 0)

    # Extract URL
    url = record.get("url", "")
    if not url and dataset_id:
        url = f"https://www.scidb.cn/en/detail?id={dataset_id}"

    # Build dataset dictionary
    dataset = {
        "dataset_id": str(dataset_id),
        "doi": doi,
        "cstr": cstr,
        "pid": pid,
        "title": title_en or title_zh,
        "title_en": title_en,
        "title_zh": title_zh,
        "description": abstract_en or abstract_zh,
        "description_en": abstract_en,
        "description_zh": abstract_zh,
        "authors": authors,
        "keywords_en": keywords_en,
        "keywords_zh": keywords_zh,
        "publication_date": publication_date,
        "created": create_time,
        "modified": update_time,
        "status": data_set_status,
        "license": license_code,
        "file_size_mb": file_size_mb,
        "file_size_bytes": file_size_bytes,
        "download_count": download_count,
        "visit_count": visit_count,
        "url": url,
        "source": "scidb",
    }

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEG BIDS datasets from SciDB (Science Data Bank)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="EEG BIDS",
        help='Search query (default: "EEG BIDS").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/scidb_datasets.json"),
        help="Output JSON file path (default: consolidated/scidb_datasets.json).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Maximum number of results to fetch (default: 10000, use 0 to fetch all pages).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of results per page (max 100, default: 100).",
    )
    parser.add_argument(
        "--include-restricted",
        action="store_true",
        help="Include restricted/private datasets (default: public only).",
    )

    args = parser.parse_args()

    # Search SciDB
    records = search_scidb(
        query=args.query,
        size=args.size,
        page_size=args.page_size,
        public_only=not args.include_restricted,
    )

    if not records:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset information
    print("\nExtracting dataset information...")
    datasets = []
    for record in records:
        try:
            dataset = extract_dataset_info(record)
            datasets.append(dataset)
        except Exception as e:
            print(
                f"Warning: Error extracting info for record {record.get('id')}: {e}",
                file=sys.stderr,
            )
            continue

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2, ensure_ascii=False)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")
    print("\nDataset Statistics:")

    # Count by status
    statuses = {}
    for ds in datasets:
        status = ds.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    print("\nBy Status:")
    for status, count in sorted(statuses.items(), key=lambda x: x[1], reverse=True):
        print(f"  {status}: {count}")

    # Datasets with files
    datasets_with_files = sum(1 for ds in datasets if ds.get("file_count", 0) > 0)
    print(f"\nDatasets with files: {datasets_with_files}/{len(datasets)}")

    # Total size
    total_size_mb = sum(ds.get("total_size_mb", 0) for ds in datasets)
    total_size_gb = round(total_size_mb / 1024, 2)
    print(f"Total Size: {total_size_gb} GB ({total_size_mb} MB)")

    # Datasets with DOI
    datasets_with_doi = sum(1 for ds in datasets if ds.get("doi", ""))
    print(f"Datasets with DOI: {datasets_with_doi}/{len(datasets)}")

    # Datasets with CSTR
    datasets_with_cstr = sum(1 for ds in datasets if ds.get("cstr", ""))
    print(f"Datasets with CSTR: {datasets_with_cstr}/{len(datasets)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
