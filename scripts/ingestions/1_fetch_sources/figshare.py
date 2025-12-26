"""Fetch EEG BIDS datasets from Figshare.

This script searches Figshare for datasets containing both "EEG" and "BIDS" keywords
using the Figshare API v2. It retrieves comprehensive metadata including DOIs,
descriptions, files, authors, and download URLs.

Output: consolidated/figshare_datasets.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Add parent paths for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eegdash.records import create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically


def search_figshare(
    query: str,
    size: int = 100,
    page_size: int = 100,
    item_type: int = 3,  # 3 = dataset
) -> list[dict[str, Any]]:
    """Search Figshare for datasets matching the query.

    Args:
        query: Search query string
        size: Maximum number of results to fetch
        page_size: Number of results per page (max 1000)
        item_type: Figshare item type (3=dataset, 1=figure, 2=media, etc.)

    Returns:
        List of Figshare article dictionaries

    """
    base_url = "https://api.figshare.com/v2/articles/search"

    print(f"Searching Figshare with query: {query}")
    print(f"Item type: {item_type} (dataset)")
    print(f"Max results to fetch: {size}")
    print(f"Results per page: {min(page_size, 1000)}")

    all_articles = []
    page = 1
    actual_page_size = min(page_size, 1000)  # Figshare max is 1000

    while True:
        print(f"\nFetching page {page}...", end=" ", flush=True)

        # Build request payload
        payload = {
            "search_for": query,
            "item_type": item_type,
            "page": page,
            "page_size": actual_page_size,
        }

        try:
            response = requests.post(
                base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"\nError fetching page {page}: {e}", file=sys.stderr)
            break

        articles = response.json()

        if not articles:
            print("No more results")
            break

        print(f"Got {len(articles)} articles")
        all_articles.extend(articles)

        # Check if we've reached the requested size (only if size > 0)
        if size > 0 and len(all_articles) >= size:
            print(f"Reached limit ({len(all_articles)} articles)")
            break

        # If we got fewer results than page_size, we've reached the end
        if len(articles) < actual_page_size:
            print("Reached end of results")
            break

        page += 1

        # Be nice to the API
        time.sleep(0.5)

    print(f"\nTotal articles fetched: {len(all_articles)}")
    # Return all articles if size=0, otherwise trim to exact size
    return all_articles if size <= 0 else all_articles[:size]


def get_article_details(article_id: int) -> dict[str, Any]:
    """Fetch detailed information for a specific article.

    Args:
        article_id: Figshare article ID

    Returns:
        Detailed article dictionary

    """
    url = f"https://api.figshare.com/v2/articles/{article_id}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(
            f"Warning: Error fetching details for article {article_id}: {e}",
            file=sys.stderr,
        )
        return {}


def detect_modalities(article: dict) -> list[str]:
    """Detect modalities from article metadata.
    
    Args:
        article: Figshare article dictionary
        
    Returns:
        List of detected modalities
    """
    modalities = []
    
    # Search in title, description, tags, categories
    search_text = " ".join([
        article.get("title", ""),
        article.get("description", ""),
        " ".join(article.get("tags", [])),
        " ".join([c.get("title", "") for c in article.get("categories", [])]),
    ]).lower()
    
    # Map keywords to modalities
    modality_keywords = {
        "eeg": ["eeg"],
        "meg": ["meg"],
        "ieeg": ["ieeg", "intracranial"],
        "emg": ["emg"],
        "fnirs": ["fnirs", "fNIRS", "nirs"],
        "lfp": ["lfp", "local field potential"],
    }
    
    for modality, keywords in modality_keywords.items():
        if any(kw in search_text for kw in keywords):
            modalities.append(modality)
    
    # Default to EEG if nothing detected (since we search for "EEG BIDS")
    if not modalities:
        modalities = ["eeg"]
    
    return modalities


def extract_dataset_info(article: dict, fetch_details: bool = False, digested_at: str | None = None) -> dict[str, Any]:
    """Extract relevant information from a Figshare article and normalize to Dataset schema.

    Args:
        article: Figshare article dictionary
        fetch_details: Whether to fetch full details (slower but more complete)
        digested_at: ISO 8601 timestamp for digested_at field

    Returns:
        Dataset schema document

    """
    # Get basic info from search results
    article_id = article.get("id", "")
    title = article.get("title", "")
    doi = article.get("doi", "")

    # Fetch full details if requested
    if fetch_details and article_id:
        details = get_article_details(article_id)
        if details:
            article = details

    # Extract metadata
    description = article.get("description", "")
    
    # Extract URLs
    url_public_html = article.get("url_public_html", "")

    # Extract authors - normalize to simple list of names
    author_names = []
    for author in article.get("authors", []):
        name = author.get("full_name", "")
        if name:
            author_names.append(name)

    # Extract license
    license_info = article.get("license", {})
    license_name = license_info.get("name", "") if license_info else None

    # Calculate total size
    total_size_bytes = sum(f.get("size", 0) for f in article.get("files", []))
    
    # Detect modalities from content
    modalities = detect_modalities(article)
    recording_modality = modalities[0] if modalities else "eeg"
    
    # Extract modified date for dataset_modified_at
    modified_date = article.get("modified_date")

    # Create Dataset document using the schema
    dataset = create_dataset(
        dataset_id=f"figshare_{article_id}",
        name=title,
        source="figshare",
        recording_modality=recording_modality,
        modalities=modalities,
        license=license_name,
        authors=author_names,
        dataset_doi=doi,
        source_url=url_public_html,
        total_files=len(article.get("files", [])),
        size_bytes=total_size_bytes if total_size_bytes > 0 else None,
        dataset_modified_at=modified_date,
        digested_at=digested_at,
    )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEG BIDS datasets from Figshare."
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
        default=Path("consolidated/figshare_datasets.json"),
        help="Output JSON file path (default: consolidated/figshare_datasets.json).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Maximum number of results to fetch (default: 1000).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of results per page (max 1000, default: 100).",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        help="Fetch full details for each article (slower but more complete).",
    )
    parser.add_argument(
        "--item-type",
        type=int,
        default=3,
        help="Figshare item type: 3=dataset, 1=figure, 2=media, etc. (default: 3).",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output).",
    )

    args = parser.parse_args()

    # Search Figshare
    articles = search_figshare(
        query=args.query,
        size=args.size,
        page_size=args.page_size,
        item_type=args.item_type,
    )

    if not articles:
        print("No articles found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset information
    print("\nExtracting dataset information...")
    datasets = []
    for idx, article in enumerate(articles, start=1):
        if args.fetch_details and idx % 10 == 0:
            print(f"Processing {idx}/{len(articles)}...", flush=True)

        try:
            dataset = extract_dataset_info(
                article, 
                fetch_details=args.fetch_details,
                digested_at=args.digested_at,
            )
            datasets.append(dataset)
        except Exception as e:
            print(
                f"Warning: Error extracting info for article {article.get('id')}: {e}",
                file=sys.stderr,
            )
            continue

        # Be nice to the API when fetching details
        if args.fetch_details:
            time.sleep(0.2)

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")
    print("\nDataset Statistics:")

    # Count by modality
    modalities_found = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities_found[mod] = modalities_found.get(mod, 0) + 1

    print("\nBy Modality:")
    for mod, count in sorted(modalities_found.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mod}: {count}")

    # Datasets with files
    datasets_with_files = sum(1 for ds in datasets if ds.get("total_files", 0) > 0)
    print(f"\nDatasets with files: {datasets_with_files}/{len(datasets)}")

    # Total size
    total_size_bytes = sum(ds.get("size_bytes", 0) or 0 for ds in datasets)
    total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
    print(f"Total Size: {total_size_gb} GB")

    # Datasets with DOI
    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi", ""))
    print(f"Datasets with DOI: {datasets_with_doi}/{len(datasets)}")

    # Datasets with authors
    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
