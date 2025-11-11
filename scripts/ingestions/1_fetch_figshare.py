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
from pathlib import Path
from typing import Any

import requests


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

        # Check if we've reached the requested size
        if len(all_articles) >= size:
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
    return all_articles[:size]  # Trim to exact size


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


def extract_dataset_info(article: dict, fetch_details: bool = False) -> dict[str, Any]:
    """Extract relevant information from a Figshare article.

    Args:
        article: Figshare article dictionary
        fetch_details: Whether to fetch full details (slower but more complete)

    Returns:
        Cleaned dataset dictionary

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
    defined_type = article.get("defined_type", 0)
    defined_type_name = article.get("defined_type_name", "")

    # Extract dates
    published_date = article.get("published_date", "")
    created_date = article.get("created_date", "")
    modified_date = article.get("modified_date", "")

    # Extract URLs
    url_public_html = article.get("url_public_html", "")
    url_public_api = article.get("url_public_api", "")

    # Extract resource info (linked DOI/resource)
    resource_title = article.get("resource_title", "")
    resource_doi = article.get("resource_doi", "")

    # Extract authors (may not be in search results)
    authors = []
    for author in article.get("authors", []):
        author_info = {
            "name": author.get("full_name", ""),
        }
        if "id" in author:
            author_info["id"] = author["id"]
        if "orcid_id" in author:
            author_info["orcid"] = author["orcid_id"]
        authors.append(author_info)

    # Extract tags/categories
    tags = article.get("tags", [])
    categories = article.get("categories", [])

    # Extract files info
    files = []
    for file_entry in article.get("files", []):
        file_info = {
            "id": file_entry.get("id", ""),
            "name": file_entry.get("name", ""),
            "size_bytes": file_entry.get("size", 0),
            "download_url": file_entry.get("download_url", ""),
        }
        if "computed_md5" in file_entry:
            file_info["md5"] = file_entry["computed_md5"]
        files.append(file_info)

    # Calculate total size
    total_size_bytes = sum(f.get("size", 0) for f in article.get("files", []))
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    # Extract license
    license_info = article.get("license", {})
    license_name = license_info.get("name", "") if license_info else ""

    # Build dataset dictionary
    dataset = {
        "dataset_id": str(article_id),
        "doi": doi,
        "title": title,
        "description": description,
        "defined_type": defined_type,
        "defined_type_name": defined_type_name,
        "published_date": published_date,
        "created_date": created_date,
        "modified_date": modified_date,
        "authors": authors,
        "tags": tags,
        "categories": categories,
        "license": license_name,
        "url": url_public_html,
        "api_url": url_public_api,
        "resource_title": resource_title,
        "resource_doi": resource_doi,
        "files": files,
        "file_count": len(files),
        "total_size_mb": total_size_mb,
        "source": "figshare",
    }

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
            dataset = extract_dataset_info(article, fetch_details=args.fetch_details)
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

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")
    print("\nDataset Statistics:")

    # Count by type
    types = {}
    for ds in datasets:
        dtype = ds.get("defined_type_name", "unknown")
        types[dtype] = types.get(dtype, 0) + 1

    print("\nBy Type:")
    for dtype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dtype}: {count}")

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

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
