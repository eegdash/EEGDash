"""Fetch neuroimaging datasets from Zenodo.

This script searches Zenodo for datasets combining neuroimaging modalities
(EEG, MEG, iEEG, ECoG, SEEG, EMG) with BIDS standards using the Zenodo REST API.
It retrieves comprehensive metadata including DOIs, descriptions, file information,
and download URLs.

Search Strategy:
- Modalities: EEG, electroencephalography, MEG, magnetoencephalography, iEEG,
              intracranial EEG, ECoG, electrocorticography, SEEG, stereo EEG,
              EMG, electromyography
- Standards: BIDS, Brain Imaging Data Structure, neuroimaging
- Logic: AND operator for balanced recall and precision

Output: consolidated/zenodo_datasets.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def search_zenodo(
    query: str,
    size: int = 100,
    all_versions: bool = False,
    resource_type: str = "dataset",
    access_status: str = "open",
    subject: str = "",
    sort: str = "bestmatch",
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Search Zenodo for records matching the query.

    Implements Zenodo REST API v1 search endpoint with comprehensive filtering.
    See: https://developers.zenodo.org/#list39

    Args:
        query: Search query string using Elasticsearch syntax (e.g., '"EEG" "BIDS"' for exact match)
        size: Maximum number of total results to fetch (will paginate)
        all_versions: Include all versions of records (default: latest only)
        resource_type: Filter by resource type (default: 'dataset'). Options: 'publication', 'dataset', 'image', 'video', 'software', etc.
        access_status: Filter by access status (default: 'open'). Options: 'open', 'embargoed', 'restricted', 'closed'
        subject: Filter by subject classification (e.g., 'EEG' to filter by subject taxonomy)
        sort: Sort order (default: 'bestmatch' for relevance). Options: 'bestmatch', 'mostrecent', '-mostrecent'
        access_token: Personal Access Token from https://zenodo.org/account/settings/applications/tokens/new/
                     Increases rate limits from 60 to 100 req/min

    Returns:
        List of Zenodo record dictionaries

    Rate Limits:
        Guest users: 60 requests/min, 2000 requests/hour
        Authenticated users: 100 requests/min, 5000 requests/hour
        Response headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

    """
    base_url = "https://zenodo.org/api/records"

    # Build base params following API documentation
    base_params = {
        "q": query,
        "sort": sort,
    }

    # Add faceted filters as URL parameters
    # Based on API docs and testing: type, subject, access_status are separate params
    if resource_type:
        base_params["type"] = resource_type
    if access_status:
        base_params["access_status"] = access_status
    if subject:
        base_params["subject"] = subject

    if all_versions:
        base_params["all_versions"] = "1"

    # Build headers with auth if provided
    headers = {
        "Accept": "application/vnd.inveniordm.v1+json",
    }

    # Check rate limit headers if available
    rate_limit_info = ""
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
        rate_limit_info = "Authenticated (100 req/min, 5000 req/hour)"
    else:
        rate_limit_info = "Guest (60 req/min, 2000 req/hour)"

    print(f"{'=' * 70}")
    print(f"Zenodo API Search - {rate_limit_info}")
    print(f"{'=' * 70}")
    print(f"Query: {query}")
    filters_list = [f"type={resource_type}"] if resource_type else []
    if access_status:
        filters_list.append(f"access_status={access_status}")
    if subject:
        filters_list.append(f"subject={subject}")
    filters_list.append(f"sort={sort}")
    print(f"Filters: {', '.join(filters_list)}")
    print(f"Max results: {size}")
    print(f"{'=' * 70}\n")

    all_records = []
    page = 1
    page_size = min(size, 100)  # API supports up to 100 per page
    consecutive_errors = 0
    total_available = None

    while len(all_records) < size:
        print(f"Page {page:3d}...", end=" ", flush=True)

        # Build params for this page
        params = base_params.copy()
        params["page"] = page
        params["size"] = page_size

        try:
            response = requests.get(
                base_url, params=params, headers=headers, timeout=120
            )

            # Handle rate limiting (429 = Too Many Requests)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
                reset_time = response.headers.get("X-RateLimit-Reset", "unknown")
                print(f"\n⚠️ Rate limited! Remaining: {remaining}, Reset: {reset_time}")
                print(f"   Waiting {retry_after} seconds...", file=sys.stderr)
                time.sleep(retry_after + 1)
                continue  # Retry this page

            response.raise_for_status()
            consecutive_errors = 0

            # Log rate limit status (helpful for monitoring)
            if page == 1 or page % 10 == 0:
                limit = response.headers.get("X-RateLimit-Limit", "N/A")
                remaining = response.headers.get("X-RateLimit-Remaining", "N/A")
                if page > 1:
                    print(f"\n[Rate Limit: {remaining}/{limit}]", end=" ")

        except requests.Timeout:
            consecutive_errors += 1
            print(f"TIMEOUT (attempt {consecutive_errors}/3)")
            if consecutive_errors >= 3:
                print("Too many timeouts. Stopping.", file=sys.stderr)
                break
            time.sleep(5)
            continue

        except requests.RequestException as e:
            consecutive_errors += 1
            print(f"ERROR: {type(e).__name__}")
            if consecutive_errors >= 3:
                print("Too many errors. Stopping.", file=sys.stderr)
                break
            time.sleep(5)
            continue

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            break

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", 0)

        if total_available is None:
            total_available = total

        if not hits:
            print("No more results")
            break

        print(
            f"{len(hits):3d} records | Total available: {total_available:5d} | Fetched so far: {len(all_records) + len(hits):5d}"
        )
        all_records.extend(hits)

        # Check if we've reached the requested size
        if len(all_records) >= size:
            break

        # Check if we've reached the end of results
        if len(hits) < page_size:
            print(f"Reached end of results at page {page}")
            break

        page += 1

        # Respect rate limiting with conservative delays
        # Guest: 60 req/min → 1 req/sec max → use 1.1s delay
        # Auth: 100 req/min → 0.6 req/sec max → use 0.7s delay
        delay = 1.1 if not access_token else 0.7
        time.sleep(delay)

    print(f"\n{'=' * 70}")
    print(
        f"Total records fetched: {len(all_records)} (requested: {size}, available: {total_available or 'unknown'})"
    )
    print(f"{'=' * 70}\n")
    return all_records[:size]  # Trim to exact size


def extract_dataset_info(record: dict) -> dict[str, Any]:
    """Extract relevant information from a Zenodo record.

    Handles both InvenioRDM v1 format (from API with vnd.inveniordm.v1+json)
    and legacy Zenodo format.

    Args:
        record: Raw Zenodo record dictionary

    Returns:
        Cleaned dataset dictionary

    """
    metadata = record.get("metadata", {})

    # Detect format based on structure
    is_inveniordm = "pids" in record  # InvenioRDM has 'pids' at top level

    # Extract DOI and IDs
    dataset_id = str(record.get("id", ""))

    if is_inveniordm:
        # InvenioRDM format
        doi = record.get("pids", {}).get("doi", {}).get("identifier", "")
        concept_doi = record.get("parent", {}).get("id", "")
    else:
        # Legacy format
        doi = record.get("doi", "")
        concept_doi = record.get("conceptdoi", "")

    # Extract basic metadata
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    publication_date = metadata.get("publication_date", "")

    # Extract creators/authors
    creators = []
    for creator in metadata.get("creators", []):
        if isinstance(creator, dict):
            creator_info = {
                "name": creator.get("person_or_org", {}).get(
                    "name", creator.get("name", "")
                )
            }
            # InvenioRDM uses affiliations array
            affiliations = creator.get("affiliations", [])
            if affiliations and isinstance(affiliations[0], dict):
                creator_info["affiliation"] = affiliations[0].get("name", "")
            elif "affiliation" in creator:
                creator_info["affiliation"] = creator["affiliation"]
            # ORCID
            person = creator.get("person_or_org", creator)
            for identifier in person.get("identifiers", []):
                if identifier.get("scheme") == "orcid":
                    creator_info["orcid"] = identifier.get("identifier", "")
            if "orcid" in person:
                creator_info["orcid"] = person["orcid"]
            creators.append(creator_info)

    # Extract keywords/subjects
    if is_inveniordm:
        # InvenioRDM uses 'subjects' array
        keywords = [
            s.get("subject", s.get("id", "")) for s in metadata.get("subjects", [])
        ]
    else:
        # Legacy uses 'keywords' array
        keywords = metadata.get("keywords", [])

    # Extract resource type
    resource_type_data = metadata.get("resource_type", {})
    if isinstance(resource_type_data, dict):
        if is_inveniordm:
            resource_type_str = resource_type_data.get("id", "")
            resource_title = resource_type_data.get("title", {}).get("en", "")
        else:
            resource_type_str = resource_type_data.get("type", "")
            resource_title = resource_type_data.get("title", "")
    else:
        resource_type_str = str(resource_type_data)
        resource_title = ""

    # Extract license
    if is_inveniordm:
        # InvenioRDM uses 'rights' array
        rights = metadata.get("rights", [])
        license_id = (
            rights[0].get("id", "") if rights and isinstance(rights[0], dict) else ""
        )
    else:
        # Legacy uses 'license' dict
        license_info = metadata.get("license", {})
        license_id = (
            license_info.get("id", "")
            if isinstance(license_info, dict)
            else str(license_info)
        )

    # Extract access rights/status
    if is_inveniordm:
        # InvenioRDM uses 'access' dict at top level
        access_data = record.get("access", {})
        access_right = access_data.get("status", "")
    else:
        # Legacy uses 'access_right' in metadata
        access_right = metadata.get("access_right", "")

    # Extract file information
    # InvenioRDM v1 format: files is a dict with 'entries' containing filename→fileinfo mapping
    # Legacy format: files is a list of file dicts
    files = []
    files_data = record.get("files", {})

    if isinstance(files_data, dict):
        # InvenioRDM v1 format
        file_entries = files_data.get("entries", {})
        if isinstance(file_entries, dict):
            # entries is a dict: {filename: file_info}
            for filename, file_info in file_entries.items():
                files.append(
                    {
                        "filename": file_info.get("key", filename),
                        "size_bytes": file_info.get("size", 0),
                        "checksum": file_info.get("checksum", ""),
                        "download_url": record.get("links", {}).get("self", "")
                        + f"/files/{filename}/content",
                    }
                )
        else:
            # entries is a list (shouldn't happen but handle gracefully)
            for file_entry in file_entries:
                files.append(
                    {
                        "filename": file_entry.get("key", ""),
                        "size_bytes": file_entry.get("size", 0),
                        "checksum": file_entry.get("checksum", ""),
                        "download_url": file_entry.get("links", {}).get("self", ""),
                    }
                )
    elif isinstance(files_data, list):
        # Legacy format: files is a list
        for file_entry in files_data:
            files.append(
                {
                    "filename": file_entry.get("key", ""),
                    "size_bytes": file_entry.get("size", 0),
                    "checksum": file_entry.get("checksum", ""),
                    "download_url": file_entry.get("links", {}).get("self", ""),
                }
            )

    # Extract links
    links = record.get("links", {})

    # Calculate total size from files structure
    if isinstance(files_data, dict):
        # InvenioRDM v1 format has total_bytes at top level
        total_size_bytes = files_data.get("total_bytes", 0)
    else:
        # Legacy format: sum from individual files
        total_size_bytes = sum(f.get("size_bytes", 0) for f in files)

    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    # Build dataset dictionary
    dataset = {
        "dataset_id": dataset_id,
        "doi": doi,
        "concept_doi": concept_doi,
        "title": title,
        "description": description,
        "publication_date": publication_date,
        "created": record.get("created", ""),
        "modified": record.get("modified", ""),
        "creators": creators,
        "keywords": keywords,
        "resource_type": resource_type_str,
        "resource_title": resource_title,
        "license": license_id,
        "access_right": access_right,
        "url": links.get("self_html", ""),
        "doi_url": links.get("doi", ""),
        "files": files,
        "file_count": len(files),
        "total_size_mb": total_size_mb,
        "source": "zenodo",
    }

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EEG BIDS datasets from Zenodo.")
    parser.add_argument(
        "--query",
        type=str,
        default='(EEG OR electroencephalography OR MEG OR magnetoencephalography OR iEEG OR "intracranial EEG" OR ECoG OR electrocorticography OR SEEG OR "stereo EEG" OR EMG OR electromyography) AND (BIDS OR "Brain Imaging Data Structure" OR neuroimaging)',
        help=(
            "Search query (default: comprehensive neuroimaging + BIDS for balanced recall and precision). "
            "Searches across all modalities (EEG, MEG, iEEG, ECoG, SEEG, EMG) combined with BIDS standards. "
            "Use quotes for multi-word terms. Example: '\"intracranial EEG\" AND BIDS'"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/zenodo_datasets.json"),
        help="Output JSON file path (default: consolidated/zenodo_datasets.json).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Maximum number of results to fetch (default: 10000, max: 9999).",
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Include all versions of records (default: latest only).",
    )
    parser.add_argument(
        "--resource-type",
        type=str,
        default="dataset",
        help='Filter by resource type (default: "dataset"). Set to empty string to disable filter.',
    )
    parser.add_argument(
        "--access-status",
        type=str,
        default="open",
        help='Filter by access status (default: "open"). Options: "open", "embargoed", "restricted", "closed". Empty to disable.',
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help='Filter by subject classification (e.g., "EEG"). Leave empty for broader coverage. Default: no subject filter.',
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="bestmatch",
        help='Sort order (default: "bestmatch"). Options: "bestmatch", "mostrecent", "-mostrecent".',
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default="v3dkycQdOlyc0gXXkeeroSIkpSyAgaTyzpsIJLw8lhQsoEw089MFICKqnxWz",
        help="Zenodo Personal Access Token (default: temp test token). Get from https://zenodo.org/account/settings/applications/tokens/new/",
    )

    args = parser.parse_args()

    # Search Zenodo with all filters
    records = search_zenodo(
        query=args.query,
        size=args.size,
        all_versions=args.all_versions,
        resource_type=args.resource_type or "",
        access_status=args.access_status or "",
        subject=args.subject or "",
        sort=args.sort,
        access_token=args.access_token,
    )

    if not records:
        print("No records found. Exiting.", file=sys.stderr)
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
        json.dump(datasets, fh, indent=2)

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 60}")
    print("\nDataset Statistics:")

    # Count by resource type
    resource_types = {}
    for ds in datasets:
        rt = ds.get("resource_type", "unknown")
        resource_types[rt] = resource_types.get(rt, 0) + 1

    print("\nBy Resource Type:")
    for rt, count in sorted(resource_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rt}: {count}")

    # Count by access right
    access_rights = {}
    for ds in datasets:
        ar = ds.get("access_right", "unknown")
        access_rights[ar] = access_rights.get(ar, 0) + 1

    print("\nBy Access Rights:")
    for ar, count in sorted(access_rights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ar}: {count}")

    # Total size
    total_size_mb = sum(ds.get("total_size_mb", 0) for ds in datasets)
    total_size_gb = round(total_size_mb / 1024, 2)
    print(f"\nTotal Size: {total_size_gb} GB ({total_size_mb} MB)")

    # Datasets with files
    datasets_with_files = sum(1 for ds in datasets if ds.get("file_count", 0) > 0)
    print(f"Datasets with files: {datasets_with_files}/{len(datasets)}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
