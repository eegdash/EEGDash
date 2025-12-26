"""Fetch neural recording BIDS datasets from Zenodo with maximum recall.

AGGRESSIVE SEARCH STRATEGY - Maximum Recall:
This script searches Zenodo broadly for ANY neurophysiology modality OR BIDS reference,
capturing all potentially relevant datasets including false positives.

Expected results: ~1,500+ datasets across all neural recording modalities.

Output: consolidated/zenodo_datasets.json (Dataset schema format)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from eegdash.records import create_dataset


# =============================================================================
# Search Function
# =============================================================================


def search_zenodo(
    max_results: int = 2000,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Search Zenodo aggressively for maximum recall.

    Uses broad search: ANY neurophysiology modality OR BIDS (maximum recall strategy).

    Args:
        max_results: Maximum datasets to fetch
        access_token: Optional Zenodo API token

    Returns:
        List of Zenodo records

    """
    query = (
        "EEG OR electroencephalography OR MEG OR magnetoencephalography OR "
        "iEEG OR 'intracranial EEG' OR ECoG OR electrocorticography OR SEEG OR "
        "stereo EEG OR EMG OR electromyography OR BIDS OR "
        "'Brain Imaging Data Structure'"
    )

    return _search_zenodo_api(
        query=query,
        max_results=max_results,
        access_token=access_token,
    )


def _search_zenodo_api(
    query: str,
    max_results: int = 2000,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Core Zenodo REST API search function.

    Args:
        query: Search query
        max_results: Maximum results to fetch
        access_token: Optional auth token

    Returns:
        List of Zenodo records

    """
    base_url = "https://zenodo.org/api/records"

    params = {
        "q": query,
        "type": "dataset",
        "access_status": "open",
        "sort": "bestmatch",
    }

    headers = {
        "Accept": "application/vnd.inveniordm.v1+json",
    }

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    rate_limit_info = "Auth (100 req/min)" if access_token else "Guest (60 req/min)"
    print(f"\n{'=' * 70}")
    print(f"Zenodo Aggressive Search - {rate_limit_info}")
    print(f"{'=' * 70}")
    print(f"Query: {query}")
    print(f"Max results: {max_results}")
    print(f"{'=' * 70}\n")

    all_records = []
    page = 1
    page_size = min(max_results, 100)
    consecutive_errors = 0

    while len(all_records) < max_results:
        print(f"Page {page:3d}...", end=" ", flush=True)

        params["page"] = page
        params["size"] = page_size

        try:
            response = requests.get(
                base_url, params=params, headers=headers, timeout=120
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(
                    f"\nRate limited! Waiting {retry_after}s...",
                    file=sys.stderr,
                )
                time.sleep(retry_after + 1)
                continue

            response.raise_for_status()
            consecutive_errors = 0

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

        if not hits:
            print("No more results")
            break

        print(f"{len(hits):3d} records | Total: {total:5d} | Fetched: {len(all_records) + len(hits):5d}")
        all_records.extend(hits)

        if len(all_records) >= max_results or len(hits) < page_size:
            break

        page += 1
        delay = 1.1 if not access_token else 0.7
        time.sleep(delay)

    print(f"\n{'=' * 70}")
    print(f"Total records fetched: {len(all_records)}")
    print(f"{'=' * 70}\n")

    return all_records[:max_results]


# =============================================================================
# Archive and Metadata Extraction
# =============================================================================


def get_archive_preview_files(
    record_id: str,
    filename: str,
    access_token: str | None = None,
) -> list[str]:
    """Extract file listing from Zenodo archive preview.

    Args:
        record_id: Zenodo record ID
        filename: Archive filename (e.g., "data.zip")
        access_token: Optional auth token

    Returns:
        List of filenames found inside archive

    """
    try:
        preview_url = f"https://zenodo.org/records/{record_id}/preview/{filename}"
        params = {"include_deleted": "0"}

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        response = requests.get(preview_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        html_content = response.text.lower()

        # Extract filenames from HTML
        file_pattern = r'<span><i class="file outline icon"></i>[^<]*?([^<]+)</span>'
        folder_pattern = r'<i class="folder icon"></i>\s*<a[^>]*>([^<]+)</a>'

        files = re.findall(file_pattern, html_content)
        folders = re.findall(folder_pattern, html_content)

        all_items = []
        for item in files + folders:
            cleaned = item.strip()
            if cleaned:
                all_items.append(cleaned)

        return all_items

    except Exception:
        return []


def enrich_with_oaipmh(
    record_id: str,
    access_token: str | None = None,
) -> dict[str, Any] | None:
    """Enrich dataset with OAI-PMH metadata.

    Args:
        record_id: Zenodo record ID
        access_token: Optional auth token

    Returns:
        Enriched metadata dictionary or None

    """
    try:
        oaipmh_url = f"https://zenodo.org/oai2d?verb=GetRecord&identifier=oai:zenodo.org:{record_id}&metadataPrefix=oai_datacite"

        response = requests.get(oaipmh_url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespaces = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "datacite": "http://datacite.org/schema/kernel-4",
        }

        enriched = {}

        # Extract subjects from datacite:subject
        subjects = []
        for subject_elem in root.findall(
            ".//datacite:subject", namespaces
        ):
            if subject_elem.text:
                subjects.append(subject_elem.text)
        if subjects:
            enriched["subjects"] = subjects

        # Extract contributors
        contributors = []
        for contrib in root.findall(
            ".//datacite:contributor", namespaces
        ):
            contrib_info = {
                "name": contrib.findtext("datacite:contributorName", "", namespaces)
            }
            contrib_type = contrib.get("contributorType")
            if contrib_type:
                contrib_info["type"] = contrib_type
            contributors.append(contrib_info)
        if contributors:
            enriched["contributors"] = contributors

        # Extract related identifiers
        related_ids = []
        for rel_id in root.findall(
            ".//datacite:relatedIdentifier", namespaces
        ):
            if rel_id.text:
                related_ids.append(
                    {
                        "identifier": rel_id.text,
                        "type": rel_id.get("relatedIdentifierType"),
                        "relation": rel_id.get("relationType"),
                    }
                )
        if related_ids:
            enriched["related_identifiers"] = related_ids

        # Extract funding
        funding = []
        for funder in root.findall(
            ".//datacite:fundingReference/datacite:funderName", namespaces
        ):
            if funder.text:
                funding.append({"funder_name": funder.text})
        if funding:
            enriched["funding"] = funding

        return enriched if enriched else None

    except Exception as e:
        print(f"  OAI-PMH error for {record_id}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Dataset Conversion
# =============================================================================


def extract_dataset_info(
    record: dict[str, Any],
    enrich_oaipmh: bool = False,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Extract dataset information from Zenodo record.

    Args:
        record: Zenodo REST API record
        enrich_oaipmh: Whether to fetch OAI-PMH enrichment
        access_token: Optional auth token

    Returns:
        Dataset information dictionary

    """
    metadata = record.get("metadata", {})
    record_id = str(record.get("id", ""))

    # DOI
    doi = record.get("pids", {}).get("doi", {}).get("identifier", "")

    # Basic metadata
    title = metadata.get("title", "")
    description = metadata.get("description", "")

    # Creators/authors
    creators = []
    for creator in metadata.get("creators", []):
        if isinstance(creator, dict):
            person = creator.get("person_or_org", {})
            name = person.get("name", creator.get("name", ""))
            if name:
                creators.append(name)

    # Keywords
    keywords = [s.get("subject", s.get("id", "")) for s in metadata.get("subjects", [])]

    # Resource type
    resource_type = metadata.get("resource_type", {}).get("id", "")

    # License
    rights = metadata.get("rights", [])
    license_info = (
        rights[0].get("id", "") if rights and isinstance(rights[0], dict) else ""
    )

    # Access status
    access_status = record.get("access", {}).get("status", "")

    # Files
    files_data = record.get("files", {})
    total_size_bytes = (
        files_data.get("total_bytes", 0) if isinstance(files_data, dict) else 0
    )
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    # Archive preview (optional)
    archive_contents = {}
    if isinstance(files_data, dict):
        file_entries = files_data.get("entries", {})
        for filename in file_entries:
            fn_lower = filename.lower()
            if any(fn_lower.endswith(ext) for ext in [".zip", ".tar.gz", ".tgz"]):
                preview = get_archive_preview_files(record_id, filename, access_token)
                if preview:
                    archive_contents[filename] = preview

    # Build dataset dictionary
    dataset_dict = {
        "zenodo_id": record_id,
        "title": title,
        "description": description[:1000] if description else None,
        "source": "zenodo",
        "source_url": record.get("links", {}).get("self_html", ""),
        "authors": creators if creators else None,
        "license": license_info or None,
        "dataset_doi": doi or None,
    }

    # Detect modalities from title, description, keywords
    combined_text = (
        (title or "")
        + " "
        + (description or "")
        + " "
        + " ".join(keywords)
    ).lower()

    modalities = []
    if any(x in combined_text for x in ["eeg", "electroencephalography"]):
        modalities.append("eeg")
    if any(x in combined_text for x in ["meg", "magnetoencephalography"]):
        modalities.append("meg")
    if any(
        x in combined_text
        for x in ["ieeg", "intracranial eeg", "ecog", "electrocorticography", "seeg"]
    ):
        modalities.append("ieeg")
    if any(x in combined_text for x in ["fnirs", "nirs"]):
        modalities.append("fnirs")
    if any(x in combined_text for x in ["emg", "electromyography"]):
        modalities.append("emg")

    # OAI-PMH enrichment (optional)
    if enrich_oaipmh:
        enriched = enrich_with_oaipmh(record_id, access_token)
        if enriched:
            dataset_dict["funding"] = enriched.get("funding")

    # Always use create_dataset for consistent schema
    primary_modality = modalities[0] if modalities else "unknown"
    dataset = create_dataset(
        dataset_id=f"zenodo_{record_id}",
        name=title or "Zenodo Dataset",
        source="zenodo",
        recording_modality=primary_modality,
        modalities=modalities if modalities else [],
        license=license_info or None,
        authors=creators if creators else None,
        source_url=record.get("links", {}).get("self_html", ""),
        dataset_doi=doi or None,
    )
    
    # Add zenodo-specific metadata
    dataset["zenodo_id"] = record_id
    dataset["resource_type"] = resource_type
    dataset["access_status"] = access_status
    
    if enrich_oaipmh:
        enriched = enrich_with_oaipmh(record_id, access_token)
        if enriched:
            dataset["funding"] = enriched.get("funding")
    
    return dataset


# =============================================================================
# Main CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from Zenodo (aggressive search)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/zenodo_datasets.json"),
        help="Output JSON file (default: zenodo_datasets.json)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Max results to fetch (default: 2000)",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default="v3dkycQdOlyc0gXXkeeroSIkpSyAgaTyzpsIJLw8lhQsoEw089MFICKqnxWz",
        help="Zenodo API token (get from https://zenodo.org/account/settings/applications/tokens/new/)",
    )
    parser.add_argument(
        "--enrich-oaipmh",
        action="store_true",
        help="Enrich with OAI-PMH metadata (slower)",
    )

    args = parser.parse_args()

    # Search Zenodo
    records = search_zenodo(max_results=args.size, access_token=args.access_token)

    if not records:
        print("No datasets found.", file=sys.stderr)
        sys.exit(1)

    # Extract datasets
    print(f"\nProcessing {len(records)} records...")
    datasets = []
    for i, record in enumerate(records, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(records)}...")

        try:
            dataset = extract_dataset_info(
                record,
                enrich_oaipmh=args.enrich_oaipmh,
                access_token=args.access_token,
            )
            if dataset:
                datasets.append(dataset)
        except Exception as e:
            print(f"  Error processing record {record.get('id')}: {e}", file=sys.stderr)
            continue

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(datasets, f, indent=2)

    # Statistics
    print(f"\n{'=' * 70}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 70}")

    modalities = {}
    for ds in datasets:
        for mod in ds.get("modalities", []):
            modalities[mod] = modalities.get(mod, 0) + 1

    if modalities:
        print("\nModalities detected:")
        for mod in sorted(modalities.keys()):
            print(f"  {mod:10s}: {modalities[mod]:4d}")

    datasets_with_doi = sum(1 for ds in datasets if ds.get("dataset_doi"))
    print(f"\nDatasets with DOI: {datasets_with_doi}/{len(datasets)}")

    datasets_with_authors = sum(1 for ds in datasets if ds.get("authors"))
    print(f"Datasets with authors: {datasets_with_authors}/{len(datasets)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
