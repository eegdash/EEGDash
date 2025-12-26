"""Fetch neural recording BIDS datasets from Zenodo with multiple strategies.

This script consolidates three previously separate Zenodo fetching approaches:

1. CONSERVATIVE (Default):
   - Balanced recall and precision
   - Searches for neuroimaging modalities (EEG, MEG, iEEG, etc.) AND BIDS keywords
   - Output: consolidated/zenodo_datasets.json

2. AGGRESSIVE:
   - Maximum recall strategy
   - Searches broadly for ANY neurophysiology modality OR BIDS reference
   - Captures ~1,570 datasets (many false positives)
   - Output: consolidated/zenodo_datasets_aggressive.json
   - Use case: Data discovery phase before filtering

3. FILTER:
   - Post-processing of aggressive results
   - Identifies genuine BIDS datasets from aggressive search results
   - Validates structure using archive preview and file analysis
   - Outputs: validated BIDS, convertible neurophysiology, rejected false positives
   - Use case: Clean dataset classification

All modes output Dataset schema format for consistency with other sources.
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
# Search Functions
# =============================================================================


def search_zenodo_conservative(
    max_results: int = 1000,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Search Zenodo with balanced approach (modalities AND BIDS).

    This is the default mode: balanced recall and precision.

    Args:
        max_results: Maximum datasets to fetch
        access_token: Optional Zenodo API token

    Returns:
        List of Zenodo records

    """
    query = (
        "(EEG OR electroencephalography OR MEG OR magnetoencephalography OR "
        "iEEG OR 'intracranial EEG' OR ECoG OR electrocorticography OR SEEG OR "
        "'stereo EEG' OR EMG OR electromyography) AND "
        "(BIDS OR 'Brain Imaging Data Structure' OR neuroimaging)"
    )

    return _search_zenodo_api(
        query=query,
        max_results=max_results,
        access_token=access_token,
        mode="conservative",
    )


def search_zenodo_aggressive(
    max_results: int = 2000,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Search Zenodo aggressively for maximum recall.

    Uses broader search: ANY modality OR BIDS (maximum recall strategy).

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
        mode="aggressive",
    )


def _search_zenodo_api(
    query: str,
    max_results: int = 1000,
    access_token: str | None = None,
    mode: str = "conservative",
) -> list[dict[str, Any]]:
    """Core Zenodo REST API search function.

    Args:
        query: Search query
        max_results: Maximum results to fetch
        access_token: Optional auth token
        mode: Search mode (conservative, aggressive, etc.)

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
    print(f"Zenodo {mode.upper()} Search - {rate_limit_info}")
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
# Filtering Functions (for AGGRESSIVE mode post-processing)
# =============================================================================


BIDS_FILE_INDICATORS = {
    "strict": [
        "dataset_description.json",
        "participants.tsv",
        "README.md",
        "_eeg.json",
        "_channels.tsv",
        "_events.tsv",
        "sub-",
        "ses-",
        "task-",
        "run-",
    ],
    "moderate": [
        "_eeg.set",
        "_eeg.bdf",
        "_eeg.edf",
        "_eeg.vhdr",
        "_meg.fif",
        "_ieeg.json",
        "derivatives/",
        "sourcedata/",
    ],
}


def classify_bids_dataset(
    record: dict[str, Any],
    enrich_oaipmh: bool = False,
    access_token: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Classify dataset as BIDS, convertible, or rejected.

    Args:
        record: Zenodo record
        enrich_oaipmh: Whether to enrich with OAI-PMH
        access_token: Optional auth token

    Returns:
        Tuple of (classification, dataset_dict)
        Classification: "bids", "neurophysiology", or "rejected"

    """
    metadata = record.get("metadata", {})
    record_id = str(record.get("id", ""))

    title = (metadata.get("title", "") or "").lower()
    description = (metadata.get("description", "") or "").lower()
    keywords = [
        (s.get("subject", s.get("id", "")) or "").lower()
        for s in metadata.get("subjects", [])
    ]
    combined_text = f"{title} {description} {' '.join(keywords)}"

    # Check for BIDS keywords
    has_bids_keyword = any(
        x in combined_text for x in ["bids", "brain imaging data structure"]
    )

    # Check archive contents for BIDS structure
    archive_bids_indicators = 0
    files_data = record.get("files", {})
    if isinstance(files_data, dict):
        file_entries = files_data.get("entries", {})
        for filename in file_entries:
            fn_lower = filename.lower()
            if any(fn_lower.endswith(ext) for ext in [".zip", ".tar.gz", ".tgz"]):
                preview = get_archive_preview_files(record_id, filename, access_token)
                for item in preview:
                    item_lower = item.lower()
                    for indicator in BIDS_FILE_INDICATORS["strict"]:
                        if indicator in item_lower:
                            archive_bids_indicators += 1

    # Classification logic
    has_neuro_content = any(
        x in combined_text
        for x in [
            "eeg",
            "meg",
            "ieeg",
            "ecog",
            "emg",
            "neurophysiology",
            "electrophysiology",
        ]
    )

    if has_bids_keyword or archive_bids_indicators >= 3:
        classification = "bids"
    elif has_neuro_content:
        classification = "neurophysiology"
    else:
        classification = "rejected"

    # Extract dataset info
    dataset_dict = extract_dataset_info(record, enrich_oaipmh, access_token)

    return classification, dataset_dict


# =============================================================================
# Main CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from Zenodo (consolidated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SEARCH MODES:

  conservative (DEFAULT): Balanced recall and precision
    Query: (modalities) AND (BIDS)
    Expected: ~100-200 high-quality BIDS datasets
    Use case: Clean dataset collection

  aggressive: Maximum recall (discovers all neurophysiology)
    Query: (modalities) OR (BIDS)
    Expected: ~1,500+ datasets (many false positives)
    Use case: Initial discovery, requires filtering

  filter: Post-process aggressive results
    Input: Aggressive search results or saved file
    Output: Classified as BIDS, neurophysiology, or rejected
    Use case: Validate and categorize large discovery sets

EXAMPLES:

  # Conservative mode (recommended for production)
  python zenodo.py --mode conservative

  # Aggressive discovery
  python zenodo.py --mode aggressive --size 2000

  # Filter aggressive results (requires input file)
  python zenodo.py --mode filter --input consolidated/zenodo_datasets_aggressive.json
""",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["conservative", "aggressive", "filter"],
        default="conservative",
        help="Search mode (default: conservative)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: zenodo_datasets_<mode>.json)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSON file for filter mode",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Max results to fetch (default: 1000 for conservative, 2000 for aggressive)",
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

    # Set default output path based on mode
    if args.output is None:
        args.output = Path(f"consolidated/zenodo_datasets_{args.mode}.json")

    # Aggressive mode: default size = 2000
    if args.mode == "aggressive" and args.size == 1000:
        args.size = 2000

    # FILTER MODE
    if args.mode == "filter":
        if not args.input or not args.input.exists():
            print(
                f"Error: Filter mode requires --input with valid file",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nLoading aggressive results from {args.input}...")
        with args.input.open() as f:
            input_data = json.load(f)

        # Classify records
        classified = {"bids": [], "neurophysiology": [], "rejected": []}
        for i, record in enumerate(input_data, 1):
            if i % 100 == 0:
                print(f"  Classified {i}/{len(input_data)}...")

            classification, dataset_dict = classify_bids_dataset(
                record,
                enrich_oaipmh=args.enrich_oaipmh,
                access_token=args.access_token,
            )
            classified[classification].append(dataset_dict)

        # Save all classifications
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        for class_name, datasets in classified.items():
            output_path = output_dir / f"zenodo_datasets_{class_name}.json"
            with output_path.open("w") as f:
                json.dump(datasets, f, indent=2)
            print(f"Saved {len(datasets)} {class_name} datasets to {output_path}")

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"Classification Results:")
        print(f"  BIDS datasets:        {len(classified['bids']):4d}")
        print(f"  Neurophysiology:      {len(classified['neurophysiology']):4d}")
        print(f"  Rejected:             {len(classified['rejected']):4d}")
        print(f"  Total:                {len(input_data):4d}")
        print(f"{'=' * 70}")

        return

    # CONSERVATIVE / AGGRESSIVE MODES
    if args.mode == "conservative":
        records = search_zenodo_conservative(max_results=args.size, access_token=args.access_token)
    else:  # aggressive
        records = search_zenodo_aggressive(max_results=args.size, access_token=args.access_token)

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
