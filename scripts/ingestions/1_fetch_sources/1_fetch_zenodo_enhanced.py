"""AGGRESSIVE Zenodo neurophysiology dataset fetcher - Maximum Recall Strategy.

PHILOSOPHY: Cast a wide net first, filter false positives later!

This script implements an aggressive multi-strategy search combining:
1. Extended terminology search (electroencephalography, magnetoencephalography, etc.)
2. All neurophysiology modalities (EEG, MEG, iEEG, ECoG, SEEG, EMG)
3. BIDS and related terms (Brain Imaging Data Structure)
4. Optional OAI-PMH enrichment for complete metadata

AGGRESSIVE QUERY RESULTS (Tested Nov 2024):
┌─────────────────────────────────────────────┬──────────┐
│ Query Strategy                              │ Results  │
├─────────────────────────────────────────────┼──────────┤
│ Extended terms (all synonyms)               │ 1,570 ✓  │
│ All modalities OR BIDS                      │ 1,391    │
│ Simple modality terms                       │ 1,324    │
│ Conservative "EEG BIDS" (old approach)      │   904    │
└─────────────────────────────────────────────┴──────────┘

STRATEGY: Use extended terminology query to maximize recall (1,570 datasets)
Then filter out false positives during post-processing based on:
- File structure (presence of BIDS-compliant files)
- Metadata keywords and descriptions
- File extensions (.eeg, .vhdr, .fif, .set, etc.)

Key findings from Zenodo API documentation analysis:
- Default behavior: Multiple terms are OR-ed (maximizes recall)
- Explicit AND/OR operators work with proper parentheses grouping
- type=dataset filter essential (reduces 7,595 → 1,570)
- access_status=open ensures only usable datasets
- Field searches available but not needed for initial broad sweep

Output: consolidated/zenodo_datasets_aggressive.json
Post-filtering: Will create consolidated/zenodo_datasets_validated.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def search_zenodo_rest_api(
    query: str,
    size: int = 1000,
    all_versions: bool = False,
    resource_type: str = "dataset",
    access_status: str = "open",
    sort: str = "bestmatch",
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """Search Zenodo using REST API for comprehensive dataset discovery.

    REST API advantages:
    - Fast pagination (100 records/page)
    - Rich filtering options
    - InvenioRDM v1 JSON format with complete file metadata
    - Good for initial discovery

    Args:
        query: Search query (default: "EEG BIDS" for broad fuzzy matching)
        size: Maximum results to fetch
        all_versions: Include all versions
        resource_type: Filter by type (default: "dataset")
        access_status: Filter by access (default: "open")
        sort: Sort order (default: "bestmatch")
        access_token: Personal Access Token for higher rate limits

    Returns:
        List of record dictionaries with InvenioRDM v1 structure

    Rate Limits:
        Guest: 60 req/min, 2000 req/hour
        Auth: 100 req/min, 5000 req/hour

    """
    base_url = "https://zenodo.org/api/records"

    params = {"q": query, "sort": sort}
    if resource_type:
        params["type"] = resource_type
    if access_status:
        params["access_status"] = access_status
    if all_versions:
        params["all_versions"] = "1"

    headers = {"Accept": "application/vnd.inveniordm.v1+json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    rate_limit_info = "Auth (100 req/min)" if access_token else "Guest (60 req/min)"
    print(f"{'=' * 70}")
    print(f"REST API Search - {rate_limit_info}")
    print(f"{'=' * 70}")
    print(f"Query: {query}")
    print(f"Filters: type={resource_type}, access_status={access_status}")
    print(f"Max results: {size}")
    print(f"{'=' * 70}\n")

    all_records = []
    page = 1
    page_size = min(size, 100)
    consecutive_errors = 0
    total_available = None

    while len(all_records) < size:
        print(f"Page {page:3d}...", end=" ", flush=True)

        params_with_page = params.copy()
        params_with_page["page"] = page
        params_with_page["size"] = page_size

        try:
            response = requests.get(
                base_url, params=params_with_page, headers=headers, timeout=120
            )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"\n⚠️  Rate limited! Waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue

            response.raise_for_status()
            consecutive_errors = 0

        except (requests.Timeout, requests.RequestException) as e:
            consecutive_errors += 1
            print(f"ERROR ({consecutive_errors}/3): {type(e).__name__}")
            if consecutive_errors >= 3:
                print("\nToo many errors. Stopping.")
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
            f"{len(hits):3d} records | Total: {total_available:5d} | Fetched: {len(all_records) + len(hits):5d}"
        )
        all_records.extend(hits)

        if len(all_records) >= size or len(hits) < page_size:
            if len(hits) < page_size:
                print(f"Reached end of results at page {page}")
            break

        page += 1

        # Conservative rate limiting
        delay = 1.1 if not access_token else 0.7
        time.sleep(delay)

    print(f"\n{'=' * 70}")
    print(
        f"Total fetched: {len(all_records)} (available: {total_available or 'unknown'})"
    )
    print(f"{'=' * 70}\n")

    return all_records[:size]


def enrich_with_oaipmh(
    record_id: str,
    access_token: str | None = None,
    metadata_format: str = "oai_datacite",
) -> dict[str, Any] | None:
    """Enrich dataset with additional metadata from OAI-PMH API.

    OAI-PMH advantages:
    - Richer metadata formats (oai_datacite, datacite, dcat, oai_dc)
    - Access to controlled vocabularies (subjects)
    - Complete contributor information
    - Related identifiers (publications, datasets)
    - Grant/funding information
    - More detailed descriptions

    Args:
        record_id: Zenodo record ID
        access_token: Optional auth token
        metadata_format: Metadata format (oai_datacite recommended)

    Returns:
        Parsed metadata dict or None if error

    Metadata formats:
        oai_datacite: Most complete (recommended)
        datacite: DataCite schema only
        oai_dc: Dublin Core (minimal)
        dcat: DCAT-AP format (includes file links)
        marc21: Legacy format

    """
    try:
        import xml.etree.ElementTree as ET

        oai_url = f"https://zenodo.org/oai2d?verb=GetRecord&identifier=oai:zenodo.org:{record_id}&metadataPrefix={metadata_format}"

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        response = requests.get(oai_url, headers=headers, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Extract DataCite metadata if using oai_datacite format
        if metadata_format == "oai_datacite":
            namespaces = {
                "oai": "http://www.openarchives.org/OAI/2.0/",
                "oai_datacite": "http://schema.datacite.org/oai/oai-1.1/",
                "datacite": "http://datacite.org/schema/kernel-4",
            }

            resource = root.find(".//datacite:resource", namespaces)
            if resource is None:
                return None

            enriched = {}

            # Extract subjects (controlled vocabularies)
            subjects = []
            for subject in resource.findall(".//datacite:subject", namespaces):
                subject_data = {"term": subject.text}
                if "subjectScheme" in subject.attrib:
                    subject_data["scheme"] = subject.attrib["subjectScheme"]
                if "schemeURI" in subject.attrib:
                    subject_data["scheme_uri"] = subject.attrib["schemeURI"]
                if "valueURI" in subject.attrib:
                    subject_data["value_uri"] = subject.attrib["valueURI"]
                subjects.append(subject_data)
            if subjects:
                enriched["subjects"] = subjects

            # Extract contributors
            contributors = []
            for contrib in resource.findall(".//datacite:contributor", namespaces):
                contrib_name = contrib.find("datacite:contributorName", namespaces)
                contrib_type = contrib.find("datacite:contributorType", namespaces)
                if contrib_name is not None:
                    contrib_data = {
                        "name": contrib_name.text,
                        "type": contrib_type.text
                        if contrib_type is not None
                        else "Other",
                    }
                    # Get affiliation
                    affiliation = contrib.find("datacite:affiliation", namespaces)
                    if affiliation is not None:
                        contrib_data["affiliation"] = affiliation.text
                    contributors.append(contrib_data)
            if contributors:
                enriched["contributors"] = contributors

            # Extract related identifiers
            related_ids = []
            for related in resource.findall(
                ".//datacite:relatedIdentifier", namespaces
            ):
                if related.text:
                    related_data = {
                        "identifier": related.text,
                        "relation_type": related.attrib.get("relationType", ""),
                        "identifier_type": related.attrib.get(
                            "relatedIdentifierType", ""
                        ),
                    }
                    if "resourceTypeGeneral" in related.attrib:
                        related_data["resource_type"] = related.attrib[
                            "resourceTypeGeneral"
                        ]
                    related_ids.append(related_data)
            if related_ids:
                enriched["related_identifiers"] = related_ids

            # Extract funding information
            funding = []
            for funder in resource.findall(".//datacite:fundingReference", namespaces):
                funder_name = funder.find("datacite:funderName", namespaces)
                if funder_name is not None:
                    funding_data = {"funder_name": funder_name.text}

                    award_number = funder.find("datacite:awardNumber", namespaces)
                    if award_number is not None:
                        funding_data["award_number"] = award_number.text

                    award_title = funder.find("datacite:awardTitle", namespaces)
                    if award_title is not None:
                        funding_data["award_title"] = award_title.text

                    funding.append(funding_data)
            if funding:
                enriched["funding"] = funding

            return enriched if enriched else None

        return None

    except Exception as e:
        print(f"OAI-PMH enrichment failed for {record_id}: {e}", file=sys.stderr)
        return None


def get_archive_preview_files(
    record_id: str, filename: str, access_token: str | None = None
) -> list[str]:
    """Extract file listing from Zenodo archive preview without downloading.

    Zenodo provides a preview endpoint that shows archive contents as HTML.
    This allows us to inspect BIDS structure inside .zip files without downloading.

    Args:
        record_id: Zenodo record ID
        filename: Archive filename (e.g., "data.zip")
        access_token: Optional auth token

    Returns:
        List of filenames found inside the archive

    """
    try:
        preview_url = f"https://zenodo.org/records/{record_id}/preview/{filename}"
        params = {"include_deleted": "0"}

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        response = requests.get(preview_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        # Simple HTML parsing to extract filenames
        html_content = response.text.lower()

        # Extract filenames from the HTML structure
        # Pattern: <span><i class="file outline icon"></i> filename</span>
        import re

        file_pattern = r'<span><i class="file outline icon"></i>[^<]*?([^<]+)</span>'
        folder_pattern = r'<i class="folder icon"></i>\s*<a[^>]*>([^<]+)</a>'

        files = re.findall(file_pattern, html_content)
        folders = re.findall(folder_pattern, html_content)

        # Clean up extracted names
        all_items = []
        for item in files:
            cleaned = item.strip()
            if cleaned:
                all_items.append(cleaned)

        for folder in folders:
            cleaned = folder.strip()
            if cleaned:
                all_items.append(cleaned + "/")

        return all_items

    except Exception:
        # Silently fail - this is a best-effort enhancement
        return []


def extract_dataset_info(
    record: dict, enrich_oaipmh: bool = False, access_token: str | None = None
) -> dict[str, Any]:
    """Extract and optionally enrich dataset information.

    Handles InvenioRDM v1 format from REST API and optionally enriches
    with OAI-PMH metadata for more complete information.

    NEW: Also extracts archive contents preview for better BIDS detection.

    Args:
        record: REST API record dictionary
        enrich_oaipmh: Whether to fetch additional metadata via OAI-PMH
        access_token: Optional auth token for OAI-PMH

    Returns:
        Comprehensive dataset dictionary

    """
    metadata = record.get("metadata", {})
    dataset_id = str(record.get("id", ""))

    # Extract DOI (InvenioRDM v1 format)
    doi = record.get("pids", {}).get("doi", {}).get("identifier", "")
    concept_doi = record.get("parent", {}).get("id", "")

    # Basic metadata
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    publication_date = metadata.get("publication_date", "")

    # Creators
    creators = []
    for creator in metadata.get("creators", []):
        if isinstance(creator, dict):
            person = creator.get("person_or_org", {})
            creator_info = {"name": person.get("name", creator.get("name", ""))}

            # Affiliations
            affiliations = creator.get("affiliations", [])
            if affiliations and isinstance(affiliations[0], dict):
                creator_info["affiliation"] = affiliations[0].get("name", "")

            # ORCID
            for identifier in person.get("identifiers", []):
                if identifier.get("scheme") == "orcid":
                    creator_info["orcid"] = identifier.get("identifier", "")

            creators.append(creator_info)

    # Keywords/subjects
    keywords = [s.get("subject", s.get("id", "")) for s in metadata.get("subjects", [])]

    # Resource type
    resource_type_data = metadata.get("resource_type", {})
    resource_type_str = resource_type_data.get("id", "")
    resource_title = resource_type_data.get("title", {}).get("en", "")

    # License
    rights = metadata.get("rights", [])
    license_id = (
        rights[0].get("id", "") if rights and isinstance(rights[0], dict) else ""
    )

    # Access status
    access_data = record.get("access", {})
    access_right = access_data.get("status", "")

    # Files (InvenioRDM v1 format)
    files = []
    archive_contents = {}  # Store archive preview contents
    files_data = record.get("files", {})

    if isinstance(files_data, dict):
        file_entries = files_data.get("entries", {})
        for filename, file_info in file_entries.items():
            file_dict = {
                "filename": file_info.get("key", filename),
                "size_bytes": file_info.get("size", 0),
                "checksum": file_info.get("checksum", ""),
                "mimetype": file_info.get("mimetype", ""),
                "download_url": record.get("links", {}).get("self", "")
                + f"/files/{filename}/content",
            }
            files.append(file_dict)

            # Check if file is an archive - if so, try to get preview
            archived_extensions = [
                ".zip",
                ".tar.gz",
                ".tgz",
                ".tar",
                ".rar",
                ".7z",
                ".gz",
            ]
            fn_lower = filename.lower()
            if any(fn_lower.endswith(ext) for ext in archived_extensions):
                # Try to get archive contents preview
                preview_files = get_archive_preview_files(
                    dataset_id, filename, access_token
                )
                if preview_files:
                    archive_contents[filename] = preview_files

    total_size_bytes = (
        files_data.get("total_bytes", 0) if isinstance(files_data, dict) else 0
    )
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
        "modified": record.get("updated", ""),
        "creators": creators,
        "keywords": keywords,
        "resource_type": resource_type_str,
        "resource_title": resource_title,
        "license": license_id,
        "access_right": access_right,
        "url": record.get("links", {}).get("self_html", ""),
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "files": files,
        "file_count": len(files),
        "total_size_mb": total_size_mb,
        "archive_contents": archive_contents,  # NEW: Archive preview contents
        "has_archive_preview": len(archive_contents) > 0,  # NEW: Flag for filtering
        "source": "zenodo",
    }

    # Enrich with OAI-PMH if requested
    if enrich_oaipmh and dataset_id:
        enriched = enrich_with_oaipmh(dataset_id, access_token)
        if enriched:
            dataset.update(
                {
                    "enriched_subjects": enriched.get("subjects", []),
                    "contributors": enriched.get("contributors", []),
                    "related_identifiers": enriched.get("related_identifiers", []),
                    "funding": enriched.get("funding", []),
                    "oaipmh_enriched": True,
                }
            )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EEG BIDS datasets from Zenodo with optional OAI-PMH enrichment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AGGRESSIVE SEARCH STRATEGY - Maximum Recall:

Query Performance (Tested with type=dataset filter):
  Extended terminology    → 1,570 results ✓ DEFAULT (Maximum recall)
  All modalities OR BIDS  → 1,391 results
  Simple "EEG BIDS"       →   904 results (old conservative approach)

The aggressive query captures ALL neurophysiology datasets by using:
  • Full terminology (electroencephalography, magnetoencephalography, etc.)
  • All modalities (EEG, MEG, iEEG, ECoG, SEEG, EMG)
  • BIDS variants (BIDS, "Brain Imaging Data Structure")
  • Boolean OR logic for maximum coverage

Expected Results Distribution:
  - True BIDS datasets: ~60-70% (need validation)
  - Neurophysiology non-BIDS: ~20-30% (can be converted)
  - False positives: ~5-10% (will filter out)

Filtering Strategy (Post-Processing):
  1. Check for BIDS-compliant file structure
  2. Validate metadata keywords (BIDS, EEG, MEG, etc.)
  3. Verify file extensions (.eeg, .vhdr, .fif, .set, .bdf, .edf)
  4. Analyze dataset descriptions for BIDS mentions
  5. Flag potential BIDS vs non-BIDS datasets

OAI-PMH Enrichment (Optional):
  Adds: controlled vocabularies, contributors, related identifiers, funding
  Trade-off: ~2x slower (requires extra API call per dataset)
  Recommended: Use --enrich-oaipmh for complete metadata extraction

Examples:
  # Maximum recall (default - gets ~1,570 datasets)
  python scripts/ingestions/1_fetch_zenodo_enhanced.py

  # With OAI-PMH enrichment (slower but complete)
  python scripts/ingestions/1_fetch_zenodo_enhanced.py --enrich-oaipmh

  # Conservative search (old approach)
  python scripts/ingestions/1_fetch_zenodo_enhanced.py --query "EEG BIDS" --size 1000

  # Custom modality search
  python scripts/ingestions/1_fetch_zenodo_enhanced.py --query "MEG OR magnetoencephalography"
""",
    )

    parser.add_argument(
        "--query",
        type=str,
        default='EEG OR electroencephalography OR MEG OR magnetoencephalography OR iEEG OR intracranial OR ECoG OR electrocorticography OR SEEG OR EMG OR electromyography OR BIDS OR "Brain Imaging Data Structure"',
        help="Search query (default: AGGRESSIVE - ALL neurophysiology OR BIDS for maximum recall)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/zenodo_datasets_aggressive.json"),
        help="Output JSON file path (default: aggressive unfiltered results).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Maximum number of results to fetch (default: 2000 to capture all ~1,570 aggressive results).",
    )
    parser.add_argument(
        "--resource-type",
        type=str,
        default="dataset",
        help='Filter by resource type (default: "dataset").',
    )
    parser.add_argument(
        "--access-status",
        type=str,
        default="open",
        help='Filter by access status (default: "open").',
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="bestmatch",
        help='Sort order (default: "bestmatch").',
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default="v3dkycQdOlyc0gXXkeeroSIkpSyAgaTyzpsIJLw8lhQsoEw089MFICKqnxWz",
        help="Zenodo Personal Access Token (default: temp test token).",
    )
    parser.add_argument(
        "--enrich-oaipmh",
        action="store_true",
        help="Enrich datasets with OAI-PMH metadata (slower but more complete).",
    )
    parser.add_argument(
        "--oaipmh-sample",
        type=int,
        default=0,
        help="Only enrich first N datasets with OAI-PMH (0 = all if --enrich-oaipmh).",
    )

    args = parser.parse_args()

    # Search via REST API
    records = search_zenodo_rest_api(
        query=args.query,
        size=args.size,
        resource_type=args.resource_type or "",
        access_status=args.access_status or "",
        sort=args.sort,
        access_token=args.access_token,
    )

    if not records:
        print("No records found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Extract dataset information
    print("\nExtracting dataset information...")
    if args.enrich_oaipmh:
        oaipmh_limit = args.oaipmh_sample if args.oaipmh_sample > 0 else len(records)
        print(f"OAI-PMH enrichment enabled for first {oaipmh_limit} datasets")
        print("(This will be slower but provides richer metadata)\n")

    datasets = []
    for i, record in enumerate(records, 1):
        try:
            enrich = args.enrich_oaipmh and (
                args.oaipmh_sample == 0 or i <= args.oaipmh_sample
            )
            dataset = extract_dataset_info(
                record, enrich_oaipmh=enrich, access_token=args.access_token
            )
            datasets.append(dataset)

            if enrich and i % 10 == 0:
                print(f"  Enriched {i}/{len(records)} datasets...")
        except Exception as e:
            print(
                f"Warning: Error extracting record {record.get('id')}: {e}",
                file=sys.stderr,
            )
            continue

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(datasets, fh, indent=2)

    # Print statistics
    print(f"\n{'=' * 70}")
    print(f"Successfully saved {len(datasets)} datasets to {args.output}")
    print(f"{'=' * 70}")

    # Modality analysis
    modalities = {
        "EEG": sum(
            1
            for d in datasets
            if "EEG" in (d.get("title", "") + " ".join(d.get("keywords", []))).upper()
            and "IEEG"
            not in (d.get("title", "") + " ".join(d.get("keywords", []))).upper()
        ),
        "MEG": sum(
            1
            for d in datasets
            if "MEG" in (d.get("title", "") + " ".join(d.get("keywords", []))).upper()
        ),
        "iEEG/ECoG": sum(
            1
            for d in datasets
            if any(
                x in (d.get("title", "") + " ".join(d.get("keywords", []))).upper()
                for x in ["IEEG", "ECOG", "INTRACRANIAL"]
            )
        ),
        "EMG": sum(
            1
            for d in datasets
            if "EMG" in (d.get("title", "") + " ".join(d.get("keywords", []))).upper()
        ),
    }

    print("\nModality Detection:")
    for mod, count in modalities.items():
        print(f"  {mod:15s}: {count:4d} datasets ({count / len(datasets) * 100:.1f}%)")

    # Metadata coverage
    print("\nMetadata Coverage:")
    print(
        f"  With DOI:       {sum(1 for d in datasets if d.get('doi')):4d}/{len(datasets)} ({sum(1 for d in datasets if d.get('doi')) / len(datasets) * 100:.1f}%)"
    )
    print(
        f"  With license:   {sum(1 for d in datasets if d.get('license')):4d}/{len(datasets)} ({sum(1 for d in datasets if d.get('license')) / len(datasets) * 100:.1f}%)"
    )
    print(
        f"  With keywords:  {sum(1 for d in datasets if d.get('keywords')):4d}/{len(datasets)} ({sum(1 for d in datasets if d.get('keywords')) / len(datasets) * 100:.1f}%)"
    )
    print(
        f"  With files:     {sum(1 for d in datasets if d.get('file_count', 0) > 0):4d}/{len(datasets)} ({sum(1 for d in datasets if d.get('file_count', 0) > 0) / len(datasets) * 100:.1f}%)"
    )

    if args.enrich_oaipmh:
        enriched_count = sum(1 for d in datasets if d.get("oaipmh_enriched"))
        print(
            f"  OAI-PMH enriched: {enriched_count:4d}/{len(datasets)} ({enriched_count / len(datasets) * 100:.1f}%)"
        )

    # Total size
    total_size_mb = sum(d.get("total_size_mb", 0) for d in datasets)
    print(f"\nTotal Size: {total_size_mb / 1024:.2f} GB ({total_size_mb:.2f} MB)")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
