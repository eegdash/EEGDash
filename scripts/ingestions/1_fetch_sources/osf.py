"""Fetch neural recording BIDS datasets from Open Science Framework (OSF).

This script searches OSF for public nodes (projects/data) containing EEG, MEG, iEEG,
or other neural recording modalities with BIDS formatting, using the OSF API v2. 
It retrieves comprehensive metadata including contributors, licenses, and project 
details, outputting in the EEGDash Dataset schema format.

Output: consolidated/osf_datasets.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Add parent paths for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eegdash.records import Dataset, create_dataset

# Add ingestions dir to path for _serialize module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _serialize import save_datasets_deterministically

OSF_API_URL = "https://api.osf.io/v2"

# License ID to name mapping (common OSF licenses)
LICENSE_NAMES = {
    "563c1cf88c5e4a3877f9e96a": "CC-BY-4.0",
    "563c1cf88c5e4a3877f9e965": "CC0-1.0",
    "563c1cf88c5e4a3877f9e968": "MIT",
    "563c1cf88c5e4a3877f9e96c": "GPL-3.0",
    "563c1cf88c5e4a3877f9e96e": "Apache-2.0",
    "563c1cf88c5e4a3877f9e967": "BSD-2-Clause",
    "563c1cf88c5e4a3877f9e969": "BSD-3-Clause",
    "563c1cf88c5e4a3877f9e96b": "CC-BY-NC-4.0",
}

# Categories we're interested in (datasets/data, not posters/presentations)
DATA_CATEGORIES = {"data", "project", "analysis", "software", "other"}

# Modality tags to search for - comprehensive keyword coverage
MODALITY_TAGS = {
    "eeg": [
        "eeg", "electroencephalography", "electroencephalogram", 
        "scalp eeg", "scalp-eeg"
    ],
    "meg": [
        "meg", "magnetoencephalography", "magnetoencephalogram"
    ],
    "emg": [
        "emg", "electromyography", "electromyogram"
    ],
    "fnirs": [
        "fnirs", "fNIRS", "nirs", "near-infrared spectroscopy", 
        "near infrared spectroscopy", "functional near-infrared"
    ],
    "lfp": [
        "lfp", "local field potential", "local field potentials", 
        "field potential", "field potentials"
    ],
    "spike": [
        "single unit", "single-unit", "multi-unit", "multiunit",
        "spike", "spike train", "neuronal firing", "unit activity", 
        "single unit activity", "multi-unit activity"
    ],
    "mea": [
        "mea", "microelectrode array", "microelectrode arrays",
        "utah array", "neuropixels", "depth electrode"
    ],
    "ieeg": [
        "ieeg", "intracranial eeg", "intracranial electroencephalography",
        "intracranial electroencephalogram", "seeg", "stereoelectroencephalography",
        "ecog", "electrocorticography", "corticography",
        "subdural electrode", "subdural grid", "subdural strip"
    ],
    "bids": [
        "bids", "brain imaging data structure", 
        "brain imaging data structures"
    ],
}


def fetch_license_name(license_id: str, timeout: float = 10.0) -> str | None:
    """Fetch license name from OSF API."""
    if license_id in LICENSE_NAMES:
        return LICENSE_NAMES[license_id]
    
    try:
        response = requests.get(
            f"{OSF_API_URL}/licenses/{license_id}/",
            timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            name = data.get("data", {}).get("attributes", {}).get("name")
            return name
    except Exception:
        pass
    return None


def fetch_contributors(node_id: str, timeout: float = 10.0) -> list[str]:
    """Fetch contributor names for a node."""
    try:
        response = requests.get(
            f"{OSF_API_URL}/nodes/{node_id}/contributors/",
            params={"embed": "users"},
            timeout=timeout
        )
        if response.status_code != 200:
            return []
        
        data = response.json()
        contributors = []
        
        for c in data.get("data", []):
            # Try embedded users first
            user_data = c.get("embeds", {}).get("users", {}).get("data", {})
            if user_data:
                name = user_data.get("attributes", {}).get("full_name")
                if name:
                    contributors.append(name)
        
        return contributors
    except Exception:
        return []


def fetch_osf_nodes(
    modalities: list[str] | None = None,
    require_bids: bool = True,
    max_results_per_modality: int = 1000,
    page_size: int = 100,
    timeout: float = 30.0,
    fetch_details: bool = True,
) -> list[Dataset]:
    """Fetch public OSF nodes with neural recording modalities.

    Args:
        modalities: List of modalities to search for (default: all)
        require_bids: Only include datasets with BIDS in tags/description
        max_results_per_modality: Maximum results per modality tag search
        page_size: Number of results per page (max 100)
        timeout: Request timeout in seconds
        fetch_details: Whether to fetch additional details (contributors, license)

    Returns:
        List of Dataset documents

    """
    if modalities is None:
        modalities = list(MODALITY_TAGS.keys())
    
    print(f"Searching for modalities: {modalities}")
    print(f"Require BIDS: {require_bids}")
    
    # Collect all search tags
    search_tags = []
    for mod in modalities:
        if mod in MODALITY_TAGS:
            search_tags.extend(MODALITY_TAGS[mod])
    
    # Also search directly for BIDS
    if require_bids:
        search_tags.append("bids")
    
    # Deduplicate
    search_tags = list(set(search_tags))
    print(f"Search tags: {search_tags}")
    
    # Track seen nodes to avoid duplicates
    seen_ids = set()
    all_datasets = []
    
    for tag in search_tags:
        print(f"\n{'='*40}")
        print(f"Searching for tag: {tag}")
        print(f"{'='*40}")
        
        datasets = fetch_nodes_by_tag(
            tag=tag,
            max_results=max_results_per_modality,
            page_size=page_size,
            timeout=timeout,
            fetch_details=fetch_details,
            require_bids=require_bids,
            seen_ids=seen_ids,
        )
        
        # Add new datasets (already filtered by seen_ids in fetch_nodes_by_tag)
        all_datasets.extend(datasets)
        
        print(f"  New unique datasets from '{tag}': {len(datasets)}")
        print(f"  Total unique so far: {len(all_datasets)}")
        
        time.sleep(0.5)  # Rate limiting between tag searches
    
    print(f"\n\nTotal unique datasets: {len(all_datasets)}")
    return all_datasets


def fetch_nodes_by_tag(
    tag: str,
    max_results: int = 1000,
    page_size: int = 100,
    timeout: float = 30.0,
    fetch_details: bool = True,
    require_bids: bool = True,
    seen_ids: set | None = None,
) -> list[Dataset]:
    """Fetch public OSF nodes with a specific tag.

    Args:
        tag: Tag to filter by
        max_results: Maximum number of results to fetch
        page_size: Number of results per page (max 100)
        timeout: Request timeout in seconds
        fetch_details: Whether to fetch additional details
        require_bids: Only include datasets mentioning BIDS
        seen_ids: Set of already-seen node IDs to skip

    Returns:
        List of Dataset documents

    """
    if seen_ids is None:
        seen_ids = set()
    
    datasets = []
    page = 1
    fetched = 0
    
    # Build initial URL with filters
    base_params = {
        "filter[public]": "true",
        "filter[tags]": tag,
        "page[size]": min(page_size, 100),
    }
    
    url = f"{OSF_API_URL}/nodes/"
    params = base_params.copy()
    
    while fetched < max_results:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}", file=sys.stderr)
            break
        
        data = response.json()
        nodes = data.get("data", [])
        
        if not nodes:
            break
        
        # Get total from meta
        meta = data.get("links", {}).get("meta", {}) or data.get("meta", {})
        total = meta.get("total", 0)
        if page == 1:
            print(f"  Total available: {total}")
        
        for node in nodes:
            if fetched >= max_results:
                break
            
            node_id = node.get("id", "")
            if node_id in seen_ids:
                continue
            
            try:
                dataset = process_node(
                    node, 
                    fetch_details=fetch_details, 
                    timeout=timeout,
                    require_bids=require_bids,
                )
                if dataset:
                    datasets.append(dataset)
                    seen_ids.add(node_id)
                    fetched += 1
            except Exception as e:
                print(f"  Warning: Error processing node {node_id}: {e}", file=sys.stderr)
                continue
        
        # Check for next page
        next_url = data.get("links", {}).get("next")
        if not next_url or fetched >= max_results:
            break
        
        url = next_url
        params = {}  # Next URL has all params
        page += 1
        time.sleep(0.3)  # Rate limiting
    
    return datasets


def process_node(
    node: dict,
    fetch_details: bool = True,
    timeout: float = 10.0,
    require_bids: bool = True,
) -> Dataset | None:
    """Process an OSF node into a Dataset document.

    Args:
        node: OSF node data
        fetch_details: Whether to fetch additional details
        timeout: Request timeout
        require_bids: Only include datasets mentioning BIDS

    Returns:
        Dataset document or None if filtered out

    """
    node_id = node.get("id", "")
    attrs = node.get("attributes", {})
    relationships = node.get("relationships", {})
    links = node.get("links", {})
    
    # Filter out non-data categories
    category = attrs.get("category", "")
    if category not in DATA_CATEGORIES:
        return None
    
    # Skip registrations, preprints, forks
    if attrs.get("registration") or attrs.get("preprint") or attrs.get("fork"):
        return None
    
    # Skip non-public
    if not attrs.get("public", False):
        return None
    
    title = attrs.get("title", "")
    description = attrs.get("description", "") or ""
    tags = attrs.get("tags", [])
    date_created = attrs.get("date_created", "")
    date_modified = attrs.get("date_modified", "")
    
    # Check for BIDS - in tags or description
    tags_lower = [t.lower() for t in tags]
    desc_lower = description.lower()
    has_bids = any("bids" in t for t in tags_lower) or "bids" in desc_lower
    
    if require_bids and not has_bids:
        return None
    
    bids_version = "unknown" if has_bids else None
    
    # Get license
    license_info = None
    license_rel = relationships.get("license", {}).get("data", {})
    if license_rel:
        license_id = license_rel.get("id")
        if license_id:
            license_info = LICENSE_NAMES.get(license_id)
            if not license_info and fetch_details:
                license_info = fetch_license_name(license_id, timeout=timeout)
    
    # Get contributors
    authors = []
    if fetch_details:
        authors = fetch_contributors(node_id, timeout=timeout)
    
    # Build URLs
    html_url = links.get("html", f"https://osf.io/{node_id}/")
    
    # Determine modalities from tags and description
    modalities = []
    combined_text = " ".join(tags_lower + [desc_lower])
    
    for mod, keywords in MODALITY_TAGS.items():
        if any(kw in combined_text for kw in keywords):
            modalities.append(mod)
    
    # If no modality detected, skip
    if not modalities:
        return None
    
    # Determine primary modality
    primary_modality = modalities[0]
    
    # Determine study domain from tags/description
    study_domain = None
    domain_keywords = {
        "attention": "attention",
        "memory": "memory",
        "language": "language",
        "motor": "motor",
        "sleep": "sleep",
        "epilepsy": "epilepsy",
        "emotion": "emotion",
        "perception": "perception",
        "cognition": "cognition",
        "learning": "learning",
        "decision": "decision-making",
        "bci": "brain-computer interface",
        "erp": "event-related potentials",
        "resting": "resting-state",
    }
    for keyword, domain in domain_keywords.items():
        if keyword in combined_text:
            study_domain = domain
            break
    
    # Create dataset
    dataset = create_dataset(
        dataset_id=f"osf_{node_id}",
        name=title,
        source="osf",
        recording_modality=primary_modality,
        modalities=modalities,
        bids_version=bids_version,
        license=license_info,
        authors=authors,
        study_domain=study_domain,
        source_url=html_url,
        dataset_modified_at=date_modified or date_created,
    )
    
    # Add OSF-specific metadata
    dataset["osf_id"] = node_id
    dataset["osf_category"] = category
    dataset["tags"] = tags
    if description:
        dataset["description"] = description[:1000]  # Truncate long descriptions
    
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch neural recording BIDS datasets from Open Science Framework."
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        help='Modalities to search for (default: all). Options: eeg, meg, ieeg, fnirs',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated/osf_datasets.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum results per modality tag (default: 500).",
    )
    parser.add_argument(
        "--no-bids",
        action="store_true",
        help="Include datasets without BIDS (not recommended).",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip fetching additional details (faster but less metadata).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--digested-at",
        type=str,
        default=None,
        help="ISO 8601 timestamp for digested_at field (for deterministic output, default: omitted for determinism)",
    )

    args = parser.parse_args()

    modalities = args.modalities
    if modalities:
        print(f"Fetching OSF datasets for modalities: {modalities}")
    else:
        print(f"Fetching OSF datasets for all modalities: {list(MODALITY_TAGS.keys())}")

    # Fetch nodes
    datasets = fetch_osf_nodes(
        modalities=modalities,
        require_bids=not args.no_bids,
        max_results_per_modality=args.max_results,
        fetch_details=not args.no_details,
        timeout=args.timeout,
    )

    if not datasets:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Add digested_at timestamp if provided
    if args.digested_at:
        for ds in datasets:
            if "timestamps" in ds:
                ds["timestamps"]["digested_at"] = args.digested_at

    # Save deterministically
    save_datasets_deterministically(datasets, args.output)

    print(f"\nSaved {len(datasets)} datasets to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(datasets)}")

    # Count by category
    categories = {}
    for d in datasets:
        cat = d.get("osf_category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    # Count by modality
    modalities = {}
    for d in datasets:
        for mod in d.get("modalities", []):
            modalities[mod] = modalities.get(mod, 0) + 1
    
    print("\nBy Modality:")
    for mod, count in sorted(modalities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mod}: {count}")

    # License coverage
    with_license = sum(1 for d in datasets if d.get("license"))
    print(f"\nDatasets with license: {with_license}/{len(datasets)}")

    # Author coverage
    with_authors = sum(1 for d in datasets if d.get("authors"))
    total_authors = len(set(a for d in datasets for a in d.get("authors", [])))
    print(f"Datasets with authors: {with_authors}/{len(datasets)}")
    print(f"Unique authors: {total_authors}")

    # BIDS coverage
    with_bids = sum(1 for d in datasets if d.get("bids_version"))
    print(f"Datasets with BIDS: {with_bids}/{len(datasets)}")

    print("=" * 60)



if __name__ == "__main__":
    main()
