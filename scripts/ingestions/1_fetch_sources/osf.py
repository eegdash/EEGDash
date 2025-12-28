"""Fetch neural recording BIDS datasets from Open Science Framework (OSF).

This script searches OSF for public nodes (projects/data) containing EEG, MEG, iEEG,
or other neural recording modalities with BIDS formatting, using the OSF API v2.
It retrieves comprehensive metadata including contributors, licenses, and project
details, outputting in the EEGDash Dataset schema format.

BIDS validation is performed by checking OSF storage for:
- Required BIDS files (dataset_description.json)
- Optional BIDS files (participants.tsv, README, etc.)
- BIDS-like subject folder patterns (sub-XX)
- BIDS dataset zip files (containing BIDS data)

Output: consolidated/osf_datasets.json
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Add ingestion paths before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bids import (
    BIDS_DATASET_ZIP_PATTERN,
    BIDS_OPTIONAL_FILES,
    BIDS_REQUIRED_FILES,
    BIDS_SUBJECT_PATTERN,
    collect_bids_matches,
)
from _http import request_json
from _serialize import (
    extract_subjects_count,
    generate_dataset_id,
    save_datasets_deterministically,
    setup_paths,
)

setup_paths()
from eegdash.records import Dataset, create_dataset

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

# Modality tags to search for - optimized for speed (core keywords only)
# Less common variants are covered by title search
MODALITY_TAGS = {
    "eeg": ["eeg", "electroencephalography", "erp"],  # erp = event-related potential
    "meg": ["meg", "magnetoencephalography"],
    "emg": ["emg", "electromyography"],
    "fnirs": ["fnirs", "fNIRS", "nirs"],
    "lfp": ["lfp", "local field potential"],
    "spike": ["spike", "single unit", "multi-unit"],
    "mea": ["mea", "microelectrode array", "neuropixels"],
    "ieeg": ["ieeg", "intracranial eeg", "seeg", "ecog"],
    "bids": ["bids"],
}

# Title search keywords - for datasets without tags
# These are searched via filter[title][icontains]
TITLE_SEARCH_KEYWORDS = {
    "eeg": [
        "EEG",
        "electroencephalography",
        "electroencephalogram",
        "event-related potential",
        "ERP CORE",
    ],
    "meg": ["MEG", "magnetoencephalography"],
    "emg": ["EMG", "electromyography"],
    "fnirs": ["fNIRS", "NIRS", "near-infrared spectroscopy"],
    "ieeg": ["iEEG", "intracranial EEG", "ECoG", "sEEG"],
    "bids": ["BIDS"],
}


def validate_bids_structure(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate BIDS structure from OSF file list.

    Checks for:
    - Required BIDS files (dataset_description.json)
    - Optional BIDS files (participants.tsv, README, etc.)
    - Subject-level folders (sub-XX)
    - BIDS dataset zip files

    Args:
        files: List of file/folder dictionaries from OSF API

    Returns:
        Dictionary with validation results:
        - is_bids: True if BIDS structure confirmed
        - bids_files_found: List of BIDS files found
        - subject_count: Number of subject folders/zips
        - has_subject_zips: True if subjects are in ZIP format
        - has_bids_zip: True if BIDS dataset ZIP found
        - bids_zip_files: List of BIDS dataset ZIP filenames

    """
    result = {
        "is_bids": False,
        "bids_files_found": [],
        "subject_count": 0,
        "has_subject_zips": False,
        "has_bids_zip": False,
        "bids_zip_files": [],
    }

    if not files:
        return result

    # Collect all file/folder names (handle both files and folders)
    all_names = []
    for f in files:
        name = f.get("name", "")
        if name:
            all_names.append(name)

    matches = collect_bids_matches(
        all_names,
        required_files=BIDS_REQUIRED_FILES,
        optional_files=BIDS_OPTIONAL_FILES,
        subject_pattern=BIDS_SUBJECT_PATTERN,
        dataset_zip_pattern=BIDS_DATASET_ZIP_PATTERN,
        dataset_zip_matcher="match",
    )
    result["bids_files_found"] = matches["required_found"] + matches["optional_found"]

    subject_files = matches["subject_files"]
    subject_zips = matches["subject_zips"]
    result["subject_count"] = len(subject_files)
    result["has_subject_zips"] = len(subject_zips) > 0

    result["bids_zip_files"] = matches["bids_zip_files"]
    result["has_bids_zip"] = len(matches["bids_zip_files"]) > 0

    # Determine if this is a valid BIDS dataset
    # Criteria: has dataset_description.json OR has subject folders OR has BIDS zip
    has_required = "dataset_description.json" in result["bids_files_found"]
    has_subjects = result["subject_count"] > 0
    has_bids_zip = result["has_bids_zip"]

    result["is_bids"] = (
        has_required
        or (has_subjects and len(result["bids_files_found"]) > 0)
        or has_bids_zip
    )

    return result


def fetch_node_files(
    node_id: str,
    storage_provider: str = "osfstorage",
    timeout: float = 10.0,
    max_files: int = 200,
) -> list[dict[str, Any]]:
    """Fetch files from OSF node storage.

    Args:
        node_id: OSF node ID
        storage_provider: Storage provider (default: osfstorage)
        timeout: Request timeout
        max_files: Maximum files to fetch

    Returns:
        List of file/folder dictionaries

    """
    files = []
    url = f"{OSF_API_URL}/nodes/{node_id}/files/{storage_provider}/"

    try:
        data, response = request_json("get", url, timeout=timeout)
        if not response or response.status_code != 200 or data is None:
            return files
        items = data.get("data", [])

        for item in items[:max_files]:
            attrs = item.get("attributes", {})
            files.append(
                {
                    "name": attrs.get("name", ""),
                    "kind": attrs.get("kind", ""),  # 'file' or 'folder'
                    "size": attrs.get("size", 0),
                    "path": attrs.get("materialized_path", ""),
                }
            )

    except Exception:
        pass

    return files


# Modality tags continued (ieeg closing moved above)


def fetch_license_name(license_id: str, timeout: float = 10.0) -> str | None:
    """Fetch license name from OSF API."""
    if license_id in LICENSE_NAMES:
        return LICENSE_NAMES[license_id]

    try:
        data, response = request_json(
            "get",
            f"{OSF_API_URL}/licenses/{license_id}/",
            timeout=timeout,
        )
        if response and response.status_code == 200 and data is not None:
            name = data.get("data", {}).get("attributes", {}).get("name")
            return name
    except Exception:
        pass
    return None


def fetch_contributors(node_id: str, timeout: float = 10.0) -> list[str]:
    """Fetch contributor names for a node."""
    try:
        data, response = request_json(
            "get",
            f"{OSF_API_URL}/nodes/{node_id}/contributors/",
            params={"embed": "users"},
            timeout=timeout,
        )
        if not response or response.status_code != 200 or data is None:
            return []
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


def fetch_node_children(
    node_id: str,
    timeout: float = 10.0,
    fetch_details: bool = True,
    require_bids: bool = True,
    validate_bids: bool = True,
    seen_ids: set | None = None,
    parent_modality: str | None = None,
) -> list[Dataset]:
    """Fetch children of an OSF node (for projects with sub-components).

    This captures datasets like ERP CORE which has separate components
    for N170, P3, N400, etc.

    Args:
        node_id: Parent node ID
        timeout: Request timeout
        fetch_details: Whether to fetch additional details
        require_bids: Only include datasets mentioning BIDS
        validate_bids: Validate BIDS by checking file structure
        seen_ids: Set of already-seen node IDs to skip
        parent_modality: Modality inherited from parent (for children without tags)

    Returns:
        List of Dataset documents from children

    """
    if seen_ids is None:
        seen_ids = set()

    datasets = []

    try:
        data, response = request_json(
            "get",
            f"{OSF_API_URL}/nodes/{node_id}/children/",
            params={"page[size]": 100},
            timeout=timeout,
        )
        if not response or response.status_code != 200 or data is None:
            return datasets
        children = data.get("data", [])

        for child in children:
            child_id = child.get("id", "")
            if child_id in seen_ids:
                continue

            try:
                dataset = process_node(
                    child,
                    fetch_details=fetch_details,
                    timeout=timeout,
                    require_bids=require_bids,
                    validate_bids=validate_bids,
                    inherit_modality=parent_modality,
                )
                if dataset:
                    datasets.append(dataset)
                    seen_ids.add(child_id)
            except Exception:
                continue

    except Exception:
        pass

    return datasets


def fetch_nodes_by_title(
    keyword: str,
    max_results: int = 100,
    page_size: int = 100,
    timeout: float = 30.0,
    fetch_details: bool = True,
    require_bids: bool = True,
    validate_bids: bool = True,
    seen_ids: set | None = None,
    fetch_children: bool = True,
) -> list[Dataset]:
    """Fetch public OSF nodes by title keyword search.

    This captures datasets that don't use tags but have relevant keywords
    in their title (like "ERP CORE", "EEG dataset", etc.).
    Also fetches children of projects to capture sub-components.

    Args:
        keyword: Keyword to search in titles
        max_results: Maximum number of results to fetch
        page_size: Number of results per page (max 100)
        timeout: Request timeout in seconds
        fetch_details: Whether to fetch additional details
        require_bids: Only include datasets mentioning BIDS
        validate_bids: Validate BIDS by checking file structure
        seen_ids: Set of already-seen node IDs to skip
        fetch_children: Also fetch children of projects found

    Returns:
        List of Dataset documents

    """
    if seen_ids is None:
        seen_ids = set()

    datasets = []
    projects_to_check_children = []  # Track projects for child fetching
    page = 1
    fetched = 0

    # Build initial URL with title filter
    base_params = {
        "filter[public]": "true",
        "filter[title][icontains]": keyword,
        "page[size]": min(page_size, 100),
    }

    url = f"{OSF_API_URL}/nodes/"
    params = base_params.copy()

    while fetched < max_results:
        try:
            data, response = request_json(
                "get",
                url,
                params=params,
                timeout=timeout,
                raise_for_status=True,
                raise_for_request=True,
            )
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}", file=sys.stderr)
            break

        if not response or data is None:
            print(f"  Error fetching page {page}: empty response", file=sys.stderr)
            break
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

            # Track projects for child fetching (with their modality)
            category = node.get("attributes", {}).get("category", "")

            try:
                dataset = process_node(
                    node,
                    fetch_details=fetch_details,
                    timeout=timeout,
                    require_bids=require_bids,
                    validate_bids=validate_bids,
                )
                if dataset:
                    datasets.append(dataset)
                    seen_ids.add(node_id)
                    fetched += 1
                    # Track projects with their primary modality for child fetching
                    if fetch_children and category == "project":
                        projects_to_check_children.append(
                            (node_id, dataset.get("recording_modality"))
                        )
            except Exception as e:
                print(
                    f"  Warning: Error processing node {node_id}: {e}", file=sys.stderr
                )
                continue

        # Check for next page
        next_url = data.get("links", {}).get("next")
        if not next_url or fetched >= max_results:
            break

        url = next_url
        params = {}  # Next URL has all params
        page += 1
        time.sleep(0.1)  # Rate limiting

    # Fetch children of projects found (inherit modality from parent)
    if fetch_children and projects_to_check_children:
        children_found = 0
        for project_id, parent_modality in projects_to_check_children[
            :20
        ]:  # Limit to avoid too many requests
            children = fetch_node_children(
                project_id,
                timeout=timeout,
                fetch_details=fetch_details,
                require_bids=require_bids,
                validate_bids=validate_bids,
                seen_ids=seen_ids,
                parent_modality=parent_modality,
            )
            datasets.extend(children)
            children_found += len(children)
            time.sleep(0.1)
        if children_found > 0:
            print(f"  + {children_found} children from projects")

    return datasets


def fetch_osf_nodes(
    modalities: list[str] | None = None,
    require_bids: bool = True,
    validate_bids: bool = True,
    max_results_per_modality: int = 1000,
    page_size: int = 100,
    timeout: float = 30.0,
    fetch_details: bool = True,
    search_titles: bool = True,
) -> list[Dataset]:
    """Fetch public OSF nodes with neural recording modalities.

    Searches both by tags and by title keywords to capture datasets
    that don't use tags (like ERP CORE).

    Args:
        modalities: List of modalities to search for (default: all)
        require_bids: Only include datasets with BIDS in tags/description
        validate_bids: Validate BIDS by checking file structure
        max_results_per_modality: Maximum results per modality tag search
        page_size: Number of results per page (max 100)
        timeout: Request timeout in seconds
        fetch_details: Whether to fetch additional details (contributors, license)
        search_titles: Also search by title keywords (captures untagged datasets)

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
        print(f"\n{'=' * 40}")
        print(f"Searching for tag: {tag}")
        print(f"{'=' * 40}")

        datasets = fetch_nodes_by_tag(
            tag=tag,
            max_results=max_results_per_modality,
            page_size=page_size,
            timeout=timeout,
            fetch_details=fetch_details,
            require_bids=require_bids,
            validate_bids=validate_bids,
            seen_ids=seen_ids,
        )

        # Add new datasets (already filtered by seen_ids in fetch_nodes_by_tag)
        all_datasets.extend(datasets)

        print(f"  New unique datasets from '{tag}': {len(datasets)}")
        print(f"  Total unique so far: {len(all_datasets)}")

        time.sleep(0.2)  # Rate limiting between tag searches

    # Also search by title keywords to capture untagged datasets
    if search_titles:
        print(f"\n{'=' * 60}")
        print("Searching by title keywords (captures untagged datasets)")
        print(f"{'=' * 60}")

        # Collect title search keywords
        title_keywords = set()
        for mod in modalities:
            if mod in TITLE_SEARCH_KEYWORDS:
                title_keywords.update(TITLE_SEARCH_KEYWORDS[mod])

        # Always search for BIDS in title
        title_keywords.add("BIDS")

        for keyword in sorted(title_keywords):
            print(f"\n{'=' * 40}")
            print(f"Title search: '{keyword}'")
            print(f"{'=' * 40}")

            datasets = fetch_nodes_by_title(
                keyword=keyword,
                max_results=max_results_per_modality,
                page_size=page_size,
                timeout=timeout,
                fetch_details=fetch_details,
                require_bids=require_bids,
                validate_bids=validate_bids,
                seen_ids=seen_ids,
            )

            all_datasets.extend(datasets)

            print(f"  New unique datasets from title '{keyword}': {len(datasets)}")
            print(f"  Total unique so far: {len(all_datasets)}")

            time.sleep(0.2)  # Rate limiting

    print(f"\n\nTotal unique datasets: {len(all_datasets)}")
    return all_datasets


def fetch_nodes_by_tag(
    tag: str,
    max_results: int = 1000,
    page_size: int = 100,
    timeout: float = 30.0,
    fetch_details: bool = True,
    require_bids: bool = True,
    validate_bids: bool = True,
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
            data, response = request_json(
                "get",
                url,
                params=params,
                timeout=timeout,
                raise_for_status=True,
                raise_for_request=True,
            )
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}", file=sys.stderr)
            break

        if not response or data is None:
            print(f"  Error fetching page {page}: empty response", file=sys.stderr)
            break
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
                    validate_bids=validate_bids,
                )
                if dataset:
                    datasets.append(dataset)
                    seen_ids.add(node_id)
                    fetched += 1
            except Exception as e:
                print(
                    f"  Warning: Error processing node {node_id}: {e}", file=sys.stderr
                )
                continue

        # Check for next page
        next_url = data.get("links", {}).get("next")
        if not next_url or fetched >= max_results:
            break

        url = next_url
        params = {}  # Next URL has all params
        page += 1
        time.sleep(0.1)  # Rate limiting

    return datasets


def process_node(
    node: dict,
    fetch_details: bool = True,
    timeout: float = 10.0,
    require_bids: bool = True,
    validate_bids: bool = True,
    inherit_modality: str | None = None,
) -> Dataset | None:
    """Process an OSF node into a Dataset document.

    Args:
        node: OSF node data
        fetch_details: Whether to fetch additional details
        timeout: Request timeout
        require_bids: Only include datasets mentioning BIDS in tags (legacy)
        validate_bids: Validate BIDS by checking actual file structure
        inherit_modality: Modality to use if none detected (inherited from parent)

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

    # Check for BIDS mentions in tags or description
    tags_lower = [t.lower() for t in tags]
    desc_lower = description.lower()
    has_bids_mention = any("bids" in t for t in tags_lower) or "bids" in desc_lower

    # Legacy behavior: skip if require_bids and no BIDS mention
    if require_bids and not has_bids_mention:
        return None

    bids_version = "unknown" if has_bids_mention else None
    bids_validation = None

    # Validate BIDS structure by checking actual files
    if validate_bids and fetch_details:
        files = fetch_node_files(node_id, timeout=timeout)
        if files:
            bids_validation = validate_bids_structure(files)
            if bids_validation["is_bids"]:
                bids_version = "validated"
            elif not has_bids_mention:
                # No BIDS mention and no BIDS structure - skip unless validate_bids is False
                pass  # Continue processing, will be flagged as non-BIDS

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

    # If no modality detected, use inherited modality from parent (for children nodes)
    if not modalities and inherit_modality:
        modalities = [inherit_modality]

    # If still no modality detected, skip
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

    # Generate SurnameYEAR dataset_id
    dataset_id = generate_dataset_id(
        source="osf",
        authors=authors,
        date=date_created,
        fallback_id=node_id,
    )

    # Extract subject count from description using shared utility
    subjects_count = extract_subjects_count(description)

    # Create dataset
    dataset = create_dataset(
        dataset_id=dataset_id,
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
        subjects_count=subjects_count if subjects_count > 0 else None,
    )

    # Add OSF-specific metadata
    dataset["osf_id"] = node_id
    dataset["osf_category"] = category
    dataset["tags"] = tags
    if description:
        dataset["description"] = description[:1000]  # Truncate long descriptions

    # Store demographics for downstream use (for manifest->digester)
    if subjects_count > 0:
        dataset["demographics"] = {
            "subjects_count": subjects_count,
            "ages": [],
        }

    # Add BIDS validation results
    if bids_validation:
        dataset["bids_validated"] = bids_validation["is_bids"]
        if bids_validation["bids_files_found"]:
            dataset["bids_files_found"] = bids_validation["bids_files_found"]
        if bids_validation["subject_count"] > 0:
            dataset["bids_subject_count"] = bids_validation["subject_count"]
            dataset["bids_has_subject_zips"] = bids_validation["has_subject_zips"]
        if bids_validation["has_bids_zip"]:
            dataset["bids_has_dataset_zip"] = True
            dataset["bids_zip_files"] = bids_validation["bids_zip_files"]
    else:
        dataset["bids_validated"] = has_bids_mention  # Fall back to tag-based

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
        help="Modalities to search for (default: all). Options: eeg, meg, ieeg, fnirs",
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
        help="Don't require BIDS mention in tags/description (searches all neural recording datasets).",
    )
    parser.add_argument(
        "--validate-bids",
        action="store_true",
        default=True,
        help="Validate BIDS by checking actual file structure (default: True).",
    )
    parser.add_argument(
        "--no-validate-bids",
        action="store_true",
        help="Skip BIDS file structure validation.",
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

    # Determine validation mode
    validate_bids = not args.no_validate_bids

    # Fetch nodes
    datasets = fetch_osf_nodes(
        modalities=modalities,
        require_bids=not args.no_bids,
        validate_bids=validate_bids,
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
    with_bids_tag = sum(1 for d in datasets if d.get("bids_version"))
    with_bids_validated = sum(1 for d in datasets if d.get("bids_validated"))
    with_subject_folders = sum(
        1 for d in datasets if d.get("bids_subject_count", 0) > 0
    )
    with_subject_zips = sum(1 for d in datasets if d.get("bids_has_subject_zips"))
    with_bids_zip = sum(1 for d in datasets if d.get("bids_has_dataset_zip"))

    print("\nBIDS Validation:")
    print(f"  With BIDS tag/mention: {with_bids_tag}/{len(datasets)}")
    print(f"  Confirmed BIDS (file check): {with_bids_validated}/{len(datasets)}")
    print(f"  With subject folders/zips: {with_subject_folders}/{len(datasets)}")
    print(f"  Using subject-level ZIPs: {with_subject_zips}/{len(datasets)}")
    print(f"  With BIDS dataset ZIP: {with_bids_zip}/{len(datasets)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
