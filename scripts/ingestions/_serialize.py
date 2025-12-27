"""Deterministic JSON serialization utilities for dataset documents.

Ensures consistent, sorted output across runs for CI/CD reproducibility.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Path constants for the ingestion scripts
_SERIALIZE_FILE = Path(__file__).resolve()
INGESTIONS_DIR = _SERIALIZE_FILE.parent
SCRIPTS_DIR = INGESTIONS_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def setup_paths() -> None:
    """Add project paths to sys.path for importing eegdash modules.

    Call this at the top of any fetch script before importing from eegdash.

    Example:
        from _serialize import setup_paths, generate_dataset_id, save_datasets_deterministically
        setup_paths()
        from eegdash.records import create_dataset

    """
    project_root_str = str(PROJECT_ROOT)
    ingestions_dir_str = str(INGESTIONS_DIR)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if ingestions_dir_str not in sys.path:
        sys.path.insert(0, ingestions_dir_str)


def extract_surname(author_name: str) -> str | None:
    """Extract surname from an author name string.

    Handles various formats:
    - "Lastname, Firstname"
    - "Firstname Lastname"
    - "Firstname Middle Lastname"
    - "F. M. Lastname"

    Args:
        author_name: Author name in various formats

    Returns:
        Extracted surname or None if extraction fails

    """
    if not author_name or not isinstance(author_name, str):
        return None

    # Clean the name
    name = author_name.strip()
    if not name:
        return None

    # Remove common suffixes like Jr., III, Ph.D., etc.
    name = re.sub(
        r",?\s*(Jr\.?|Sr\.?|III?|IV|Ph\.?D\.?|M\.?D\.?|et al\.?)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Handle "Lastname, Firstname" format
    if "," in name:
        parts = name.split(",")
        surname = parts[0].strip()
    else:
        # Handle "Firstname Lastname" or "Firstname Middle Lastname"
        parts = name.split()
        if len(parts) == 0:
            return None
        # Last part is likely the surname (unless it's an initial like "F.")
        surname = parts[-1].strip()
        # If last part is too short (likely initial), try the one before
        if len(surname) <= 2 and len(parts) > 1:
            surname = parts[-2].strip()

    # Remove any remaining punctuation
    surname = re.sub(r"[^\w\s-]", "", surname)

    # Normalize unicode (convert accented chars to ascii)
    surname = (
        unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode("ascii")
    )

    # Capitalize properly
    surname = surname.strip().title()

    return surname if len(surname) > 1 else None


def extract_year(date_string: str | None) -> str | None:
    """Extract year from various date formats.

    Handles:
    - ISO format: "2024-01-15T10:30:00Z"
    - Date only: "2024-01-15"
    - Year only: "2024"
    - DOI-style dates

    Args:
        date_string: Date string in various formats

    Returns:
        Four-digit year string or None

    """
    if not date_string:
        return None

    # Try to find a 4-digit year
    match = re.search(r"(19|20)\d{2}", str(date_string))
    if match:
        return match.group(0)
    return None


def generate_dataset_id(
    source: str,
    authors: list[str] | None = None,
    date: str | None = None,
    fallback_id: str | None = None,
    index: int | None = None,
) -> str:
    """Generate a dataset ID in SurnameYEAR format.

    Examples:
    - ["John Smith"], "2024-01-15" -> "Smith2024"
    - ["Alice Johnson", "Bob Brown"], "2023" -> "Johnson2023"
    - No authors, "2024" with fallback="12345" -> "figshare_12345"

    Args:
        source: Source name (figshare, zenodo, osf, etc.)
        authors: List of author names
        date: Date string (various formats supported)
        fallback_id: Fallback ID to use if SurnameYEAR cannot be generated
        index: Optional index to append for disambiguation (e.g., "Smith2024_2")

    Returns:
        Dataset ID in SurnameYEAR format or source_fallback format

    """
    surname = None
    year = None

    # Try to get surname from first author
    if authors and len(authors) > 0:
        for author in authors:
            surname = extract_surname(author)
            if surname:
                break

    # Try to get year
    year = extract_year(date)

    # Generate ID
    if surname and year:
        dataset_id = f"{surname}{year}"
    elif surname:
        dataset_id = surname
    elif year and fallback_id:
        dataset_id = f"{source}_{fallback_id}"
    elif fallback_id:
        dataset_id = f"{source}_{fallback_id}"
    else:
        dataset_id = f"{source}_unknown"

    # Add index for disambiguation if needed
    if index is not None and index > 0:
        dataset_id = f"{dataset_id}_{index + 1}"

    return dataset_id


def deduplicate_dataset_ids(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add suffixes to duplicate dataset_id values.

    When multiple datasets have the same dataset_id (e.g., "Smith2024"),
    they are renamed to "Smith2024", "Smith2024_2", "Smith2024_3", etc.

    Args:
        datasets: List of dataset dictionaries with dataset_id field

    Returns:
        List with deduplicated dataset_id values

    """
    # Count occurrences of each base ID
    id_counts: dict[str, int] = {}
    id_indices: dict[str, int] = {}

    for dataset in datasets:
        base_id = dataset.get("dataset_id", "unknown")
        id_counts[base_id] = id_counts.get(base_id, 0) + 1

    # Second pass: assign unique IDs
    result = []
    for dataset in datasets:
        base_id = dataset.get("dataset_id", "unknown")

        if id_counts[base_id] > 1:
            # Multiple datasets with same ID - add suffix
            idx = id_indices.get(base_id, 0)
            id_indices[base_id] = idx + 1

            if idx == 0:
                # First occurrence keeps original ID
                new_id = base_id
            else:
                # Subsequent occurrences get _2, _3, etc.
                new_id = f"{base_id}_{idx + 1}"

            dataset = dataset.copy()
            dataset["dataset_id"] = new_id

        result.append(dataset)

    return result


def normalize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Normalize a dataset document for deterministic serialization.

    - Sorts all lists (except ages which should preserve order)
    - Removes None values
    - Ensures consistent key ordering through JSON round-trip

    Args:
        dataset: Dataset dictionary to normalize

    Returns:
        Normalized dataset dictionary

    """

    def _normalize(obj: Any) -> Any:
        """Recursively normalize a value."""
        if isinstance(obj, dict):
            # Remove None values, recursively normalize nested objects
            normalized = {k: _normalize(v) for k, v in obj.items() if v is not None}
            return normalized
        elif isinstance(obj, list):
            # Recursively normalize items, but keep ages unsorted (order matters)
            return [_normalize(item) for item in obj]
        return obj

    return _normalize(dataset)


def save_datasets_deterministically(
    datasets: list[dict[str, Any]],
    output_path: Path,
    deduplicate_ids: bool = True,
) -> None:
    """Save datasets to JSON with deterministic ordering.

    - Deduplicates dataset_id values (adds _2, _3 suffixes for duplicates)
    - Normalizes datasets (removes None values)
    - Sorts list by dataset_id for consistent ordering
    - Writes with sorted keys
    - Compact output without extra whitespace

    Args:
        datasets: List of dataset dictionaries
        output_path: Path to write JSON file
        deduplicate_ids: Whether to add suffixes to duplicate IDs (default: True)

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate dataset IDs if requested
    if deduplicate_ids:
        datasets = deduplicate_dataset_ids(datasets)

    # Normalize each dataset
    normalized = [normalize_dataset(ds) for ds in datasets]

    # Sort by dataset_id for consistent ordering across runs
    normalized.sort(key=lambda d: d.get("dataset_id", ""))

    # Write with sorted keys for deterministic output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, sort_keys=True, ensure_ascii=False)
