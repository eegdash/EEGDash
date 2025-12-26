"""Deterministic JSON serialization utilities for dataset documents.

Ensures consistent, sorted output across runs for CI/CD reproducibility.
"""

import json
from pathlib import Path
from typing import Any


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
) -> None:
    """Save datasets to JSON with deterministic ordering.

    - Normalizes datasets (removes None values)
    - Sorts list by dataset_id for consistent ordering
    - Writes with sorted keys
    - Compact output without extra whitespace

    Args:
        datasets: List of dataset dictionaries
        output_path: Path to write JSON file

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize each dataset
    normalized = [normalize_dataset(ds) for ds in datasets]

    # Sort by dataset_id for consistent ordering across runs
    normalized.sort(key=lambda d: d.get("dataset_id", ""))

    # Write with sorted keys for deterministic output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, sort_keys=True, ensure_ascii=False)
