"""Shared helpers for BIDS structure validation in ingestion scripts."""

from __future__ import annotations

import re
from typing import Pattern

# Common BIDS indicator files (case-insensitive matching)
BIDS_REQUIRED_FILES = ["dataset_description.json"]
BIDS_OPTIONAL_FILES = [
    "participants.tsv",
    "participants.json",
    "readme",
    "readme.md",
    "readme.txt",
    "changes",
]

# Patterns for BIDS subject folders/zips
BIDS_SUBJECT_PATTERN = re.compile(r"^sub-[a-zA-Z0-9]+(?:\.zip)?$", re.IGNORECASE)

# Patterns for BIDS dataset zips (contains BIDS data inside)
BIDS_DATASET_ZIP_PATTERN = re.compile(
    r"(?:bids|dataset).*\.zip$|.*_bids\.zip$", re.IGNORECASE
)


def collect_bids_matches(
    file_names: list[str],
    required_files: list[str],
    optional_files: list[str],
    subject_pattern: Pattern[str] | None = BIDS_SUBJECT_PATTERN,
    dataset_zip_pattern: Pattern[str] | None = None,
    dataset_zip_matcher: str = "match",
) -> dict[str, list[str]]:
    """Collect BIDS file matches from a list of file names.

    Args:
        file_names: File names to inspect.
        required_files: Required BIDS file names (case-insensitive).
        optional_files: Optional BIDS file names (case-insensitive).
        subject_pattern: Regex for BIDS subject folders/zips.
        dataset_zip_pattern: Regex for BIDS dataset zip files.
        dataset_zip_matcher: "match" or "search" for dataset zip pattern checks.

    Returns:
        Dict with required/optional matches and subject/zip file lists.

    """
    names = [name for name in file_names if name]
    names_lower = [name.lower() for name in names]

    required_found = [f for f in required_files if f.lower() in names_lower]
    optional_found = [f for f in optional_files if f.lower() in names_lower]

    subject_files: list[str] = []
    subject_zips: list[str] = []
    if subject_pattern:
        for name in names:
            if subject_pattern.match(name):
                subject_files.append(name)
                if name.lower().endswith(".zip"):
                    subject_zips.append(name)

    bids_zip_files: list[str] = []
    if dataset_zip_pattern:
        matcher = (
            dataset_zip_pattern.match
            if dataset_zip_matcher == "match"
            else dataset_zip_pattern.search
        )
        for name in names:
            if matcher(name):
                bids_zip_files.append(name)

    return {
        "required_found": required_found,
        "optional_found": optional_found,
        "subject_files": subject_files,
        "subject_zips": subject_zips,
        "bids_zip_files": bids_zip_files,
    }
