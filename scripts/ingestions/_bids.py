"""Shared helpers for BIDS structure validation in ingestion scripts."""

from __future__ import annotations

import re
from typing import Any, Pattern

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


def validate_bids_structure_from_names(
    file_names: list[str],
    *,
    required_files: list[str] = BIDS_REQUIRED_FILES,
    optional_files: list[str] = BIDS_OPTIONAL_FILES,
    subject_pattern: Pattern[str] | None = BIDS_SUBJECT_PATTERN,
    dataset_zip_pattern: Pattern[str] | None = None,
    dataset_zip_matcher: str = "match",
    subject_min_count: int = 2,
    subject_requires_bids_files: bool = False,
    include_subject_files: bool = False,
    subject_files_limit: int = 10,
    bids_zip_files_limit: int = 5,
) -> dict[str, Any]:
    """Validate BIDS structure based on a list of file or folder names.

    The goal is to provide a consistent, minimal BIDS heuristic across ingestion
    sources (Figshare/OSF/Zenodo/etc.) while keeping source-specific knobs.

    Returns a dict compatible with existing ingestion outputs.
    """
    matches = collect_bids_matches(
        file_names,
        required_files=required_files,
        optional_files=optional_files,
        subject_pattern=subject_pattern,
        dataset_zip_pattern=dataset_zip_pattern,
        dataset_zip_matcher=dataset_zip_matcher,
    )

    bids_files_found = matches["required_found"] + matches["optional_found"]
    subject_files = matches["subject_files"]
    subject_count = len(subject_files)
    has_subject_zips = len(matches["subject_zips"]) > 0

    has_required = len(matches["required_found"]) > 0
    has_subjects = subject_count >= subject_min_count
    subject_ok = has_subjects and (
        len(bids_files_found) > 0 if subject_requires_bids_files else True
    )

    bids_zips = matches["bids_zip_files"]
    has_bids_zip = len(bids_zips) > 0

    is_bids = has_required or subject_ok or has_bids_zip

    result: dict[str, Any] = {
        "is_bids": is_bids,
        "bids_files_found": bids_files_found,
        "subject_count": subject_count,
        "has_subject_zips": has_subject_zips,
    }

    if include_subject_files:
        result["subject_files"] = subject_files[:subject_files_limit]

    if dataset_zip_pattern is not None:
        result["has_bids_zip"] = has_bids_zip
        result["bids_zip_files"] = bids_zips[:bids_zip_files_limit]

    return result


def validate_bids_structure_from_files(
    files: list[dict[str, Any]],
    *,
    name_key: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate BIDS structure from a list of file dicts."""
    names = [f.get(name_key, "") for f in files] if files else []
    return validate_bids_structure_from_names(names, **kwargs)
