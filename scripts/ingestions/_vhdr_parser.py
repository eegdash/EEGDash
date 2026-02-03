"""Parse metadata from BrainVision VHDR header files.

This module provides a fallback mechanism to extract EEG metadata
(sampling_frequency, nchans, ch_names) from BrainVision VHDR header files
when BIDS sidecar files (JSON/TSV) are unavailable.

Includes robust handling for common issues:
- BIDS filename mismatches (original vs renamed files)
- Typos in file references
- Missing companion files (.vmrk, .eeg)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from _parser_utils import is_broken_symlink, read_with_encoding_fallback

# Add eegdash to path if not already importable
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from eegdash.dataset.io import _find_best_matching_file

logger = logging.getLogger(__name__)


def parse_vhdr_metadata(vhdr_path: Path | str) -> dict[str, Any] | None:
    r"""Parse metadata from BrainVision VHDR header file.

    Extracts sampling frequency, number of channels, and channel names
    from a VHDR file. This is useful as a fallback when BIDS sidecar
    files are missing.

    Parameters
    ----------
    vhdr_path : Path | str
        Path to the VHDR header file.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with keys:
        - sampling_frequency: float (in Hz)
        - nchans: int
        - ch_names: list[str]
        Returns None if file cannot be parsed.

    Notes
    -----
    VHDR files are INI-like text files with sections:
    - [Common Infos]: Contains NumberOfChannels and SamplingInterval
    - [Channel Infos]: Contains channel definitions (Ch1=name,,resolution,unit)

    SamplingInterval is in microseconds, so Hz = 1,000,000 / interval.

    Channel names may contain escaped commas (\\1 -> ,).

    For broken git-annex symlinks in OpenNeuro datasets, this function
    will automatically fetch the file content from S3.

    """
    vhdr_path = Path(vhdr_path)

    # Check if file exists (either directly or as a broken symlink that we can fetch from S3)
    if not vhdr_path.exists() and not is_broken_symlink(vhdr_path):
        return None

    # Read file content with encoding fallback (includes S3 fallback for git-annex)
    content = read_with_encoding_fallback(vhdr_path)
    if content is None:
        return None

    # Parse the VHDR content
    result: dict[str, Any] = {}

    # Extract NumberOfChannels from [Common Infos] section
    nchans_match = re.search(
        r"^\s*NumberOfChannels\s*=\s*(\d+)",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if nchans_match:
        result["nchans"] = int(nchans_match.group(1))

    # Extract SamplingInterval from [Common Infos] section
    # SamplingInterval is in microseconds
    sampling_interval_match = re.search(
        r"^\s*SamplingInterval\s*=\s*([\d.]+)",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if sampling_interval_match:
        interval_us = float(sampling_interval_match.group(1))
        if interval_us > 0:
            # Convert microseconds to Hz: Hz = 1,000,000 / interval_us
            result["sampling_frequency"] = 1_000_000.0 / interval_us

    # Extract channel names from [Channel Infos] section
    # Format: Ch1=Fp1,,0.0488281,µV or Ch1=Fp1,ref,0.1,µV
    ch_names = []

    # Find [Channel Infos] section
    channel_section_match = re.search(
        r"\[Channel\s*Infos?\](.+?)(?:\[|$)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if channel_section_match:
        channel_section = channel_section_match.group(1)

        # Match channel definitions: Ch1=name,ref,resolution,unit
        # The name is everything before the first comma (or empty/escaped)
        channel_pattern = re.compile(
            r"^\s*Ch(\d+)\s*=\s*([^,]*)",
            re.MULTILINE | re.IGNORECASE,
        )

        channels = []
        for match in channel_pattern.finditer(channel_section):
            ch_num = int(match.group(1))
            ch_name = match.group(2).strip()

            # Handle escaped commas: \1 means comma in BrainVision format
            ch_name = ch_name.replace("\\1", ",")

            # If channel name is empty, use generic name
            if not ch_name:
                ch_name = f"Ch{ch_num}"

            channels.append((ch_num, ch_name))

        # Sort by channel number and extract names
        channels.sort(key=lambda x: x[0])
        ch_names = [name for _, name in channels]

    if ch_names:
        result["ch_names"] = ch_names
        # If we found channel names but not nchans, derive it
        if "nchans" not in result:
            result["nchans"] = len(ch_names)

    # Return None if we couldn't extract any useful metadata
    if not result:
        return None

    return result


def extract_vhdr_references(vhdr_path: Path | str) -> dict[str, str | None]:
    """Extract DataFile and MarkerFile references from VHDR header.

    Parameters
    ----------
    vhdr_path : Path | str
        Path to the VHDR header file.

    Returns
    -------
    dict[str, str | None]
        Dictionary with keys:
        - datafile: Referenced .eeg data file name (or None)
        - markerfile: Referenced .vmrk marker file name (or None)
        - datafile_exists: bool - whether the referenced data file exists
        - markerfile_exists: bool - whether the referenced marker file exists

    """
    vhdr_path = Path(vhdr_path)
    result = {
        "datafile": None,
        "markerfile": None,
        "datafile_exists": False,
        "markerfile_exists": False,
    }

    # Read file content
    content = read_with_encoding_fallback(vhdr_path)
    if content is None:
        return result

    # Extract DataFile reference
    datafile_match = re.search(
        r"^\s*DataFile\s*=\s*(.+?)\s*$",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if datafile_match:
        result["datafile"] = datafile_match.group(1).strip()
        data_path = vhdr_path.parent / result["datafile"]
        result["datafile_exists"] = data_path.exists()

    # Extract MarkerFile reference
    markerfile_match = re.search(
        r"^\s*MarkerFile\s*=\s*(.+?)\s*$",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if markerfile_match:
        result["markerfile"] = markerfile_match.group(1).strip()
        marker_path = vhdr_path.parent / result["markerfile"]
        result["markerfile_exists"] = marker_path.exists()

    return result


def diagnose_vhdr_issues(vhdr_path: Path | str) -> dict[str, Any]:
    """Diagnose issues with a BrainVision VHDR file and its companions.

    Checks for common problems like:
    - Missing .eeg or .vmrk files
    - BIDS filename mismatches
    - Typos in file references
    - Suggests fixes when possible

    Parameters
    ----------
    vhdr_path : Path | str
        Path to the VHDR header file.

    Returns
    -------
    dict[str, Any]
        Diagnosis with keys:
        - status: "ok" | "fixable" | "missing_files" | "error"
        - issues: list of issue descriptions
        - suggested_fixes: dict mapping referenced names to actual files
        - metadata: extracted metadata (if available)
        - can_extract_metadata: bool - whether metadata extraction is possible

    """
    vhdr_path = Path(vhdr_path)
    result = {
        "status": "ok",
        "issues": [],
        "suggested_fixes": {},
        "metadata": None,
        "can_extract_metadata": False,
    }

    # Check VHDR file exists
    if not vhdr_path.exists() and not is_broken_symlink(vhdr_path):
        result["status"] = "error"
        result["issues"].append(f"VHDR file not found: {vhdr_path}")
        return result

    # Try to extract metadata (this works even without companion files)
    metadata = parse_vhdr_metadata(vhdr_path)
    if metadata:
        result["metadata"] = metadata
        result["can_extract_metadata"] = True

    # Check file references
    refs = extract_vhdr_references(vhdr_path)
    directory = vhdr_path.parent

    issues_found = False

    # Check data file
    if refs["datafile"]:
        if not refs["datafile_exists"]:
            issues_found = True
            result["issues"].append(f"DataFile reference not found: {refs['datafile']}")

            # Try to find matching file using shared fuzzy matching from eegdash.dataset.io
            match_name = _find_best_matching_file(directory, refs["datafile"], ".eeg")
            if match_name:
                result["suggested_fixes"][refs["datafile"]] = match_name
                result["issues"][-1] += f" (suggest: {match_name})"
            else:
                # Check if there's ANY .eeg file
                eeg_files = list(directory.glob("*.eeg"))
                if eeg_files:
                    result["issues"][-1] += f" ({len(eeg_files)} .eeg file(s) exist)"

    # Check marker file
    if refs["markerfile"]:
        if not refs["markerfile_exists"]:
            issues_found = True
            result["issues"].append(
                f"MarkerFile reference not found: {refs['markerfile']}"
            )

            # Try to find matching file using shared fuzzy matching from eegdash.dataset.io
            match_name = _find_best_matching_file(
                directory, refs["markerfile"], ".vmrk"
            )
            if match_name:
                result["suggested_fixes"][refs["markerfile"]] = match_name
                result["issues"][-1] += f" (suggest: {match_name})"
            else:
                # Check if there's ANY .vmrk file
                vmrk_files = list(directory.glob("*.vmrk"))
                if vmrk_files:
                    result["issues"][-1] += f" ({len(vmrk_files)} .vmrk file(s) exist)"

    # Determine overall status
    if issues_found:
        if result["suggested_fixes"]:
            result["status"] = "fixable"
        else:
            result["status"] = "missing_files"
    else:
        result["status"] = "ok"

    return result


def parse_vhdr_metadata_robust(vhdr_path: Path | str) -> dict[str, Any] | None:
    """Parse VHDR metadata with enhanced error reporting.

    Like parse_vhdr_metadata, but also includes diagnostic information
    about any issues found with the file.

    Parameters
    ----------
    vhdr_path : Path | str
        Path to the VHDR header file.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with standard metadata keys plus:
        - _diagnosis: dict with issue diagnosis
        - _status: "ok" | "fixable" | "missing_files" | "error"

    """
    vhdr_path = Path(vhdr_path)

    # Get basic metadata
    metadata = parse_vhdr_metadata(vhdr_path)
    if metadata is None:
        metadata = {}

    # Add diagnosis
    diagnosis = diagnose_vhdr_issues(vhdr_path)
    metadata["_diagnosis"] = diagnosis
    metadata["_status"] = diagnosis["status"]

    # If we have no metadata and no diagnosis, return None
    if not metadata.get("nchans") and not metadata.get("sampling_frequency"):
        if diagnosis["status"] == "error":
            return None

    return metadata if metadata else None


def find_datasets_needing_redigestion(output_dir: Path) -> list[str]:
    """Find datasets with VHDR files that are missing metadata.

    Scans the digestion output directory for datasets that have records
    with VHDR files (.vhdr extension) but missing sampling_frequency
    or nchans values.

    Parameters
    ----------
    output_dir : Path
        Directory containing digested dataset JSON files.

    Returns
    -------
    list[str]
        List of dataset IDs that may benefit from re-digestion with
        VHDR metadata extraction.

    """
    datasets_needing_redigestion = []

    for dataset_dir in output_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        records_file = dataset_dir / f"{dataset_dir.name}_records.json"
        if not records_file.exists():
            continue

        try:
            with open(records_file) as f:
                records_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        records = records_data.get("records", [])
        for record in records:
            bids_relpath = record.get("bids_relpath", "")

            # Check if this is a VHDR file
            if not bids_relpath.lower().endswith(".vhdr"):
                continue

            # Check if metadata is missing
            has_sfreq = record.get("sampling_frequency") is not None
            has_nchans = record.get("nchans") is not None

            if not has_sfreq or not has_nchans:
                datasets_needing_redigestion.append(dataset_dir.name)
                break  # Only need to flag dataset once

    return sorted(set(datasets_needing_redigestion))
