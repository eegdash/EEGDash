"""Parse metadata from BrainVision VHDR header files.

This module provides a fallback mechanism to extract EEG metadata
(sampling_frequency, nchans, ch_names) from BrainVision VHDR header files
when BIDS sidecar files (JSON/TSV) are unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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

    """
    vhdr_path = Path(vhdr_path)

    # Check if file exists and is readable
    # For git-annex broken symlinks, resolve() will point to non-existent target
    if not vhdr_path.exists():
        return None

    # Handle broken symlinks (git-annex)
    try:
        resolved = vhdr_path.resolve()
        if not resolved.exists():
            return None
    except (OSError, RuntimeError):
        return None

    # Read file content with encoding fallback
    content = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            content = vhdr_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, OSError):
            continue

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
