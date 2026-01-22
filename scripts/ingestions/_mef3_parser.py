"""Parse metadata from MEF3 (Multiscale Electrophysiology Format) files.

MEF3 is a directory-based format for electrophysiology data.
Structure: dataset.mefd/channel.timd/segment.segd/segment.{tmet,tidx,tdat}

Reference: https://github.com/msel-source/meflib
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def parse_mef3_metadata(mefd_path: Path | str) -> dict[str, Any] | None:
    """Parse metadata from MEF3 directory.

    Extracts sampling frequency, number of channels, and channel names
    from a MEF3 (.mefd) directory.

    Parameters
    ----------
    mefd_path : Path | str
        Path to the .mefd directory.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with keys:
        - sampling_frequency: float (in Hz)
        - nchans: int
        - ch_names: list[str]
        Returns None if directory cannot be parsed.

    Notes
    -----
    MEF3 structure:
    - dataset.mefd/           (main directory)
      - channel.timd/         (time series metadata directory, one per channel)
        - segment.segd/       (segment directory)
          - segment.tmet      (metadata file)
          - segment.tidx      (time index)
          - segment.tdat      (data)

    The .tmet files contain binary metadata including sampling frequency.
    Channel names are derived from .timd directory names.

    """
    mefd_path = Path(mefd_path)

    # Check if directory exists
    if not mefd_path.exists():
        return None

    # Must be a directory ending in .mefd
    if not mefd_path.is_dir() or not mefd_path.suffix.lower() == ".mefd":
        return None

    result: dict[str, Any] = {}
    ch_names = []
    sampling_frequencies = []

    # Find all .timd directories (each represents a channel)
    try:
        timd_dirs = sorted(
            [
                d
                for d in mefd_path.iterdir()
                if d.is_dir() and d.suffix.lower() == ".timd"
            ]
        )
    except (OSError, RuntimeError):
        return None

    if not timd_dirs:
        return None

    for timd_dir in timd_dirs:
        # Channel name is the directory name without .timd extension
        ch_name = timd_dir.stem
        ch_names.append(ch_name)

        # Try to get sampling frequency from first segment's .tmet file
        if not sampling_frequencies:
            sfreq = _extract_sfreq_from_timd(timd_dir)
            if sfreq is not None:
                sampling_frequencies.append(sfreq)

    if ch_names:
        result["ch_names"] = ch_names
        result["nchans"] = len(ch_names)

    if sampling_frequencies:
        result["sampling_frequency"] = sampling_frequencies[0]

    # Return None if we couldn't extract any useful metadata
    if not result:
        return None

    return result


def _extract_sfreq_from_timd(timd_dir: Path) -> float | None:
    """Extract sampling frequency from a .timd directory.

    Parameters
    ----------
    timd_dir : Path
        Path to the .timd directory.

    Returns
    -------
    float | None
        Sampling frequency in Hz, or None if not found.

    """
    # Find first .segd directory
    try:
        segd_dirs = [
            d for d in timd_dir.iterdir() if d.is_dir() and d.suffix.lower() == ".segd"
        ]
    except (OSError, RuntimeError):
        return None

    if not segd_dirs:
        return None

    segd_dir = segd_dirs[0]

    # Find .tmet file in segment directory
    tmet_files = list(segd_dir.glob("*.tmet"))
    if not tmet_files:
        return None

    tmet_file = tmet_files[0]

    # Check if file exists and is readable
    if not tmet_file.exists():
        return None

    # Handle broken symlinks
    try:
        resolved = tmet_file.resolve()
        if not resolved.exists():
            return None
    except (OSError, RuntimeError):
        return None

    # Parse MEF3 metadata file
    # MEF3 .tmet files have a specific binary format
    # The sampling frequency is stored as a double at a specific offset
    try:
        return _parse_tmet_sampling_frequency(tmet_file)
    except Exception:
        return None


def _parse_tmet_sampling_frequency(tmet_path: Path) -> float | None:
    """Parse sampling frequency from MEF3 .tmet file.

    MEF3 .tmet files contain time series metadata in a binary format.
    The sampling frequency is stored as a double-precision float.

    Parameters
    ----------
    tmet_path : Path
        Path to the .tmet file.

    Returns
    -------
    float | None
        Sampling frequency in Hz, or None if not found.

    """
    try:
        with open(tmet_path, "rb") as f:
            # MEF3 .tmet file structure (simplified):
            # - Universal header (1024 bytes)
            # - Time series metadata section header
            # - Sampling frequency is at offset 272 in the metadata section
            # (after the 1024-byte universal header)

            # Read the file
            data = f.read()

            if len(data) < 1300:  # Minimum size for valid .tmet
                return None

            # Try to find sampling frequency
            # In MEF3 format, sampling_frequency is a double at specific offset
            # Offset varies by MEF version, try common locations

            # MEF 3.0 layout: sampling_frequency at offset 1024 + 248 = 1272
            for offset in [1272, 1280, 1288, 272, 280]:
                if offset + 8 <= len(data):
                    try:
                        sfreq = struct.unpack("<d", data[offset : offset + 8])[0]
                        # Sanity check - sampling frequency should be reasonable
                        if 0.1 < sfreq < 1000000:
                            return sfreq
                    except struct.error:
                        continue

            return None

    except Exception:
        return None
