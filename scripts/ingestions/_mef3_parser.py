"""Parse metadata from MEF3 (Multiscale Electrophysiology Format) files.

MEF3 is a directory-based format for electrophysiology data.
Structure: dataset.mefd/channel.timd/segment.segd/segment.{tmet,tidx,tdat}

Reference: https://github.com/msel-source/meflib
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any

from _parser_utils import validate_file_path

logger = logging.getLogger(__name__)


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

    n_times_value: int | None = None
    for timd_dir in timd_dirs:
        # Channel name is the directory name without .timd extension
        ch_name = timd_dir.stem
        ch_names.append(ch_name)

        # Try to get sampling frequency from first segment's .tmet file
        if not sampling_frequencies:
            sfreq = _extract_sfreq_from_timd(timd_dir)
            if sfreq is not None:
                sampling_frequencies.append(sfreq)
                # n_times comes from the same channel's .tmet header.
                n_times_value = _extract_nsamples_from_timd(timd_dir, sfreq)

    if ch_names:
        result["ch_names"] = ch_names
        result["nchans"] = len(ch_names)

    if sampling_frequencies:
        result["sampling_frequency"] = sampling_frequencies[0]

    if n_times_value is not None:
        result["n_times"] = n_times_value

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

    # Validate file path (handles broken symlinks from git-annex)
    if not validate_file_path(tmet_file):
        return None

    # Parse MEF3 metadata file
    # MEF3 .tmet files have a specific binary format
    # The sampling frequency is stored as a double at a specific offset
    try:
        return _parse_tmet_sampling_frequency(tmet_file)
    except (OSError, struct.error, ValueError) as e:
        # OSError covers file-not-found / permission. struct.error fires
        # when the .tmet is truncated below the header offset. ValueError
        # protects against pathological data passing the size check but
        # holding NaN or infinity. All recoverable.
        logger.debug("Could not parse .tmet at %s: %s", tmet_file, e)
        return None


# MEF 3.0 time-series-metadata-section-2 field layout: number_of_samples (si8)
# sits 200 bytes after the sampling_frequency (sf8) field, and number_of_blocks
# (si8) another 8 bytes on. We locate sampling_frequency by value, then read these.
_MEF3_NSAMPLES_OFFSET_FROM_SFREQ = 200
_MEF3_NBLOCKS_OFFSET_FROM_SFREQ = 208
_MEF3_NSAMPLES_SANITY_MAX = 10**12


def _extract_nsamples_from_timd(timd_dir: Path, sfreq: float) -> int | None:
    """Extract number_of_samples (n_times) from a .timd directory's first .tmet."""
    try:
        segd_dirs = [
            d for d in timd_dir.iterdir() if d.is_dir() and d.suffix.lower() == ".segd"
        ]
    except (OSError, RuntimeError):
        return None
    if not segd_dirs:
        return None
    tmet_files = list(segd_dirs[0].glob("*.tmet"))
    if not tmet_files:
        return None
    if not validate_file_path(tmet_files[0]):
        return None
    try:
        return _parse_tmet_number_of_samples(tmet_files[0], sfreq)
    except (OSError, struct.error, ValueError) as e:
        logger.debug("Could not parse number_of_samples at %s: %s", tmet_files[0], e)
        return None


def _parse_tmet_number_of_samples(tmet_path: Path, sfreq: float) -> int | None:
    """Read number_of_samples (n_times) from a MEF3 .tmet, validated against number_of_blocks.

    ``number_of_samples`` is at a spec-fixed +200 bytes from the
    sampling_frequency double. We find the sfreq offset by matching the known
    value (canonical 8720 first, then the historical offsets, then a full scan),
    then accept the si8 only when it is internally consistent with
    ``number_of_blocks`` — a coincidental sfreq match almost never yields a
    plausible (samples, blocks) pair, so a WRONG value never ships. Never raises.
    """
    if sfreq is None or sfreq <= 0:
        return None
    try:
        with open(tmet_path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 1300:
        return None

    nblk_end = _MEF3_NBLOCKS_OFFSET_FROM_SFREQ + 8
    candidate_offsets = [8720, 1272, 1280, 1288, 272, 280]
    candidate_offsets += list(range(1024, len(data) - 8, 8))
    for off in candidate_offsets:
        if off < 0 or off + nblk_end > len(data):
            continue
        try:
            if struct.unpack("<d", data[off : off + 8])[0] != sfreq:
                continue
            nos = struct.unpack(
                "<q",
                data[
                    off + _MEF3_NSAMPLES_OFFSET_FROM_SFREQ : off
                    + _MEF3_NSAMPLES_OFFSET_FROM_SFREQ
                    + 8
                ],
            )[0]
            nblk = struct.unpack(
                "<q",
                data[
                    off + _MEF3_NBLOCKS_OFFSET_FROM_SFREQ : off
                    + _MEF3_NBLOCKS_OFFSET_FROM_SFREQ
                    + 8
                ],
            )[0]
        except struct.error:
            continue
        if not (0 < nos < _MEF3_NSAMPLES_SANITY_MAX):
            continue
        # Consistency: blocks present and each spans a sane sample count
        # (1 .. 60 s of data). Guards against a coincidental sfreq match.
        if nblk <= 0:
            continue
        samples_per_block = nos / nblk
        if not (1.0 <= samples_per_block <= sfreq * 60.0):
            continue
        return int(nos)
    return None


def _parse_tmet_sampling_frequency(tmet_path: Path) -> float | None:
    """Parse sampling frequency from MEF3 .tmet file.

    MEF3 .tmet files contain time series metadata in a binary format
    documented at https://github.com/msel-source/meflib_3p0. The
    sampling frequency is stored as a little-endian double in the
    time-series metadata section, AFTER the 1024-byte universal header.

    The exact offset depends on the structure version + padding
    choices the encoder made. Production MEF 3.0 files written by the
    Mayo Foundation tooling place it at offset 8720 (verified against
    OpenNeuro ds003708's CC0 fixture); older / synthetic files have
    been observed at 272, 280, 1272, 1280, 1288.

    Rather than guess at a fixed offset, this implementation scans
    every 8-byte-aligned position past the universal header and
    accepts the first value within the sanity range (0.1 - 1_000_000 Hz)
    that's also an integer multiple of 0.5 Hz (a near-universal
    convention for clinical recording rates: 256/512/1024/2048/etc).

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
            data = f.read()

            if len(data) < 1300:  # Minimum size for valid .tmet
                return None

            # Fast-path: try the historically-observed offsets first
            # so non-MEF-3.0 / synthetic test files keep working.
            for offset in (1272, 1280, 1288, 272, 280):
                if offset + 8 > len(data):
                    continue
                try:
                    sfreq = struct.unpack("<d", data[offset : offset + 8])[0]
                except struct.error:
                    continue
                if 0.1 < sfreq < 1_000_000:
                    return sfreq

            # Slow-path: scan every 8-byte-aligned position past the
            # universal header. Pick the first half-Hz multiple in the
            # sanity range — that filters out random doubles that happen
            # to fall in 100-1000 (e.g., MD5 hash bytes interpreted as
            # doubles).
            for offset in range(1024, len(data) - 8, 8):
                try:
                    sfreq = struct.unpack("<d", data[offset : offset + 8])[0]
                except struct.error:
                    continue
                if not (0.1 < sfreq < 1_000_000):
                    continue
                # Half-Hz multiple filter — production clinical / research
                # recording rates are all integer or half-integer Hz.
                if (sfreq * 2.0) != int(sfreq * 2.0):
                    continue
                return sfreq

            return None

    except (OSError, struct.error) as e:
        # OSError = file disappeared or unreadable; struct.error = the
        # binary layout doesn't match any of the offset hypotheses above.
        logger.debug("Failed to unpack .tmet %s: %s", tmet_path, e)
        return None
