"""Parse metadata from EEGLAB .set files.

This module provides a fallback mechanism to extract EEG metadata
(sampling_frequency, nchans, ch_names) from EEGLAB .set header files
when BIDS sidecar files (JSON/TSV) are unavailable or when the
companion .fdt data file is missing.

EEGLAB .set files are MATLAB .mat files that can store data in two ways:
1. Embedded: Data stored directly in the .set file
2. External: Data stored in a companion .fdt file, .set contains metadata only
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_set_metadata(set_path: Path | str) -> dict[str, Any] | None:
    """Parse metadata from EEGLAB .set file.

    Extracts sampling frequency, number of channels, and channel names
    from a .set file without loading the actual data. This works even
    when the companion .fdt file is missing.

    Parameters
    ----------
    set_path : Path | str
        Path to the .set file.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with keys:
        - sampling_frequency: float (in Hz)
        - nchans: int
        - ch_names: list[str] (if available)
        - n_samples: int (if available)
        - duration: float in seconds (if available)
        - has_fdt: bool - whether companion .fdt file exists
        Returns None if file cannot be parsed.

    Notes
    -----
    Uses scipy.io.loadmat for MATLAB v5 format files.
    For newer HDF5-based .set files, uses h5py.

    """
    set_path = Path(set_path)

    if not set_path.exists():
        return None

    result: dict[str, Any] = {}

    # Check for companion .fdt file
    fdt_path = set_path.with_suffix(".fdt")
    result["has_fdt"] = fdt_path.exists()

    # Try scipy.io first (for MATLAB v5 format)
    try:
        import scipy.io

        mat = scipy.io.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)

        if "EEG" in mat:
            eeg = mat["EEG"]

            # Extract sampling rate
            if hasattr(eeg, "srate"):
                srate = eeg.srate
                if hasattr(srate, "item"):
                    srate = srate.item()
                result["sampling_frequency"] = float(srate)

            # Extract number of channels
            if hasattr(eeg, "nbchan"):
                nbchan = eeg.nbchan
                if hasattr(nbchan, "item"):
                    nbchan = nbchan.item()
                result["nchans"] = int(nbchan)

            # Extract number of samples/points
            if hasattr(eeg, "pnts"):
                pnts = eeg.pnts
                if hasattr(pnts, "item"):
                    pnts = pnts.item()
                result["n_samples"] = int(pnts)

            # Calculate duration if we have both
            if "sampling_frequency" in result and "n_samples" in result:
                if result["sampling_frequency"] > 0:
                    result["duration"] = (
                        result["n_samples"] / result["sampling_frequency"]
                    )

            # Extract channel names from chanlocs structure
            if hasattr(eeg, "chanlocs") and eeg.chanlocs is not None:
                try:
                    chanlocs = eeg.chanlocs
                    ch_names = []

                    # chanlocs can be a single struct or array of structs
                    if hasattr(chanlocs, "__iter__") and not isinstance(chanlocs, str):
                        for ch in chanlocs:
                            if hasattr(ch, "labels"):
                                label = ch.labels
                                if hasattr(label, "item"):
                                    label = label.item()
                                ch_names.append(str(label))
                    elif hasattr(chanlocs, "labels"):
                        # Single channel case
                        label = chanlocs.labels
                        if hasattr(label, "item"):
                            label = label.item()
                        ch_names.append(str(label))

                    if ch_names:
                        result["ch_names"] = ch_names
                        # Update nchans if not set
                        if "nchans" not in result:
                            result["nchans"] = len(ch_names)
                except Exception as e:
                    logger.debug("Could not extract channel names: %s", e)

            # Check for external data file reference
            if hasattr(eeg, "datfile") and eeg.datfile:
                datfile = eeg.datfile
                if hasattr(datfile, "item"):
                    datfile = datfile.item()
                result["external_datafile"] = str(datfile)

            if result:
                logger.debug(
                    "Extracted from .set: sfreq=%s, nchans=%s, has_fdt=%s",
                    result.get("sampling_frequency"),
                    result.get("nchans"),
                    result.get("has_fdt"),
                )
                return result

    except ImportError:
        logger.debug("scipy not available for .set parsing")
    except Exception as e:
        logger.debug("scipy.io.loadmat failed: %s", e)

    # Try h5py for HDF5-based .set files (newer EEGLAB format)
    try:
        import h5py

        with h5py.File(str(set_path), "r") as f:
            # Navigate to EEG structure
            if "EEG" in f:
                eeg = f["EEG"]

                # Extract sampling rate
                if "srate" in eeg:
                    srate = eeg["srate"][0, 0]
                    result["sampling_frequency"] = float(srate)

                # Extract number of channels
                if "nbchan" in eeg:
                    nbchan = eeg["nbchan"][0, 0]
                    result["nchans"] = int(nbchan)

                # Extract number of points
                if "pnts" in eeg:
                    pnts = eeg["pnts"][0, 0]
                    result["n_samples"] = int(pnts)

                # Calculate duration
                if "sampling_frequency" in result and "n_samples" in result:
                    if result["sampling_frequency"] > 0:
                        result["duration"] = (
                            result["n_samples"] / result["sampling_frequency"]
                        )

                if result:
                    return result

    except ImportError:
        logger.debug("h5py not available for .set parsing")
    except Exception as e:
        logger.debug("h5py parsing failed: %s", e)

    return None if not result else result


def diagnose_set_issues(set_path: Path | str) -> dict[str, Any]:
    """Diagnose issues with an EEGLAB .set file and its companion .fdt.

    Parameters
    ----------
    set_path : Path | str
        Path to the .set file.

    Returns
    -------
    dict[str, Any]
        Diagnosis with keys:
        - status: "ok" | "missing_fdt" | "error"
        - issues: list of issue descriptions
        - metadata: extracted metadata (if available)
        - can_extract_metadata: bool

    """
    set_path = Path(set_path)
    result = {
        "status": "ok",
        "issues": [],
        "metadata": None,
        "can_extract_metadata": False,
    }

    if not set_path.exists():
        result["status"] = "error"
        result["issues"].append(f".set file not found: {set_path}")
        return result

    # Try to extract metadata
    metadata = parse_set_metadata(set_path)
    if metadata:
        result["metadata"] = metadata
        result["can_extract_metadata"] = True

        # Check for missing .fdt
        if not metadata.get("has_fdt", True):
            if metadata.get("external_datafile"):
                result["status"] = "missing_fdt"
                result["issues"].append(
                    f"Missing .fdt file: {metadata['external_datafile']}"
                )
            else:
                # Data might be embedded, which is fine
                pass
    else:
        result["status"] = "error"
        result["issues"].append("Could not parse .set file")

    return result
