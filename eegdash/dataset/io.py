"""Input/Output utilities for EEG datasets.

This module contains helper functions for managing EEG data files,
specifically for fixing common issues in BIDS datasets and handling
file system operations.
"""

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..logging import logger


def _find_best_matching_file(
    directory: Path, target_name: str, extension: str, threshold: float = 0.5
) -> str | None:
    """Find the best matching file in a directory using fuzzy string matching.

    Parameters
    ----------
    directory : Path
        Directory to search in.
    target_name : str
        The filename we're looking for (may have typos or different naming).
    extension : str
        File extension to filter by (e.g., ".eeg", ".vmrk").
    threshold : float
        Minimum similarity ratio (0-1) to consider a match.

    Returns
    -------
    str | None
        The best matching filename, or None if no good match found.

    """
    candidates = list(directory.glob(f"*{extension}"))
    if not candidates:
        return None

    # If only one file with this extension, use it
    if len(candidates) == 1:
        return candidates[0].name

    # Find best match using similarity ratio
    target_stem = Path(target_name).stem.lower()
    best_match = None
    best_ratio = threshold

    for candidate in candidates:
        candidate_stem = candidate.stem.lower()
        ratio = SequenceMatcher(None, target_stem, candidate_stem).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate.name

    return best_match


class _VHDRPointerFixer:
    """Helper class to fix VHDR pointers with state tracking."""

    def __init__(self, vhdr_path: Path):
        self.vhdr_path = vhdr_path
        self.data_dir = vhdr_path.parent
        self.changes = False

    def replace(self, match) -> str:
        """Regex replacement callback."""
        key = match.group(1)  # DataFile or MarkerFile
        old_val = match.group(2).strip()

        # If the pointed file exists, do nothing
        if (self.data_dir / old_val).exists():
            return match.group(0)

        # Determine expected extension
        ext = ".vmrk" if key == "MarkerFile" else ".eeg"

        # Strategy 1: Check if BIDS filename exists (same stem as .vhdr)
        bids_name = self.vhdr_path.with_suffix(ext).name
        if (self.data_dir / bids_name).exists():
            self.changes = True
            logger.info(
                f"Auto-repairing {self.vhdr_path.name}: {key}={old_val} -> {bids_name}"
            )
            return f"{key}={bids_name}"

        # Strategy 2: Fuzzy match - find best matching file with same extension
        # This handles typos (rsub- vs sub-, sternbeg vs sternberg) and
        # BIDS renames where original filename was completely different
        fuzzy_match = _find_best_matching_file(self.data_dir, old_val, ext)
        if fuzzy_match and (self.data_dir / fuzzy_match).exists():
            self.changes = True
            logger.info(
                f"Auto-repairing {self.vhdr_path.name}: {key}={old_val} -> {fuzzy_match} (fuzzy match)"
            )
            return f"{key}={fuzzy_match}"

        return match.group(0)


def _repair_vhdr_pointers(vhdr_path: Path) -> bool:
    """Fix VHDR file pointing to internal filenames instead of BIDS filenames.

    Checks if the DataFile and MarkerFile pointers in a .vhdr file refer
    to files that exist. If not, checks if files matching the BIDS naming
    convention exist in the same directory and updates the pointers accordingly.

    Parameters
    ----------
    vhdr_path : Path
        Path to the .vhdr file to check and repair.

    Returns
    -------
    bool
        True if changes were made, False otherwise.

    """
    if not vhdr_path.exists() or vhdr_path.suffix != ".vhdr":
        return False

    try:
        content = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read VHDR {vhdr_path}: {e}")
        return False

    fixer = _VHDRPointerFixer(vhdr_path)

    # Common section for VHDR
    # Case insensitive keys, value until end of line
    new_content = re.sub(
        r"(DataFile|MarkerFile)\s*=\s*(.*)",
        fixer.replace,
        content,
        flags=re.IGNORECASE,
    )

    if fixer.changes:
        try:
            vhdr_path.write_text(new_content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed to write repaired VHDR {vhdr_path}: {e}")
            return False

    return False


# TO-DO: fix this when mne-bids is fixed
def _ensure_coordsystem_symlink(data_dir: Path) -> None:
    """Ensure coordsystem.json exists in the data directory using symlinks if needed.

    MNE-BIDS can be strict about coordsystem.json location. If it's missing
    in the EEG directory but present in the subject root, this function
    creates a symlink.

    Parameters
    ----------
    data_dir : Path
        The directory containing the EEG data files (e.g. sub-01/eeg/).

    """
    if not data_dir.exists():
        return

    try:
        # Check if we have electrodes.tsv here (implies need for coordsystem)
        electrodes_files = list(data_dir.glob("*_electrodes.tsv"))
        if not electrodes_files:
            return

        # Check if we lack coordsystem.json here
        coordsystem_files = list(data_dir.glob("*_coordsystem.json"))
        if coordsystem_files:
            return

        # Look for coordsystem in parent (subject root)
        # We assume the data_dir is sub-XX/eeg etc, so parent is sub-XX
        subject_root = data_dir.parent
        root_coordsystems = list(subject_root.glob("*_coordsystem.json"))

        if root_coordsystems:
            src = root_coordsystems[0]
            # match naming convention if possible, or just use src name
            dst = data_dir / src.name

            if not dst.exists():
                # Use relative path for portability
                rel_target = os.path.relpath(src, dst.parent)
                try:
                    # Clean up potential broken symlink (FileExistsError otherwise)
                    dst.unlink(missing_ok=True)
                    dst.symlink_to(rel_target)
                    logger.debug(f"Created coordsystem symlink: {dst} -> {rel_target}")
                except Exception as e:
                    logger.warning(f"Failed to link coordsystem: {e}")

    except Exception as e:
        logger.warning(f"Error checking coordsystem symlinks: {e}")


def _generate_vmrk_stub(vmrk_path: Path, vhdr_name: str) -> bool:
    """Generate a minimal VMRK marker file stub.

    BrainVision requires a .vmrk file to exist. This creates a minimal
    valid stub with just the DataFile reference.

    Parameters
    ----------
    vmrk_path : Path
        Path where the VMRK file should be created.
    vhdr_name : str
        Name of the corresponding VHDR file.

    Returns
    -------
    bool
        True if file was generated, False on failure.

    """
    content = f"""Brain Vision Data Exchange Marker File Version 1.0
; Generated by EEGDash - minimal stub

[Common Infos]
Codepage=UTF-8
DataFile={vhdr_name.replace(".vhdr", ".eeg")}

[Marker Infos]
; No markers defined
"""
    try:
        vmrk_path.write_text(content, encoding="utf-8")
        logger.info(f"Generated VMRK stub: {vmrk_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to write VMRK stub {vmrk_path}: {e}")
        return False


def _generate_vhdr_from_metadata(
    vhdr_path: Path,
    record: dict[str, Any],
) -> bool:
    """Generate a minimal VHDR file from database record metadata.

    When a VHDR file is missing but we have the required metadata
    (channel names, sampling frequency, number of channels) from the
    database record, this function creates a minimal valid VHDR header
    that MNE can read.

    Parameters
    ----------
    vhdr_path : Path
        Path where the VHDR file should be created.
    record : dict
        Database record containing ch_names, sampling_frequency, nchans.

    Returns
    -------
    bool
        True if file was generated, False if insufficient metadata.

    """
    # Check if we have required metadata
    ch_names = record.get("ch_names")
    sfreq = record.get("sampling_frequency")
    nchans = record.get("nchans")

    if not ch_names or not sfreq or not nchans:
        logger.warning(f"Cannot generate VHDR: missing metadata for {vhdr_path.name}")
        return False

    # Validate that ch_names length matches nchans
    if len(ch_names) != nchans:
        logger.warning(
            f"Cannot generate VHDR: ch_names length ({len(ch_names)}) "
            f"!= nchans ({nchans}) for {vhdr_path.name}"
        )
        return False

    # Derive companion file names from VHDR path
    data_file = vhdr_path.with_suffix(".eeg").name
    marker_file = vhdr_path.with_suffix(".vmrk").name

    # Convert Hz to microseconds (BrainVision uses sampling interval in µs)
    sampling_interval = 1_000_000 / sfreq

    # Build VHDR content
    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; Generated by EEGDash from database metadata",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data_file}",
        f"MarkerFile={marker_file}",
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={nchans}",
        f"SamplingInterval={sampling_interval}",
        "",
        "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32",
        "",
        "[Channel Infos]",
    ]

    # Add channel definitions
    # Format: ChN=Name,Reference,Resolution,Unit
    for i, name in enumerate(ch_names, 1):
        lines.append(f"Ch{i}={name},,0.1,µV")

    # Write file
    try:
        # Ensure parent directory exists
        vhdr_path.parent.mkdir(parents=True, exist_ok=True)
        vhdr_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated VHDR from metadata: {vhdr_path.name}")

        # Also generate VMRK stub if missing
        vmrk_path = vhdr_path.with_suffix(".vmrk")
        if not vmrk_path.exists():
            _generate_vmrk_stub(vmrk_path, vhdr_path.name)

        return True
    except Exception as e:
        logger.error(f"Failed to write generated VHDR {vhdr_path}: {e}")
        return False


def _repair_snirf_bids_metadata(snirf_path: Path, record: dict[str, Any]) -> bool:
    """Fix BIDS metadata files for SNIRF (fNIRS) datasets.

    This function attempts to fix common BIDS compliance issues in fNIRS datasets:
    1. Regenerates channels.tsv from the SNIRF file's actual channel names
    2. Fixes malformed entries in scans.tsv (NaN timestamps, short strings)

    Parameters
    ----------
    snirf_path : Path
        Path to the SNIRF file.
    record : dict
        The database record for this file.

    Returns
    -------
    bool
        True if any repairs were made, False otherwise.

    """
    if not snirf_path.exists() or snirf_path.suffix.lower() != ".snirf":
        return False

    data_dir = snirf_path.parent
    repairs_made = False

    # ==== Fix 1: Regenerate channels.tsv from SNIRF channel names ====
    try:
        from mne.io import read_raw_snirf

        # Load SNIRF to get actual channel names
        raw = read_raw_snirf(str(snirf_path), preload=False)
        ch_names = raw.ch_names
        ch_types = raw.get_channel_types()

        # Find the channels.tsv file for this recording
        # It can be named *_channels.tsv or just channels.tsv
        channels_files = list(data_dir.glob("*_channels.tsv"))
        if not channels_files:
            channels_files = list(data_dir.glob("channels.tsv"))

        for channels_tsv in channels_files:
            # Check if this channels.tsv matches our SNIRF file
            # (by checking if the basename patterns match)
            snirf_base = snirf_path.stem.replace("_nirs", "")
            tsv_base = channels_tsv.stem.replace("_channels", "")

            if snirf_base.startswith(tsv_base) or tsv_base in snirf_base:
                # Read existing channels.tsv to preserve structure
                import pandas as pd

                try:
                    existing_df = pd.read_csv(channels_tsv, sep="\t")

                    # Check if channel names match
                    if list(existing_df["name"]) != ch_names:
                        repairs_made = True

                        # Regenerate with correct channel names
                        new_df = pd.DataFrame(
                            {
                                "name": ch_names,
                                "type": [t.upper() for t in ch_types],
                                "units": ["V"] * len(ch_names),
                            }
                        )
                        new_df.to_csv(channels_tsv, sep="\t", index=False)
                        logger.info(
                            f"Regenerated {channels_tsv.name} with {len(ch_names)} channels "
                            f"from SNIRF file"
                        )

                except Exception as e:
                    logger.warning(f"Could not repair channels.tsv: {e}")

    except ImportError:
        logger.warning("Cannot repair SNIRF metadata: mne not available")
    except Exception as e:
        logger.warning(f"Error reading SNIRF for channel repair: {e}")

    # ==== Fix 2: Remove malformed scans.tsv entries ====
    try:
        scans_files = list(data_dir.parent.glob("*_scans.tsv"))
        if not scans_files:
            scans_files = list(data_dir.parent.glob("scans.tsv"))

        for scans_tsv in scans_files:
            try:
                import pandas as pd

                df = pd.read_csv(scans_tsv, sep="\t")

                if "acq_time" in df.columns:
                    # Check for malformed timestamps: NaN, empty, or very short strings
                    malformed = df["acq_time"].apply(
                        lambda x: pd.isna(x)
                        or (isinstance(x, str) and (len(x) < 10 or x.strip() == ""))
                    )

                    if malformed.any():
                        repairs_made = True

                        # Convert column to string type, then replace malformed values
                        df["acq_time"] = df["acq_time"].astype(str)
                        df.loc[malformed, "acq_time"] = "n/a"
                        df.to_csv(scans_tsv, sep="\t", index=False)
                        logger.info(
                            f"Fixed {malformed.sum()} malformed timestamps in {scans_tsv.name}"
                        )

            except Exception as e:
                logger.warning(f"Could not repair scans.tsv: {e}")

    except Exception as e:
        logger.warning(f"Error fixing scans.tsv: {e}")

    return repairs_made
