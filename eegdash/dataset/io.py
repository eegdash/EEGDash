"""Input/Output utilities for EEG datasets.

This module contains helper functions for managing EEG data files,
specifically for fixing common issues in BIDS datasets and handling
file system operations.
"""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from mne.io import BaseRaw

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
        else:
            # No coordsystem.json found anywhere — generate a minimal one
            # Infer the coordinate system from the electrodes filename
            # (e.g. "sub-01_ses-01_space-CapTrak_electrodes.tsv")
            _generate_coordsystem_json(electrodes_files[0])

    except Exception as e:
        logger.warning(f"Error checking coordsystem symlinks: {e}")


def _generate_coordsystem_json(electrodes_tsv: Path) -> bool:
    """Generate a minimal coordsystem.json from the electrodes.tsv filename.

    BIDS requires coordsystem.json whenever electrodes.tsv exists. Some
    OpenNeuro datasets omit it. This generates a minimal valid one by
    extracting the coordinate system from the ``space-<label>`` entity
    in the electrodes filename.

    Parameters
    ----------
    electrodes_tsv : Path
        Path to the electrodes.tsv file.

    Returns
    -------
    bool
        True if the file was generated, False otherwise.

    """
    try:
        name = electrodes_tsv.stem  # e.g. sub-01_ses-01_space-CapTrak_electrodes
        # Extract space entity
        match = re.search(r"space-([A-Za-z0-9]+)", name)
        coord_system = match.group(1) if match else "Other"

        # Build the coordsystem.json filename by replacing _electrodes with
        # _coordsystem and keeping the rest of the BIDS entities
        coordsystem_name = name.replace("_electrodes", "_coordsystem") + ".json"
        coordsystem_path = electrodes_tsv.parent / coordsystem_name

        coordsystem_data = {
            "EEGCoordinateSystem": coord_system,
            "EEGCoordinateUnits": "m",
        }

        coordsystem_path.write_text(json.dumps(coordsystem_data, indent=2))
        logger.info(
            f"Generated minimal coordsystem.json: {coordsystem_path.name} "
            f"(system={coord_system})"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to generate coordsystem.json: {e}")
        return False


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


def _repair_tsv_encoding(data_dir: Path) -> bool:
    """Fix TSV files with non-UTF-8 encoding (e.g., Latin-1).

    Some datasets have channels.tsv files saved with Latin-1 encoding
    (common when using µ for microvolts). This converts them to UTF-8.

    Parameters
    ----------
    data_dir : Path
        Directory containing TSV files to check.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    repaired_any = False
    for tsv_path in data_dir.glob("*.tsv"):
        try:
            tsv_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            for encoding in ("cp1252", "latin-1"):
                try:
                    content = tsv_path.read_text(encoding=encoding)
                    tsv_path.write_text(content, encoding="utf-8")
                    logger.info(
                        f"Repaired TSV encoding: {tsv_path.name} ({encoding} -> UTF-8)"
                    )
                    repaired_any = True
                    break
                except Exception:
                    continue
        except Exception:
            pass
    return repaired_any


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


def _load_epoched_eeglab_as_raw(set_path: Path) -> BaseRaw:
    """Load an epoched EEGLAB .set file and convert to continuous Raw.

    EEGLAB .set files can contain epoched (segmented) data, which
    ``mne.io.read_raw_eeglab()`` rejects. This function loads epochs
    and concatenates them into a continuous ``RawArray``.

    Uses two strategies:
    1. ``mne.read_epochs_eeglab()`` — standard approach
    2. Direct scipy/h5py loading as fallback when MNE fails (some datasets
       have event structures that confuse MNE's epoch reader)

    Parameters
    ----------
    set_path : Path
        Path to the .set file containing epoched data.

    Returns
    -------
    mne.io.RawArray
        Continuous raw data created by concatenating all epochs.

    """
    import mne

    logger.info(f"Loading epoched EEGLAB file as continuous: {set_path.name}")

    # Strategy 1: try MNE's epoch reader
    try:
        epochs = mne.read_epochs_eeglab(str(set_path), verbose="ERROR")
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        continuous = np.concatenate(data, axis=-1)
        return mne.io.RawArray(continuous, epochs.info, verbose="ERROR")
    except Exception as mne_err:
        logger.warning(
            f"MNE epoch reader failed ({mne_err}), trying direct scipy loading..."
        )

    # Strategy 2: load directly via scipy/h5py
    return _load_set_via_scipy(set_path)


def _load_set_via_scipy(set_path: Path) -> BaseRaw:
    """Load an EEGLAB .set file directly using scipy or h5py.

    Fallback when ``mne.read_epochs_eeglab()`` fails. Reads the raw
    MATLAB structure, extracts data and channel info, and builds a
    ``RawArray``.

    Parameters
    ----------
    set_path : Path
        Path to the .set file.

    Returns
    -------
    mne.io.RawArray
        Continuous raw data.

    """
    import mne

    eeg = None

    # Try scipy first (MATLAB v5 format)
    try:
        from scipy.io import loadmat

        mat = loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
        if "EEG" in mat:
            eeg = mat["EEG"]
        elif "srate" in mat and "nbchan" in mat:
            # Flat structure: fields stored as top-level variables (not nested)
            from types import SimpleNamespace

            eeg = SimpleNamespace(
                srate=mat["srate"],
                nbchan=mat["nbchan"],
                pnts=mat["pnts"],
                trials=mat["trials"],
                data=mat["data"],
                chanlocs=mat.get("chanlocs"),
                datfile=mat.get("datfile", ""),
            )
    except Exception:
        pass

    # Try h5py (MATLAB v7.3 / HDF5 format)
    if eeg is None:
        try:
            import h5py

            with h5py.File(str(set_path), "r") as h5:
                # h5py structure is different — extract data manually
                data = np.array(h5["EEG"]["data"])
                srate = float(np.array(h5["EEG"]["srate"]).flat[0])
                nbchan = int(np.array(h5["EEG"]["nbchan"]).flat[0])
                # Try to extract channel names
                ch_names = []
                try:
                    chanlocs = h5["EEG"]["chanlocs"]
                    labels = chanlocs["labels"]
                    for ref in labels[0]:
                        name = "".join(chr(c) for c in h5[ref][:].flat)
                        ch_names.append(name.strip())
                except Exception:
                    ch_names = [f"EEG{i:03d}" for i in range(nbchan)]

                if data.ndim == 3:
                    # (n_channels, n_points, n_trials) or permuted
                    if data.shape[0] == nbchan:
                        continuous = data.reshape(nbchan, -1)
                    else:
                        data = np.transpose(data, (2, 1, 0))
                        continuous = data.reshape(nbchan, -1)
                else:
                    continuous = data

                info = mne.create_info(
                    ch_names=ch_names[:nbchan], sfreq=srate, ch_types="eeg"
                )
                return mne.io.RawArray(continuous, info, verbose="ERROR")
        except Exception:
            pass

    if eeg is None:
        raise ValueError(f"Cannot read .set file: {set_path}")

    # Extract from scipy's mat_struct
    srate = float(eeg.srate)
    nbchan = int(eeg.nbchan)
    pnts = int(eeg.pnts)
    trials = int(eeg.trials)

    # Handle data that may be in a separate .fdt file
    raw_data = eeg.data
    if isinstance(raw_data, str) or (hasattr(raw_data, "dtype") and raw_data.size == 0):
        # Data is in a .fdt companion file — try several strategies to find it
        fdt_path = None

        # Strategy 1: matching .fdt with same stem as .set file
        stem_fdt = set_path.with_suffix(".fdt")
        if stem_fdt.exists():
            fdt_path = stem_fdt

        # Strategy 2: explicit datfile field
        if fdt_path is None:
            datfile = getattr(eeg, "datfile", "")
            if datfile and isinstance(datfile, str):
                candidate = set_path.parent / datfile
                if candidate.exists():
                    fdt_path = candidate

        # Strategy 3: the data field itself (may be a filename)
        if fdt_path is None and isinstance(raw_data, str):
            candidate = set_path.parent / raw_data
            if candidate.exists():
                fdt_path = candidate

        # Strategy 4: any .fdt in the same directory (only if single match)
        if fdt_path is None:
            fdt_candidates = list(set_path.parent.glob("*.fdt"))
            if len(fdt_candidates) == 1:
                fdt_path = fdt_candidates[0]
        if fdt_path.exists():
            logger.info(f"Loading data from companion .fdt file: {fdt_path.name}")
            data = np.fromfile(str(fdt_path), dtype=np.float32)
            data = data.reshape(nbchan, pnts * trials, order="F")
            if trials > 1:
                # Already flattened to (n_channels, total_points)
                pass
        else:
            raise ValueError(f"Cannot find .fdt companion file for {set_path}")
    else:
        data = np.array(raw_data, dtype=np.float64)

    # Extract channel names
    ch_names = []
    try:
        chanlocs = eeg.chanlocs
        if chanlocs is not None and hasattr(chanlocs, "__iter__") and len(chanlocs) > 0:
            for ch in chanlocs:
                label = getattr(ch, "labels", None) or f"EEG{len(ch_names):03d}"
                ch_names.append(str(label).strip())
        elif chanlocs is not None and hasattr(chanlocs, "labels"):
            label = getattr(chanlocs, "labels", "EEG000")
            ch_names = [str(label).strip()]
    except Exception:
        pass
    if len(ch_names) != nbchan:
        ch_names = [f"EEG{i:03d}" for i in range(nbchan)]

    # Reshape data: (n_channels, n_points, n_trials) → (n_channels, total_points)
    if data.ndim == 3:
        # EEGLAB stores as (n_channels, n_points, n_trials)
        continuous = data.reshape(nbchan, -1)
    elif data.ndim == 2:
        continuous = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    info = mne.create_info(ch_names=ch_names[:nbchan], sfreq=srate, ch_types="eeg")
    logger.info(
        f"Loaded via scipy: {nbchan} channels, {continuous.shape[1]} samples, {srate} Hz"
    )
    return mne.io.RawArray(continuous, info, verbose="ERROR")


def _repair_electrodes_tsv(data_dir: Path) -> bool:
    """Fix electrodes.tsv files with 'n/a' in coordinate columns.

    Some datasets have ``n/a`` in the x, y, z columns of electrodes.tsv,
    which causes MNE-BIDS to fail with "Illegal data in EEG position file".
    This replaces ``n/a`` with ``0.0`` in coordinate columns.

    Parameters
    ----------
    data_dir : Path
        Directory containing electrodes.tsv files.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    repaired_any = False
    for tsv_path in data_dir.glob("*_electrodes.tsv"):
        try:
            content = tsv_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if not lines:
                continue

            header = lines[0].split("\t")
            coord_cols = {
                i
                for i, col in enumerate(header)
                if col.strip().lower() in ("x", "y", "z")
            }
            if not coord_cols:
                continue

            new_lines = [lines[0]]
            changed = False
            for line in lines[1:]:
                if not line.strip():
                    new_lines.append(line)
                    continue
                fields = line.split("\t")
                for i in coord_cols:
                    if i < len(fields) and fields[i].strip().lower() == "n/a":
                        fields[i] = "0.0"
                        changed = True
                new_lines.append("\t".join(fields))

            if changed:
                tsv_path.write_text("\n".join(new_lines), encoding="utf-8")
                logger.info(f"Repaired n/a coordinates in {tsv_path.name}")
                repaired_any = True

        except Exception as e:
            logger.warning(f"Failed to repair electrodes.tsv {tsv_path}: {e}")

    return repaired_any


def _repair_tsv_decimal_separators(data_dir: Path) -> bool:
    """Fix TSV files using comma as decimal separator instead of dot.

    Some European datasets use comma as decimal separator (e.g., ``5,004``
    instead of ``5.004``). This converts commas to dots in numeric columns.

    Parameters
    ----------
    data_dir : Path
        Directory containing TSV files to check.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    # Only check files likely to have numeric columns
    target_files = (
        list(data_dir.glob("*_electrodes.tsv"))
        + list(data_dir.glob("*_channels.tsv"))
        + list(data_dir.glob("*_events.tsv"))
    )

    repaired_any = False
    # Pattern: digit(s), comma, digit(s) — but not inside quoted strings
    comma_decimal_re = re.compile(r"(?<!\w)(\d+),(\d+)(?!\w)")

    for tsv_path in target_files:
        try:
            content = tsv_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if len(lines) < 2:
                continue

            # Skip header, check data lines
            new_content = lines[0] + "\n"
            changed = False
            for line in lines[1:]:
                new_line = comma_decimal_re.sub(r"\1.\2", line)
                if new_line != line:
                    changed = True
                new_content += new_line + "\n"

            if changed:
                # Remove trailing newline to match original
                new_content = new_content.rstrip("\n")
                if content.endswith("\n"):
                    new_content += "\n"
                tsv_path.write_text(new_content, encoding="utf-8")
                logger.info(f"Repaired decimal separators in {tsv_path.name}")
                repaired_any = True

        except Exception as e:
            logger.warning(f"Failed to repair decimal separators in {tsv_path}: {e}")

    return repaired_any


def _repair_tsv_na_values(data_dir: Path) -> bool:
    """Fix TSV files with 'n/a' in numeric columns that cause NaN-to-int errors.

    Some datasets have ``n/a`` in columns like ``sampling_frequency``,
    ``low_cutoff``, ``high_cutoff``, ``notch`` in channels.tsv. When pandas
    reads these as float NaN and MNE tries to convert to int, it fails.
    This replaces problematic ``n/a`` values with valid defaults.

    Parameters
    ----------
    data_dir : Path
        Directory containing TSV files to check.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    # Columns where n/a causes NaN-to-int conversion errors
    numeric_columns = {
        "sampling_frequency",
        "low_cutoff",
        "high_cutoff",
        "notch",
        "status",
    }
    # Defaults: 0 is safe for cutoff/notch (means "not applied")
    default_value = "0"

    repaired_any = False
    for tsv_path in data_dir.glob("*_channels.tsv"):
        try:
            content = tsv_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if not lines:
                continue

            header = lines[0].split("\t")
            # Find which columns are numeric and might have n/a
            target_cols = {
                i
                for i, col in enumerate(header)
                if col.strip().lower() in numeric_columns
            }
            if not target_cols:
                continue

            new_lines = [lines[0]]
            changed = False
            for line in lines[1:]:
                if not line.strip():
                    new_lines.append(line)
                    continue
                fields = line.split("\t")
                for i in target_cols:
                    if i < len(fields) and fields[i].strip().lower() == "n/a":
                        fields[i] = default_value
                        changed = True
                new_lines.append("\t".join(fields))

            if changed:
                tsv_path.write_text("\n".join(new_lines), encoding="utf-8")
                logger.info(f"Repaired n/a numeric values in {tsv_path.name}")
                repaired_any = True

        except Exception as e:
            logger.warning(f"Failed to repair n/a values in {tsv_path}: {e}")

    return repaired_any


def _repair_events_tsv_na_duration(data_dir: Path) -> bool:
    """Fix events.tsv with invalid values that cause MNE annotation errors.

    Handles three cases:

    1. ``n/a`` in the ``duration`` column → replaced with ``0``
    2. ``NaN`` in the ``onset`` column → row is removed (no valid timestamp)
    3. ``n/a`` in the ``sample`` column → replaced with ``0``

    Parameters
    ----------
    data_dir : Path
        Directory containing TSV files to check.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    repaired_any = False
    for tsv_path in data_dir.glob("*_events.tsv"):
        try:
            content = tsv_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if not lines:
                continue

            header_lower = [c.strip().lower() for c in lines[0].split("\t")]

            # Find column indices (all optional)
            dur_idx = (
                header_lower.index("duration") if "duration" in header_lower else -1
            )
            onset_idx = header_lower.index("onset") if "onset" in header_lower else -1
            sample_idx = (
                header_lower.index("sample") if "sample" in header_lower else -1
            )

            if dur_idx == -1 and onset_idx == -1 and sample_idx == -1:
                continue

            new_lines = [lines[0]]
            changed = False
            for line in lines[1:]:
                if not line.strip():
                    new_lines.append(line)
                    continue
                fields = line.split("\t")

                # Drop rows with NaN onset — events without timestamps
                # can't be used for MNE annotations
                if onset_idx >= 0 and onset_idx < len(fields):
                    val = fields[onset_idx].strip().lower()
                    if val in ("nan", "n/a"):
                        changed = True
                        continue  # skip this row entirely

                # Fix n/a in duration column
                if dur_idx >= 0 and dur_idx < len(fields):
                    if fields[dur_idx].strip().lower() == "n/a":
                        fields[dur_idx] = "0"
                        changed = True

                # Fix n/a in sample column
                if sample_idx >= 0 and sample_idx < len(fields):
                    if fields[sample_idx].strip().lower() == "n/a":
                        fields[sample_idx] = "0"
                        changed = True

                new_lines.append("\t".join(fields))

            if changed:
                tsv_path.write_text("\n".join(new_lines), encoding="utf-8")
                logger.info(f"Repaired invalid values in {tsv_path.name}")
                repaired_any = True

        except Exception as e:
            logger.warning(f"Failed to repair events.tsv {tsv_path}: {e}")

    return repaired_any


def _repair_ctf_eeg_position_file(ds_dir: Path) -> bool:
    """Fix CTF .eeg position files containing only 'n/a'.

    MNE's CTF reader expects the ``.eeg`` file (inside a ``.ds`` directory)
    to contain lines with exactly 5 space-separated fields:
    ``<id> <label> <x> <y> <z>``. Some datasets have a placeholder ``n/a``
    instead. Since MNE gracefully handles *missing* ``.eeg`` files (returns
    ``None``), the simplest fix is to truncate the malformed file to zero
    bytes so the reader skips it.

    Parameters
    ----------
    ds_dir : Path
        Path to the ``.ds`` CTF directory.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not ds_dir.exists() or not ds_dir.is_dir():
        return False

    repaired = False
    for eeg_file in ds_dir.glob("*.eeg"):
        try:
            content = eeg_file.read_text(encoding="utf-8", errors="replace").strip()
            # If the file only contains n/a (possibly with whitespace),
            # truncate it so MNE treats it as "no EEG positions"
            if content.lower() == "n/a":
                eeg_file.write_text("", encoding="utf-8")
                logger.info(
                    f"Repaired CTF .eeg position file {eeg_file.name}: "
                    f"replaced 'n/a' with empty file"
                )
                repaired = True
        except Exception as e:
            logger.warning(f"Failed to repair CTF .eeg file {eeg_file}: {e}")

    return repaired


def _repair_vhdr_missing_markerfile(vhdr_path: Path) -> bool:
    """Fix VHDR files missing a MarkerFile entry in [Common Infos].

    Some BrainVision datasets omit the ``MarkerFile=`` line, causing a
    KeyError when MNE tries to read the marker file reference.
    This adds the entry and generates a VMRK stub if needed.

    Parameters
    ----------
    vhdr_path : Path
        Path to the VHDR file.

    Returns
    -------
    bool
        True if repairs were made, False otherwise.

    """
    if not vhdr_path.exists() or vhdr_path.suffix != ".vhdr":
        return False

    try:
        content = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read VHDR {vhdr_path}: {e}")
        return False

    # Check if MarkerFile is already present
    if re.search(r"MarkerFile\s*=", content, re.IGNORECASE):
        return False

    # Check if [Common Infos] section exists
    if "[Common Infos]" not in content:
        return False

    # Add MarkerFile entry after DataFile (or after [Common Infos] if no DataFile)
    vmrk_name = vhdr_path.with_suffix(".vmrk").name
    datafile_match = re.search(r"(DataFile\s*=.*)", content, re.IGNORECASE)

    if datafile_match:
        insert_point = datafile_match.end()
        new_content = (
            content[:insert_point]
            + f"\nMarkerFile={vmrk_name}"
            + content[insert_point:]
        )
    else:
        new_content = content.replace(
            "[Common Infos]", f"[Common Infos]\nMarkerFile={vmrk_name}"
        )

    try:
        vhdr_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Added MarkerFile entry to {vhdr_path.name}")
    except Exception as e:
        logger.error(f"Failed to write repaired VHDR {vhdr_path}: {e}")
        return False

    # Generate VMRK stub if it doesn't exist
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    if not vmrk_path.exists():
        _generate_vmrk_stub(vmrk_path, vhdr_path.name)

    return True


def _load_raw_direct(filepath: Path, **kwargs) -> BaseRaw:
    """Load raw data directly via MNE readers, bypassing MNE-BIDS validation.

    Used as a fallback when ``mne_bids.read_raw_bids()`` fails due to BIDS
    validation errors (illegal dates, HPI issues, etc.) but the underlying
    data file is readable.

    Parameters
    ----------
    filepath : Path
        Path to the data file.
    **kwargs
        Additional arguments passed to the MNE reader function.

    Returns
    -------
    mne.io.BaseRaw
        The loaded raw data.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    """
    import mne

    ext = filepath.suffix.lower()
    # Map extensions to MNE reader functions
    readers = {
        ".fif": mne.io.read_raw_fif,
        ".set": mne.io.read_raw_eeglab,
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".gdf": mne.io.read_raw_gdf,
        ".cnt": mne.io.read_raw_cnt,
    }

    reader = readers.get(ext)
    if reader is None:
        raise ValueError(f"No direct reader available for extension '{ext}'")

    logger.info(f"Loading {filepath.name} directly via MNE (bypassing BIDS)")
    # Filter kwargs to only pass what the reader accepts
    reader_kwargs = {"preload": False, "verbose": "ERROR"}
    if ext == ".fif" and kwargs.get("allow_maxshield"):
        reader_kwargs["allow_maxshield"] = True
    elif ext == ".set":
        pass  # read_raw_eeglab doesn't take allow_maxshield

    return reader(str(filepath), **reader_kwargs)
