"""Input/Output utilities for EEG datasets.

This module contains helper functions for managing EEG data files,
specifically for fixing common issues in BIDS datasets and handling
file system operations.
"""

import configparser
import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path
from time import strptime
from typing import Any

import mne
import numpy as np
import pandas as pd
from dateutil.parser import parse as _dateutil_parse
from dateutil.parser import parserinfo
from mne.io.eeglab import _eeglab as mne_eeglab
from scipy.io import loadmat

from ..io import READERS, get_reader, tolerant
from ..io import eeglab as eeglab_io
from ..io.bids import tsv_encoding
from ..logging import logger


class _MultiLocaleParserInfo(parserinfo):
    """``parserinfo`` subclass that recognizes non-English month abbreviations.

    Covers German, Italian, French, Spanish, and Dutch short-month names
    commonly found in CTF MEG date headers.
    """

    MONTHS = [
        ("Jan", "Ene", "Gen"),  # January
        ("Feb", "Fév", "Fev"),  # February
        ("Mar", "Mär", "Mrt"),  # March
        ("Apr", "Avr"),  # April
        ("May", "Mai", "Mei"),  # May
        ("Jun", "Giu"),  # June
        ("Jul",),  # July
        ("Aug", "Aoû", "Aou", "Ago"),  # August
        ("Sep", "Set"),  # September
        ("Oct", "Okt", "Ott"),  # October
        ("Nov",),  # November
        ("Dec", "Dez", "Dic"),  # December
    ]


def _convert_time_with_numeric_dash(date_str: str, time_str: str, *, orig):
    """Try numeric dash date formats, then locale month names, then delegate to orig."""
    # Pass 1: purely numeric dash dates (e.g. 14-10-1925)
    for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            date = strptime(date_str.strip(), fmt)
            normalized = f"{date.tm_mday:02d}/{date.tm_mon:02d}/{date.tm_year}"
            return orig(normalized, time_str)
        except ValueError:
            continue

    # Pass 2: locale month abbreviations (e.g. 14-Okt-1925, 14-Ott-1925)
    try:
        dt = _dateutil_parse(date_str.strip(), parserinfo=_MultiLocaleParserInfo())
        normalized = f"{dt.day:02d}/{dt.month:02d}/{dt.year}"
        return orig(normalized, time_str)
    except (ValueError, TypeError):
        pass

    return orig(date_str, time_str)


_ANNEX_KEY_RE = re.compile(r"^(SHA256E|MD5E)-s\d+--[0-9a-f]+\.", flags=re.IGNORECASE)

# Regex matching common git-annex placeholder content.
# When git-annex content is unavailable, the file may contain only a short
# placeholder string like ``--corrupted--`` or ``/annex/objects/...``.
_ANNEX_PLACEHOLDER_RE = re.compile(
    r"^\s*(--corrupted--|/annex/objects/|\.git/annex/)", flags=re.IGNORECASE
)

# Maximum file size (bytes) to consider when checking for placeholder content.
# Real BrainVision data files are always larger than this.
_PLACEHOLDER_MAX_SIZE = 256

# BIDS datatype → coordinate-system JSON key prefix (used by coordsystem
# generation/validation helpers).
_COORD_PREFIX = {"eeg": "EEG", "ieeg": "iEEG", "meg": "MEG", "emg": "EMG"}


def _is_annex_placeholder(path: Path) -> bool:
    """Return True if *path* is a git-annex placeholder (not real data).

    Git-annex stores symlinks to content-addressed blobs.  When the actual
    content was never fetched (or has been dropped), the blob may contain a
    short placeholder string like ``--corrupted--``.  This function checks
    for that pattern so callers can raise an informative error instead of
    letting MNE crash on unparsable content.
    """
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
        if size > _PLACEHOLDER_MAX_SIZE:
            return False
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return True  # empty file is also not real data
        return bool(_ANNEX_PLACEHOLDER_RE.match(content))
    except Exception:
        return False


def _prepare_writable_path(path: Path) -> bool:
    """Ensure *path* can be written to by the current process.

    If the path is a symlink (common with git-annex), the symlink is removed
    so a new regular file can be created in its place.  The function also
    verifies that the parent directory is writable.

    Returns ``True`` if the path is (now) writable, ``False`` otherwise.
    """
    try:
        parent = path.parent
        # Check parent directory is writable *before* removing anything
        if not os.access(parent, os.W_OK):
            return False
        # Remove symlink so we can create a fresh regular file
        if path.is_symlink():
            os.unlink(path)
        return True
    except OSError:
        return False


def _ensure_coordsystem_symlink(data_dir: Path) -> None:
    """Ensure coordsystem.json exists in the data directory using symlinks if needed.

    MNE-BIDS can be strict about coordsystem.json location. If it's missing
    in the data directory but present in the subject root, this function
    creates a symlink. Otherwise it generates a minimal one with the correct
    datatype-specific keys (EEG vs iEEG vs MEG).

    Parameters
    ----------
    data_dir : Path
        The directory containing the data files (e.g. sub-01/eeg/, sub-01/ieeg/).

    """
    if not data_dir.exists():
        return

    # Infer BIDS datatype from directory name (eeg, ieeg, meg, …)
    datatype = data_dir.name

    try:
        # Check if we have electrodes.tsv here (implies need for coordsystem)
        electrodes_files = list(data_dir.glob("*_electrodes.tsv"))
        if not electrodes_files:
            return

        # Check if coordsystem.json exists and has correct keys for the datatype
        coordsystem_files = list(data_dir.glob("*_coordsystem.json"))
        if coordsystem_files:
            prefix = _COORD_PREFIX.get(datatype, "EEG")
            expected_key = f"{prefix}CoordinateSystem"
            try:
                content = json.loads(coordsystem_files[0].read_text(encoding="utf-8"))
                if expected_key in content:
                    return  # Valid — nothing to do
            except Exception:
                pass
            # Keys missing/wrong or file unreadable — overwrite with correct keys
            logger.info(
                "coordsystem.json has wrong keys for %s datatype, regenerating...",
                datatype,
            )
            _generate_coordsystem_json(electrodes_files[0], datatype=datatype)
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
                    logger.debug(
                        "Created coordsystem symlink: %s -> %s", dst, rel_target
                    )
                except Exception as e:
                    logger.warning("Failed to link coordsystem: %s", e)
        else:
            # No coordsystem.json found anywhere — generate a minimal one
            # Infer the coordinate system from the electrodes filename
            # (e.g. "sub-01_ses-01_space-CapTrak_electrodes.tsv")
            _generate_coordsystem_json(electrodes_files[0], datatype=datatype)

    except Exception as e:
        logger.warning("Error checking coordsystem symlinks: %s", e)


def _generate_coordsystem_json(electrodes_tsv: Path, datatype: str = "eeg") -> bool:
    """Generate a minimal coordsystem.json from the electrodes.tsv filename.

    BIDS requires coordsystem.json whenever electrodes.tsv exists. Some
    OpenNeuro datasets omit it. This generates a minimal valid one by
    extracting the coordinate system from the ``space-<label>`` entity
    in the electrodes filename.

    The JSON keys are datatype-specific per BIDS:

    - iEEG: ``iEEGCoordinateSystem`` / ``iEEGCoordinateUnits``
    - MEG:  ``MEGCoordinateSystem``  / ``MEGCoordinateUnits``
    - EEG (default): ``EEGCoordinateSystem`` / ``EEGCoordinateUnits``

    Parameters
    ----------
    electrodes_tsv : Path
        Path to the electrodes.tsv file.
    datatype : str
        BIDS datatype (``"eeg"``, ``"ieeg"``, or ``"meg"``).

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

        # BIDS uses datatype-specific keys for coordinate system metadata
        prefix = _COORD_PREFIX.get(datatype)
        if prefix is None:
            logger.warning(
                f"Unexpected datatype {datatype!r} for coordsystem generation, "
                f"expected one of {set(_COORD_PREFIX)}. Defaulting to EEG keys."
            )
            prefix = "EEG"

        coordsystem_data = {
            f"{prefix}CoordinateSystem": coord_system,
            f"{prefix}CoordinateUnits": "m",
        }

        coordsystem_path.write_text(json.dumps(coordsystem_data, indent=2))
        logger.info(
            f"Generated minimal coordsystem.json: {coordsystem_path.name} "
            f"(datatype={datatype}, system={coord_system})"
        )
        return True
    except Exception as e:
        logger.warning("Failed to generate coordsystem.json: %s", e)
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
        logger.warning("Cannot generate VHDR: missing metadata for %s", vhdr_path.name)
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
        # Handle symlinks (e.g. git-annex) — remove so we can create a
        # regular file; also verify directory is writable.
        if not _prepare_writable_path(vhdr_path):
            logger.warning(
                "Cannot write VHDR %s: parent directory not writable",
                vhdr_path,
            )
            return False
        vhdr_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Generated VHDR from metadata: %s", vhdr_path.name)
        return True
    except Exception as e:
        logger.error("Failed to write generated VHDR %s: %s", vhdr_path, e)
        return False


def _generate_vhdr_from_sibling(vhdr_path: Path) -> bool:
    """Generate a VHDR by borrowing metadata from a valid sibling VHDR.

    When the database record lacks ``ch_names``/``nchans`` (e.g. because the
    file was already corrupted at ingestion time), this function searches the
    dataset for another valid ``.vhdr`` file with parseable channel info and
    uses it to regenerate the corrupted header.

    Parameters
    ----------
    vhdr_path : Path
        Path where the VHDR file should be (re)created.

    Returns
    -------
    bool
        True if file was generated, False if no valid sibling was found.

    """
    # Walk up to the dataset root (two levels: eeg/ -> sub-XX/ -> dataset/)
    # but search broadly — any .vhdr in the dataset tree will do.
    dataset_root = vhdr_path.parent.parent.parent
    if not dataset_root.exists():
        return False

    # Ensure the target path is writable (handle symlinks / read-only dirs)
    if not _prepare_writable_path(vhdr_path):
        logger.warning(
            "Cannot write VHDR %s: parent directory not writable",
            vhdr_path,
        )
        return False

    # Limit search depth to avoid unbounded traversal on large/network
    # filesystems.  BIDS datasets have data at sub-XX/<datatype>/*.vhdr
    # (depth 2 from dataset root), so depth 3 covers all reasonable layouts.
    _MAX_VHDR_CANDIDATES = 50
    candidates_checked = 0
    for sibling in dataset_root.rglob("*.vhdr"):
        if sibling == vhdr_path:
            continue
        candidates_checked += 1
        if candidates_checked > _MAX_VHDR_CANDIDATES:
            break
        try:
            content = sibling.read_text(encoding="utf-8")
            if "[Common Infos]" not in content and "[Common infos]" not in content:
                continue

            # Validate that the sibling has the same number of channels
            # as the target .eeg file to avoid silent channel misattribution.
            sibling_cfg = configparser.ConfigParser()
            sibling_cfg.read_string(content)
            try:
                sibling_nchans = int(
                    sibling_cfg.get("Common Infos", "NumberOfChannels")
                )
            except (configparser.Error, ValueError):
                continue  # Can't determine channel count — skip this sibling

            target_eeg = vhdr_path.with_suffix(".eeg")
            if target_eeg.exists() and target_eeg.stat().st_size > 0:
                # BrainVision uses 4 bytes per sample (float32) by default
                eeg_size = target_eeg.stat().st_size
                if eeg_size % sibling_nchans != 0:
                    logger.debug(
                        "Skipping sibling %s: channel count %d doesn't "
                        "divide .eeg file size evenly",
                        sibling.name,
                        sibling_nchans,
                    )
                    continue

            # Replace DataFile and MarkerFile pointers to match target name
            data_file = vhdr_path.with_suffix(".eeg").name
            marker_file = vhdr_path.with_suffix(".vmrk").name
            sibling_data = sibling.with_suffix(".eeg").name
            sibling_marker = sibling.with_suffix(".vmrk").name

            content = content.replace(
                f"DataFile={sibling_data}", f"DataFile={data_file}"
            )
            content = content.replace(
                f"MarkerFile={sibling_marker}", f"MarkerFile={marker_file}"
            )

            vhdr_path.write_text(content, encoding="utf-8")
            logger.info(
                "Generated VHDR %s from sibling %s",
                vhdr_path.name,
                sibling.name,
            )
            return True
        except Exception:
            continue

    logger.warning("No valid sibling VHDR found for %s", vhdr_path.name)
    return False


def _repair_snirf_bids_metadata(snirf_path: Path, record: dict[str, Any]) -> bool:
    """Fix BIDS metadata files for SNIRF (fNIRS) datasets.

    Regenerates channels.tsv from the SNIRF file's actual channel names.
    (Malformed scans.tsv ``acq_time`` values need no repair: MNE-BIDS 0.20
    warns and keeps the file's meas_date.)

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
                try:
                    existing_df = pd.read_csv(
                        channels_tsv, sep="\t", encoding=tsv_encoding(channels_tsv)
                    )

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
                    logger.warning("Could not repair channels.tsv: %s", e)

    except ImportError:
        logger.warning("Cannot repair SNIRF metadata: mne not available")
    except Exception as e:
        logger.warning("Error reading SNIRF for channel repair: %s", e)

    return repairs_made


def _repair_channels_tsv(data_dir: Path) -> bool:
    r"""Repair empty or malformed ``*_channels.tsv`` files in *data_dir*.

    MNE-BIDS skips channel metadata when ``channels.tsv`` is absent but
    raises ``KeyError: 'name'`` when the file exists and is empty or lacks
    the required ``name`` column.  This function fixes such files:

    * **Empty / whitespace-only** → removed so MNE-BIDS skips channel
      metadata gracefully (an empty file has no data to preserve).
    * **Has content but no ``name`` column** → the first column header is
      renamed to ``name`` (per BIDS spec the first column must be ``name``).

    Returns ``True`` if any file was repaired.
    """
    if not data_dir.is_dir():
        return False

    repaired_any = False

    try:
        tsv_files = list(data_dir.glob("*_channels.tsv"))
    except Exception:
        return False

    for tsv_path in tsv_files:
        try:
            content = tsv_path.read_text(encoding=tsv_encoding(tsv_path))
            stripped = content.strip()

            if not stripped:
                # Empty or whitespace-only — remove so MNE-BIDS skips
                # channel metadata gracefully (no data to preserve).
                tsv_path.unlink()
                logger.info("Removed empty channels.tsv: %s", tsv_path.name)
                repaired_any = True
                continue

            parts = stripped.split("\n", 1)
            columns = parts[0].split("\t")

            if "name" not in [c.strip() for c in columns]:
                # First column must be 'name' per BIDS spec — rename it
                columns[0] = "name"
                new_header = "\t".join(columns)
                if len(parts) > 1:
                    new_content = new_header + "\n" + parts[1] + "\n"
                else:
                    new_content = new_header + "\n"
                tsv_path.write_text(new_content, encoding="utf-8")
                logger.info(
                    "Repaired channels.tsv (renamed first column to 'name'): %s",
                    tsv_path.name,
                )
                repaired_any = True
        except Exception as exc:
            logger.warning("Failed to repair %s: %s", tsv_path.name, exc)

    return repaired_any


def _deduplicate_channel_names(ch_names: list[str]) -> list[str]:
    """Return a list of channel names with duplicates made unique.

    For each duplicate name, appends ``-0``, ``-1``, … so that all names
    are distinct.  Non-duplicate names are left unchanged.

    Parameters
    ----------
    ch_names : list of str
        Original channel names (may contain duplicates).

    Returns
    -------
    list of str
        Channel names with duplicates disambiguated.

    """
    counts = Counter(ch_names)
    duplicates = {name for name, cnt in counts.items() if cnt > 1}
    if not duplicates:
        return ch_names

    seen: dict[str, int] = {}
    result: list[str] = []
    for name in ch_names:
        if name in duplicates:
            idx = seen.get(name, 0)
            result.append(f"{name}-{idx}")
            seen[name] = idx + 1
        else:
            result.append(name)
    return result


def _repair_participants_tsv_ids(bids_root: Path) -> bool:
    """Align ``participant_id`` values in ``participants.tsv`` with ``sub-*`` folders.

    Some BIDS datasets have mismatches between folder names and
    ``participants.tsv`` entries — different zero-padding (``sub-001`` vs
    ``sub-01``), stripped prefixes (``sub-FFE001`` vs ``sub-001``), or
    missing rows entirely. ``mne_bids.read_raw_bids`` performs a strict
    lookup that fails when these diverge.

    This function uses the same fuzzy-matching logic as the digestion
    pipeline (``_match_subject_fallback``) to rename mismatched TSV entries
    and adds placeholder rows for folders that have no TSV entry at all.

    Parameters
    ----------
    bids_root : Path
        Root directory of the BIDS dataset.

    Returns
    -------
    bool
        True if the file was repaired, False otherwise.

    """
    from .bids_dataset import _match_subject_fallback

    participants_tsv = bids_root / "participants.tsv"
    if not participants_tsv.exists():
        return False

    # Collect actual sub-* folder names
    folder_ids = {
        d.name for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
    }
    if not folder_ids:
        return False

    try:
        content = participants_tsv.read_text(encoding="utf-8-sig")
        lines = content.splitlines()
    except Exception:
        return False
    if not lines:
        return False

    header = lines[0].split("\t")
    try:
        pid_idx = header.index("participant_id")
    except ValueError:
        return False

    n_cols = len(header)

    # --- Pass 1: rename mismatched TSV entries to match folders ---
    # For each folder not yet in TSV, try to find a TSV entry that
    # fuzzy-matches it via _match_subject_fallback (same logic as digestion).
    tsv_ids = []
    for line in lines[1:]:
        cols = line.split("\t")
        if pid_idx < len(cols):
            tsv_ids.append(cols[pid_idx].strip())
    tsv_index = pd.Index(tsv_ids)

    # Build mapping: tsv_id -> folder_id for entries that need renaming.
    # Two directions are tried so that _match_subject_fallback's Tier 3
    # (prefix-stripping) works regardless of which side has the prefix.
    rename_map: dict[str, str] = {}
    matched_tsv_ids: set[str] = set()
    matched_folder_ids: set[str] = set()

    # Direction 1 (folder → TSV): for each folder missing from TSV, find
    # the TSV entry it matches.
    for folder_id in folder_ids:
        if folder_id in tsv_ids:
            matched_tsv_ids.add(folder_id)
            matched_folder_ids.add(folder_id)
            continue
        subj_val = folder_id.removeprefix("sub-")
        matched_tsv = _match_subject_fallback(folder_id, subj_val, tsv_index)
        if matched_tsv is not None and matched_tsv not in matched_tsv_ids:
            rename_map[matched_tsv] = folder_id
            matched_tsv_ids.add(matched_tsv)
            matched_folder_ids.add(folder_id)

    # Direction 2 (TSV → folder): for each TSV entry still unmatched, see
    # if it matches a folder (covers cases like sub-SD_1010 → sub-1010
    # where the TSV has a prefix the folder lacks).
    folder_index = pd.Index(sorted(folder_ids))
    for tsv_id in tsv_ids:
        if tsv_id in matched_tsv_ids or tsv_id in folder_ids:
            continue
        subj_val = tsv_id.removeprefix("sub-")
        matched_folder = _match_subject_fallback(tsv_id, subj_val, folder_index)
        if (
            matched_folder is not None
            and matched_folder not in matched_folder_ids
            and tsv_id not in rename_map
        ):
            rename_map[tsv_id] = matched_folder
            matched_tsv_ids.add(tsv_id)
            matched_folder_ids.add(matched_folder)

    # Apply renames to lines
    new_lines = [lines[0]]
    changed = False
    for line in lines[1:]:
        cols = line.split("\t")
        if pid_idx >= len(cols):
            new_lines.append(line)
            continue

        tsv_id = cols[pid_idx].strip()
        if tsv_id in rename_map:
            cols[pid_idx] = rename_map[tsv_id]
            new_lines.append("\t".join(cols))
            changed = True
        else:
            new_lines.append(line)

    # --- Pass 2: add placeholder rows for folders still missing ---
    # Collect all participant_ids now present after renames
    current_ids = set()
    for line in new_lines[1:]:
        cols = line.split("\t")
        if pid_idx < len(cols):
            current_ids.add(cols[pid_idx].strip())

    missing_folders = sorted(folder_ids - current_ids)
    for folder_id in missing_folders:
        placeholder = ["n/a"] * n_cols
        placeholder[pid_idx] = folder_id
        new_lines.append("\t".join(placeholder))
        changed = True

    if not changed:
        return False

    try:
        participants_tsv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info(
            "Repaired participants.tsv (aligned participant_id with sub-* folder names)"
        )
        return True
    except Exception as e:
        logger.warning("Failed to write repaired participants.tsv: %s", e)
        return False


def _clamping_crop(self, *args, orig_crop, **kwargs):
    """Patched ``Annotations.crop`` that clamps negative durations first."""
    if hasattr(self, "duration") and self.duration is not None:
        mask = self.duration < 0
        if np.any(mask):
            n_neg = int(np.sum(mask))
            logger.warning(
                "Clamping %d annotation(s) with negative duration to 0.",
                n_neg,
            )
            self.duration[mask] = 0.0
    return orig_crop(self, *args, **kwargs)


@contextmanager
def _fix_negative_annotation_durations():
    """Context manager that patches ``mne.Annotations.crop`` to clamp
    negative durations to zero before the internal assertion.

    Some EEGLAB ``.set`` files contain events whose computed duration is
    negative (e.g. overlapping boundary events).  MNE's
    ``Annotations.crop`` asserts ``(self.duration >= 0).all()``, which
    crashes during ``read_raw_eeglab`` before we ever get the Raw object
    back.  This patch silently clamps those durations so loading can
    succeed.

    .. warning::
       This patches a **class method** on ``mne.Annotations``, so it is
       **not thread-safe**.  It is safe with ``multiprocessing`` (separate
       interpreters) but must never be used from concurrent threads.
    """
    _orig_crop = mne.Annotations.crop
    # partialmethod, not partial: a partial is no descriptor, so
    # ``annot.crop(0, t)`` got ``self=0`` ("'int' has no len()", on006923).
    mne.Annotations.crop = partialmethod(_clamping_crop, orig_crop=_orig_crop)
    try:
        yield
    finally:
        mne.Annotations.crop = _orig_crop


def _load_raw_direct(filepath: Path):  # -> mne.io.BaseRaw
    """Load a data file directly via MNE readers, bypassing MNE-BIDS.

    Used as a fallback when MNE-BIDS entity validation rejects the
    filename (e.g. hyphens in ``task-`` entities) but the underlying
    data file is perfectly readable.

    Parameters
    ----------
    filepath : Path
        Path to the data file.

    Returns
    -------
    mne.io.BaseRaw
        The loaded Raw object.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    """
    ext = filepath.suffix.lower()
    if ext not in READERS:
        raise ValueError(
            f"No direct reader for extension {ext!r}. Supported: {sorted(READERS)}"
        )
    reader = get_reader(ext)

    logger.warning(
        "Falling back to direct %s reader for %s (bypassing MNE-BIDS).",
        ext,
        filepath.name,
    )

    # Split FIF files: MNE needs on_split_missing="warn" to load a
    # partial split without crashing when other splits are absent.
    if ext == ".fif":
        return reader(
            str(filepath), preload=False, verbose="ERROR", on_split_missing="warn"
        )

    # Try loading normally first; if annotations have negative durations
    # (common in EEGLAB .set files with malformed event boundaries),
    # retry with a patch that clamps them to zero.  The assertion
    # `assert (self.duration >= 0).all()` inside MNE's Annotations.crop
    # produces a bare AssertionError (no message text), so we match by
    # checking the traceback for "annotations" / "crop" / "duration".
    try:
        return reader(str(filepath), preload=False, verbose="ERROR")
    except AssertionError:
        import traceback

        tb_text = traceback.format_exc().lower()
        if "duration" in tb_text or "annotation" in tb_text or "crop" in tb_text:
            logger.warning(
                "Negative annotation durations detected in %s; "
                "retrying with duration clamping.",
                filepath.name,
            )
            with _fix_negative_annotation_durations():
                return reader(str(filepath), preload=False, verbose="ERROR")
        raise


def _parse_set_metadata(set_path: Path) -> dict:
    """Extract metadata from an EEGLAB .set file.

    Returns dict with keys: srate, nbchan, pnts, ch_names, data (if embedded).
    Raises ValueError if the file cannot be parsed.
    """
    set_path = Path(set_path)

    # Try scipy.io first (MATLAB v5 format)
    try:
        import scipy.io

        mat = scipy.io.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)
        if "EEG" not in mat:
            raise ValueError("No EEG structure found in .set file")

        eeg = mat["EEG"]
        result: dict = {}

        for attr in ("srate", "nbchan", "pnts"):
            val = getattr(eeg, attr, None)
            if val is None:
                raise ValueError(f"Missing required field '{attr}' in .set")
            if hasattr(val, "item"):
                val = val.item()
            result[attr] = float(val) if attr == "srate" else int(val)

        # Channel names from chanlocs
        ch_names = []
        if hasattr(eeg, "chanlocs") and eeg.chanlocs is not None:
            chanlocs = eeg.chanlocs
            if hasattr(chanlocs, "__iter__") and not isinstance(chanlocs, str):
                for ch in chanlocs:
                    if hasattr(ch, "labels"):
                        label = ch.labels
                        if hasattr(label, "item"):
                            label = label.item()
                        ch_names.append(str(label))
            elif hasattr(chanlocs, "labels"):
                label = chanlocs.labels
                if hasattr(label, "item"):
                    label = label.item()
                ch_names.append(str(label))
        result["ch_names"] = ch_names if ch_names else []

        # Check for embedded data
        if hasattr(eeg, "data") and isinstance(eeg.data, np.ndarray):
            if eeg.data.ndim >= 2:
                result["data"] = eeg.data.astype(np.float32)

        return result

    except ImportError:
        pass
    except ValueError:
        raise
    except Exception:
        pass

    # Try h5py for HDF5-based .set files
    try:
        import h5py

        with h5py.File(str(set_path), "r") as f:
            if "EEG" not in f:
                raise ValueError("No EEG structure found in .set file (HDF5)")

            eeg = f["EEG"]
            result = {}

            for key in ("srate", "nbchan", "pnts"):
                if key not in eeg:
                    raise ValueError(f"Missing required field '{key}' in .set")
                result[key] = (
                    float(eeg[key][0, 0]) if key == "srate" else int(eeg[key][0, 0])
                )

            result["ch_names"] = []
            return result

    except ImportError:
        pass
    except ValueError:
        raise

    raise ValueError("Cannot parse .set file: neither scipy nor h5py succeeded")


def _read_bids_channels_tsv(
    data_dir: Path,
) -> tuple[list[str], list[str]] | None:
    """Read channel names and types from a BIDS channels.tsv sidecar.

    Returns (ch_names, ch_types) or None if no channels.tsv found.
    """
    import csv

    tsv_files = list(data_dir.glob("*_channels.tsv"))
    if not tsv_files:
        return None

    tsv_path = tsv_files[0]
    bids_to_mne = {
        "EEG": "eeg",
        "EOG": "eog",
        "ECG": "ecg",
        "EMG": "emg",
        "MISC": "misc",
        "STIM": "stim",
        "REF": "eeg",
    }

    ch_names = []
    ch_types = []
    with open(tsv_path, newline="", encoding=tsv_encoding(tsv_path)) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("name", "").strip()
            if name:
                ch_names.append(name)
                bids_type = row.get("type", "EEG").strip().upper()
                ch_types.append(bids_to_mne.get(bids_type, "eeg"))

    return (ch_names, ch_types) if ch_names else None


def _load_raw_eeglab_fallback(set_path: Path, bids_root: Path | None = None):
    """Load EEGLAB .set/.fdt bypassing MNE's read_raw_eeglab.

    Fallback for when MNE's reader fails on non-standard .set structures.
    Extracts metadata from .set, reads raw data from .fdt, returns RawArray.
    """
    set_path = Path(set_path)
    meta = _parse_set_metadata(set_path)

    n_channels = meta["nbchan"]
    n_samples = meta["pnts"]
    sfreq = meta["srate"]

    # --- Resolve data array ---
    data = None
    fdt_path = set_path.with_suffix(".fdt")

    if fdt_path.exists():
        expected_size = n_channels * n_samples * 4  # float32
        actual_size = fdt_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"FDT size mismatch: expected {expected_size} bytes "
                f"({n_channels} ch x {n_samples} pts x 4), got {actual_size}"
            )
        data = np.fromfile(str(fdt_path), dtype=np.float32).reshape(
            (n_channels, n_samples), order="C"
        )
    elif "data" in meta:
        data = meta["data"]
        # Ensure correct shape
        if data.shape != (n_channels, n_samples):
            if data.size == n_channels * n_samples:
                data = data.reshape((n_channels, n_samples))
            else:
                raise ValueError(
                    f"Embedded data shape {data.shape} does not match "
                    f"metadata ({n_channels} ch x {n_samples} pts)"
                )
    else:
        raise ValueError(
            f"No data source: .fdt file not found at {fdt_path} "
            "and no embedded data in .set"
        )

    # --- Resolve channel names and types ---
    ch_names = None
    ch_types = None

    # Try BIDS sidecar first
    if bids_root or set_path.parent:
        bids_info = _read_bids_channels_tsv(set_path.parent)
        if bids_info and len(bids_info[0]) == n_channels:
            ch_names, ch_types = bids_info

    # Fall back to .set chanlocs
    if ch_names is None and meta["ch_names"] and len(meta["ch_names"]) == n_channels:
        ch_names = meta["ch_names"]

    # Generate defaults
    if ch_names is None:
        ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]

    if ch_types is None:
        ch_types = ["eeg"] * n_channels

    # --- Deduplicate channel names if needed ---
    ch_names = _deduplicate_channel_names(ch_names)

    # --- Scale: EEGLAB stores microvolts, MNE expects volts ---
    data = data.astype(np.float64) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    logger.warning(
        "Loaded %s via EEGLAB fallback (%d ch, %.1f s, %.0f Hz).",
        set_path.name,
        n_channels,
        n_samples / sfreq,
        sfreq,
    )
    return raw


def _load_raw_from_eeglab_epochs(set_path: Path):
    """Load an EEGLAB .set file containing epoched data as continuous Raw.

    When MNE raises ``TypeError`` because the ``.set`` file contains multiple
    trials (epochs), this function uses ``mne.io.read_epochs_eeglab`` and
    concatenates the epoch data into a continuous ``RawArray``.

    If ``read_epochs_eeglab`` itself fails (e.g. non-standard epoch structs
    missing ``eventtype``, or empty event arrays causing ``IndexError``),
    a manual fallback reads the raw data directly from the ``.set``/``.fdt``
    files using scipy, bypassing MNE's epoch parser entirely.

    Both paths share the same degenerate-data guard, channel-name resolution,
    and ``RawArray`` creation logic.

    Parameters
    ----------
    set_path : Path
        Path to the EEGLAB .set file.

    Returns
    -------
    mne.io.BaseRaw
        Continuous Raw object created from concatenated epochs.

    Raises
    ------
    ValueError
        If the file cannot be parsed or the data is too short.

    """
    set_path = Path(set_path)
    logger.info("Loading EEGLAB epoch file via read_epochs_eeglab: %s", set_path.name)

    # ---- Try MNE's read_epochs_eeglab first ----
    used_mne = False
    try:
        # tolerant(): the own .fdt, as for Raw (eegdash.io.eeglab)
        with tolerant():
            epochs = mne.io.read_epochs_eeglab(str(set_path), verbose="ERROR")
        used_mne = True
    except (AttributeError, IndexError, KeyError) as epoch_error:
        # Common failures:
        # - AttributeError: 'Bunch' object has no attribute 'eventtype'
        #   (non-standard epoch struct with custom fields instead of
        #   EEGLAB's standard eventtype/eventlatency/eventduration)
        # - IndexError: list index out of range
        #   (empty event/epoch arrays but trials > 1)
        # - KeyError: missing expected field in epoch struct
        logger.warning(
            "read_epochs_eeglab failed (%s: %s) — "
            "falling back to manual epoch data loading.",
            type(epoch_error).__name__,
            epoch_error,
        )

    if used_mne:
        # --- MNE path: extract data + metadata from Epochs object ---
        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        total_samples = n_epochs * n_times
        sfreq = epochs.info["sfreq"]

        # Concatenate epochs into continuous data (channel-major order).
        # epochs._data is (n_epochs, n_channels, n_times) — already in memory
        # since read_epochs_eeglab always preloads. We access it directly to
        # avoid the copy that get_data() would create.
        epoch_data = epochs._data  # no copy — direct reference
        data_concat = np.empty((n_channels, total_samples), dtype=epoch_data.dtype)
        for i in range(n_epochs):
            data_concat[:, i * n_times : (i + 1) * n_times] = epoch_data[i]

        # read_epochs_eeglab already converts µV → V, so no extra scaling.
        info = epochs.info.copy()
        event_id = epochs.event_id
        events = epochs.events.copy() if epochs.event_id else None
        tmin = epochs.tmin

        # Free the epochs object and its data before creating RawArray
        del epoch_data, epochs

    else:
        # --- Manual path: read raw data via scipy from .set/.fdt ---
        from scipy.io import loadmat as _loadmat

        # Use struct_as_record=False for attribute-style access,
        # and squeeze_me=True to collapse singleton dimensions.
        mat = _loadmat(str(set_path), squeeze_me=True, struct_as_record=False)

        # Handle both nested (mat["EEG"].field) and flat (mat["field"]) layouts.
        if "EEG" in mat:
            eeg = mat["EEG"]
            _get = lambda key: getattr(eeg, key, None)  # noqa: E731
        elif "nbchan" in mat and "pnts" in mat:
            _get = lambda key: mat.get(key)  # noqa: E731
        else:
            raise ValueError(
                f"Cannot parse epoched .set file '{set_path.name}': "
                "neither 'EEG' struct nor flat fields found"
            )

        n_channels = int(_get("nbchan"))
        n_times = int(_get("pnts"))  # samples per epoch
        n_epochs = int(_get("trials"))
        sfreq = float(_get("srate"))

        if n_epochs <= 0 or n_times <= 0 or n_channels <= 0:
            raise ValueError(
                f"Invalid dimensions in '{set_path.name}': "
                f"nbchan={n_channels}, pnts={n_times}, trials={n_epochs}"
            )

        total_samples = n_times * n_epochs

        # --- Resolve data array ---
        data_ref = _get("data")

        if isinstance(data_ref, str):
            # External .fdt file: the own one first (on004348)
            fdt_path = eeglab_io.own_data_file(set_path, data_ref)
            if not fdt_path.exists():
                fdt_path = set_path.with_suffix(".fdt")
            if not fdt_path.exists():
                raise ValueError(
                    f"No .fdt file found for '{set_path.name}' "
                    f"(expected '{fdt_path.name}')"
                )
            expected_size = n_channels * n_times * n_epochs * _EEGLAB_BYTES_PER_SAMPLE
            actual_size = fdt_path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"FDT size mismatch for epoched data: expected {expected_size} "
                    f"bytes ({n_channels} ch x {n_times} pts x {n_epochs} trials x 4), "
                    f"got {actual_size}"
                )
            raw_data = np.fromfile(str(fdt_path), dtype=np.float32)
        elif isinstance(data_ref, np.ndarray):
            # Embedded data — may be 2-D (nbchan, pnts*trials) or 3-D
            raw_data = data_ref.astype(np.float32).ravel(order="F")
        else:
            raise ValueError(
                f"No data source in '{set_path.name}': "
                f"data field is {type(data_ref).__name__}"
            )

        expected_elements = n_channels * n_times * n_epochs
        if raw_data.size != expected_elements:
            raise ValueError(
                f"Data element count mismatch: expected {expected_elements} "
                f"({n_channels} x {n_times} x {n_epochs}), got {raw_data.size}"
            )

        # Reshape: EEGLAB stores as (nbchan, pnts, trials) in Fortran order.
        # We want (nbchan, pnts * trials) — continuous concatenation of epochs.
        data_2d = raw_data.reshape((n_channels, total_samples), order="F")

        # Scale µV → V (manual path only; MNE path already converts)
        data_concat = data_2d.astype(np.float64) * _EEGLAB_UV_TO_V

        # --- Resolve channel names ---
        ch_names = None
        bids_info = _read_bids_channels_tsv(set_path.parent)
        if bids_info and len(bids_info[0]) == n_channels:
            ch_names, ch_types = bids_info
        else:
            ch_types = None

        if ch_names is None:
            # Try chanlocs from .set
            chanlocs = _get("chanlocs")
            if chanlocs is not None and hasattr(chanlocs, "__len__"):
                names = []
                for cl in np.atleast_1d(chanlocs):
                    label = getattr(cl, "labels", None)
                    if label is not None:
                        if hasattr(label, "item"):
                            label = label.item()
                        names.append(str(label).strip())
                if len(names) == n_channels:
                    ch_names = names

        if ch_names is None:
            ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]

        if ch_types is None:
            ch_types = ["eeg"] * n_channels

        ch_names = _deduplicate_channel_names(ch_names)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        event_id = None
        events = None
        tmin = 0.0

    # ==== Shared: degenerate-data guard ====
    min_total_samples = max(int(2 * sfreq), 100)
    if total_samples < min_total_samples:
        raise ValueError(
            f"Epoched EEGLAB file '{set_path.name}' has only "
            f"{n_epochs} epoch(s) x {n_times} samples = {total_samples} "
            f"total samples (< {min_total_samples} minimum at {sfreq} Hz). "
            f"Recording is too short for processing."
        )

    # ==== Shared: create RawArray ====
    raw = mne.io.RawArray(data_concat, info, verbose="ERROR")

    # Recalculate event sample positions for the concatenated timeline.
    # The original events[:, 0] references the ORIGINAL continuous recording,
    # but our raw is the concatenated epoch data.  Each epoch i occupies
    # samples [i*n_times, (i+1)*n_times) and the trigger sits at
    # -tmin seconds from the epoch start.
    if event_id:
        offset = int(round(-tmin * sfreq))
        new_events = events
        new_events[:, 0] = np.arange(n_epochs) * n_times + offset
        event_desc = {v: k for k, v in event_id.items()}
        annotations = mne.annotations_from_events(
            events=new_events,
            event_desc=event_desc,
            sfreq=sfreq,
            orig_time=raw.info.get("meas_date"),
        )
        raw.set_annotations(annotations)

    fallback_tag = "" if used_mne else " [manual fallback]"
    raw.info["description"] = (
        f"Converted from {n_epochs} epochs ({n_times / sfreq:.3f}s each){fallback_tag}"
    )

    logger.warning(
        "Loaded EEGLAB epoch file as continuous raw%s: %s "
        "(%d epochs x %d ch x %d samples = %d total samples).",
        fallback_tag,
        set_path.name,
        n_epochs,
        n_channels,
        n_times,
        total_samples,
    )
    return raw


# EEGLAB .fdt stores single-precision floats (4 bytes per sample)
_EEGLAB_BYTES_PER_SAMPLE = 4
# EEGLAB stores data in microvolts; MNE expects volts
_EEGLAB_UV_TO_V = 1e-6


def _eeglab_get_first_eeg(mat_root: dict) -> dict | None:
    """Get the first EEG struct from a loaded .set (EEG or ALLEEG) as a dict."""
    if "ALLEEG" in mat_root:
        arr = np.atleast_1d(np.asarray(mat_root["ALLEEG"], dtype=object))
        if arr.size == 0:
            return None
        first = arr.flat[0]
    elif "EEG" in mat_root:
        first = mat_root["EEG"]
    elif isinstance(mat_root, dict) and "nbchan" in mat_root and "pnts" in mat_root:
        # pymatreader / v7.3 sometimes returns EEG fields at top level (no EEG/ALLEEG key)
        return mat_root
    else:
        return None
    if isinstance(first, dict):
        out = first.get("EEG", first)
        return out if isinstance(out, dict) else first
    if hasattr(first, "dtype") and getattr(first.dtype, "names", None):
        out = {n: first[n] for n in first.dtype.names}
        nested = out.get("EEG")
        if isinstance(nested, dict):
            return nested
        if hasattr(nested, "dtype") and getattr(nested.dtype, "names", None):
            return {n: nested[n] for n in nested.dtype.names}
        return out
    if hasattr(first, "_fieldnames"):
        return {n: getattr(first, n) for n in first._fieldnames}
    return None


def _eeglab_load_first_eeg(set_path: Path) -> dict | None:
    """Load a .set file and return the first EEG struct as a dict, or None."""
    try:
        mat = loadmat(
            str(set_path),
            squeeze_me=True,
            mat_dtype=False,
            struct_as_record=False,
        )
        eeg = _eeglab_get_first_eeg(mat)
        if eeg is not None and isinstance(eeg, dict):
            return eeg
    except Exception as e:
        logger.debug("Could not read .set with scipy: %s", e)
    try:
        mat = mne_eeglab._readmat(str(set_path), preload=True)
        eeg = _eeglab_get_first_eeg(mat)
        return eeg if isinstance(eeg, dict) else None
    except Exception as e:
        logger.debug("Could not read .set with MNE _readmat: %s", e)
        return None


def _eeglab_fdt_path(set_path: Path, eeg: dict) -> Path | None:
    """Return the path to the companion .fdt when data is external (string), else None."""
    data_ref = eeg.get("data")
    if not isinstance(data_ref, str):
        return None
    return eeglab_io.own_data_file(set_path, data_ref)


def _eeglab_ch_names_from_eeg(eeg: dict, nbchan: int) -> list[str]:
    """Build channel names from EEG struct chanlocs; fallback to CH1, CH2, ..."""
    chanlocs = eeg.get("chanlocs", [])
    n_loc = len(np.atleast_1d(chanlocs)) if chanlocs is not None else 0
    ch_names = []
    for i in range(nbchan):
        if i < n_loc:
            c = np.atleast_1d(chanlocs)[i]
            try:
                if isinstance(c, dict) and "labels" in c:
                    lab = c["labels"]
                elif (
                    hasattr(c, "dtype")
                    and getattr(c.dtype, "names", None)
                    and "labels" in (c.dtype.names or ())
                ):
                    lab = c["labels"]
                else:
                    lab = None
                if lab is not None:
                    ch_names.append(str(lab).strip())
                    continue
            except Exception:
                pass
        ch_names.append(f"CH{i + 1}")
    return ch_names


def _load_raw_eeglab_alleeg(filepath: Path):
    """Load an EEGLAB .set that contains an ALLEEG array (use first dataset).

    Fallback when MNE raises NotImplementedError for ALLEEG. Returns an
    MNE Raw instance built from the first EEG in the array. Supports
    both v5/v7 and v7.3 .set files via the shared loader.

    Parameters
    ----------
    filepath : Path
        Path to the .set file.

    Returns
    -------
    mne.io.Raw
        Raw instance (preload=True).

    """
    eeg = _eeglab_load_first_eeg(filepath)
    if eeg is None:
        raise ValueError("No EEG or ALLEEG found in .set file")

    nbchan = int(eeg.get("nbchan", 1))
    pnts = int(eeg.get("pnts", 1))
    srate = float(eeg.get("srate", 1.0))
    ch_names = _eeglab_ch_names_from_eeg(eeg, nbchan)

    data_ref = eeg.get("data")
    if isinstance(data_ref, str):
        data_fname = _eeglab_fdt_path(filepath, eeg)
        data = np.fromfile(data_fname, dtype="<f4", count=nbchan * pnts)
        present = data.size // nbchan  # complete samples; never pad with zeros
        if present < pnts:
            if present == 0:
                raise ValueError(f"{Path(data_fname).name} holds no complete sample")
            logger.warning(
                "Truncated EEGLAB data file %s: %d of %d samples present; "
                "reading those only.",
                Path(data_fname).name,
                present,
                pnts,
            )
            pnts = present
        data = data[: nbchan * pnts].astype(np.float64)
        data = data.reshape(nbchan, pnts, order="F")
    else:
        data = np.asarray(data_ref, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(nbchan, -1, order="F")
        elif data.shape[0] != nbchan or data.shape[1] != pnts:
            data = data[:nbchan, :pnts]

    ch_names = _deduplicate_channel_names(ch_names)
    info = mne.create_info(ch_names, srate, ch_types="eeg")
    raw = mne.io.RawArray(data * _EEGLAB_UV_TO_V, info, verbose="ERROR")
    logger.warning(
        "Loaded ALLEEG .set using first dataset: %s",
        filepath.name,
    )
    return raw
