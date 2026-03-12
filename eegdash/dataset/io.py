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
from time import strptime
from typing import Any

import mne
import numpy as np
from dateutil.parser import parse as _dateutil_parse
from dateutil.parser import parserinfo
from mne.io.eeglab import _eeglab as mne_eeglab
from scipy.io import loadmat, savemat

from ..logging import logger


class _MultiLocaleParserInfo(parserinfo):
    """``parserinfo`` subclass that recognises non-English month abbreviations.

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


_ANNEX_KEY_RE = re.compile(r"^(SHA256E|MD5E)-s\d+--[0-9a-f]+\.", flags=re.IGNORECASE)


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

        is_annex_key = bool(_ANNEX_KEY_RE.match(old_val))

        # If the pointed file exists AND it's not an annex key, keep it.
        # Annex keys are always rewritten: the key-based filename only
        # resolves inside a git-annex repo, so outside that context
        # (e.g. S3 download cache) it will always fail.
        if not is_annex_key and (self.data_dir / old_val).exists():
            return match.group(0)

        # Determine expected extension
        ext = ".vmrk" if key == "MarkerFile" else ".eeg"

        # Strategy 1: Use BIDS filename (same stem as .vhdr).
        # For annex keys we always rewrite — the target file need not
        # exist yet (a VMRK stub may be generated right after repair).
        bids_name = self.vhdr_path.with_suffix(ext).name
        if not _ANNEX_KEY_RE.match(bids_name) and (
            is_annex_key or (self.data_dir / bids_name).exists()
        ):
            self.changes = True
            logger.info(
                f"Auto-repairing {self.vhdr_path.name}: {key}={old_val} -> {bids_name}"
            )
            return f"{key}={bids_name}"

        # Strategy 2: Fuzzy match - find best matching file with same extension
        # This handles typos (rsub- vs sub-, sternbeg vs sternberg) and
        # BIDS renames where original filename was completely different
        fuzzy_match = _find_best_matching_file(self.data_dir, old_val, ext)
        if (
            fuzzy_match
            and (self.data_dir / fuzzy_match).exists()
            and not _ANNEX_KEY_RE.match(fuzzy_match)
        ):
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


def _repair_vhdr_missing_markerfile(vhdr_path: Path) -> bool:
    """Add a missing MarkerFile entry to a VHDR ``[Common Infos]`` section.

    Some BrainVision files omit ``MarkerFile`` entirely, which causes
    ``configparser.NoOptionError`` in MNE's reader.  This inserts the
    entry and generates a VMRK stub if the marker file does not exist.

    Parameters
    ----------
    vhdr_path : Path
        Path to the VHDR file.

    Returns
    -------
    bool
        True if the VHDR was modified, False otherwise.

    """
    if not vhdr_path.exists() or vhdr_path.suffix != ".vhdr":
        return False

    try:
        content = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    # Check if MarkerFile is already present (case-insensitive)
    if re.search(r"(?i)^\s*MarkerFile\s*=", content, re.MULTILINE):
        return False

    # Find [Common Infos] section and insert MarkerFile after it
    match = re.search(r"(?i)(\[Common Infos\][^\[]*)", content)
    if not match:
        return False

    vmrk_name = vhdr_path.with_suffix(".vmrk").name
    section = match.group(1)
    # Insert MarkerFile at end of section (before next section or EOF)
    new_section = section.rstrip("\n") + f"\nMarkerFile={vmrk_name}\n"
    new_content = content[: match.start()] + new_section + content[match.end() :]

    try:
        vhdr_path.write_text(new_content, encoding="utf-8")
        logger.info("Added missing MarkerFile=%s to %s", vmrk_name, vhdr_path.name)
    except Exception as e:
        logger.warning("Failed to write MarkerFile to VHDR: %s", e)
        return False

    # Ensure the VMRK file exists
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    if not vmrk_path.exists():
        _generate_vmrk_stub(vmrk_path, vhdr_path.name)

    return True


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
            _COORD_PREFIX = {"eeg": "EEG", "ieeg": "iEEG", "meg": "MEG"}
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
                    logger.debug(f"Created coordsystem symlink: {dst} -> {rel_target}")
                except Exception as e:
                    logger.warning(f"Failed to link coordsystem: {e}")
        else:
            # No coordsystem.json found anywhere — generate a minimal one
            # Infer the coordinate system from the electrodes filename
            # (e.g. "sub-01_ses-01_space-CapTrak_electrodes.tsv")
            _generate_coordsystem_json(electrodes_files[0], datatype=datatype)

    except Exception as e:
        logger.warning(f"Error checking coordsystem symlinks: {e}")


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
        _COORD_PREFIX = {"eeg": "EEG", "ieeg": "iEEG", "meg": "MEG"}
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


# Match digits separated by a comma that looks like a European decimal separator,
# e.g. "4,988" → "4.988".  This intentionally treats ALL digit,digit patterns as
# decimals.  Thousands separators like "10,000" would also be rewritten, but in
# practice European-locale datasets that use comma-as-decimal use dot or space for
# thousands grouping, so the ambiguity does not arise in real BIDS TSV files.
_DECIMAL_COMMA_RE = re.compile(r"(?<!\w)(\d+),(\d+)(?!\w)")


def _repair_tsv_decimal_separators(data_dir: Path) -> bool:
    """Fix TSV files using comma as decimal separator instead of dot.

    Some European datasets use comma as decimal separator (e.g., ``5,004``
    instead of ``5.004``).  This converts commas to dots in numeric columns
    of TSV files that are likely to contain floating-point values.

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

    # Only target TSV files that typically contain numeric columns
    target_patterns = ("*_events.tsv", "*_electrodes.tsv", "*_channels.tsv")
    tsv_paths = []
    for pat in target_patterns:
        tsv_paths.extend(data_dir.glob(pat))

    if not tsv_paths:
        return False

    repaired_any = False
    for tsv_path in tsv_paths:
        try:
            content = tsv_path.read_text(encoding="utf-8")
        except Exception:
            continue

        new_content = _DECIMAL_COMMA_RE.sub(r"\1.\2", content)
        if new_content != content:
            try:
                tsv_path.write_text(new_content, encoding="utf-8")
                logger.info(
                    f"Repaired decimal separators (comma -> dot): {tsv_path.name}"
                )
                repaired_any = True
            except Exception as e:
                logger.warning(f"Failed to write repaired TSV {tsv_path.name}: {e}")

    return repaired_any


_TRAILING_FIELD_WS_RE = re.compile(r" +(?=[\t\r\n]|\Z)")


def _repair_tsv_na_whitespace(data_dir: Path) -> bool:
    """Strip trailing spaces from fields in TSV files.

    Some datasets have ``n/a`` values padded with spaces (e.g.,
    ``'n/a      '``).  MNE-BIDS only recognizes exact ``n/a`` as the BIDS
    missing-value marker, so padded variants cause ``float()`` conversion
    errors.  This strips trailing spaces before every tab delimiter,
    carriage-return, or line-feed in all TSV files under *data_dir*,
    preserving original line endings.

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
            content = tsv_path.read_text(encoding="utf-8")
        except Exception:
            continue

        new_content = _TRAILING_FIELD_WS_RE.sub("", content)
        if new_content != content:
            try:
                tsv_path.write_text(new_content, encoding="utf-8")
                logger.info(
                    "Stripped trailing whitespace from TSV fields: %s",
                    tsv_path.name,
                )
                repaired_any = True
            except Exception as e:
                logger.warning("Failed to write repaired TSV %s: %s", tsv_path.name, e)

    return repaired_any


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

    for sibling in dataset_root.rglob("*.vhdr"):
        if sibling == vhdr_path:
            continue
        try:
            content = sibling.read_text(encoding="utf-8")
            if "[Common Infos]" not in content and "[Common infos]" not in content:
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

            # Also generate VMRK stub if missing
            vmrk_path = vhdr_path.with_suffix(".vmrk")
            if not vmrk_path.exists():
                _generate_vmrk_stub(vmrk_path, vhdr_path.name)

            return True
        except Exception:
            continue

    logger.warning("No valid sibling VHDR found for %s", vhdr_path.name)
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
    if _repair_scans_tsv_timestamps(data_dir):
        repairs_made = True

    return repairs_made


def _repair_scans_tsv_timestamps(data_dir: Path) -> bool:
    """Fix invalid timestamps in ``*_scans.tsv`` files.

    ``mne_bids.read_raw_bids()`` parses ``acq_time`` values via
    ``datetime.fromisoformat()``.  Some datasets contain malformed
    timestamps (seconds >= 60, NaN values, truncated strings) that crash
    the parser.  This function replaces any unparsable ``acq_time``
    entries with ``"n/a"`` so that loading can proceed.

    Parameters
    ----------
    data_dir : Path
        The directory containing the data file (e.g. ``sub-01/eeg/``).
        Scans files are searched in both ``data_dir`` *and* its parent
        (session-level ``scans.tsv``).

    Returns
    -------
    bool
        True if any timestamps were repaired, False otherwise.

    """
    from datetime import datetime

    search_dirs = [data_dir]
    if data_dir.parent.exists():
        search_dirs.append(data_dir.parent)

    repaired_any = False
    seen: set[Path] = set()

    for d in search_dirs:
        for scans_tsv in list(d.glob("*_scans.tsv")) + list(d.glob("scans.tsv")):
            if scans_tsv in seen:
                continue
            seen.add(scans_tsv)

            try:
                lines = scans_tsv.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            if not lines:
                continue

            header = lines[0].split("\t")
            try:
                acq_idx = header.index("acq_time")
            except ValueError:
                continue

            new_lines = [lines[0]]
            changed = False
            for line in lines[1:]:
                cols = line.split("\t")
                if acq_idx < len(cols):
                    ts = cols[acq_idx].strip()
                    if ts and ts.lower() != "n/a":
                        try:
                            datetime.fromisoformat(ts)
                        except (ValueError, TypeError):
                            cols[acq_idx] = "n/a"
                            changed = True
                new_lines.append("\t".join(cols))

            if changed:
                try:
                    scans_tsv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    logger.info(f"Repaired invalid timestamps in {scans_tsv.name}")
                    repaired_any = True
                except Exception as e:
                    logger.warning(
                        f"Failed to write repaired scans.tsv {scans_tsv.name}: {e}"
                    )

    return repaired_any


def _repair_events_tsv_nan_samples(data_dir: Path) -> bool:
    """Drop rows with NaN onset/sample from ``*_events.tsv`` files.

    Some EEGLAB datasets export events with NaN onset and sample values.
    ``mne_bids.read_raw_bids()`` tries to convert the ``sample`` column
    to ``int`` and crashes on NaN.  This removes those rows so loading
    can proceed — the underlying data is fine, only the event markers
    are incomplete.

    Parameters
    ----------
    data_dir : Path
        Directory containing the events TSV files.

    Returns
    -------
    bool
        True if any files were repaired, False otherwise.

    """
    if not data_dir.exists():
        return False

    repaired_any = False
    for events_tsv in data_dir.glob("*_events.tsv"):
        try:
            lines = events_tsv.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        if len(lines) < 2:
            continue

        header = lines[0]
        cols = header.split("\t")

        # Find onset or sample column
        onset_idx = None
        for col_name in ("onset", "sample"):
            if col_name in cols:
                onset_idx = cols.index(col_name)
                break
        if onset_idx is None:
            continue

        new_lines = [header]
        dropped = 0
        for line in lines[1:]:
            fields = line.split("\t")
            if onset_idx < len(fields):
                val = fields[onset_idx].strip().lower()
                if val in ("nan", "n/a", ""):
                    dropped += 1
                    continue
            new_lines.append(line)

        if dropped > 0:
            try:
                events_tsv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                logger.info(
                    "Repaired %s: dropped %d rows with NaN onset/sample",
                    events_tsv.name,
                    dropped,
                )
                repaired_any = True
            except Exception as e:
                logger.warning("Failed to write repaired %s: %s", events_tsv.name, e)

    return repaired_any


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
            content = tsv_path.read_text(encoding="utf-8-sig")
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
    import pandas as pd

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
        logger.warning(f"Failed to write repaired participants.tsv: {e}")
        return False


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
    _EXT_TO_READER = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
        ".cnt": mne.io.read_raw_cnt,
        ".ds": mne.io.read_raw_ctf,
    }

    ext = filepath.suffix.lower()
    reader = _EXT_TO_READER.get(ext)
    if reader is None:
        raise ValueError(
            f"No direct reader for extension {ext!r}. "
            f"Supported: {sorted(_EXT_TO_READER)}"
        )

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

    return reader(str(filepath), preload=False, verbose="ERROR")


def _parse_set_metadata(set_path: Path) -> dict:
    """Extract metadata from an EEGLAB .set file.

    Returns dict with keys: srate, nbchan, pnts, ch_names, data (if embedded).
    Raises ValueError if the file cannot be parsed.
    """
    import numpy as np

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
    with open(tsv_path, newline="") as f:
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
    import mne
    import numpy as np

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

    Parameters
    ----------
    set_path : Path
        Path to the EEGLAB .set file.

    Returns
    -------
    mne.io.BaseRaw
        Continuous Raw object created from concatenated epochs.

    """
    import mne
    import numpy as np

    set_path = Path(set_path)
    logger.info("Loading EEGLAB epoch file via read_epochs_eeglab: %s", set_path.name)

    epochs = mne.io.read_epochs_eeglab(str(set_path), verbose="ERROR")
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    # Concatenate epochs into continuous data (channel-major order)
    data_concat = np.ascontiguousarray(data.transpose(1, 0, 2)).reshape(n_channels, -1)

    # read_epochs_eeglab already converts µV → V, so no extra scaling needed
    raw = mne.io.RawArray(data_concat, epochs.info, verbose="ERROR")

    # Recalculate event sample positions for the concatenated timeline.
    # epochs.events[:, 0] references the ORIGINAL continuous recording,
    # but our raw is the concatenated epoch data.  Each epoch i occupies
    # samples [i*n_times, (i+1)*n_times) and the trigger sits at
    # -tmin seconds from the epoch start.
    if epochs.event_id:
        offset = int(round(-epochs.tmin * epochs.info["sfreq"]))
        new_events = epochs.events.copy()
        new_events[:, 0] = np.arange(n_epochs) * n_times + offset
        event_desc = {v: k for k, v in epochs.event_id.items()}
        annotations = mne.annotations_from_events(
            events=new_events,
            event_desc=event_desc,
            sfreq=epochs.info["sfreq"],
            orig_time=raw.info.get("meas_date"),
        )
        raw.set_annotations(annotations)

    raw.info["description"] = (
        f"Converted from {n_epochs} epochs ({n_times / epochs.info['sfreq']:.3f}s each)"
    )

    logger.warning(
        "Loaded EEGLAB epoch file as continuous raw: %s "
        "(%d epochs × %d ch × %d samples → %d total samples).",
        set_path.name,
        n_epochs,
        n_channels,
        n_times,
        n_epochs * n_times,
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
    p = set_path.parent / data_ref
    if not p.exists():
        p = set_path.with_suffix(".fdt")
    return p


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


def _repair_eeglab_fdt(set_path: Path) -> bool:
    """Update .set header so pnts matches the actual .fdt size (no data change).

    When the companion .fdt is truncated, rewrites the .set file so the
    declared number of samples (pnts) matches the bytes in the .fdt. The
    standard MNE reader can then load the file; no padding, no custom loader.

    Reads with scipy first (v5/v7), then MNE _readmat (v7.3). Always writes
    a flat v5/v7 .set via scipy so all formats are supported.

    Parameters
    ----------
    set_path : Path
        Path to the .set file.

    Returns
    -------
    bool
        True if the .set was modified, False otherwise.

    """
    if not set_path.exists() or set_path.suffix.lower() != ".set":
        return False

    eeg = _eeglab_load_first_eeg(set_path)
    if eeg is None:
        return False

    nbchan = int(eeg.get("nbchan", 1))
    pnts = int(eeg.get("pnts", 1))
    fdt_path = _eeglab_fdt_path(set_path, eeg)
    if fdt_path is None or not fdt_path.exists():
        return False
    expected_bytes = nbchan * pnts * _EEGLAB_BYTES_PER_SAMPLE
    fdt_size = fdt_path.stat().st_size
    if fdt_size >= expected_bytes:
        return False

    actual_pnts = fdt_size // _EEGLAB_BYTES_PER_SAMPLE // nbchan
    if actual_pnts <= 0:
        return False

    srate = float(eeg.get("srate", 1.0)) or 1.0
    repaired = dict(eeg)
    repaired["pnts"] = actual_pnts
    repaired["xmax"] = (
        actual_pnts - 1
    ) / srate  # EEGLAB xmax is last time point in sec

    try:
        savemat(str(set_path), repaired, do_compression=False)
        logger.info(
            "Repaired EEGLAB .set header: %s (pnts %s -> %s).",
            set_path.name,
            pnts,
            actual_pnts,
        )
        return True
    except Exception as e:
        logger.warning("Failed to rewrite .set %s: %s", set_path.name, e)
        return False


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
        if data.size != nbchan * pnts:
            data = np.pad(
                data.astype(np.float64),
                (0, nbchan * pnts - data.size),
                mode="constant",
                constant_values=0,
            )
        else:
            data = data.astype(np.float64)
        data = data.reshape(nbchan, pnts, order="F")
    else:
        data = np.asarray(data_ref, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(nbchan, -1, order="F")
        elif data.shape[0] != nbchan or data.shape[1] != pnts:
            data = data[:nbchan, :pnts]

    info = mne.create_info(ch_names, srate, ch_types="eeg")
    raw = mne.io.RawArray(data * _EEGLAB_UV_TO_V, info, verbose="ERROR")
    logger.warning(
        "Loaded ALLEEG .set using first dataset: %s",
        filepath.name,
    )
    return raw
