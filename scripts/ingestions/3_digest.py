#!/usr/bin/env python3
"""Digest BIDS datasets and generate JSON records for MongoDB.

This script produces two types of documents:
- **Dataset**: One per dataset (metadata for discovery/filtering)
- **Record**: One per file (metadata for loading data)

Usage:
    # Digest all cloned datasets
    python 3_digest.py --input data/cloned --output digestion_output

    # Digest specific datasets
    python 3_digest.py --input data/cloned --output digestion_output --datasets ds002718 ds005506

    # Digest with parallel processing
    python 3_digest.py --input data/cloned --output digestion_output --workers 4
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from _fingerprint import fingerprint_from_files, fingerprint_from_manifest
from tqdm import tqdm

from eegdash.records import create_dataset, create_record

# Storage configuration per source
# Each source has a backend type and base URL pattern
STORAGE_CONFIGS = {
    "openneuro": {"backend": "s3", "base": "s3://openneuro.org"},
    "nemar": {"backend": "s3", "base": "s3://nemar"},  # NEMAR uses S3 too
    "gin": {"backend": "https", "base": "https://gin.g-node.org"},
    "figshare": {"backend": "https", "base": "https://figshare.com/ndownloader/files"},
    "zenodo": {"backend": "https", "base": "https://zenodo.org/records"},
    "osf": {"backend": "https", "base": "https://files.osf.io"},
    "scidb": {"backend": "https", "base": "https://www.scidb.cn"},
    "datarn": {"backend": "webdav", "base": "https://webdav.data.ru.nl"},
}

# Default config for unknown sources
DEFAULT_STORAGE_CONFIG = {"backend": "https", "base": "https://unknown"}

# Datasets to explicitly ignore during ingestion
EXCLUDED_DATASETS = {
    "ABUDUKADI",
    "ABUDUKADI_2",
    "ABUDUKADI_3",
    "ABUDUKADI_4",
    "AILIJIANG",
    "AILIJIANG_3",
    "AILIJIANG_4",
    "AILIJIANG_5",
    "AILIJIANG_7",
    "AILIJIANG_8",
    "BAIHETI",
    "BAIHETI_2",
    "BAIHETI_3",
    "BIAN_3",
    "BIN_27",
    "BLIX",
    "BOJIN",
    "BOUSSAGOL",
    "AISHENG",
    "ACHOLA",
    "ANASHKIN",
    "ANJUM",
    "BARBIERI",
    "BIN_8",
    "BIN_9",
    "BING_4",
    "BING_8",
    "BOWEN_4",
    "AZIZAH",
    "BAO",
    "BAO-YOU",
    "BAO_2",
    "BENABBOU",
    "BING",
    "BOXIN",
}


def get_storage_config(source: str) -> dict:
    """Get storage configuration for a source."""
    return STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)


def get_storage_base(dataset_id: str, source: str) -> str:
    """Get storage base URL for a dataset."""
    config = get_storage_config(source)
    return f"{config['base']}/{dataset_id}"


def get_storage_backend(source: str) -> str:
    """Get storage backend type for a source."""
    config = get_storage_config(source)
    return config["backend"]


def detect_source(dataset_dir: Path) -> str:
    """Detect source from manifest.json or dataset structure."""
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest.get("source", "openneuro")
        except Exception:
            pass

    # Fallback: check dataset_id pattern
    dataset_id = dataset_dir.name
    if dataset_id.startswith("ds"):
        return "openneuro"
    elif "EEGManyLabs" in dataset_id:
        return "gin"

    return "openneuro"


def extract_dataset_metadata(
    bids_dataset,
    dataset_id: str,
    source: str,
    digested_at: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Extract Dataset-level metadata from a BIDS dataset.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object
    dataset_id : str
        Dataset identifier
    source : str
        Source name (openneuro, nemar, etc.)
    digested_at : str
        ISO 8601 timestamp
    metadata : dict | None
        Optional metadata from source (e.g. from GraphQL/API)

    Returns
    -------
    dict
        Dataset schema compliant metadata

    """
    bids_root = Path(bids_dataset.bidsdir)
    metadata = metadata or {}

    # Read dataset_description.json
    description = {}
    desc_path = bids_root / "dataset_description.json"
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                description = json.load(f)
        except Exception:
            pass

    # Read README file
    readme = None
    for readme_name in ["README", "README.md", "README.txt", "readme", "readme.md"]:
        readme_path = bids_root / readme_name
        if readme_path.exists():
            try:
                raw_readme = readme_path.read_text(encoding="utf-8")
                # Clean up readme: strip lines and join
                readme = "\n".join(
                    [line.rstrip() for line in raw_readme.splitlines() if line.strip()]
                )
                break
            except Exception:
                pass

    # Extract basic metadata
    name = description.get("Name", dataset_id)
    bids_version = description.get("BIDSVersion")
    license_info = description.get("License")
    authors = description.get("Authors", [])
    funding = description.get("Funding", [])
    dataset_doi = description.get("DatasetDOI")

    # Get files and detect modalities
    files = bids_dataset.get_files()
    modalities = set()
    tasks = set()
    tasks = set()
    sessions = set()
    subjects = set()

    for bids_file in files:
        mod = bids_dataset.get_bids_file_attribute("modality", bids_file)
        mod_canon = normalize_modality(mod)
        if mod_canon:
            modalities.add(mod_canon)

        # Only collect subjects/tasks/sessions if file belongs to a neuro modality
        if mod_canon in NEURO_MODALITIES:
            task = bids_dataset.get_bids_file_attribute("task", bids_file)
            if task:
                tasks.add(task)
            session = bids_dataset.get_bids_file_attribute("session", bids_file)
            if session:
                sessions.add(session)
            subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
            if subject:
                subjects.add(subject)

    # Determine recording modalities (list of canonical names)
    # Filter to only include NEURO_MODALITIES for the summary field
    recording_modalities = sorted(
        list({m for m in modalities if m in NEURO_MODALITIES})
    )
    if not recording_modalities:
        recording_modalities = ["eeg"]

    # Read participants.tsv for demographics
    subjects_count = 0
    ages = []
    sex_distribution = {}
    handedness_distribution = {}

    participants_path = bids_root / "participants.tsv"
    if participants_path.exists():
        try:
            df = pd.read_csv(
                participants_path, sep="\t", dtype="string", keep_default_na=False
            )
            subjects_count = len(df)

            # Extract ages
            age_col = None
            for col in ["age", "Age", "AGE"]:
                if col in df.columns:
                    age_col = col
                    break
            if age_col:
                for val in df[age_col]:
                    try:
                        age = int(float(val))
                        if 0 < age < 120:
                            ages.append(age)
                    except (ValueError, TypeError):
                        pass

            # Extract sex distribution
            sex_col = None
            for col in ["sex", "Sex", "SEX", "gender", "Gender"]:
                if col in df.columns:
                    sex_col = col
                    break
            if sex_col:
                for val in df[sex_col]:
                    val_lower = str(val).lower().strip()
                    if val_lower in ("m", "male"):
                        sex_distribution["m"] = sex_distribution.get("m", 0) + 1
                    elif val_lower in ("f", "female"):
                        sex_distribution["f"] = sex_distribution.get("f", 0) + 1
                    elif val_lower and val_lower not in (
                        "n/a",
                        "na",
                        "nan",
                        "unknown",
                        "",
                    ):
                        sex_distribution["o"] = sex_distribution.get("o", 0) + 1

            # Extract handedness distribution
            hand_col = None
            for col in ["handedness", "Handedness", "hand", "Hand"]:
                if col in df.columns:
                    hand_col = col
                    break
            if hand_col:
                for val in df[hand_col]:
                    val_lower = str(val).lower().strip()
                    if val_lower in ("r", "right"):
                        handedness_distribution["r"] = (
                            handedness_distribution.get("r", 0) + 1
                        )
                    elif val_lower in ("l", "left"):
                        handedness_distribution["l"] = (
                            handedness_distribution.get("l", 0) + 1
                        )
                    elif val_lower in ("a", "ambidextrous"):
                        handedness_distribution["a"] = (
                            handedness_distribution.get("a", 0) + 1
                        )

        except Exception:
            pass

    # Count subjects from directories if participants.tsv based count is inconsistent or we prefer file based
    # User request: "count only subject from the modalities that are validated"
    # So we should prioritize the count derived from valid files (len(subjects))
    if subjects:
        subjects_count = len(subjects)
    elif subjects_count == 0:
        subjects_count = len(
            [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    # Check for derivatives (processed data)
    data_processed = (bids_root / "derivatives").exists()

    # Build source URL
    source_url = None
    if source == "openneuro":
        source_url = f"https://openneuro.org/datasets/{dataset_id}"
    elif source == "nemar":
        source_url = f"https://nemar.org/dataexplorer/detail/{dataset_id}"
    elif source == "gin":
        source_url = f"https://gin.g-node.org/EEGManyLabs/{dataset_id}"

    # Extract timestamps from metadata if available
    dataset_created_at = metadata.get("dataset_created_at")
    dataset_modified_at = metadata.get("dataset_modified_at")
    senior_author = metadata.get("senior_author")
    contact_info = metadata.get("contact_info")

    if not dataset_modified_at:
        # Try to get from manifest timestamps dict if present
        ts = metadata.get("timestamps", {})
        if isinstance(ts, dict):
            dataset_modified_at = ts.get("dataset_modified_at")
            dataset_created_at = ts.get("dataset_created_at") or dataset_created_at

    # Extract size_bytes
    size_bytes = metadata.get("size_bytes")

    # Create Dataset document
    dataset = create_dataset(
        dataset_id=dataset_id,
        name=name,
        source=source,
        readme=readme,
        recording_modality=recording_modalities,
        datatypes=sorted(modalities) if modalities else recording_modalities,
        bids_version=bids_version,
        license=license_info,
        authors=authors if isinstance(authors, list) else [authors] if authors else [],
        funding=funding if isinstance(funding, list) else [funding] if funding else [],
        dataset_doi=dataset_doi,
        tasks=sorted(tasks),
        sessions=sorted(sessions),
        total_files=len(files),
        size_bytes=size_bytes,
        data_processed=data_processed,
        subjects_count=subjects_count,
        ages=ages,
        age_mean=sum(ages) / len(ages) if ages else None,
        sex_distribution=sex_distribution if sex_distribution else None,
        handedness_distribution=handedness_distribution
        if handedness_distribution
        else None,
        source_url=source_url,
        digested_at=digested_at,
        dataset_created_at=dataset_created_at,
        dataset_modified_at=dataset_modified_at,
        senior_author=senior_author,
        contact_info=contact_info,
    )

    return dict(dataset)


def extract_record(
    bids_dataset,
    bids_file: str,
    dataset_id: str,
    source: str,
    digested_at: str,
) -> dict[str, Any]:
    """Extract Record metadata for a single BIDS file.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object
    bids_file : str
        Path to the BIDS file
    dataset_id : str
        Dataset identifier
    source : str
        Source name (openneuro, nemar, etc.)
    digested_at : str
        ISO 8601 timestamp

    Returns
    -------
    dict
        Record schema compliant metadata

    """
    # Get BIDS entities
    subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
    session = bids_dataset.get_bids_file_attribute("session", bids_file)
    task = bids_dataset.get_bids_file_attribute("task", bids_file)
    run = bids_dataset.get_bids_file_attribute("run", bids_file)
    modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    mod_canon = normalize_modality(modality) or "eeg"

    # Get BIDS relative path (without dataset prefix)
    bids_relpath = str(bids_dataset.get_relative_bidspath(bids_file))
    if bids_relpath.startswith(f"{dataset_id}/"):
        bids_relpath = bids_relpath[len(dataset_id) + 1 :]

    # Determine datatype and suffix
    datatype = mod_canon
    suffix = mod_canon

    # Get storage info
    storage_base = get_storage_base(dataset_id, source)
    storage_backend = get_storage_backend(source)

    # Extract technical metadata from BIDS sidecars via EEGBIDSDataset
    sampling_frequency = bids_dataset.get_bids_file_attribute("sfreq", bids_file)
    nchans = bids_dataset.get_bids_file_attribute("nchans", bids_file)
    ntimes = bids_dataset.get_bids_file_attribute("ntimes", bids_file)

    # Convert sfreq to float and nchans to int if found
    if sampling_frequency:
        sampling_frequency = float(sampling_frequency)
    if nchans:
        nchans = int(nchans)
    if ntimes:
        ntimes = int(ntimes)

    # Extract channel names if channels.tsv exists
    ch_names = None
    try:
        ch_names = bids_dataset.channel_labels(bids_file)
    except Exception:
        pass

    # Find dependency files (channels.tsv, events.tsv, etc.) for storage manifest
    dep_keys = []
    bids_file_path = Path(bids_file)
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]

    # BIDS sidecar files
    for dep_suffix in [
        "_channels.tsv",
        "_events.tsv",
        "_electrodes.tsv",
        "_coordsystem.json",
        "_eeg.json",
    ]:
        dep_file = parent_dir / f"{base_name}{dep_suffix}"
        if dep_file.exists() or dep_file.is_symlink():
            try:
                dep_relpath = dep_file.relative_to(bids_dataset.bidsdir)
                dep_keys.append(str(dep_relpath))
            except ValueError:
                pass

    # Format-specific companion files (e.g., .fdt for EEGLAB .set files)
    ext = bids_file_path.suffix.lower()
    if ext == ".set":
        # EEGLAB .set files may have a companion .fdt file with the same stem
        fdt_file = bids_file_path.with_suffix(".fdt")
        if fdt_file.exists() or fdt_file.is_symlink():
            try:
                fdt_relpath = fdt_file.relative_to(bids_dataset.bidsdir)
                dep_keys.append(str(fdt_relpath))
            except ValueError:
                pass
    elif ext == ".vhdr":
        # BrainVision .vhdr files have .vmrk (markers) and .eeg (data) companions
        # First try standard BIDS names (matching the .vhdr stem)
        found_bv_exts = set()
        for bv_ext in [".vmrk", ".eeg", ".dat"]:
            bv_file = bids_file_path.with_suffix(bv_ext)
            if bv_file.exists() or bv_file.is_symlink():
                try:
                    bv_relpath = bv_file.relative_to(bids_dataset.bidsdir)
                    dep_keys.append(str(bv_relpath))
                    found_bv_exts.add(bv_ext)
                except ValueError:
                    pass

    # Create record using the schema
    record = create_record(
        dataset=dataset_id,
        storage_base=storage_base,
        bids_relpath=bids_relpath,
        subject=subject,
        session=session,
        task=task,
        run=str(run) if run is not None else None,
        dep_keys=dep_keys,
        datatype=datatype,
        suffix=suffix,
        storage_backend=storage_backend,
        recording_modality=[mod_canon],
        ch_names=ch_names,
        sampling_frequency=sampling_frequency,
        nchans=nchans,
        ntimes=ntimes,
        digested_at=digested_at,
    )

    return dict(record)


# =============================================================================
# API-Only Digest (for OSF, Figshare, Zenodo - no actual files on disk)
# =============================================================================


def parse_bids_entities_from_path(filepath: str) -> dict[str, Any]:
    """Extract BIDS entities from a file path without needing actual files.

    Parameters
    ----------
    filepath : str
        BIDS-style file path (e.g., "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.set")

    Returns
    -------
    dict
        Extracted BIDS entities (subject, session, task, run, modality, etc.)

    """
    import re

    entities = {}
    filename = Path(filepath).name
    filepath_lower = filepath.lower()
    filename_lower = filename.lower()

    # Extract entities from filename using BIDS naming convention
    # Format: sub-<label>[_ses-<label>][_task-<label>][_run-<index>][_<suffix>].<extension>

    # Subject
    sub_match = re.search(r"sub-([^_/.]+)", filepath)
    if sub_match:
        entities["subject"] = sub_match.group(1)

    # Session
    ses_match = re.search(r"ses-([^_/.]+)", filepath)
    if ses_match:
        entities["session"] = ses_match.group(1)

    # Task
    task_match = re.search(r"task-([^_/.]+)", filepath)
    if task_match:
        val = task_match.group(1)
        # needed because of ds004841
        # if "run-" in val:
        #    val = val.split("run-")[0]
        entities["task"] = val

    # Run
    run_match = re.search(r"run-([^_/.]+)", filepath)
    if run_match:
        entities["run"] = run_match.group(1)

    # Acquisition
    acq_match = re.search(r"acq-([^_/]+)", filepath)
    if acq_match:
        entities["acquisition"] = acq_match.group(1)

    # --- Fallback for non-BIDS datasets ---
    # If no subject or task was found using BIDS convention, use folder/filename as fallback
    if "subject" not in entities:
        # Take the parent directory name if it's not the root or a standard bids folder
        parts = Path(filepath).parts
        if len(parts) > 1:
            # Avoid using standard modality folders as subject names
            possible_sub = parts[0]
            if (
                possible_sub.lower() not in MODALITY_DETECTION_TARGETS
                and possible_sub.lower()
                not in (
                    "anat",
                    "derivatives",
                    "sourcedata",
                    "code",
                    "raw",
                    "data",
                    "files",
                )
            ):
                entities["subject"] = possible_sub

    if "task" not in entities:
        stem = Path(filepath).stem
        # Clean up common suffixes
        for mod in MODALITY_DETECTION_TARGETS:
            if stem.lower().endswith(f"_{mod}"):
                stem = stem[: -(len(mod) + 1)]
                break
        # Avoid generic task names
        if stem.lower() not in (
            "dataset",
            "manifest",
            "readme",
            "participants",
            "scans",
            "dataset_description",
            "samples",
            "data",
        ) and not stem.lower().startswith(("sub-", "ses-")):
            entities["task"] = stem

    # Determine modality/datatype from path or filename
    for mod_target in MODALITY_DETECTION_TARGETS:
        if f"/{mod_target}/" in filepath_lower or f"_{mod_target}." in filename_lower:
            ent_mod = normalize_modality(mod_target)
            if ent_mod:
                entities["modality"] = ent_mod
                entities["datatype"] = ent_mod
                break

    if "modality" not in entities:
        if "/anat/" in filepath_lower:
            entities["modality"] = "anat"
            entities["datatype"] = "anat"
        elif "/func/" in filepath_lower:
            entities["modality"] = "func"
            entities["datatype"] = "func"
        elif "/fmap/" in filepath_lower:
            entities["modality"] = "fmap"
            entities["datatype"] = "fmap"
        elif "/beh/" in filepath_lower:
            entities["modality"] = "beh"
            entities["datatype"] = "beh"

    # If no modality found via indicators, use detect_modality_from_path as fallback
    if "modality" not in entities:
        entities["modality"] = detect_modality_from_path(filepath)
        entities["datatype"] = entities["modality"]

    # Extract suffix (last part before extension)
    suffix_match = re.search(r"_([^_]+)\.[^.]+$", filename)
    if suffix_match:
        entities["suffix"] = suffix_match.group(1)

    return entities


# Semantic mapping to canonical BIDS modalities
MODALITY_CANONICAL_MAP = {
    "nirs": "fnirs",
    "fnirs": "fnirs",
    "spike": "ieeg",
    "lfp": "ieeg",
    "mea": "ieeg",
}

# Supported canonical neurophysiology modalities
NEURO_MODALITIES = ("eeg", "meg", "ieeg", "emg", "fnirs")

# Modalities we care about detecting (including aliases/variants)
MODALITY_DETECTION_TARGETS = (
    "eeg",
    "meg",
    "ieeg",
    "emg",
    "nirs",
    "fnirs",
    "spike",
    "lfp",
    "mea",
)

# CTF MEG uses .ds directories containing multiple files
# We should only match the .ds directory path, not files inside
CTF_INTERNAL_EXTENSIONS = {
    ".meg4",
    ".res4",
    ".hc",
    ".infods",
    ".acq",
    ".hist",
    ".newds",
}


def normalize_modality(modality: str | None) -> str | None:
    """Normalize modality to canonical BIDS terms."""
    if not modality:
        return None
    lowered = modality.lower()
    if lowered == "bids":
        return None
    return MODALITY_CANONICAL_MAP.get(lowered, lowered)


NEURO_DATA_EXTENSIONS: set[str] | None = None


def _load_neuro_data_extensions() -> set[str]:
    """Lazily load neurophysiology data extensions from MNE-BIDS."""
    global NEURO_DATA_EXTENSIONS
    if NEURO_DATA_EXTENSIONS is not None:
        return NEURO_DATA_EXTENSIONS

    # Avoid numba cache issues on CI/help by setting cache dir before import.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache") / "numba"))

    from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS

    extensions: set[str] = set()
    for modality, exts in ALLOWED_DATATYPE_EXTENSIONS.items():
        if modality in NEURO_MODALITIES or modality in MODALITY_DETECTION_TARGETS:
            extensions.update(exts)

    NEURO_DATA_EXTENSIONS = extensions
    return extensions

    return False


def is_neuro_data_file(filepath: str) -> bool:
    """Check if file is a neurophysiology data file (EEG, MEG, iEEG, EMG, fNIRS).

    Uses MNE-BIDS ALLOWED_DATATYPE_EXTENSIONS for extension detection.
    Handles CTF MEG .ds directories specially - only matches the .ds path,
    not internal files like .meg4, .res4, etc.

    Parameters
    ----------
    filepath : str
        Path to check (BIDS-style relative path)

    Returns
    -------
    bool
        True if file appears to be neurophysiology data

    """
    filepath_lower = filepath.lower()

    # Skip files inside CTF .ds directories (we want the .ds directory itself)
    # e.g., skip "sub-01_meg.ds/sub-01_meg.meg4" but keep "sub-01_meg.ds"
    if ".ds/" in filepath_lower:
        return False

    # Also skip CTF internal files by extension
    for ext in CTF_INTERNAL_EXTENSIONS:
        if filepath_lower.endswith(ext):
            return False

    # Check for modality indicators (makes detection more robust for non-BIDS)
    for modality in MODALITY_DETECTION_TARGETS:
        if f"/{modality}/" in filepath_lower or f"_{modality}." in filepath_lower:
            return True

    # Check for data file extensions from MNE-BIDS
    data_exts = _load_neuro_data_extensions()
    is_data_ext = any(filepath_lower.endswith(ext) for ext in data_exts)

    # If NO modality indicator, RELAX:
    # Still count it if it's a known data extension (typical for Zenodo/SciDB/etc.)
    if is_data_ext:
        return True

    # Also allow .zip files if they seem to be subject/session zips (common on Zenodo)
    # OR if it's a generic "Dataset.zip" or variants which often hold the raw data
    if filepath_lower.endswith(".zip"):
        if (
            "sub-" in filepath_lower
            or "ses-" in filepath_lower
            or "dataset.zip" in filepath_lower
            or "data.zip" in filepath_lower
        ):
            return True

    return False


def detect_modality_from_path(filepath: str) -> str:
    """Detect the recording modality from a file path.

    Parameters
    ----------
    filepath : str
        BIDS-style file path

    Returns
    -------
    str
        Detected modality (eeg, meg, ieeg, emg, nirs) or 'eeg' as default

    """
    filepath_lower = filepath.lower()

    for modality in MODALITY_DETECTION_TARGETS:
        # Check folder pattern first (more reliable)
        if f"/{modality}/" in filepath_lower:
            return normalize_modality(modality) or "eeg"
        # Check suffix pattern
        if f"_{modality}." in filepath_lower:
            return normalize_modality(modality) or "eeg"

    return "eeg"  # Default fallback


# Keep old function name as alias for backward compatibility
def is_eeg_data_file(filepath: str) -> bool:
    """Check if file is a neurophysiology data file. Alias for is_neuro_data_file."""
    return is_neuro_data_file(filepath)


def digest_from_manifest(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a dataset from its manifest.json without requiring actual files.

    This is used for API-only sources (OSF, Figshare, Zenodo) where we have
    file listings but no actual files on disk.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    input_dir : Path
        Directory containing cloned datasets (with manifest.json)
    output_dir : Path
        Directory for output JSON files

    Returns
    -------
    dict
        Summary of digestion results

    """
    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id
    manifest_path = dataset_dir / "manifest.json"

    if not manifest_path.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "manifest.json not found",
        }

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load manifest: {e}",
        }

    source = manifest.get("source", "osf")
    digested_at = datetime.now(timezone.utc).isoformat()

    # Get file list from manifest
    files = manifest.get("files", [])

    if not files:
        return {
            "status": "empty",
            "dataset_id": dataset_id,
            "reason": "no files in manifest",
        }

    # Check for ZIP contents (files extracted from subject ZIPs)
    zip_contents = manifest.get("zip_contents", [])

    # Determine storage base based on source
    # First check if manifest has explicit storage_base (from fetch step)
    storage_base = manifest.get("storage_base")

    if not storage_base:
        # Build storage base from source configuration
        config = get_storage_config(source)
        base = config["base"]

        if source == "figshare":
            # Figshare: use source_url from external_links if available
            source_url = manifest.get("external_links", {}).get("source_url", "")
            storage_base = source_url if source_url else f"{base}/{dataset_id}"
        elif source == "zenodo":
            # Zenodo: use zenodo_id if available
            zenodo_id = manifest.get("zenodo_id", dataset_id)
            storage_base = f"{base}/{zenodo_id}"
        elif source == "osf":
            # OSF: use osf_id if available
            osf_id = manifest.get("osf_id", dataset_id)
            storage_base = f"{base}/{osf_id}"
        elif source == "gin":
            # GIN: include organization in path
            org = manifest.get("organization", "EEGManyLabs")
            storage_base = f"{base}/{org}/{dataset_id}"
        else:
            # Default: use base/dataset_id pattern
            storage_base = f"{base}/{dataset_id}"

    # Collect BIDS entities from file paths (both direct files and ZIP contents)
    subjects = set()
    sessions = set()
    tasks = set()
    modalities = set()

    all_paths = []
    for f in files:
        filepath = f.get("path", "") if isinstance(f, dict) else f
        all_paths.append(filepath)

        # Check if this file has extracted ZIP contents
        if isinstance(f, dict) and f.get("_zip_contents"):
            for zf in f["_zip_contents"]:
                zpath = zf.get("path", "") if isinstance(zf, dict) else zf
                all_paths.append(zpath)

    # Also add any separately stored zip_contents
    for zpath in zip_contents:
        if isinstance(zpath, dict):
            all_paths.append(zpath.get("path", ""))
        else:
            all_paths.append(zpath)

    # Extract BIDS entities from all paths
    for filepath in all_paths:
        entities = parse_bids_entities_from_path(filepath)

        # Always track modalities found
        if entities.get("modality"):
            modalities.add(entities["modality"])

        # Only count subjects/sessions/tasks for supported neuro modalities
        if entities.get("modality") in NEURO_MODALITIES:
            if entities.get("subject"):
                subjects.add(entities["subject"])
            if entities.get("session"):
                sessions.add(entities["session"])
            if entities.get("task"):
                tasks.add(entities["task"])

    # Get demographics from manifest
    demographics = manifest.get("demographics", {})
    # User request: "count only subject from the modalities that are validated"
    # So we prefer the count from valid files (len(subjects)) if available.
    # If not found from files, check manifest's demographic subjects_count or bids_subject_count.
    subjects_count = (
        len(subjects)
        if subjects
        else (
            demographics.get("subjects_count", 0)
            or manifest.get("bids_subject_count", 0)
        )
    )

    # Fallback: Try to fetch metadata files if counts/tasks are missing
    if subjects_count == 0 or not tasks:
        import urllib.error
        import urllib.request

        # Look for key metadata files in the file list
        desc_url = None
        participants_url = None

        for f in files:
            if isinstance(f, dict):
                path = f.get("path", "").lower()
                url = f.get("download_url")
                if path.endswith("dataset_description.json"):
                    desc_url = url
                elif path.endswith("participants.tsv"):
                    participants_url = url

        # Try dataset_description.json first
        if desc_url:
            try:
                with urllib.request.urlopen(desc_url, timeout=10) as response:
                    desc_data = json.loads(response.read().decode("utf-8"))
                    # Some datasets put subject count here
                    if "Subjects" in desc_data:  # heuristic
                        subjects_count = int(desc_data["Subjects"])
            except Exception as e:
                print(f"Failed to fetch/parse dataset_description.json: {e}")

        # Try participants.tsv if still 0
        if subjects_count == 0 and participants_url:
            try:
                with urllib.request.urlopen(participants_url, timeout=10) as response:
                    content = response.read().decode("utf-8")
                    lines = [l for l in content.splitlines() if l.strip()]
                    # Subtract header
                    if len(lines) > 1:
                        subjects_count = len(lines) - 1
            except Exception as e:
                print(f"Failed to fetch/parse participants.tsv: {e}")
    ages = demographics.get("ages", [])

    # Get DOI from manifest (OSF enhanced clone provides this)
    dataset_doi = manifest.get("dataset_doi")
    if not dataset_doi:
        identifiers = manifest.get("identifiers", {})
        dataset_doi = identifiers.get("doi")

    # Get source URL (prefer explicit URL, fallback to OSF URL)
    source_url = manifest.get("external_links", {}).get("source_url")
    if not source_url:
        source_url = manifest.get("external_links", {}).get("osf_url")

    fingerprint = fingerprint_from_manifest(dataset_id, source, manifest)

    # Determine recording modalities
    recording_modality_val = manifest.get("recording_modality")
    if isinstance(recording_modality_val, str):
        # Support both + and , as separators and wrap in list
        recording_modality_val = [
            m.strip()
            for m in recording_modality_val.replace("+", ",").split(",")
            if m.strip()
        ]

    if recording_modality_val:
        # Normalize and filter
        recording_modality_val = [normalize_modality(m) for m in recording_modality_val]
        recording_modality_val = sorted(
            list({m for m in recording_modality_val if m in NEURO_MODALITIES})
        )
    if not recording_modality_val:
        recording_modality_val = sorted(
            list({m for m in modalities if m in NEURO_MODALITIES})
        ) or ["eeg"]

    # Use first neuro modality for BIDS path construction in placeholders
    primary_mod = recording_modality_val[0] if recording_modality_val else "eeg"

    # Build Dataset document
    dataset_doc = create_dataset(
        dataset_id=dataset_id,
        name=manifest.get("name"),
        source=source,
        readme=manifest.get("readme"),
        recording_modality=recording_modality_val,
        datatypes=sorted(manifest.get("modalities", list(modalities))),
        bids_version=None,  # Not available from API
        license=manifest.get("license"),
        authors=manifest.get("authors", []),
        funding=manifest.get("funding", []),
        dataset_doi=dataset_doi,
        tasks=sorted(manifest.get("tasks") or sorted(list(tasks))),
        sessions=sorted(manifest.get("sessions") or sorted(list(sessions))),
        total_files=len(files),
        subjects_count=subjects_count,
        ages=ages,
        age_mean=sum(ages) / len(ages) if ages else None,
        study_domain=manifest.get("study_domain"),
        source_url=source_url,
        digested_at=digested_at,
    )
    dataset_doc["ingestion_fingerprint"] = fingerprint

    # Generate Records for EEG data files
    records = []
    errors = []

    # First pass: Extract unique CTF .ds directory paths from files inside them
    # CTF MEG stores data in .ds directories, but manifests list individual files
    ctf_ds_dirs = set()
    for file_info in files:
        if isinstance(file_info, dict):
            filepath = file_info.get("path", "")
        else:
            filepath = file_info

        filepath_lower = filepath.lower()
        if ".ds/" in filepath_lower:
            # Extract the .ds directory path (everything up to and including .ds)
            ds_idx = filepath_lower.index(".ds/") + 3  # +3 to include ".ds"
            ds_path = filepath[:ds_idx]  # Use original case
            ctf_ds_dirs.add(ds_path)

    # Create records for CTF .ds directories
    for ds_path in ctf_ds_dirs:
        try:
            entities = parse_bids_entities_from_path(ds_path)
            detected_modality = detect_modality_from_path(ds_path)

            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=ds_path,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend=get_storage_backend(source),
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            records.append(dict(record))
        except Exception as e:
            errors.append({"file": ds_path, "error": str(e)})

    for file_info in files:
        if isinstance(file_info, dict):
            filepath = file_info.get("path", "")
            download_url = file_info.get("download_url")
            file_size = file_info.get("size", 0)
        else:
            filepath = file_info
            download_url = None
            file_size = 0

        # Check if this is a ZIP file with extracted contents
        zip_file_contents = None
        if isinstance(file_info, dict):
            zip_file_contents = file_info.get("_zip_contents", [])

        # If this is a ZIP with contents, create records from ZIP contents instead
        if zip_file_contents:
            zip_name = file_info.get("name", "")
            zip_download_url = file_info.get("download_url", "")

            for zf in zip_file_contents:
                zf_path = zf.get("path", "") if isinstance(zf, dict) else zf
                zf_size = zf.get("size", 0) if isinstance(zf, dict) else 0

                # Only create records for neurophysiology data files inside ZIPs
                if not is_neuro_data_file(zf_path):
                    continue

                try:
                    entities = parse_bids_entities_from_path(zf_path)
                    detected_modality = detect_modality_from_path(zf_path)

                    record = create_record(
                        dataset=dataset_id,
                        storage_base=storage_base,
                        bids_relpath=zf_path,
                        subject=entities.get("subject"),
                        session=entities.get("session"),
                        task=entities.get("task"),
                        run=entities.get("run"),
                        dep_keys=[],
                        datatype=entities.get("datatype", detected_modality),
                        suffix=entities.get("suffix", detected_modality),
                        storage_backend="https",
                        recording_modality=[detected_modality],
                        digested_at=digested_at,
                    )

                    # Store ZIP info for download
                    if zip_download_url:
                        record["container_url"] = zip_download_url
                        record["container_type"] = "zip"
                        record["container_name"] = zip_name
                    if zf_size:
                        record["file_size"] = zf_size

                    records.append(dict(record))
                except Exception as e:
                    errors.append({"file": zf_path, "error": str(e)})
            continue  # Skip to next file (we've processed the ZIP contents)

        # Handle ZIP files without extracted contents
        import re

        if filepath.lower().endswith(".zip"):
            # Pattern 1: Subject ZIP files like sub-01.zip
            subject_match = re.match(
                r"^(sub-[a-zA-Z0-9]+)\.zip$", filepath, re.IGNORECASE
            )
            if subject_match:
                subject_id = subject_match.group(1)
                # Infer recording modality from dataset modalities (already in recording_modality_val)

                try:
                    # Create a placeholder record for this subject
                    record = create_record(
                        dataset=dataset_id,
                        storage_base=storage_base,
                        bids_relpath=f"{subject_id}/{primary_mod}/{subject_id}_{primary_mod}.set",  # Placeholder path
                        subject=subject_id.replace("sub-", ""),
                        session=None,
                        task=None,
                        run=None,
                        dep_keys=[],
                        datatype=primary_mod,
                        suffix=primary_mod,
                        storage_backend="https",
                        recording_modality=recording_modality_val,
                        digested_at=digested_at,
                    )

                    # Store ZIP info
                    if download_url:
                        record["container_url"] = download_url
                        record["container_type"] = "zip"
                        record["container_name"] = filepath
                        record["zip_contains_bids"] = (
                            True  # Flag that this ZIP contains BIDS data
                        )
                    if file_size:
                        record["container_size"] = file_size

                    records.append(dict(record))
                except Exception as e:
                    errors.append({"file": filepath, "error": str(e)})
                continue

            # Pattern 2: BIDS data ZIPs (e.g., data_bids.zip, *_bids_EEG.zip, bids_data.zip)
            # These commonly contain BIDS-formatted data but we can't peek inside
            bids_zip_patterns = [
                r".*bids.*\.zip$",  # Contains 'bids' anywhere
                r".*_eeg\.zip$",  # Ends with _eeg.zip
                r".*_meg\.zip$",  # Ends with _meg.zip
                r".*_ieeg\.zip$",  # Ends with _ieeg.zip
                r".*dataset.*\.zip$",  # Contains 'dataset'
                r".*rawdata.*\.zip$",  # Contains 'rawdata'
                r".*data\.zip$",  # Simple data.zip
                r".*eeg.*\.zip$",  # Contains 'eeg' anywhere
                r".*meg.*\.zip$",  # Contains 'meg' anywhere
                r".*nirs.*\.zip$",  # Contains 'nirs' anywhere
                r".*fnirs.*\.zip$",  # Contains 'fnirs' anywhere
            ]

            filepath_lower = filepath.lower()
            is_bids_zip = any(re.match(p, filepath_lower) for p in bids_zip_patterns)

            if is_bids_zip:
                # Try to infer subject count from manifest demographics
                demographics = manifest.get("demographics", {})
                inferred_subjects = demographics.get("subjects_count", 0)

                # If we have subject count, create individual subject records
                if inferred_subjects and inferred_subjects > 0:
                    for sub_idx in range(
                        1, min(inferred_subjects + 1, 201)
                    ):  # Cap at 200 subjects
                        sub_id = (
                            f"{sub_idx:02d}"
                            if inferred_subjects < 100
                            else f"{sub_idx:03d}"
                        )

                        try:
                            record = create_record(
                                dataset=dataset_id,
                                storage_base=storage_base,
                                bids_relpath=f"sub-{sub_id}/{primary_mod}/sub-{sub_id}_{primary_mod}.set",
                                subject=sub_id,
                                session=None,
                                task=manifest.get("tasks", [None])[0]
                                if manifest.get("tasks")
                                else None,
                                run=None,
                                dep_keys=[],
                                datatype=primary_mod,
                                suffix=primary_mod,
                                storage_backend="https",
                                recording_modality=recording_modality_val,
                                digested_at=digested_at,
                            )

                            # Store ZIP info for download
                            if download_url:
                                record["container_url"] = download_url
                                record["container_type"] = "zip"
                                record["container_name"] = filepath
                                record["needs_extraction"] = True
                                record["inferred_from_metadata"] = True
                            if file_size:
                                record["container_size"] = file_size

                            records.append(dict(record))
                        except Exception as e:
                            errors.append({"file": f"sub-{sub_id}", "error": str(e)})
                else:
                    # No subject count - create single placeholder record
                    try:
                        record = create_record(
                            dataset=dataset_id,
                            storage_base=storage_base,
                            bids_relpath=f"__ZIP__/{filepath}",
                            subject=None,
                            session=None,
                            task=None,
                            run=None,
                            dep_keys=[],
                            datatype=primary_mod,
                            suffix=primary_mod,
                            storage_backend=get_storage_backend(source),
                            recording_modality=recording_modality_val,
                            digested_at=digested_at,
                        )

                        if download_url:
                            record["container_url"] = download_url
                            record["container_type"] = "zip"
                            record["container_name"] = filepath
                            record["needs_extraction"] = True
                        if file_size:
                            record["container_size"] = file_size

                        records.append(dict(record))
                    except Exception as e:
                        errors.append({"file": filepath, "error": str(e)})
                continue

        # Check if this is a neurophysiology data file (treat symlinks as files)
        if not is_neuro_data_file(filepath):
            continue

        try:
            entities = parse_bids_entities_from_path(filepath)
            detected_modality = detect_modality_from_path(filepath)

            # Build the record
            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=filepath,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                dep_keys=[],  # Can't determine dependencies without actual files
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend=get_storage_backend(source),
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )

            # Add download URL if available
            if download_url:
                record["download_url"] = download_url
            if file_size:
                record["file_size"] = file_size

            records.append(dict(record))
        except Exception as e:
            errors.append({"file": filepath, "error": str(e)})

    # Also process standalone zip_contents (from clone script)
    for zpath in zip_contents:
        if isinstance(zpath, dict):
            filepath = zpath.get("path", "")
            file_size = zpath.get("size", 0)
        else:
            filepath = zpath
            file_size = 0

        if not is_neuro_data_file(filepath):
            continue

        try:
            entities = parse_bids_entities_from_path(filepath)
            detected_modality = detect_modality_from_path(filepath)

            record = create_record(
                dataset=dataset_id,
                storage_base=storage_base,
                bids_relpath=filepath,
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=entities.get("task"),
                run=entities.get("run"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend="https",
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )

            if file_size:
                record["file_size"] = file_size

            records.append(dict(record))
        except Exception as e:
            errors.append({"file": filepath, "error": str(e)})

    # Create output directory
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Save Dataset document
    dataset_path = dataset_output_dir / f"{dataset_id}_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dict(dataset_doc), f, indent=2)

    # Save Records document
    records_path = dataset_output_dir / f"{dataset_id}_records.json"
    records_data = {
        "dataset": dataset_id,
        "source": source,
        "digested_at": digested_at,
        "record_count": len(records),
        "records": records,
    }
    with open(records_path, "w") as f:
        json.dump(records_data, f, indent=2)

    # Save summary
    summary = {
        "status": "success" if records else "no_neuro_files",
        "dataset_id": dataset_id,
        "source": source,
        "record_count": len(records),
        "total_files": len(files),
        "error_count": len(errors),
        "dataset_file": str(dataset_path),
        "records_file": str(records_path),
        "digest_method": "manifest_only",
    }

    summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def digest_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a single dataset and generate JSON output.

    Produces:
    - {dataset_id}_dataset.json: Dataset-level metadata
    - {dataset_id}_records.json: Per-file Record metadata

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    input_dir : Path
        Directory containing cloned datasets
    output_dir : Path
        Directory for output JSON files

    Returns
    -------
    dict
        Summary of digestion results

    """
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id

    if not dataset_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "directory not found",
        }

    # Check if this is an API-only source (has manifest.json but no actual files)
    manifest_path = dataset_dir / "manifest.json"
    has_manifest = manifest_path.exists()

    # Check if there are actual EEG files or symlinks (git-annex uses broken symlinks)
    # We accept both real files and symlinks for metadata extraction
    has_actual_files = any(
        f.suffix in [".set", ".edf", ".bdf", ".vhdr", ".fif", ".cnt"]
        for f in dataset_dir.rglob("*")
        if f.is_file() or f.is_symlink()  # Include symlinks for git-annex
    )

    # For API-only sources, use manifest-based digestion
    if has_manifest and not has_actual_files:
        return digest_from_manifest(dataset_id, input_dir, output_dir)

    # Detect source
    source = detect_source(dataset_dir)

    # Generate timestamp
    digested_at = datetime.now(timezone.utc).isoformat()

    # Load BIDS dataset
    try:
        bids_dataset = EEGBIDSDataset(
            data_dir=str(dataset_dir),
            dataset=dataset_id,
            allow_symlinks=True,
        )
    except Exception as e:
        # Fallback to manifest-based digestion if BIDS parsing fails
        if has_manifest:
            return digest_from_manifest(dataset_id, input_dir, output_dir)
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load BIDS dataset: {e}",
        }

    files = bids_dataset.get_files()
    if not files:
        # Fallback to manifest-based digestion if no files found
        if has_manifest:
            return digest_from_manifest(dataset_id, input_dir, output_dir)
        return {
            "status": "empty",
            "dataset_id": dataset_id,
            "reason": "no neurophysiology files found",
        }

    manifest_data = None
    if has_manifest:
        try:
            with open(manifest_path) as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    # Extract Dataset metadata
    try:
        dataset_meta = extract_dataset_metadata(
            bids_dataset, dataset_id, source, digested_at, metadata=manifest_data
        )
    except Exception as e:
        dataset_meta = {
            "dataset_id": dataset_id,
            "source": source,
            "error": str(e),
        }

    try:
        file_paths = [Path(str(f)) for f in files]
        fingerprint = fingerprint_from_files(
            dataset_id, source, file_paths, dataset_dir
        )
        dataset_meta["ingestion_fingerprint"] = fingerprint
    except Exception:
        pass

    # Extract Record metadata for each file
    records = []
    errors = []

    for bids_file in files:
        # Filter out non-neuro data files (sidecars, etc.)
        if not is_neuro_data_file(str(bids_file)):
            continue

        try:
            record = extract_record(
                bids_dataset, bids_file, dataset_id, source, digested_at
            )
            records.append(record)
        except Exception as e:
            errors.append({"file": str(bids_file), "error": str(e)})

    if not records:
        # Fallback to manifest-based digestion if no records extracted
        if has_manifest:
            return digest_from_manifest(dataset_id, input_dir, output_dir)
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No records extracted",
            "errors": errors,
        }

    # Create output directory
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Save Dataset document
    dataset_path = dataset_output_dir / f"{dataset_id}_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_meta, f, indent=2, default=_json_serializer)

    # Save Records document
    records_path = dataset_output_dir / f"{dataset_id}_records.json"
    records_data = {
        "dataset": dataset_id,
        "source": source,
        "digested_at": digested_at,
        "record_count": len(records),
        "records": records,
    }
    with open(records_path, "w") as f:
        json.dump(records_data, f, indent=2, default=_json_serializer)

    # Save summary
    summary = {
        "status": "success",
        "dataset_id": dataset_id,
        "source": source,
        "record_count": len(records),
        "error_count": len(errors),
        "dataset_file": str(dataset_path),
        "records_file": str(records_path),
    }

    summary_path = dataset_output_dir / f"{dataset_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _json_serializer(obj):
    """Handle non-serializable objects."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return sorted(list(obj))
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def find_datasets(input_dir: Path, datasets: list[str] | None = None) -> list[str]:
    """Find dataset IDs in input directory."""
    if datasets:
        return datasets

    found = []
    for d in input_dir.iterdir():
        if (
            d.is_dir()
            and d.name not in ("__pycache__", ".git")
            and d.name not in EXCLUDED_DATASETS
        ):
            # Check if it has a manifest.json (API-based sources)
            # or dataset_description.json (git-cloned BIDS datasets)
            if (d / "manifest.json").exists() or (
                d / "dataset_description.json"
            ).exists():
                found.append(d.name)

    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Digest BIDS datasets and generate Dataset + Record JSON for MongoDB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cloned"),
        help="Directory containing cloned datasets (default: data/cloned/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digestion_output"),
        help="Output directory for JSON files (default: digestion_output/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset IDs to digest (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of datasets to process (for testing)",
    )

    args = parser.parse_args()

    # Find datasets
    dataset_ids = find_datasets(args.input, args.datasets)
    if args.limit:
        dataset_ids = dataset_ids[: args.limit]

    print(f"Found {len(dataset_ids)} datasets to digest")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process datasets
    results = []
    stats = {"success": 0, "error": 0, "skipped": 0, "empty": 0}

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(digest_dataset, ds_id, args.input, args.output): ds_id
                for ds_id in dataset_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Digesting"
            ):
                result = future.result()
                results.append(result)
                status = result.get("status", "error")
                stats[status] = stats.get(status, 0) + 1
    else:
        # Sequential processing
        for ds_id in tqdm(dataset_ids, desc="Digesting"):
            result = digest_dataset(ds_id, args.input, args.output)
            results.append(result)
            status = result.get("status", "error")
            stats[status] = stats.get(status, 0) + 1

    # Save batch summary
    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_datasets": len(dataset_ids),
        "stats": stats,
        "total_records": sum(
            r.get("record_count", 0) for r in results if r.get("status") == "success"
        ),
    }

    batch_summary_path = args.output / "BATCH_SUMMARY.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("DIGESTION SUMMARY")
    print("=" * 60)
    print(f"  Success:  {stats['success']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Empty:    {stats['empty']}")
    print(f"  Error:    {stats['error']}")
    print(f"\nTotal records: {batch_summary['total_records']}")
    print(f"Batch summary: {batch_summary_path}")
    print("=" * 60)

    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
