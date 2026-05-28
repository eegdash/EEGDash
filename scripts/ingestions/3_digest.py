#!/usr/bin/env python3
"""Digest BIDS datasets and generate JSON records for MongoDB.

Produces one Dataset doc (discovery/filtering) and one Record doc per file (loading metadata).
See ``python 3_digest.py --help``.
"""

import argparse
import csv as _csv
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

# Avoid numba cache issues by setting cache dir before importing MNE.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache") / "numba"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS
from eegdash.dataset.bids_dataset import _COMPANION_FILES
from eegdash.dataset.io import _repair_participants_tsv_ids
from eegdash.schemas import (
    Storage,
    create_dataset,
    create_record,
)
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS

logger = logging.getLogger(__name__)

from _constants import (
    CTF_INTERNAL_EXTENSIONS,
    EXCLUDED_DATASETS,
    MEF3_INTERNAL_DIRS,
    MEF3_INTERNAL_EXTENSIONS,
    MODALITY_CANONICAL_MAP,
    MODALITY_DETECTION_TARGETS,
    NEURO_MODALITIES,
)
from _digest_config import load_digest_config_from_argv
from _file_utils import (
    get_annex_file_size,
)
from _fingerprint import fingerprint_from_files, fingerprint_from_manifest
from _metadata_cascade import (  # noqa: F401 — re-export for back-compat
    CascadeContext,
    MetadataCascade,
    _parse_fif_with_mne,
    extract_sfreq_nchans_from_channels_tsv,
    extract_sfreq_nchans_from_modality_sidecar,
    sum_bids_channel_counts,
)
from _montage import _walk_up_find as _walk
from _montage import extract_layout
from _parser_utils import _http_client
from digest_telemetry import TelemetryEvent, auto_configure_from_env, get_emitter
from record_enumerator import (
    EnumerationResult,
    ManifestEnumerator,
    RecordEnumerator,
    get_record_enumerator,
    write_dataset_outputs,
)

auto_configure_from_env()

from source_adapter import SourceAdapter, get_source_adapter

DEFAULT_DATASET_TIMEOUT_SECONDS = 2 * 60
WORKER_POLL_INTERVAL_SECONDS = 1.0
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5.0
RESULT_QUEUE_TIMEOUT_SECONDS = 5.0


def _source_from_dataset_id(dataset_id: str) -> str:
    """Infer source from dataset_id prefix pattern (ds* → openneuro, nm* → nemar)."""
    if dataset_id.startswith("ds") and dataset_id[2:].isdigit():
        return "openneuro"
    if dataset_id.startswith("nm") and dataset_id[2:].isdigit():
        return "nemar"
    if "EEGManyLabs" in dataset_id:
        return "gin"
    if dataset_id.startswith("EEG2025"):
        return "nemar"
    return "unknown"


def _reconcile_source(
    manifest_src: str | None, dataset_id: str, *, context: str
) -> str:
    """Trust dataset_id pattern over manifest source to prevent S3 bucket misrouting."""
    pattern_src = _source_from_dataset_id(dataset_id)
    if (
        manifest_src
        and pattern_src not in (None, "unknown")
        and manifest_src != pattern_src
    ):
        print(
            f"WARNING [{context}]: {dataset_id} manifest source={manifest_src!r} "
            f"disagrees with id-pattern source={pattern_src!r}; using pattern.",
            file=sys.stderr,
        )
        return pattern_src
    if manifest_src:
        return manifest_src
    return pattern_src


def detect_source(dataset_dir: Path) -> str:
    """Detect source from manifest.json or dataset structure."""
    dataset_id = dataset_dir.name
    manifest_path = dataset_dir / "manifest.json"
    manifest_src: str | None = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_src = manifest.get("source")
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            manifest_src = None

    return _reconcile_source(manifest_src, dataset_id, context="detect_source")


# Companion files required for different formats
COMPANION_FILE_REQUIREMENTS = {
    ".vhdr": {
        "required": [".eeg", ".dat"],  # Need at least one of these
        "optional": [".vmrk"],
        "mode": "any",  # any = at least one required file must exist
    },
    ".set": {
        "required": [".fdt"],
        "optional": [],
        "mode": "optional",  # optional = may or may not have .fdt (data can be in .set)
    },
}


def _file_exists_or_symlink(path: Path, allow_symlinks: bool = True) -> bool:
    """Return True if path exists or is a broken git-annex symlink."""
    if path.exists():
        return True
    if allow_symlinks and path.is_symlink():
        return True
    return False


def validate_companion_files(
    file_path: Path, allow_symlinks: bool = True
) -> dict[str, Any]:
    """Check that required companion files exist for a data file (e.g. .eeg for .vhdr)."""
    result = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "found": [],
        "warnings": [],
        "errors": [],
    }

    ext = file_path.suffix.lower()
    requirements = COMPANION_FILE_REQUIREMENTS.get(ext)

    if not requirements:
        # No companion file requirements for this format
        return result

    parent_dir = file_path.parent
    stem = file_path.stem

    # Check required companions
    required_exts = requirements.get("required", [])
    mode = requirements.get("mode", "all")

    found_required = []
    for req_ext in required_exts:
        companion_path = parent_dir / f"{stem}{req_ext}"
        if _file_exists_or_symlink(companion_path, allow_symlinks):
            found_required.append(req_ext)
            result["found"].append(str(companion_path.name))

    # Validate based on mode
    if mode == "any":
        # At least one required file must exist
        if required_exts and not found_required:
            result["valid"] = False
            result["missing_required"] = required_exts
            result["errors"].append(f"Missing data file: need one of {required_exts}")
    elif mode == "all":
        # All required files must exist
        missing = [e for e in required_exts if e not in found_required]
        if missing:
            result["valid"] = False
            result["missing_required"] = missing
            result["errors"].append(f"Missing required files: {missing}")
    elif mode == "optional":
        # Required files are not strictly required (e.g., .set can contain data)
        missing = [e for e in required_exts if e not in found_required]
        if missing:
            result["warnings"].append(f"Optional companion files not found: {missing}")
            result["missing_optional"].extend(missing)

    # Check optional companions
    optional_exts = requirements.get("optional", [])
    for opt_ext in optional_exts:
        companion_path = parent_dir / f"{stem}{opt_ext}"
        if _file_exists_or_symlink(companion_path, allow_symlinks):
            result["found"].append(str(companion_path.name))
        else:
            result["missing_optional"].append(opt_ext)

    # Special case for BrainVision: try to read VHDR to check referenced files
    if ext == ".vhdr" and not result["valid"]:
        # Try to get more information from the VHDR file
        try:
            from _vhdr_parser import extract_vhdr_references

            refs = extract_vhdr_references(file_path)
            if refs.get("datafile"):
                data_file = refs["datafile"]
                result["errors"].append(
                    f"VHDR references missing data file: {data_file}"
                )
        except (OSError, ValueError, UnicodeDecodeError, KeyError):
            pass

    return result


def _read_bids_readme(bids_root: Path) -> str | None:
    """Return cleaned README text, or None if absent / unreadable."""
    for readme_name in ("README", "README.md", "README.txt", "readme", "readme.md"):
        readme_path = bids_root / readme_name
        if not readme_path.exists():
            continue
        try:
            raw_readme = readme_path.read_text(encoding="utf-8")
            return "\n".join(
                [line.rstrip() for line in raw_readme.splitlines() if line.strip()]
            )
        except (OSError, UnicodeDecodeError):
            continue
    return None


def _read_participants_demographics(
    bids_root: Path,
) -> tuple[int, list[int], dict[str, int], dict[str, int]]:
    """Read participants.tsv and return (subjects_count, ages, sex_distribution, handedness_distribution)."""
    subjects_count = 0
    ages: list[int] = []
    sex_distribution: dict[str, int] = {}
    handedness_distribution: dict[str, int] = {}

    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        return subjects_count, ages, sex_distribution, handedness_distribution

    try:
        df = pd.read_csv(
            participants_path, sep="\t", dtype="string", keep_default_na=False
        )
        subjects_count = len(df)

        age_col = next(
            (col for col in ("age", "Age", "AGE") if col in df.columns), None
        )
        if age_col:
            for val in df[age_col]:
                try:
                    age = int(float(val))
                    if 0 < age < 120:
                        ages.append(age)
                except (ValueError, TypeError):
                    pass

        sex_col = next(
            (
                col
                for col in ("sex", "Sex", "SEX", "gender", "Gender")
                if col in df.columns
            ),
            None,
        )
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

        hand_col = next(
            (
                col
                for col in ("handedness", "Handedness", "hand", "Hand")
                if col in df.columns
            ),
            None,
        )
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
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
    ):
        pass

    return subjects_count, ages, sex_distribution, handedness_distribution


_BIDS_GLOBAL_FILES: tuple[str, ...] = (
    "participants.tsv",
    "participants.json",
    "samples.tsv",
    "samples.json",
    "README",
    "README.md",
    "README.txt",
    "CHANGES",
    "CHANGES.md",
    "LICENSE",
    "authors.tsv",
    "dataset_description.json",
)


def _build_global_storage_info(
    dataset_id: str, source: str, bids_root: Path
) -> "Storage | None":
    """Build the Dataset-level storage doc (backend, base URL, root-level dep_keys)."""
    if source not in ("openneuro", "nemar", "gin") and source not in STORAGE_CONFIGS:
        return None

    cfg = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)
    storage_base = f"{cfg['base']}/{dataset_id}"
    storage_backend = cfg["backend"]

    dep_keys: list[str] = []
    raw_key = "dataset_description.json"  # default "main" file
    found_files: set[str] = set()

    # 1. Check explicit list (case-sensitive then case-insensitive).
    for fname in _BIDS_GLOBAL_FILES:
        fpath = bids_root / fname
        if fpath.exists():
            found_files.add(fname)
            if fname == "dataset_description.json":
                raw_key = fname
            else:
                dep_keys.append(fname)
            continue
        # Case-insensitive fallback
        try:
            found = next(
                x.name for x in bids_root.iterdir() if x.name.lower() == fname.lower()
            )
            found_files.add(found)
            if found.lower() == "dataset_description.json":
                raw_key = found
            else:
                dep_keys.append(found)
        except StopIteration:
            pass

    # 2. Scan for other root-level BIDS files (sidecars, etc.).
    ignored_files = {"manifest.json", ".ds_store"}
    for item in bids_root.iterdir():
        if not item.is_file():
            continue
        item_name = item.name
        if (
            item_name in found_files
            or item_name.lower() in ignored_files
            or item_name.startswith(".")
        ):
            continue
        if item_name.lower().endswith(
            (".json", ".tsv", ".txt", ".md", ".yaml", ".yml")
        ):
            dep_keys.append(item_name)
            found_files.add(item_name)

    return {
        "backend": storage_backend,  # type: ignore[typeddict-item]
        "base": storage_base,
        "raw_key": raw_key,
        "dep_keys": sorted(set(dep_keys)),
    }


def extract_dataset_metadata(
    bids_dataset,
    dataset_id: str,
    source: str,
    digested_at: str,
    metadata: dict | None = None,
    source_adapter: "SourceAdapter | None" = None,
) -> dict[str, Any]:
    """Extract Dataset-level metadata from a BIDS dataset."""
    metadata = metadata or {}
    bids_root = Path(bids_dataset.bidsdir)
    # Read dataset_description.json
    description = {}
    desc_path = bids_root / "dataset_description.json"
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                description = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    readme = _read_bids_readme(bids_root)

    name = description.get("Name", dataset_id)
    bids_version = description.get("BIDSVersion")
    license_info = description.get("License")
    authors = description.get("Authors", [])
    funding = description.get("Funding", [])
    dataset_doi = description.get("DatasetDOI")

    files = bids_dataset.get_files()
    modalities = set()
    tasks = set()
    sessions = set()
    subjects = set()

    for bids_file in files:
        mod = bids_dataset.get_bids_file_attribute("modality", bids_file)
        mod_canon = normalize_modality(mod)
        if mod_canon:
            modalities.add(mod_canon)

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

    recording_modalities = sorted(
        list({m for m in modalities if m in NEURO_MODALITIES})
    )
    if not recording_modalities:
        recording_modalities = ["eeg"]

    (
        subjects_count,
        ages,
        sex_distribution,
        handedness_distribution,
    ) = _read_participants_demographics(bids_root)
    participants_path = bids_root / "participants.tsv"

    if participants_path.exists() and subjects_count > 0:
        folder_subjects = {
            d.name
            for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        }
        if folder_subjects and subjects_count != len(folder_subjects):
            logging.warning(
                "%s: participants.tsv has %d rows but found %d sub-* folders "
                "(possible naming convention mismatch)",
                dataset_id,
                subjects_count,
                len(folder_subjects),
            )

    # Prefer subject count from validated neuro files over participants.tsv row count.
    if subjects:
        subjects_count = len(subjects)
    elif subjects_count == 0:
        subjects_count = len(
            [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        )

    data_processed = (bids_root / "derivatives").exists()

    if source_adapter is not None:
        source_url = source_adapter.dataset_url()
    else:
        source_url = None
    if source_url is None and source == "gin":
        source_url = f"https://gin.g-node.org/EEGManyLabs/{dataset_id}"

    dataset_created_at = metadata.get("dataset_created_at")
    dataset_modified_at = metadata.get("dataset_modified_at")
    senior_author = metadata.get("senior_author")
    contact_info = metadata.get("contact_info")

    is_clinical = metadata.get("is_clinical")
    clinical_purpose = metadata.get("clinical_purpose")

    if not dataset_modified_at:
        # Try to get from manifest timestamps dict if present
        ts = metadata.get("timestamps", {})
        if isinstance(ts, dict):
            dataset_modified_at = ts.get("dataset_modified_at")
            dataset_created_at = ts.get("dataset_created_at") or dataset_created_at

    # Prefer API-supplied size_bytes; don't trust manifest total_size (wrong for git-annex).
    size_bytes = metadata.get("size_bytes")
    if size_bytes is None and bids_root.exists():
        size_bytes = sum(
            get_annex_file_size(f)
            for f in bids_root.rglob("*")
            if f.is_file() or f.is_symlink()
        )

    storage_info = _build_global_storage_info(dataset_id, source, bids_root)

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
        storage=storage_info,
        # Clinical classification
        is_clinical=is_clinical,
        clinical_purpose=clinical_purpose,
    )

    description_extras = _extract_dataset_description_extras(description)
    dataset.update(description_extras)

    return dict(dataset)


_PROV_MNE_BIDS = "mne_bids"
_PROV_MODALITY_SIDECAR = "modality_sidecar"
_PROV_CHANNELS_TSV = "channels_tsv"
_PROV_BINARY_PARSER = "binary_parser"
_PROV_MNE_FALLBACK = "mne_fallback"

_METADATA_FIELDS: tuple[str, ...] = (
    "sampling_frequency",
    "nchans",
    "ntimes",
    "ch_names",
)


def _empty_provenance() -> dict[str, str | None]:
    """A fresh provenance dict — all fields unattributed."""
    return {field: None for field in _METADATA_FIELDS}


def _stamp_provenance(
    provenance: dict[str, str | None],
    source: str,
    *,
    field: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Set provenance[field] = source if this step was the first to fill it."""
    if old_value is None and new_value is not None and provenance[field] is None:
        provenance[field] = source


# camelCase sidecar key → snake_case Record field (highest-leverage BIDS fields only).
_BIDS_SIDECAR_RECORD_FIELDS: dict[str, str] = {
    "PowerLineFrequency": "power_line_frequency",
    "EEGReference": "eeg_reference",
    "iEEGReference": "ieeg_reference",
    "SoftwareFilters": "software_filters",
    "HardwareFilters": "hardware_filters",
    "Manufacturer": "manufacturer",
    "ManufacturersModelName": "manufacturers_model_name",
    "EEGPlacementScheme": "eeg_placement_scheme",
    "CapManufacturer": "cap_manufacturer",
    "CapManufacturersModelName": "cap_manufacturers_model_name",
    "InstitutionName": "institution_name",
    "RecordingType": "recording_type",
    "RecordingDuration": "recording_duration",
    "EEGGround": "eeg_ground",
}


def _extract_bids_sidecar_fields(bids_dataset: Any, bids_file: str) -> dict[str, Any]:
    """Extract BIDS sidecar fields (PowerLineFrequency, EEGReference, …) into structured Record fields."""
    out: dict[str, Any] = {}

    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
        modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    except (AttributeError, TypeError):
        return {}

    json_pattern = f"*_{modality}.json"
    sidecar = _walk(data_file, bids_root, json_pattern)
    if sidecar is None or not sidecar.exists():
        return {}

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    for sidecar_key, record_key in _BIDS_SIDECAR_RECORD_FIELDS.items():
        val = data.get(sidecar_key)
        if val is None or val == "" or val == [] or val == {}:
            continue
        out[record_key] = val
    return out


def _extract_channel_status_counts(bids_dataset: Any, bids_file: str) -> dict[str, Any]:
    """Return bad channel names and count from channels.tsv, or {} if absent/unreadable."""
    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
    except (AttributeError, TypeError):
        return {}

    tsv = _walk(data_file, bids_root, "*_channels.tsv")
    if tsv is None:
        tsv = _walk(data_file, bids_root, "channels.tsv")
    if tsv is None or not tsv.exists():
        return {}

    bad_channels: list[str] = []
    try:
        with open(tsv, encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return {}
            status_col = next(
                (c for c in reader.fieldnames if c.lower() == "status"), None
            )
            name_col = next((c for c in reader.fieldnames if c.lower() == "name"), None)
            if status_col is None or name_col is None:
                return {}
            for row in reader:
                status = str(row.get(status_col, "")).strip().lower()
                if status == "bad":
                    name = str(row.get(name_col, "")).strip()
                    if name:
                        bad_channels.append(name)
    except (OSError, _csv.Error, UnicodeDecodeError):
        return {}

    if not bad_channels:
        return {"bad_channels": [], "bad_channels_count": 0}
    return {"bad_channels": bad_channels, "bad_channels_count": len(bad_channels)}


_BIDS_DESCRIPTION_DATASET_FIELDS: dict[str, str] = {
    "Acknowledgements": "acknowledgements",
    "HowToAcknowledge": "how_to_acknowledge",
    "EthicsApprovals": "ethics_approvals",
    "ReferencesAndLinks": "references_and_links",
    "GeneratedBy": "generated_by",
    "SourceDatasets": "source_datasets",
}


def _extract_dataset_description_extras(
    description: dict[str, Any],
) -> dict[str, Any]:
    """Pull extra dataset_description.json fields not already captured by extract_dataset_metadata."""
    out: dict[str, Any] = {}
    for desc_key, ds_key in _BIDS_DESCRIPTION_DATASET_FIELDS.items():
        val = description.get(desc_key)
        if val is None or val == "" or val == []:
            continue
        out[ds_key] = val
    return out


def _extract_technical_metadata(
    bids_dataset: Any, bids_file: str
) -> tuple[
    float | None,
    int | None,
    int | None,
    list[str] | None,
    bool,
    bool,
    dict[str, str | None],
]:
    """Delegate to MetadataCascade; returns (sfreq, nchans, ntimes, ch_names, fif_is_split, fif_continuations_ok, provenance)."""
    ctx = CascadeContext(bids_dataset=bids_dataset, bids_file=bids_file)
    result = MetadataCascade().run(ctx)
    return (
        result.sampling_frequency,
        result.nchans,
        result.ntimes,
        result.ch_names,
        result.fif_is_split,
        result.fif_continuations_ok,
        result.provenance,
    )


_DEP_SUFFIXES: tuple[str, ...] = (
    "_channels.tsv",
    "_events.tsv",
    "_events.json",
    "_electrodes.tsv",
    "_coordsystem.json",
    "_eeg.json",
    # NIRS-specific sidecars
    "_optodes.tsv",
    "_optodes.json",
    "_nirs.json",
)


def _build_dep_keys(
    bids_file_path: Path,
    bids_root: Path,
    fif_is_split: bool,
    fif_continuations_ok: bool,
) -> tuple[list[str], bool, bool]:
    """Return (dep_keys, fif_is_split, fif_continuations_ok) for one BIDS file."""
    dep_keys: list[str] = []
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]

    search_dirs = [parent_dir]
    if parent_dir.name in NEURO_MODALITIES or parent_dir.name in {
        "eeg",
        "meg",
        "ieeg",
        "beh",
        "nirs",
    }:
        search_dirs.append(parent_dir.parent)

    base_names_to_search = [base_name]
    session_base = re.sub(r"_task-[^_]+", "", base_name)
    session_base = re.sub(r"_run-[^_]+", "", session_base)
    session_base = re.sub(r"_acq-[^_]+", "", session_base)
    if session_base != base_name:
        base_names_to_search.append(session_base)

    for search_dir in search_dirs:
        for dep_suffix in _DEP_SUFFIXES:
            for search_base in base_names_to_search:
                dep_file = search_dir / f"{search_base}{dep_suffix}"
                if dep_file.exists() or dep_file.is_symlink():
                    try:
                        dep_keys.append(str(dep_file.relative_to(bids_root)))
                    except ValueError:
                        pass

    ext = bids_file_path.suffix.lower()
    for comp_ext in _COMPANION_FILES.get(ext, []):
        comp_file = bids_file_path.with_suffix(comp_ext)
        try:
            dep_keys.append(str(comp_file.relative_to(bids_root)))
        except ValueError:
            pass

    if ext == ".fif":
        if not fif_is_split:
            cont_check = bids_file_path.parent / f"{bids_file_path.stem}-1{ext}"
            if cont_check.exists() or cont_check.is_symlink():
                fif_is_split = True
        if fif_is_split:
            fif_continuations_ok = True
            for i in range(1, 100):
                cont = bids_file_path.parent / f"{bids_file_path.stem}-{i}{ext}"
                if not cont.exists() and not cont.is_symlink():
                    break
                if cont.is_symlink() and not cont.resolve().exists():
                    fif_continuations_ok = False
                try:
                    dep_keys.append(str(cont.relative_to(bids_root)))
                except ValueError:
                    pass

    return dep_keys, fif_is_split, fif_continuations_ok


def _clamp_metadata_extremes(
    sampling_frequency: float | None,
    nchans: int | None,
    ch_names: list[str] | None,
    bids_relpath: str,
    provenance: dict[str, str | None] | None = None,
) -> tuple[float | None, int | None]:
    """Zero out impossible sfreq/nchans values and warn on suspicious extremes (>1 MHz, >10000 channels)."""
    if sampling_frequency is not None:
        if sampling_frequency <= 0:
            logging.warning(
                "Invalid sampling_frequency <= 0 for %s: %s",
                bids_relpath,
                sampling_frequency,
            )
            sampling_frequency = None
            if provenance is not None:
                provenance["sampling_frequency"] = None
        elif sampling_frequency > 1_000_000:
            logging.warning(
                "Suspicious sampling_frequency > 1MHz for %s: %s",
                bids_relpath,
                sampling_frequency,
            )

    if nchans is not None:
        if nchans <= 0:
            logging.warning("Invalid nchans <= 0 for %s: %s", bids_relpath, nchans)
            nchans = None
            if provenance is not None:
                provenance["nchans"] = None
        elif nchans > 10000:
            logging.warning(
                "Suspicious nchans > 10000 for %s: %s", bids_relpath, nchans
            )

    if ch_names and nchans and len(ch_names) != nchans:
        logging.debug(
            "ch_names count (%d) != nchans (%d) for %s",
            len(ch_names),
            nchans,
            bids_relpath,
        )

    return sampling_frequency, nchans


def extract_record(
    bids_dataset,
    bids_file: str,
    dataset_id: str,
    source: str,
    digested_at: str,
    apex_sidecar_inline: dict[str, str] | None = None,
    source_adapter: "SourceAdapter | None" = None,
) -> dict[str, Any]:
    """Extract Record metadata for a single BIDS file."""
    subject = bids_dataset.get_bids_file_attribute("subject", bids_file)
    session = bids_dataset.get_bids_file_attribute("session", bids_file)
    task = bids_dataset.get_bids_file_attribute("task", bids_file)
    run = bids_dataset.get_bids_file_attribute("run", bids_file)
    acquisition = bids_dataset.get_bids_file_attribute("acquisition", bids_file)
    modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    mod_canon = normalize_modality(modality) or "eeg"

    bids_relpath = strip_dataset_prefix(
        str(bids_dataset.get_relative_bidspath(bids_file)), dataset_id
    )

    datatype = mod_canon
    suffix = mod_canon

    cfg = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)
    storage_base = f"{cfg['base']}/{dataset_id}"
    storage_backend = cfg["backend"]

    (
        sampling_frequency,
        nchans,
        ntimes,
        ch_names,
        fif_is_split,
        fif_continuations_ok,
        metadata_provenance,
    ) = _extract_technical_metadata(bids_dataset, bids_file)
    bids_file_path = Path(bids_file)

    dep_keys, fif_is_split, fif_continuations_ok = _build_dep_keys(
        bids_file_path,
        Path(bids_dataset.bidsdir),
        fif_is_split,
        fif_continuations_ok,
    )
    ext = bids_file_path.suffix.lower()

    companion_validation = validate_companion_files(bids_file_path, allow_symlinks=True)
    data_integrity_issues = []

    if not companion_validation["valid"]:
        for error in companion_validation["errors"]:
            logging.warning("Data integrity issue for %s: %s", bids_relpath, error)
            data_integrity_issues.append(error)

    for warning in companion_validation.get("warnings", []):
        logging.info("Companion file note for %s: %s", bids_relpath, warning)

    if ext == ".fif" and fif_is_split and not fif_continuations_ok:
        data_integrity_issues.append(
            "Split FIF: continuation files not available in source"
        )

    sampling_frequency, nchans = _clamp_metadata_extremes(
        sampling_frequency,
        nchans,
        ch_names,
        bids_relpath,
        provenance=metadata_provenance,
    )

    # TODO(scale): apex sidecars (dataset_description.json, README, etc.) are duplicated
    # across every record. Move into a per-dataset side-collection when >100 MB inline
    # payload or >500 KB participants.tsv is encountered.
    if source_adapter is None:
        source_adapter = get_source_adapter(source, dataset_id, bids_dataset.bidsdir)
    bids_root_path = bids_dataset.bidsdir
    dep_paths = [bids_root_path / dep for dep in dep_keys]
    annex_keys, sidecar_inline = source_adapter.resolve_storage_extensions(
        Path(bids_file), dep_paths
    )
    if apex_sidecar_inline:
        for k, v in apex_sidecar_inline.items():
            sidecar_inline.setdefault(k, v)

    record = create_record(
        dataset=dataset_id,
        storage_base=storage_base,
        bids_relpath=bids_relpath,
        subject=subject,
        session=session,
        task=task,
        run=str(run) if run is not None else None,
        acquisition=acquisition,
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
        annex_keys=annex_keys or None,
        sidecar_inline=sidecar_inline or None,
    )

    participant_tsv = bids_dataset.subject_participant_tsv(bids_file)
    if participant_tsv:
        has_real_data = any(v not in (None, "n/a") for v in participant_tsv.values())
        if not has_real_data:
            logging.debug(
                "No participant match for %s, storing column skeleton", bids_relpath
            )
        for k, v in participant_tsv.items():
            if k == "participant_id":
                continue
            if isinstance(v, str):
                try:
                    if v.strip():
                        participant_tsv[k] = float(v)
                except (ValueError, TypeError):
                    pass
        record["participant_tsv"] = participant_tsv

    if data_integrity_issues:
        record["_data_integrity_issues"] = data_integrity_issues
        record["_has_missing_files"] = True
    else:
        record["_has_missing_files"] = False

    if any(v is not None for v in metadata_provenance.values()):
        record["_metadata_provenance"] = metadata_provenance

    sidecar_extras = _extract_bids_sidecar_fields(bids_dataset, bids_file)
    record.update(sidecar_extras)

    channel_status = _extract_channel_status_counts(bids_dataset, bids_file)
    record.update(channel_status)

    return dict(record)


# =============================================================================
# API-Only Digest (for OSF, Figshare, Zenodo - no actual files on disk)
# =============================================================================


def parse_bids_entities_from_path(filepath: str) -> dict[str, Any]:
    """Extract BIDS entities (subject, session, task, run, modality, …) from a file path."""
    entities = {}
    filename = Path(filepath).name
    filepath_lower = filepath.lower()
    filename_lower = filename.lower()

    sub_match = re.search(r"sub-([^_/.]+)", filepath)
    if sub_match:
        entities["subject"] = sub_match.group(1)

    ses_match = re.search(r"ses-([^_/.]+)", filepath)
    if ses_match:
        entities["session"] = ses_match.group(1)

    task_match = re.search(r"task-([^_/.]+)", filepath)
    if task_match:
        val = task_match.group(1)
        # needed because of ds004841
        # if "run-" in val:
        #    val = val.split("run-")[0]
        entities["task"] = val

    run_match = re.search(r"run-([^_/.]+)", filepath)
    if run_match:
        entities["run"] = run_match.group(1)

    acq_match = re.search(r"acq-([^_/]+)", filepath)
    if acq_match:
        entities["acquisition"] = acq_match.group(1)

    # Fallback for non-BIDS datasets: use folder/filename as subject/task when not found.
    if "subject" not in entities:
        parts = Path(filepath).parts
        if len(parts) > 1:
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

    if "modality" not in entities:
        entities["modality"] = detect_modality_from_path(filepath)
        entities["datatype"] = entities["modality"]

    suffix_match = re.search(r"_([^_]+)\.[^.]+$", filename)
    if suffix_match:
        entities["suffix"] = suffix_match.group(1)

    return entities


def strip_dataset_prefix(bids_relpath: str, dataset_id: str) -> str:
    """Strip the leading ``<dataset_id>/`` from a BIDS relative path if present."""
    prefix = f"{dataset_id}/"
    if bids_relpath.startswith(prefix):
        return bids_relpath[len(prefix) :]
    return bids_relpath


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
    """Return MNE-BIDS neurophysiology data extensions (module-level cached)."""
    global NEURO_DATA_EXTENSIONS
    if NEURO_DATA_EXTENSIONS is not None:
        return NEURO_DATA_EXTENSIONS

    extensions: set[str] = set()
    for modality, exts in ALLOWED_DATATYPE_EXTENSIONS.items():
        if modality in NEURO_MODALITIES or modality in MODALITY_DETECTION_TARGETS:
            extensions.update(exts)

    NEURO_DATA_EXTENSIONS = extensions
    return extensions


def is_neuro_data_file(filepath: str) -> bool:
    """Return True if the path is a neurophysiology data file (EEG, MEG, iEEG, EMG, fNIRS)."""
    filepath_lower = filepath.lower()

    if "/derivatives/" in filepath_lower or filepath_lower.startswith("derivatives/"):
        return False

    sidecar_extensions = {
        ".json",
        ".tsv",
        ".txt",
        ".md",
        ".html",
        ".pdf",
        ".csv",
    }
    for ext in sidecar_extensions:
        if filepath_lower.endswith(ext):
            return False

    if ".ds/" in filepath_lower:
        return False

    for ext in CTF_INTERNAL_EXTENSIONS:
        if filepath_lower.endswith(ext):
            return False

    if ".mefd/" in filepath_lower:
        return False

    for ext in MEF3_INTERNAL_EXTENSIONS:
        if filepath_lower.endswith(ext):
            return False

    for internal_dir in MEF3_INTERNAL_DIRS:
        if (
            filepath_lower.endswith(internal_dir)
            or f"{internal_dir}/" in filepath_lower
        ):
            return False

    for modality in MODALITY_DETECTION_TARGETS:
        if f"/{modality}/" in filepath_lower or f"_{modality}." in filepath_lower:
            return True

    data_exts = _load_neuro_data_extensions()
    is_data_ext = any(filepath_lower.endswith(ext) for ext in data_exts)

    if is_data_ext:
        return True

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
    """Detect the recording modality from a BIDS-style file path; defaults to 'eeg'."""
    filepath_lower = filepath.lower()

    for modality in MODALITY_DETECTION_TARGETS:
        if f"/{modality}/" in filepath_lower:
            return normalize_modality(modality) or "eeg"
        # Check suffix pattern
        if f"_{modality}." in filepath_lower:
            return normalize_modality(modality) or "eeg"

    return "eeg"


def _determine_manifest_storage_base(
    source: str,
    dataset_id: str,
    manifest: dict,
) -> str:
    """Resolve the canonical storage.base for a manifest-only ingest; rejects cross-source misrouting."""
    storage_base = manifest.get("storage_base")

    if not storage_base:
        base = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)["base"]

        if source == "figshare":
            # Figshare: use source_url from external_links if available.
            source_url = manifest.get("external_links", {}).get("source_url", "")
            return source_url if source_url else f"{base}/{dataset_id}"
        if source == "zenodo":
            zenodo_id = manifest.get("zenodo_id", dataset_id)
            return f"{base}/{zenodo_id}"
        if source == "osf":
            osf_id = manifest.get("osf_id", dataset_id)
            return f"{base}/{osf_id}"
        if source == "gin":
            org = manifest.get("organization", "EEGManyLabs")
            return f"{base}/{org}/{dataset_id}"
        return f"{base}/{dataset_id}"

    expected_prefix = STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG).get(
        "base", ""
    )
    if expected_prefix and not str(storage_base).startswith(expected_prefix):
        print(
            f"WARNING [digest]: {dataset_id} manifest storage_base="
            f"{storage_base!r} does not start with {expected_prefix!r} "
            f"for source={source!r}; rebuilding from source config.",
            file=sys.stderr,
        )
        return f"{expected_prefix}/{dataset_id}"

    return storage_base


def _collect_bids_entities_from_paths(
    files: list, zip_contents: list
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Parse BIDS entities from all file paths (including ZIP contents) into (subjects, sessions, tasks, modalities)."""
    subjects: set[str] = set()
    sessions: set[str] = set()
    tasks: set[str] = set()
    modalities: set[str] = set()

    all_paths: list[str] = []
    for f in files:
        filepath = f.get("path", "") if isinstance(f, dict) else f
        all_paths.append(filepath)
        # Check if this file has extracted ZIP contents
        if isinstance(f, dict) and f.get("_zip_contents"):
            for zf in f["_zip_contents"]:
                zpath = zf.get("path", "") if isinstance(zf, dict) else zf
                all_paths.append(zpath)

    for zpath in zip_contents:
        all_paths.append(zpath.get("path", "") if isinstance(zpath, dict) else zpath)

    for filepath in all_paths:
        entities = parse_bids_entities_from_path(filepath)
        if entities.get("modality"):
            modalities.add(entities["modality"])
        if entities.get("modality") in NEURO_MODALITIES:
            if entities.get("subject"):
                subjects.add(entities["subject"])
            if entities.get("session"):
                sessions.add(entities["session"])
            if entities.get("task"):
                tasks.add(entities["task"])

    return subjects, sessions, tasks, modalities


def _fetch_subject_count_via_http(files: list, fallback: int) -> int:
    """Last-resort HTTP fetch of dataset_description.json or participants.tsv for subject count."""
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

    subjects_count = fallback
    if desc_url:
        try:
            resp = _http_client().get(desc_url, timeout=10)
            resp.raise_for_status()
            desc_data = json.loads(resp.content.decode("utf-8"))
            if "Subjects" in desc_data:  # heuristic
                subjects_count = int(desc_data["Subjects"])
        except (
            httpx.HTTPError,
            RuntimeError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            KeyError,
        ) as e:
            print(f"Failed to fetch/parse dataset_description.json: {e}")

    if subjects_count == 0 and participants_url:
        try:
            resp = _http_client().get(participants_url, timeout=10)
            resp.raise_for_status()
            content = resp.content.decode("utf-8")
            lines = [line for line in content.splitlines() if line.strip()]
            if len(lines) > 1:  # subtract the header
                subjects_count = len(lines) - 1
        except (
            httpx.HTTPError,
            RuntimeError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as e:
            print(f"Failed to fetch/parse participants.tsv: {e}")

    return subjects_count


def _build_zip_extracted_records(
    file_info: dict,
    dataset_id: str,
    storage_base: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records for neuro files inside a peeked ZIP (_zip_contents); stamps container_url."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    zip_name = file_info.get("name", "")
    zip_download_url = file_info.get("download_url", "")

    for zf in file_info.get("_zip_contents", []):
        zf_path = zf.get("path", "") if isinstance(zf, dict) else zf
        zf_size = zf.get("size", 0) if isinstance(zf, dict) else 0

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
                acquisition=entities.get("acquisition"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend="https",
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            if zip_download_url:
                record["container_url"] = zip_download_url
                record["container_type"] = "zip"
                record["container_name"] = zip_name
            if zf_size:
                record["file_size"] = zf_size
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": zf_path, "error": str(e)})

    return records, errors


def _build_subject_zip_record(
    file_info: dict,
    dataset_id: str,
    storage_base: str,
    primary_mod: str,
    recording_modality_val: list[str],
    digested_at: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Synthesize a placeholder Record for a sub-<id>.zip archive."""
    filepath = file_info.get("path", "")
    download_url = file_info.get("download_url")
    file_size = file_info.get("size", 0)

    subject_match = re.match(r"^(sub-[a-zA-Z0-9]+)\.zip$", filepath, re.IGNORECASE)
    if not subject_match:
        return None, []
    subject_id = subject_match.group(1)
    try:
        record = create_record(
            dataset=dataset_id,
            storage_base=storage_base,
            bids_relpath=(f"{subject_id}/{primary_mod}/{subject_id}_{primary_mod}.set"),
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
        if download_url:
            record["container_url"] = download_url
            record["container_type"] = "zip"
            record["container_name"] = filepath
            record["zip_contains_bids"] = True
        if file_size:
            record["container_size"] = file_size
        return dict(record), []
    except (KeyError, ValueError, TypeError) as e:
        return None, [{"file": filepath, "error": str(e)}]


_BIDS_DATA_ZIP_PATTERNS: tuple[str, ...] = (
    r".*bids.*\.zip$",
    r".*_eeg\.zip$",
    r".*_meg\.zip$",
    r".*_ieeg\.zip$",
    r".*dataset.*\.zip$",
    r".*rawdata.*\.zip$",
    r".*data\.zip$",
    r".*eeg.*\.zip$",
    r".*meg.*\.zip$",
    r".*nirs.*\.zip$",
    r".*fnirs.*\.zip$",
)


def _is_bids_data_zip(filepath: str) -> bool:
    """Return True if the filename matches a known BIDS-bundle ZIP pattern (e.g. ``*_eeg.zip``, ``data.zip``)."""
    fp_lower = filepath.lower()
    return any(re.match(p, fp_lower) for p in _BIDS_DATA_ZIP_PATTERNS)


def _build_bids_data_zip_records(
    file_info: dict,
    manifest: dict,
    dataset_id: str,
    storage_base: str,
    source: str,
    primary_mod: str,
    recording_modality_val: list[str],
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records for a BIDS-bundled ZIP (one per inferred subject, capped at 200, or a single placeholder)."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    filepath = file_info.get("path", "")
    download_url = file_info.get("download_url")
    file_size = file_info.get("size", 0)

    demographics = manifest.get("demographics", {})
    inferred_subjects = demographics.get("subjects_count", 0)

    if inferred_subjects and inferred_subjects > 0:
        for sub_idx in range(1, min(inferred_subjects + 1, 201)):
            sub_id = f"{sub_idx:02d}" if inferred_subjects < 100 else f"{sub_idx:03d}"
            try:
                record = create_record(
                    dataset=dataset_id,
                    storage_base=storage_base,
                    bids_relpath=(
                        f"sub-{sub_id}/{primary_mod}/sub-{sub_id}_{primary_mod}.set"
                    ),
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
                if download_url:
                    record["container_url"] = download_url
                    record["container_type"] = "zip"
                    record["container_name"] = filepath
                    record["needs_extraction"] = True
                    record["inferred_from_metadata"] = True
                if file_size:
                    record["container_size"] = file_size
                records.append(dict(record))
            except (KeyError, ValueError, TypeError) as e:
                errors.append({"file": f"sub-{sub_id}", "error": str(e)})
        return records, errors

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
            storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                "backend"
            ],
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
    except (KeyError, ValueError, TypeError) as e:
        errors.append({"file": filepath, "error": str(e)})

    return records, errors


def _build_regular_manifest_record(
    file_info: dict | str,
    dataset_id: str,
    storage_base: str,
    source: str,
    digested_at: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Synthesize a Record for a regular (non-ZIP) neuro file from the manifest."""
    if isinstance(file_info, dict):
        filepath = file_info.get("path", "")
        download_url = file_info.get("download_url")
        file_size = file_info.get("size", 0)
    else:
        filepath = file_info
        download_url = None
        file_size = 0

    if not is_neuro_data_file(filepath):
        return None, []

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
            acquisition=entities.get("acquisition"),
            dep_keys=[],
            datatype=entities.get("datatype", detected_modality),
            suffix=entities.get("suffix", detected_modality),
            storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                "backend"
            ],
            recording_modality=[detected_modality],
            digested_at=digested_at,
        )
        if download_url:
            record["download_url"] = download_url
        if file_size:
            record["file_size"] = file_size
        return dict(record), []
    except (KeyError, ValueError, TypeError) as e:
        return None, [{"file": filepath, "error": str(e)}]


def _build_standalone_zip_content_records(
    zip_contents: list,
    dataset_id: str,
    storage_base: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize Records from the manifest's top-level zip_contents array."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
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
                acquisition=entities.get("acquisition"),
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
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": filepath, "error": str(e)})
    return records, errors


def _build_ctf_ds_records(
    files: list,
    dataset_id: str,
    storage_base: str,
    source: str,
    digested_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate CTF .ds file entries to one Record per .ds directory."""
    ctf_ds_dirs: set[str] = set()
    for file_info in files:
        filepath = (
            file_info.get("path", "") if isinstance(file_info, dict) else file_info
        )
        filepath_lower = filepath.lower()
        if ".ds/" in filepath_lower:
            # Extract the .ds directory path (everything up to and including ".ds")
            ds_idx = filepath_lower.index(".ds/") + 3  # +3 to include ".ds"
            ctf_ds_dirs.add(filepath[:ds_idx])  # use original case

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
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
                acquisition=entities.get("acquisition"),
                dep_keys=[],
                datatype=entities.get("datatype", detected_modality),
                suffix=entities.get("suffix", detected_modality),
                storage_backend=STORAGE_CONFIGS.get(source, DEFAULT_STORAGE_CONFIG)[
                    "backend"
                ],
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"file": ds_path, "error": str(e)})

    return records, errors


def _enumerate_via_manifest(
    dataset_id: str,
    manifest: dict,
    digested_at: str,
) -> tuple[EnumerationResult, int]:
    """Build an EnumerationResult from a parsed manifest.json and return (result, total_files)."""
    source = _reconcile_source(
        manifest.get("source"), dataset_id, context="digest_from_manifest"
    )

    files = manifest.get("files", [])
    if not files:
        return (
            EnumerationResult(
                dataset_meta={"dataset_id": dataset_id, "source": source},
                digest_method="manifest_only",
            ),
            0,
        )

    zip_contents = manifest.get("zip_contents", [])

    storage_base = _determine_manifest_storage_base(source, dataset_id, manifest)

    subjects, sessions, tasks, modalities = _collect_bids_entities_from_paths(
        files, zip_contents
    )

    demographics = manifest.get("demographics", {})
    # Prefer subject count from validated neuro files over manifest demographics.
    subjects_count = (
        len(subjects)
        if subjects
        else (
            demographics.get("subjects_count", 0)
            or manifest.get("bids_subject_count", 0)
        )
    )

    if subjects_count == 0 or not tasks:
        subjects_count = _fetch_subject_count_via_http(files, fallback=subjects_count)
    ages = demographics.get("ages", [])

    dataset_doi = manifest.get("dataset_doi")
    if not dataset_doi:
        identifiers = manifest.get("identifiers", {})
        dataset_doi = identifiers.get("doi")

    source_url = manifest.get("external_links", {}).get("source_url")
    if not source_url:
        source_url = manifest.get("external_links", {}).get("osf_url")

    fingerprint = fingerprint_from_manifest(dataset_id, source, manifest)

    recording_modality_val = manifest.get("recording_modality")
    if isinstance(recording_modality_val, str):
        recording_modality_val = [
            m.strip()
            for m in recording_modality_val.replace("+", ",").split(",")
            if m.strip()
        ]

    if recording_modality_val:
        recording_modality_val = [normalize_modality(m) for m in recording_modality_val]
        recording_modality_val = sorted(
            list({m for m in recording_modality_val if m in NEURO_MODALITIES})
        )
    if not recording_modality_val:
        recording_modality_val = sorted(
            list({m for m in modalities if m in NEURO_MODALITIES})
        ) or ["eeg"]

    primary_mod = recording_modality_val[0] if recording_modality_val else "eeg"

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

    records = []
    errors = []

    ctf_records, ctf_errors = _build_ctf_ds_records(
        files, dataset_id, storage_base, source, digested_at
    )
    records.extend(ctf_records)
    errors.extend(ctf_errors)

    for file_info in files:
        if isinstance(file_info, dict) and file_info.get("_zip_contents"):
            sub_records, sub_errors = _build_zip_extracted_records(
                file_info, dataset_id, storage_base, digested_at
            )
            records.extend(sub_records)
            errors.extend(sub_errors)
            continue

        filepath = (
            file_info.get("path", "") if isinstance(file_info, dict) else file_info
        )

        if filepath.lower().endswith(".zip") and isinstance(file_info, dict):
            if re.match(r"^(sub-[a-zA-Z0-9]+)\.zip$", filepath, re.IGNORECASE):
                rec, errs = _build_subject_zip_record(
                    file_info,
                    dataset_id,
                    storage_base,
                    primary_mod,
                    recording_modality_val,
                    digested_at,
                )
                if rec is not None:
                    records.append(rec)
                errors.extend(errs)
                continue

            if _is_bids_data_zip(filepath):
                sub_records, sub_errors = _build_bids_data_zip_records(
                    file_info,
                    manifest,
                    dataset_id,
                    storage_base,
                    source,
                    primary_mod,
                    recording_modality_val,
                    digested_at,
                )
                records.extend(sub_records)
                errors.extend(sub_errors)
                continue

        rec, errs = _build_regular_manifest_record(
            file_info, dataset_id, storage_base, source, digested_at
        )
        if rec is not None:
            records.append(rec)
        errors.extend(errs)

    extra_records, extra_errors = _build_standalone_zip_content_records(
        zip_contents, dataset_id, storage_base, digested_at
    )
    records.extend(extra_records)
    errors.extend(extra_errors)

    return (
        EnumerationResult(
            dataset_meta=dict(dataset_doc),
            records=records,
            errors=errors,
            montages={},  # manifest path produces no montages
            digest_method="manifest_only",
        ),
        len(files),
    )


def _attach_montage_to_record(
    record: dict[str, Any],
    bids_file: Any,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    dataset_id: str,
    digested_at: str,
    montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Extract layout, stamp montage_hash on record, and populate montages; MEG layouts cached by (dataset_id, nchans)."""
    record_datatype = (record.get("datatype") or "").lower()
    errors: list[dict[str, Any]] = []

    cache_key: tuple[str, int] | None = None
    record_nchans = record.get("nchans")
    if (
        montage_cache is not None
        and record_datatype == "meg"
        and isinstance(record_nchans, int)
        and record_nchans > 0
    ):
        cache_key = (dataset_id, record_nchans)
        cached = montage_cache.get(cache_key)
        if cached is not None:
            cached_hash, cached_doc = cached
            record["montage_hash"] = cached_hash
            if cached_hash not in montages:
                montages[cached_hash] = cached_doc
            return errors

    try:
        layout_result = extract_layout(
            Path(str(bids_file)), dataset_dir, datatype=record_datatype
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        record["montage_hash"] = None
        errors.append(
            {"file": str(bids_file), "error": f"layout extraction failed: {exc}"}
        )
        return errors

    if layout_result is None:
        record["montage_hash"] = None
        return errors

    h, doc = layout_result
    record["montage_hash"] = h
    if h not in montages:
        doc["first_seen"] = digested_at
        doc["representative_dataset"] = dataset_id
        subject_entity = record.get("entities", {}).get("subject")
        if subject_entity:
            doc["representative_subject"] = f"sub-{subject_entity}"
        montages[h] = doc

    if cache_key is not None:
        montage_cache[cache_key] = (h, montages[h])

    return errors


def _build_one_record_from_bids(
    bids_dataset: Any,
    bids_file: Any,
    dataset_id: str,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Extract one Record, attach its montage, emit telemetry; returns (record_or_None, errors)."""
    errors: list[dict[str, Any]] = []
    try:
        record = extract_record(
            bids_dataset,
            bids_file,
            dataset_id,
            source,
            digested_at,
            source_adapter=source_adapter,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        errors.append({"file": str(bids_file), "error": str(e)})
        get_emitter().emit(
            TelemetryEvent(
                event_kind="record_failed",
                dataset_id=dataset_id,
                record_id=str(bids_file),
                payload={"bids_file": str(bids_file), "error": str(e)},
            )
        )
        return None, errors

    if any("Split FIF" in issue for issue in record.get("_data_integrity_issues", [])):
        errors.append(
            {
                "file": str(bids_file),
                "error": "Split FIF without continuation files — skipped",
            }
        )
        get_emitter().emit(
            TelemetryEvent(
                event_kind="record_failed",
                dataset_id=dataset_id,
                record_id=str(bids_file),
                payload={
                    "bids_file": str(bids_file),
                    "error": "Split FIF without continuation files — skipped",
                },
            )
        )
        return None, errors

    errors.extend(
        _attach_montage_to_record(
            record,
            bids_file,
            dataset_dir,
            montages,
            dataset_id,
            digested_at,
            montage_cache=montage_cache,
        )
    )

    get_emitter().emit(
        TelemetryEvent(
            event_kind="record_built",
            dataset_id=dataset_id,
            record_id=record.get("bids_relpath"),
            payload={
                "bids_relpath": record.get("bids_relpath"),
                "datatype": record.get("datatype"),
                "sampling_frequency": record.get("sampling_frequency"),
                "nchans": record.get("nchans"),
                "ntimes": record.get("ntimes"),
                "metadata_provenance": record.get("_metadata_provenance"),
            },
        )
    )
    return record, errors


def _enumerate_via_bids(
    dataset_dir: Path,
    dataset_id: str,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    bids_dataset,
    manifest_data: dict | None = None,
) -> EnumerationResult:
    """Walk the BIDS filesystem and build an EnumerationResult."""
    files = bids_dataset.get_files()
    if not files:
        return EnumerationResult(
            dataset_meta={"dataset_id": dataset_id, "source": source},
            digest_method="bids_filesystem",
        )

    try:
        dataset_meta = extract_dataset_metadata(
            bids_dataset,
            dataset_id,
            source,
            digested_at,
            metadata=manifest_data,
            source_adapter=source_adapter,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
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
    except (OSError, ValueError, KeyError):
        pass

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    montages: dict[str, dict[str, Any]] = {}

    # MEG layouts are device-defined and identical for the same nchans; cache to avoid re-extraction.
    meg_montage_cache: dict[tuple[str, int], tuple[str, dict[str, Any]]] = {}

    for bids_file in files:
        if not is_neuro_data_file(str(bids_file)):
            continue
        record, per_file_errors = _build_one_record_from_bids(
            bids_dataset=bids_dataset,
            bids_file=bids_file,
            dataset_id=dataset_id,
            source=source,
            source_adapter=source_adapter,
            digested_at=digested_at,
            dataset_dir=dataset_dir,
            montages=montages,
            montage_cache=meg_montage_cache,
        )
        errors.extend(per_file_errors)
        if record is not None:
            records.append(record)

    return EnumerationResult(
        dataset_meta=dataset_meta,
        records=records,
        errors=errors,
        montages=montages,
        digest_method="bids_filesystem",
    )


def _emit_dataset_finished(dataset_id: str, summary: dict[str, Any]) -> None:
    """Emit a dataset_finished telemetry event."""
    get_emitter().emit(
        TelemetryEvent(
            event_kind="dataset_finished",
            dataset_id=dataset_id,
            payload={
                "status": summary.get("status"),
                "record_count": summary.get("record_count"),
                "error_count": summary.get("error_count"),
                "digest_method": summary.get("digest_method"),
                "integrity_issues_count": summary.get("integrity_issues_count"),
                "montage_count": summary.get("montage_count"),
                "total_files": summary.get("total_files"),
            },
        )
    )


def _run_enumerator_with_manifest_fallback(
    enumerator: RecordEnumerator,
    *,
    dataset_id: str,
    dataset_dir: Path,
    source: str,
    source_adapter: SourceAdapter,
    digested_at: str,
    has_manifest: bool,
) -> tuple[EnumerationResult | None, dict[str, Any] | None]:
    """Run enumerator.enumerate() with ManifestEnumerator fallback; returns (result, None) or (None, summary)."""
    try:
        result = enumerator.enumerate()
    except (
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PermissionError,
    ) as exc:
        if has_manifest and not isinstance(enumerator, ManifestEnumerator):
            logger.info(
                "BIDS load failed for %s (%s: %s); falling back to manifest path",
                dataset_id,
                type(exc).__name__,
                exc,
            )
            fallback = ManifestEnumerator(
                dataset_id, dataset_dir, source, source_adapter, digested_at
            )
            try:
                return fallback.enumerate(), None
            except Exception as fb_exc:  # noqa: BLE001
                logger.warning(
                    "Manifest fallback raised %s for %s after BIDS %s: %s",
                    type(fb_exc).__name__,
                    dataset_id,
                    type(exc).__name__,
                    fb_exc,
                )
                return None, {
                    "status": "error",
                    "dataset_id": dataset_id,
                    "error": (
                        f"BIDS path raised {type(exc).__name__}: {exc}; "
                        f"manifest fallback also raised "
                        f"{type(fb_exc).__name__}: {fb_exc}"
                    ),
                }
        return None, {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load BIDS dataset: {exc}",
        }

    if (
        not result.records
        and has_manifest
        and not isinstance(enumerator, ManifestEnumerator)
    ):
        logger.info(
            "BIDS path produced 0 records for %s (errors=%d); "
            "falling back to manifest path",
            dataset_id,
            len(result.errors),
        )
        fallback = ManifestEnumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )
        try:
            fb_result = fallback.enumerate()
        except Exception as fb_exc:  # noqa: BLE001
            logger.warning(
                "Manifest fallback raised %s for %s (BIDS produced 0 records): %s",
                type(fb_exc).__name__,
                dataset_id,
                fb_exc,
            )
            return None, {
                "status": "error",
                "dataset_id": dataset_id,
                "error": (
                    f"BIDS path returned no records; manifest fallback "
                    f"raised {type(fb_exc).__name__}: {fb_exc}"
                ),
                "errors": result.errors,
            }
        if result.errors:
            fb_result.errors = list(fb_result.errors or []) + list(result.errors)
        return fb_result, None

    return result, None


def _summarise_empty_or_error(
    dataset_id: str,
    result: EnumerationResult,
) -> dict[str, Any]:
    """Return an "empty" or "error" summary dict when no records were produced."""
    structural_errors = [
        e for e in result.errors if e.get("status") not in (None, "skipped", "warning")
    ]
    if structural_errors:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No records extracted",
            "errors": result.errors,
        }

    total_files = result.total_files
    if total_files == 0:
        reason = "no files in manifest"
    elif total_files is not None:
        reason = "no records extracted"
    else:
        reason = "no neurophysiology files found"
    empty_summary: dict[str, Any] = {
        "status": "empty",
        "dataset_id": dataset_id,
        "reason": reason,
    }
    if result.errors:
        empty_summary["errors"] = result.errors
    return empty_summary


def _check_dataset_skip_conditions(
    dataset_id: str,
    dataset_dir: Path,
    dataset_output_dir: Path,
) -> dict[str, Any] | None:
    """Return a skip summary if output exists or input dir is missing, else None."""
    if dataset_output_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "already digested",
        }
    if not dataset_dir.exists():
        return {
            "status": "skipped",
            "dataset_id": dataset_id,
            "reason": "directory not found",
        }
    return None


def digest_dataset(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a single dataset; write dataset/records/montages/summary JSON and return the summary."""
    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id

    skip = _check_dataset_skip_conditions(dataset_id, dataset_dir, dataset_output_dir)
    if skip is not None:
        return skip

    source = detect_source(dataset_dir)
    digested_at = datetime.now(timezone.utc).isoformat()
    _repair_participants_tsv_ids(dataset_dir)
    source_adapter = get_source_adapter(source, dataset_id, dataset_dir)

    get_emitter().emit(
        TelemetryEvent(
            event_kind="dataset_started",
            dataset_id=dataset_id,
            payload={"source": source, "dataset_dir": str(dataset_dir)},
        )
    )

    summary: dict[str, Any] | None = None
    try:
        has_manifest = (dataset_dir / "manifest.json").exists()
        enumerator = get_record_enumerator(
            dataset_id, dataset_dir, source, source_adapter, digested_at
        )

        result, fallback_summary = _run_enumerator_with_manifest_fallback(
            enumerator,
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            source=source,
            source_adapter=source_adapter,
            digested_at=digested_at,
            has_manifest=has_manifest,
        )
        if fallback_summary is not None:
            summary = fallback_summary
            return fallback_summary

        assert (
            result is not None
        )  # exactly one of (result, fallback_summary) is non-None

        if not result.records:
            summary = _summarise_empty_or_error(dataset_id, result)
            return summary

        summary = write_dataset_outputs(
            dataset_output_dir,
            result,
            dataset_id=dataset_id,
            source=source,
            digested_at=digested_at,
            total_files=result.total_files,
        )
        return summary
    finally:
        if summary is None:
            summary = {
                "status": "error",
                "dataset_id": dataset_id,
                "error": "digest_dataset raised an unhandled exception",
            }
        _emit_dataset_finished(dataset_id, summary)


def _json_serializer(obj):
    """Custom JSON serializer for numpy scalars, ndarrays, Paths, sets, and pandas types."""
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
    """Return dataset IDs from input_dir that have a manifest or dataset_description."""
    if datasets:
        return datasets

    found = []
    for d in input_dir.iterdir():
        if (
            d.is_dir()
            and d.name not in ("__pycache__", ".git")
            and d.name not in EXCLUDED_DATASETS
        ):
            if (d / "manifest.json").exists() or (
                d / "dataset_description.json"
            ).exists():
                found.append(d.name)

    return sorted(found)


def _positive_float(value: str) -> float:
    """Argparse type validator for positive floats."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _dataset_boundary_profile(dataset_id: str, input_dir: Path) -> str:
    """Return a structural profile string for stall-boundary diagnostics."""
    dataset_dir = input_dir / dataset_id
    if not dataset_dir.exists():
        return f"{dataset_id}: missing directory"

    manifest_path = dataset_dir / "manifest.json"
    description_path = dataset_dir / "dataset_description.json"
    parts = [
        f"{dataset_id}:",
        f"pattern_source={_source_from_dataset_id(dataset_id)}",
        f"manifest={manifest_path.exists()}",
        f"dataset_description={description_path.exists()}",
    ]

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_source = manifest.get("source")
            files = manifest.get("files", [])
            zip_contents = manifest.get("zip_contents", [])
            files_count = len(files) if isinstance(files, list) else "n/a"
            zip_contents_count = (
                len(zip_contents) if isinstance(zip_contents, list) else "n/a"
            )
            parts.extend(
                [
                    f"manifest_source={manifest_source!r}",
                    f"manifest_files={files_count}",
                    f"zip_contents={zip_contents_count}",
                ]
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parts.append(f"manifest_error={exc}")

    try:
        root_entries = list(dataset_dir.iterdir())
        root_dirs = sum(1 for p in root_entries if p.is_dir())
        root_files = sum(1 for p in root_entries if p.is_file() or p.is_symlink())
        parts.extend([f"root_dirs={root_dirs}", f"root_files={root_files}"])
    except (OSError, PermissionError) as exc:
        parts.append(f"root_scan_error={exc}")

    return " ".join(parts)


def print_stall_boundary_diagnostics(dataset_ids: list[str], input_dir: Path) -> None:
    """Print structural profiles around the #287 deterministic stall boundary."""
    boundary_indices = [285, 286]  # 0-based: completed #286 and next #287.
    present = [idx for idx in boundary_indices if idx < len(dataset_ids)]
    if not present:
        return

    print("Stall-boundary diagnostics:")
    for idx in present:
        profile = _dataset_boundary_profile(dataset_ids[idx], input_dir)
        print(f"  #{idx + 1} (0-based {idx}): {profile}")


def _worker_error_result(
    dataset_id: str,
    error: str,
    *,
    elapsed_seconds: float | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    """Build a standardised error result dict for batch summary accounting."""
    result: dict[str, Any] = {
        "status": "error",
        "dataset_id": dataset_id,
        "error": error,
    }
    if elapsed_seconds is not None:
        result["elapsed_seconds"] = round(elapsed_seconds, 3)
    if traceback_text:
        result["traceback"] = traceback_text
    return result


def _digest_dataset_worker(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    result_queue: Any,
) -> None:
    """Subprocess entry: digest one dataset and put the result dict (or error) on the queue."""
    try:
        try:
            result = digest_dataset(dataset_id, input_dir, output_dir)
            if not isinstance(result, dict):
                result = _worker_error_result(
                    dataset_id,
                    f"digest_dataset returned {type(result).__name__}, expected dict",
                )
        except BaseException as exc:  # noqa: BLE001 — worker boundary: parent decides retry/shutdown
            result = _worker_error_result(
                dataset_id,
                f"{type(exc).__name__}: {exc}",
                traceback_text=traceback.format_exc(),
            )
        result_queue.put(result, timeout=RESULT_QUEUE_TIMEOUT_SECONDS)
    finally:
        logging.shutdown()


def _start_digest_process(
    ctx: mp.context.BaseContext,
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
    *,
    position: int,
    total: int,
) -> dict[str, Any]:
    """Spawn a child process for one dataset and return the active-job dict."""
    result_queue = ctx.Queue(maxsize=100)
    process = ctx.Process(
        target=_digest_dataset_worker,
        args=(dataset_id, input_dir, output_dir, result_queue),
        name=f"digest-{position}-{dataset_id}",
    )
    process.start()
    return {
        "dataset_id": dataset_id,
        "position": position,
        "total": total,
        "process": process,
        "queue": result_queue,
        "started_at": time.monotonic(),
    }


def _close_active_resources(active: dict[str, Any]) -> None:
    """Release process and queue handles (best-effort, non-blocking)."""
    result_queue = active.get("queue")
    if result_queue is not None:
        try:
            result_queue.close()
        except (OSError, ValueError, AttributeError):
            pass

    process = active.get("process")
    if process is not None:
        try:
            process.close()
        except (OSError, ValueError, AttributeError):
            pass


def _terminate_active_process(active: dict[str, Any], reason: str) -> None:
    """Terminate a child process, escalating to SIGKILL if SIGTERM is ignored."""
    process = active["process"]
    dataset_id = active["dataset_id"]

    if process.is_alive():
        tqdm.write(f"[digest] terminating {dataset_id}: {reason}")
        process.terminate()
        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)

    if process.is_alive():
        tqdm.write(f"[digest] killing {dataset_id}: terminate did not exit")
        process.kill()
        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)

    if process.is_alive():
        tqdm.write(f"[digest] warning {dataset_id}: process still alive after kill")


def _collect_finished_process(active: dict[str, Any]) -> dict[str, Any]:
    """Collect one completed child result from the queue with an explicit timeout."""
    process = active["process"]
    dataset_id = active["dataset_id"]
    elapsed = time.monotonic() - active["started_at"]

    try:
        result = active["queue"].get(timeout=RESULT_QUEUE_TIMEOUT_SECONDS)
    except queue.Empty:
        result = _worker_error_result(
            dataset_id,
            f"worker exited without returning a result (exitcode={process.exitcode})",
            elapsed_seconds=elapsed,
        )
    except (OSError, EOFError, ValueError) as exc:
        result = _worker_error_result(
            dataset_id,
            f"failed to collect worker result: {type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
        )

    process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
    if process.is_alive():
        _terminate_active_process(active, "process still alive after result collection")

    if isinstance(result, dict):
        result.setdefault("dataset_id", dataset_id)
        result.setdefault("elapsed_seconds", round(elapsed, 3))
        return result

    return _worker_error_result(
        dataset_id,
        f"worker returned {type(result).__name__}, expected dict",
        elapsed_seconds=elapsed,
    )


def _timeout_active_process(
    active: dict[str, Any],
    dataset_timeout: float,
) -> dict[str, Any]:
    """Kill a stalled dataset worker and return an error summary."""
    elapsed = time.monotonic() - active["started_at"]
    dataset_id = active["dataset_id"]
    _terminate_active_process(
        active,
        f"dataset exceeded {dataset_timeout:.1f}s timeout",
    )
    return _worker_error_result(
        dataset_id,
        f"dataset exceeded {dataset_timeout:.1f}s timeout",
        elapsed_seconds=elapsed,
    )


def process_datasets_with_watchdog(
    dataset_ids: list[str],
    input_dir: Path,
    output_dir: Path,
    *,
    workers: int,
    dataset_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Process datasets with per-dataset subprocess isolation and per-operation timeouts."""
    total = len(dataset_ids)
    max_workers = max(1, workers)
    ctx = mp.get_context()
    active: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    stats = {"success": 0, "error": 0, "skipped": 0, "empty": 0}
    next_index = 0

    with tqdm(total=total, desc="Digesting") as progress:
        try:
            while next_index < total or active:
                while next_index < total and len(active) < max_workers:
                    dataset_id = dataset_ids[next_index]
                    active_job = _start_digest_process(
                        ctx,
                        dataset_id,
                        input_dir,
                        output_dir,
                        position=next_index + 1,
                        total=total,
                    )
                    active[id(active_job["process"])] = active_job
                    next_index += 1

                finished: list[tuple[int, dict[str, Any]]] = []
                now = time.monotonic()
                for key, active_job in list(active.items()):
                    dataset_id = active_job["dataset_id"]
                    process = active_job["process"]
                    elapsed = now - active_job["started_at"]

                    # Drain queue non-blocking first to prevent deadlock on a full queue.
                    result_queue = active_job["queue"]
                    try:
                        result = result_queue.get_nowait()
                        print(
                            f"[QUEUE] Got result from {dataset_id} after {elapsed:.1f}s",
                            flush=True,
                        )
                        finished.append((key, result))
                        process.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
                        continue
                    except (queue.Empty, OSError, EOFError, ValueError) as e:
                        print(
                            f"[QUEUE] No result yet for {dataset_id}: {type(e).__name__}",
                            flush=True,
                        )

                    if process.is_alive():
                        if elapsed > dataset_timeout:
                            result = _timeout_active_process(
                                active_job, dataset_timeout
                            )
                            finished.append((key, result))
                        continue

                    result = _collect_finished_process(active_job)
                    finished.append((key, result))

                if not finished:
                    time.sleep(WORKER_POLL_INTERVAL_SECONDS)
                    continue

                for key, result in finished:
                    active_job = active.pop(key, None)
                    if active_job is not None:
                        _close_active_resources(active_job)
                    results.append(result)
                    status = result.get("status", "error")
                    stats[status] = stats.get(status, 0) + 1
                    if status == "error":
                        tqdm.write(
                            f"[digest] error {result.get('dataset_id')}: "
                            f"{result.get('error', 'unknown error')}"
                        )
                    progress.update(1)
        except KeyboardInterrupt:
            for active_job in list(active.values()):
                _terminate_active_process(active_job, "interrupted")
                _close_active_resources(active_job)
            raise

    return results, stats


def main():
    try:
        cfg = load_digest_config_from_argv()
    except ValidationError as exc:
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    dataset_ids = find_datasets(cfg.input, cfg.datasets)
    if cfg.limit:
        dataset_ids = dataset_ids[: cfg.limit]

    print(f"Found {len(dataset_ids)} datasets to digest")
    print(f"Workers: {cfg.workers}")
    print(f"Dataset timeout: {cfg.dataset_timeout:g}s")
    print_stall_boundary_diagnostics(dataset_ids, cfg.input)
    print("=" * 60)

    cfg.output.mkdir(parents=True, exist_ok=True)

    results, stats = process_datasets_with_watchdog(
        dataset_ids,
        cfg.input,
        cfg.output,
        workers=cfg.workers,
        dataset_timeout=cfg.dataset_timeout,
    )

    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_datasets": len(dataset_ids),
        "stats": stats,
        "total_records": sum(
            r.get("record_count", 0) for r in results if r.get("status") == "success"
        ),
    }

    batch_summary_path = cfg.output / "BATCH_SUMMARY.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

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
