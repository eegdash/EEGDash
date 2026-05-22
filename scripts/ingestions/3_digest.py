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
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid numba cache issues by setting cache dir before importing MNE.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache") / "numba"))

import numpy as np
import pandas as pd

# Storage configuration: imported from the canonical home in eegdash.
# Was previously duplicated as ``STORAGE_CONFIGS`` here with a "Keep
# aligned" comment — see Phase 8 S1.thick (2026-05) for the consolidation.
# The eegdash side documents NEMAR's "nemar" backend marker (filenames
# SHA-resolved by git-annex; not directly fetchable).
from eegdash.dataset._source_inference import (
    STORAGE_CONFIGS,
    get_storage_backend,
    get_storage_base,
    get_storage_config,
)
from eegdash.dataset.bids_dataset import _COMPANION_FILES
from eegdash.dataset.io import _repair_participants_tsv_ids
from eegdash.schemas import (
    Storage,
    create_dataset,
    create_record,
)
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS
from tqdm import tqdm

from _constants import (
    CTF_INTERNAL_EXTENSIONS,
    MEF3_INTERNAL_DIRS,
    MEF3_INTERNAL_EXTENSIONS,
    MODALITY_CANONICAL_MAP,
    MODALITY_DETECTION_TARGETS,
    NEURO_MODALITIES,
)
from _file_utils import (
    get_annex_file_size,
)
from _fingerprint import fingerprint_from_files, fingerprint_from_manifest

# Technical-metadata cascade (SPRINT-2026-05-22 Task 3). The 5-step
# cascade (mne_bids → modality_sidecar → channels_tsv → binary_parser →
# mne_fallback) lives in ``_metadata_cascade.py``; this file's
# ``_extract_technical_metadata`` is a thin delegator below. The
# helpers and FIF parser are re-exported from here for back-compat
# with ``tests/test_digest_extractions.py`` and other call sites that
# imported them via ``3_digest``. (The per-extension binary parser
# registry lives in ``_format_parser_registry``; the cascade imports
# it directly — see ``_metadata_cascade.py``.)
from _metadata_cascade import (  # noqa: F401 — re-export for back-compat
    _parse_fif_with_mne,
    extract_sfreq_nchans_from_channels_tsv,
    extract_sfreq_nchans_from_modality_sidecar,
    sum_bids_channel_counts,
)
from _montage import extract_layout

# Telemetry (ROADMAP P1.1). Default emitter is NullEmitter so digest
# behaviour is unchanged when telemetry is disabled. Configure by
# setting $EEGDASH_TELEMETRY_PATH or calling configure_telemetry() in
# the caller.
from digest_telemetry import TelemetryEvent, auto_configure_from_env, get_emitter

# Record-enumeration Seam (Phase 8 S1.thick stage 2). Both legacy
# functions in this file write their JSON outputs via this shared
# helper, so the per-Dataset JSON shapes are documented in ONE place.
# Stage 3 will collapse digest_dataset + digest_from_manifest into a
# single orchestrator — see ROBUSTNESS/STAGE-3-PLAN.md for the
# fixture-based verification approach.
from record_enumerator import (
    EnumerationResult,
    ManifestEnumerator,
    get_record_enumerator,
    write_dataset_outputs,
)

# Install the env-driven emitter at module import time. Cheap if
# $EEGDASH_TELEMETRY_PATH isn't set (returns the NullEmitter).
auto_configure_from_env()

# Per-Source ingest behaviour (Phase 8 S1.thick). The orchestrator builds
# one Adapter per Dataset; extract_dataset_metadata + extract_record
# consume it via the optional ``source_adapter`` kwarg. The old
# ``source: str`` / ``apex_sidecar_inline`` parameters remain for
# back-compat; when ``source_adapter`` is provided it wins and the 4
# if-ladders that used to live in this file go through the Adapter.
from source_adapter import SourceAdapter, get_source_adapter

DEFAULT_DATASET_TIMEOUT_SECONDS = 2 * 60
WORKER_POLL_INTERVAL_SECONDS = 1.0
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5.0
RESULT_QUEUE_TIMEOUT_SECONDS = 5.0

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
    # OpenNeuro IDs that now redirect to other datasets on openneuro.org.
    # ds004929 and ds005930 are fNIRS-only (no EEG); ds005407 redirects.
    "ds004929",
    "ds005407",
    "ds005930",
}


# get_storage_config / get_storage_base / get_storage_backend now live in
# eegdash.dataset._source_inference and are imported above. The duplicated
# definitions that used to live here were removed in Phase 8 S1.thick.


def _source_from_dataset_id(dataset_id: str) -> str:
    """Infer source from a dataset_id pattern.

    Mirrors the pattern-based detection used in ``2_clone.py`` so the two
    stages stay aligned (OpenNeuro ``dsNNNNNN``, NEMAR ``nmNNNNNN``).
    """
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
    """Resolve the authoritative source for a dataset.

    When the manifest carries a source but the dataset_id pattern implies
    a different one (the bug behind the NEMAR mislabel — manifests said
    ``openneuro`` for ``nm*`` ids), trust the pattern and warn loudly.
    Without this guardrail the bad value flows straight into Mongo and we
    get records pointing at the wrong S3 bucket.
    """
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
            # Manifest is optional; absent or malformed -> fall back to
            # heuristic source detection in _reconcile_source.
            manifest_src = None

    return _reconcile_source(manifest_src, dataset_id, context="detect_source")


# _parse_edf_with_mne moved to _format_parser_registry.py (ROADMAP
# P2.2). The cascade reaches it via get_parser_for_extension(".edf")
# and get_parser_for_extension(".bdf") — same MNE-based reader, one
# canonical home for all single-dict format parsers.


# ``_parse_fif_with_mne`` moved to ``_metadata_cascade.py`` (SPRINT-2026-05-22
# Task 3). Re-exported at the top of this module via the
# ``from _metadata_cascade import ...`` block so existing call sites
# (``_extract_technical_metadata`` delegator + tests/test_digest_*) keep
# working unchanged.


# Companion files required for different formats
# These are critical files without which the data cannot be loaded
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
    """Check if file exists, considering symlinks.

    Parameters
    ----------
    path : Path
        Path to check.
    allow_symlinks : bool
        If True, accept broken symlinks (git-annex) as "existing".

    Returns
    -------
    bool
        True if file exists or is a symlink (when allow_symlinks is True).

    """
    if path.exists():
        return True
    if allow_symlinks and path.is_symlink():
        return True
    return False


def validate_companion_files(
    file_path: Path, allow_symlinks: bool = True
) -> dict[str, Any]:
    """Validate that required companion files exist for a data file.

    Parameters
    ----------
    file_path : Path
        Path to the primary data file (e.g., .vhdr, .set)
    allow_symlinks : bool
        If True, accept broken symlinks (git-annex) as "existing"

    Returns
    -------
    dict
        Validation result with keys:
        - valid: bool - True if file can be processed
        - missing_required: list[str] - Missing required companion files
        - missing_optional: list[str] - Missing optional companion files
        - found: list[str] - Found companion files
        - warnings: list[str] - Warning messages
        - errors: list[str] - Error messages

    """
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
            # .vhdr is a permissive INI; extract_vhdr_references already
            # tolerates malformed input. Anything escaping is
            # filesystem-level and not worth blocking on.
            pass

    return result


def _read_bids_readme(bids_root: Path) -> str | None:
    """Return cleaned README text, or None if absent / unreadable.

    Tries the standard BIDS README filenames in order:
    ``README``, ``README.md``, ``README.txt``, ``readme``, ``readme.md``.

    Pure I/O. Phase 8 follow-up — was inline at the top of
    :func:`extract_dataset_metadata`.
    """
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
            # README absent / unreadable / non-UTF8. Try the next
            # candidate filename in the list.
            continue
    return None


def _read_participants_demographics(
    bids_root: Path,
) -> tuple[int, list[int], dict[str, int], dict[str, int]]:
    """Read ``participants.tsv``; return ``(count, ages, sex, handedness)``.

    Demographics aggregation from BIDS ``participants.tsv``:
    - ``subjects_count``: row count (0 if absent / malformed)
    - ``ages``: list of valid integer ages (0 < age < 120)
    - ``sex_distribution``: ``{"m": N, "f": N, "o": N}`` (only keys
      with > 0 entries appear)
    - ``handedness_distribution``: ``{"r": N, "l": N, "a": N}``
      (right / left / ambidextrous)

    Tolerant of malformed files: any pandas / encoding error yields
    empty demographics (the dataset metadata still gets emitted).

    Phase 8 follow-up — was inline in :func:`extract_dataset_metadata`.
    """
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
        # participants.tsv malformed / unreadable / non-UTF8 / wrong
        # encoding. Demographics stay empty; the rest of dataset
        # metadata still gets emitted.
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
    """Build the Dataset-level ``storage`` doc.

    Walks the BIDS root looking for canonical global files (the
    explicit ordered list in :data:`_BIDS_GLOBAL_FILES`) + any other
    root-level BIDS metadata files (sidecars, TSVs, JSONs). Returns a
    :class:`Storage` dict with:
    - ``backend``: per-Source storage backend marker
    - ``base``: per-Source storage base URL
    - ``raw_key``: ``"dataset_description.json"`` (the "main" file)
    - ``dep_keys``: sorted list of other root-level metadata files

    Returns ``None`` if the source has no storage config — keeps the
    pre-extraction behaviour.

    Phase 8 follow-up — was inline in :func:`extract_dataset_metadata`.
    """
    if source not in ("openneuro", "nemar", "gin") and source not in STORAGE_CONFIGS:
        return None

    storage_base = get_storage_base(dataset_id, source)
    storage_backend = get_storage_backend(source)

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
            # dataset_description.json missing or malformed → empty
            # description dict; downstream code defaults each field.
            pass

    readme = _read_bids_readme(bids_root)

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

    (
        subjects_count,
        ages,
        sex_distribution,
        handedness_distribution,
    ) = _read_participants_demographics(bids_root)
    participants_path = bids_root / "participants.tsv"

    # Warn when participants.tsv row count differs from sub-* folder count.
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

    # Build source URL via the SourceAdapter. OpenNeuro and NEMAR each
    # return their canonical landing page; gin still has no Adapter
    # override (ADR 0001) so the fallback handles its EEGManyLabs sub-path.
    if source_adapter is not None:
        source_url = source_adapter.dataset_url()
    else:
        source_url = None
    if source_url is None and source == "gin":
        # gin's project-sub-path doesn't fit the generic <base>/<id>
        # shape; keep the inline builder until gin gets an Adapter.
        source_url = f"https://gin.g-node.org/EEGManyLabs/{dataset_id}"

    # Extract timestamps from metadata if available
    dataset_created_at = metadata.get("dataset_created_at")
    dataset_modified_at = metadata.get("dataset_modified_at")
    senior_author = metadata.get("senior_author")
    contact_info = metadata.get("contact_info")

    # Extract clinical classification fields
    is_clinical = metadata.get("is_clinical")
    clinical_purpose = metadata.get("clinical_purpose")

    if not dataset_modified_at:
        # Try to get from manifest timestamps dict if present
        ts = metadata.get("timestamps", {})
        if isinstance(ts, dict):
            dataset_modified_at = ts.get("dataset_modified_at")
            dataset_created_at = ts.get("dataset_created_at") or dataset_created_at

    # Extract size_bytes — prefer source API value (e.g. OpenNeuro sets this
    # during fetch), otherwise compute from local files resolving git-annex
    # pointers.  Do NOT trust manifest "total_size" — it was computed from
    # clone pointer files and is wrong for git-annex datasets.
    size_bytes = metadata.get("size_bytes")
    if size_bytes is None and bids_root.exists():
        size_bytes = sum(
            get_annex_file_size(f)
            for f in bids_root.rglob("*")
            if f.is_file() or f.is_symlink()
        )

    storage_info = _build_global_storage_info(dataset_id, source, bids_root)

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
        storage=storage_info,
        # Clinical classification
        is_clinical=is_clinical,
        clinical_purpose=clinical_purpose,
    )

    # C6.1 — extra BIDS-spec dataset_description.json fields beyond the
    # basic name / license / authors set already passed to create_dataset.
    # Covers Acknowledgements, HowToAcknowledge, EthicsApprovals,
    # ReferencesAndLinks, GeneratedBy, SourceDatasets. See
    # ROBUSTNESS/BIDS-GAP-AUDIT.md for the rationale.
    description_extras = _extract_dataset_description_extras(description)
    dataset.update(description_extras)

    return dict(dataset)


# Provenance source names. These are the values that show up in a
# Record's ``_metadata_provenance`` field. See ROBUSTNESS/ROADMAP.md
# P0.1 for the motivation and PROGRESS-8 for the implementation
# narrative.
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
    """Record ``source`` as the provenance of ``field`` if this step filled it.

    "Filled" means: the field was None before this step and is not
    None after. The first source to fill a field wins — later sources
    don't overwrite (mirrors the cascade's ``X = X or new_X`` pattern).
    """
    if old_value is None and new_value is not None and provenance[field] is None:
        provenance[field] = source


# ─── BIDS sidecar enrichment (C6.1 — BIDS-GAP-AUDIT.md) ────────────────────


# Per-modality sidecar fields we surface on each Record (BIDS spec).
# Mapping is camelCase-in-sidecar → snake_case-in-Record. Kept narrow
# (highest-leverage fields per BIDS-GAP-AUDIT.md); easy to extend
# without breaking older Records because RecordModel uses extra="allow".
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
    """Pull BIDS-spec sidecar fields beyond the technical-metadata cascade.

    The cascade in ``_extract_technical_metadata`` only reads
    sampling_frequency / nchans / ntimes / ch_names. This helper picks
    up the OTHER required + recommended fields from the modality
    sidecar (e.g. PowerLineFrequency, EEGReference, SoftwareFilters)
    so they're queryable in MongoDB rather than just embedded as bytes
    in ``sidecar_inline``.

    Reads the modality JSON sidecar directly (BIDS inheritance: data
    dir → session dir → subject dir → root) because
    ``get_bids_file_attribute`` only handles a hardcoded list of keys
    (sfreq / nchans / ntimes / subject / etc.) — arbitrary BIDS
    fields like PowerLineFrequency aren't surfaced through that API.

    Returns a dict of {snake_case_field: value}. Empty dict if no
    sidecar found / nothing extractable. None values are NOT included
    (so callers can ``record.update(...)`` without overwriting).

    See ``BIDS-GAP-AUDIT.md`` for the full field list + rationale.
    """
    out: dict[str, Any] = {}

    # Resolve the BIDS root + modality
    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
        modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    except (AttributeError, TypeError):
        return {}

    # BIDS sidecar naming: ``<entities>_<modality>.json``. Walk up from
    # the data file's parent to bids_root looking for the first match.
    try:
        from _montage import _walk_up_find as _walk
    except ImportError:
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
    """Pull per-channel status counts from channels.tsv (good / bad).

    Returns ``{}`` when channels.tsv is absent or has no status column.
    Otherwise returns ``{"bad_channels": [...], "bad_channels_count": N}``.

    This is the structured counterpart to the raw bytes already inlined
    via ``sidecar_inline``: instead of forcing every consumer to re-parse
    the TSV, we surface the count + names at digest time.
    """
    import csv as _csv

    # Walk up from the data file to find channels.tsv (BIDS inheritance).
    try:
        bids_root = Path(bids_dataset.bidsdir)
        data_file = Path(bids_file)
    except (AttributeError, TypeError):
        return {}

    # Reuse _montage.py's _walk_up_find — same BIDS-inheritance semantics.
    try:
        from _montage import _walk_up_find as _walk
    except ImportError:
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
            # Locate the status column case-insensitively
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
        # status column exists but no bad channels → still report 0
        return {"bad_channels": [], "bad_channels_count": 0}
    return {"bad_channels": bad_channels, "bad_channels_count": len(bad_channels)}


# Dataset-level fields we surface from dataset_description.json beyond
# the basic name / license / authors set already captured.
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
    """Pull extra BIDS-spec fields from dataset_description.json.

    Returns a dict of {snake_case_field: value} for fields beyond the
    basic name / license / authors / funding set already captured by
    ``extract_dataset_metadata``. Empty values omitted.

    See ``BIDS-GAP-AUDIT.md`` for the rationale (attribution +
    regulatory metadata).
    """
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
    """Resolve sfreq / nchans / ntimes / ch_names + their provenance.

    Thin delegator to :class:`_metadata_cascade.MetadataCascade` — see
    that module for the 5-step cascade (mne_bids → modality_sidecar →
    channels_tsv → binary_parser → mne_fallback) and provenance
    semantics. SPRINT-2026-05-22 Task 3.
    """
    from _metadata_cascade import CascadeContext, MetadataCascade

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
    """Return ``(dep_keys, fif_is_split, fif_continuations_ok)`` for one BIDS file.

    Searches for three classes of companion files relative to
    ``bids_file_path`` and resolves them to relpaths under ``bids_root``:

    1. **BIDS sidecars** (``_channels.tsv``, ``_events.tsv``,
       ``_electrodes.tsv``, ``_coordsystem.json``, ``_eeg.json``,
       NIRS optodes, …) — checked under the file's parent directory
       AND under the next-level-up (BIDS inheritance). Two base-name
       variants are tried: the full filename's base and a
       session-level base (with task / run / acq entities stripped).

    2. **Format-specific companions** (``.fdt`` for ``.set`` files,
       ``.eeg`` / ``.vmrk`` for ``.vhdr`` files, …) — sourced from
       :data:`_COMPANION_FILES`. Always added regardless of disk
       presence so the runtime download manifest can fetch them even
       when the local clone is incomplete.

    3. **Split FIF continuations** (``filename-1.fif``,
       ``filename-2.fif``, …) — detected by checking the filesystem;
       if ``filename-1.fif`` exists, walk continuations up to 99.
       Sets ``fif_continuations_ok=False`` if any continuation is a
       broken git-annex symlink.

    Phase 8 follow-up — was ~100 LOC inline in :func:`extract_record`.
    """
    dep_keys: list[str] = []
    parent_dir = bids_file_path.parent
    base_name = bids_file_path.stem.rsplit("_", 1)[0]

    # BIDS sidecars: search the file's dir AND one level up (for inheritance)
    search_dirs = [parent_dir]
    if parent_dir.name in NEURO_MODALITIES or parent_dir.name in {
        "eeg",
        "meg",
        "ieeg",
        "beh",
        "nirs",
    }:
        search_dirs.append(parent_dir.parent)

    # Two base-name variants: the full one, and a session-level one
    # (with task / run / acq entities stripped). Session-level sidecars
    # like optodes apply to all runs in a session.
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

    # Format-specific companions (always added — runtime needs them).
    ext = bids_file_path.suffix.lower()
    for comp_ext in _COMPANION_FILES.get(ext, []):
        comp_file = bids_file_path.with_suffix(comp_ext)
        try:
            dep_keys.append(str(comp_file.relative_to(bids_root)))
        except ValueError:
            pass

    # Split-FIF continuations.
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
    """Sanity-check sfreq + nchans; reject impossible values, warn on suspicious.

    - ``sampling_frequency``: must be > 0 (reject); warn if > 1 MHz.
    - ``nchans``: must be > 0 (reject); warn if > 10 000.
    - ``ch_names`` count vs ``nchans``: debug-log on mismatch.

    Returns ``(sfreq, nchans)`` with rejected values replaced by None.
    When ``provenance`` is supplied, the rejected field's provenance
    entry is also cleared — maintains the invariant
    "provenance is None iff value is None" for the final Record.

    Phase 8 follow-up — was inline in :func:`extract_record`.
    """
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
    acquisition = bids_dataset.get_bids_file_attribute("acquisition", bids_file)
    modality = bids_dataset.get_bids_file_attribute("modality", bids_file) or "eeg"
    mod_canon = normalize_modality(modality) or "eeg"

    # Get BIDS relative path (without dataset prefix)
    bids_relpath = strip_dataset_prefix(
        str(bids_dataset.get_relative_bidspath(bids_file)), dataset_id
    )

    # Determine datatype and suffix
    datatype = mod_canon
    suffix = mod_canon

    # Get storage info
    storage_base = get_storage_base(dataset_id, source)
    storage_backend = get_storage_backend(source)

    # Run the 4-step technical-metadata cascade (sidecar / channels.tsv /
    # binary parser / MNE). Returns the 4 metadata values + 2 FIF flags +
    # the provenance dict (P0.1 — cascade-with-provenance).
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

    # Validate companion files exist
    companion_validation = validate_companion_files(bids_file_path, allow_symlinks=True)
    data_integrity_issues = []

    if not companion_validation["valid"]:
        for error in companion_validation["errors"]:
            logging.warning("Data integrity issue for %s: %s", bids_relpath, error)
            data_integrity_issues.append(error)

    for warning in companion_validation.get("warnings", []):
        logging.info("Companion file note for %s: %s", bids_relpath, warning)

    # Flag split FIF files with missing continuation files
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

    # NEMAR fast-path enrichment for the runtime; see Storage TypedDict
    # in eegdash/schemas.py for the contract.
    #
    # Apex BIDS files (participants.tsv, dataset_description.json, README,
    # …) live once per dataset but mne-bids/braindecode expects them on
    # disk per recording. We get them via the caller-provided
    # apex_sidecar_inline (computed once per dataset; see digest_dataset)
    # rather than re-reading from disk per record. They're stored on every
    # record so each load is self-contained, at the cost of duplicated
    # bytes in MongoDB.
    #
    # TODO(scale): two known sources of byte duplication in MongoDB:
    #   (1) session-level sidecars (events.json shared across runs in a
    #       session) — multi-KB × N runs in same session;
    #   (2) apex sidecars (dataset_description.json, README, etc.)
    #       duplicated across every record in a dataset.
    # The structurally-correct fix is to move both classes into a per-
    # dataset side-collection (`nemar_dataset_sidecars` keyed by
    # (dataset_id, relpath)) referenced by record. Trigger to revisit:
    # >100 MB inline payload for a single dataset, OR an HBN/M3CV-style
    # ingest with a >500 KB participants.tsv.
    # Per-Source annex-key + inline-sidecar resolution via the
    # SourceAdapter (Phase 8 S1.thick — was a NEMAR if-ladder here).
    # OpenNeuro/Default return ({}, {}); NEMAR populates from its apex
    # cache + per-file annex-key resolution.
    if source_adapter is None:
        # Lazy-build for callers that haven't migrated. Should be
        # unreachable in production (digest_dataset always passes one).
        source_adapter = get_source_adapter(source, dataset_id, bids_dataset.bidsdir)
    bids_root_path = bids_dataset.bidsdir
    dep_paths = [bids_root_path / dep for dep in dep_keys]
    annex_keys, sidecar_inline = source_adapter.resolve_storage_extensions(
        Path(bids_file), dep_paths
    )
    # ``apex_sidecar_inline`` is the legacy parameter; merge any caller-
    # supplied entries that the Adapter didn't already provide.
    if apex_sidecar_inline:
        for k, v in apex_sidecar_inline.items():
            sidecar_inline.setdefault(k, v)

    # Create record using the schema
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

    # Restore participant_tsv metadata if available
    participant_tsv = bids_dataset.subject_participant_tsv(bids_file)
    if participant_tsv:
        has_real_data = any(v not in (None, "n/a") for v in participant_tsv.values())
        if not has_real_data:
            logging.debug(
                "No participant match for %s, storing column skeleton", bids_relpath
            )
        # Convert numeric strings to floats for better API/Client compatibility
        # but preserve participant_id as string
        for k, v in participant_tsv.items():
            if k == "participant_id":
                continue
            if isinstance(v, str):
                try:
                    # Only convert if it's a simple number
                    if v.strip():
                        participant_tsv[k] = float(v)
                except (ValueError, TypeError):
                    pass
        record["participant_tsv"] = participant_tsv

    # Add data integrity information if there are issues
    if data_integrity_issues:
        record["_data_integrity_issues"] = data_integrity_issues
        record["_has_missing_files"] = True
    else:
        record["_has_missing_files"] = False

    # Stamp cascade provenance (P0.1). Surfaces WHICH extractor filled
    # each technical-metadata field — debuggability win for support
    # tickets where a Record has the wrong sampling_frequency. Only
    # attached when at least one field was resolved (avoids polluting
    # Records that fell through the cascade entirely).
    if any(v is not None for v in metadata_provenance.values()):
        record["_metadata_provenance"] = metadata_provenance

    # C6.1 — capture the BIDS-spec sidecar fields beyond the technical-
    # metadata cascade (PowerLineFrequency, EEGReference, SoftwareFilters,
    # Manufacturer, EEGPlacementScheme, etc.). Surfaces them as
    # structured Record fields so they're MongoDB-queryable rather
    # than just embedded as bytes in sidecar_inline. See
    # ROBUSTNESS/BIDS-GAP-AUDIT.md for the rationale.
    sidecar_extras = _extract_bids_sidecar_fields(bids_dataset, bids_file)
    record.update(sidecar_extras)

    # Per-channel status counts from channels.tsv (good / bad).
    # ``_bids.py:count_bad_channels`` exists but wasn't being called
    # from extract_record — fixed in C6.1.
    channel_status = _extract_channel_status_counts(bids_dataset, bids_file)
    record.update(channel_status)

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


# ``_BIDS_CHANNEL_COUNT_FIELDS`` and ``sum_bids_channel_counts`` moved
# to ``_metadata_cascade.py`` (SPRINT-2026-05-22 Task 3) together with
# the modality-sidecar / channels.tsv inheritance-walk helpers — they
# all serve the technical-metadata cascade and nothing else. The name
# ``sum_bids_channel_counts`` is re-exported at the top of this module
# for back-compat with ``tests/test_digest_extractions.py``.


def strip_dataset_prefix(bids_relpath: str, dataset_id: str) -> str:
    """Strip the dataset_id leading directory from a BIDS relative path.

    BIDS files are normally stored as ``<dataset_id>/sub-XX/...``. The
    ``Record.bids_relpath`` field is normalised to omit the dataset
    prefix (the dataset is already known from ``Record.dataset``).

    Parameters
    ----------
    bids_relpath : str
        Path possibly prefixed by ``<dataset_id>/``.
    dataset_id : str
        Dataset accession (e.g. ``"ds002893"``).

    Returns
    -------
    str
        Path with the leading ``<dataset_id>/`` removed if present;
        unchanged otherwise.

    Examples
    --------
    >>> strip_dataset_prefix("ds002893/sub-001/eeg/x.set", "ds002893")
    'sub-001/eeg/x.set'
    >>> strip_dataset_prefix("sub-001/eeg/x.set", "ds002893")
    'sub-001/eeg/x.set'
    """
    prefix = f"{dataset_id}/"
    if bids_relpath.startswith(prefix):
        return bids_relpath[len(prefix) :]
    return bids_relpath


# Modality-sidecar + channels.tsv inheritance-walk helpers moved to
# _metadata_cascade.py (SPRINT-2026-05-22 Task 3). Re-exported at
# the top of this module for back-compat.


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
    """Load neurophysiology data extensions from MNE-BIDS (cached)."""
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

    # Skip files in derivatives folders - these are processed data, not raw recordings
    # Common patterns: /derivatives/, /derivative/, derivatives at root level
    if "/derivatives/" in filepath_lower or filepath_lower.startswith("derivatives/"):
        return False

    # Skip BIDS sidecar/metadata files - these are never data files
    # even when located in modality folders
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

    # Skip files inside CTF .ds directories (we want the .ds directory itself)
    # e.g., skip "sub-01_meg.ds/sub-01_meg.meg4" but keep "sub-01_meg.ds"
    if ".ds/" in filepath_lower:
        return False

    # Also skip CTF internal files by extension
    for ext in CTF_INTERNAL_EXTENSIONS:
        if filepath_lower.endswith(ext):
            return False

    # Skip files inside MEF3 .mefd directories (we want the .mefd directory itself)
    # e.g., skip "sub-01_ieeg.mefd/LTG9.timd/LTG9-000000.segd/LTG9-000000.tdat"
    # but keep "sub-01_ieeg.mefd"
    if ".mefd/" in filepath_lower:
        return False

    # Also skip MEF3 internal files by extension
    for ext in MEF3_INTERNAL_EXTENSIONS:
        if filepath_lower.endswith(ext):
            return False

    # Skip MEF3 internal directories that may appear in archive listings
    for internal_dir in MEF3_INTERNAL_DIRS:
        if (
            filepath_lower.endswith(internal_dir)
            or f"{internal_dir}/" in filepath_lower
        ):
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


def _determine_manifest_storage_base(
    source: str,
    dataset_id: str,
    manifest: dict,
) -> str:
    """Resolve the canonical ``storage.base`` for a manifest-only ingest.

    Three input modes:

    1. Manifest has explicit ``storage_base`` set (from the fetch
       step). We sanity-check it against the resolved source's config
       prefix and rebuild if it doesn't match — defends against the
       pre-PR-#327 bug where ``s3://openneuro.org/<id>`` was written
       for NEMAR datasets.
    2. Source has a per-Source URL builder (figshare / zenodo / osf /
       gin) that uses extra manifest fields beyond ``dataset_id``.
    3. Default: ``<source_base>/<dataset_id>``.

    Phase 8 follow-up: extracted from the inline body of
    :func:`_enumerate_via_manifest`. Pure function (no side effects);
    easy to unit-test.
    """
    storage_base = manifest.get("storage_base")

    if not storage_base:
        config = get_storage_config(source)
        base = config["base"]

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
            # GIN: include organization in path.
            org = manifest.get("organization", "EEGManyLabs")
            return f"{base}/{org}/{dataset_id}"
        # Default: use base/dataset_id pattern.
        return f"{base}/{dataset_id}"

    # Sanity-check explicit storage_base against the resolved source.
    # 2_clone.py used to write ``s3://openneuro.org/<id>`` for any
    # git-cloned dataset, including NEMAR — that value reached MongoDB
    # as ``storage.base`` and we'd 404 on download. Reject the mismatch
    # here so the bad value never gets written again.
    expected_prefix = get_storage_config(source).get("base", "")
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
    """Walk all file paths (direct + ZIP contents) and parse BIDS entities.

    Phase 8 follow-up extraction. Pure function — given the manifest's
    ``files`` + the separately-stored ``zip_contents`` array, returns
    four sets:
    - subjects, sessions, tasks: only counted for paths whose modality
      matches :data:`NEURO_MODALITIES` (filters out non-EEG sidecars).
    - modalities: any modality encountered in the paths.
    """
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

    # Also add any separately stored zip_contents.
    for zpath in zip_contents:
        all_paths.append(zpath.get("path", "") if isinstance(zpath, dict) else zpath)

    for filepath in all_paths:
        entities = parse_bids_entities_from_path(filepath)
        if entities.get("modality"):
            modalities.add(entities["modality"])
        # Only count subjects/sessions/tasks for supported neuro modalities.
        if entities.get("modality") in NEURO_MODALITIES:
            if entities.get("subject"):
                subjects.add(entities["subject"])
            if entities.get("session"):
                sessions.add(entities["session"])
            if entities.get("task"):
                tasks.add(entities["task"])

    return subjects, sessions, tasks, modalities


def _fetch_subject_count_via_http(files: list, fallback: int) -> int:
    """Try to fetch dataset_description.json / participants.tsv over HTTP.

    Last-resort fallback used by the manifest path when the BIDS-entity
    walk and the manifest's own ``demographics.subjects_count`` /
    ``bids_subject_count`` both came up empty. Searches the manifest's
    file list for the canonical metadata filenames and attempts a 10s
    HTTP fetch of each.

    Returns the discovered count, or ``fallback`` if nothing worked.
    Errors are logged via ``print`` (preserves the pre-extraction
    behaviour exactly).

    Phase 8 follow-up extraction. NOT a pure function — does network I/O.
    """
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
            with urllib.request.urlopen(desc_url, timeout=10) as response:
                desc_data = json.loads(response.read().decode("utf-8"))
                if "Subjects" in desc_data:  # heuristic
                    subjects_count = int(desc_data["Subjects"])
        except (
            urllib.error.URLError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
            KeyError,
        ) as e:
            print(f"Failed to fetch/parse dataset_description.json: {e}")

    if subjects_count == 0 and participants_url:
        try:
            with urllib.request.urlopen(participants_url, timeout=10) as response:
                content = response.read().decode("utf-8")
                lines = [line for line in content.splitlines() if line.strip()]
                if len(lines) > 1:  # subtract the header
                    subjects_count = len(lines) - 1
        except (
            urllib.error.URLError,
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
    """Synthesize Records for the neuro files inside a peeked ZIP.

    When the fetch step peeks into a ZIP and lists its contents in
    ``_zip_contents``, we emit one Record per neuro file inside, with
    the parent ZIP's URL stamped as ``container_url`` so the runtime
    knows to fetch the ZIP and extract on demand.

    Pure function (no side effects). Returns ``(records, errors)``.
    Phase 8 follow-up — was inline inside ``_enumerate_via_manifest``.
    """
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
    """Synthesize one Record for a ``sub-<id>.zip`` archive in the manifest.

    Subject-ZIP archives (one ZIP per subject) carry the whole subject's
    data inside. We emit a placeholder Record with a guessed BIDS path;
    the runtime downloads and extracts on demand. Marked with
    ``zip_contains_bids = True`` so consumers know to peek inside.

    Returns ``(record_or_None, errors)``. ``record`` is None when
    ``create_record`` failed (the error is captured).
    """
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
    """Whether the filename matches one of the known BIDS-bundle ZIP shapes.

    BIDS-data ZIPs (``data_bids.zip``, ``*_eeg.zip``, ...) bundle the
    whole dataset; we emit inferred per-subject Records based on the
    manifest's demographics counts.
    """
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
    """Synthesize Records for a BIDS-bundled ZIP using manifest demographics.

    Two sub-cases:
    1. ``demographics.subjects_count`` is known: emit one Record per
       inferred subject (capped at 200) with placeholder BIDS paths
       and ``needs_extraction = True``.
    2. Subject count unknown: emit a single placeholder Record with
       ``bids_relpath = __ZIP__/<filename>``.

    Both stamp ``container_url`` so the runtime can fetch + extract.
    """
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    filepath = file_info.get("path", "")
    download_url = file_info.get("download_url")
    file_size = file_info.get("size", 0)

    demographics = manifest.get("demographics", {})
    inferred_subjects = demographics.get("subjects_count", 0)

    if inferred_subjects and inferred_subjects > 0:
        # Cap at 200 subjects to prevent runaway record creation.
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

    # Subject count unknown — emit a single placeholder Record.
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
    """Synthesize one Record for a regular (non-ZIP) neuro file in the manifest.

    Returns ``(record_or_None, errors)``. The record's
    ``storage_backend`` comes from the per-Source table (unlike the
    ZIP branches, which hard-code ``"https"``).
    """
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
            storage_backend=get_storage_backend(source),
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
    """Synthesize Records from the manifest's top-level ``zip_contents`` array.

    Some clone scripts (notably the OSF enhanced fetch) write extracted
    ZIP contents into a separate ``manifest["zip_contents"]`` array
    rather than embedding them under each file's ``_zip_contents``.
    This helper handles that shape.
    """
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
    """Synthesize one Record per CTF ``.ds`` directory referenced in the manifest.

    CTF MEG datasets store one recording as a *directory* (``run-01_meg.ds``)
    containing many small files. The manifest enumerates the inner files
    individually, but downstream Records are per-recording, so we
    de-duplicate to the containing ``.ds`` paths and emit one Record each.

    Returns ``(records, errors)``. ``errors`` is the list of per-file
    diagnostics from ``create_record`` failures.

    Phase 8 follow-up extraction. Was an inline two-pass loop inside
    :func:`_enumerate_via_manifest`.
    """
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
                storage_backend=get_storage_backend(source),
                recording_modality=[detected_modality],
                digested_at=digested_at,
            )
            records.append(dict(record))
        except (KeyError, ValueError, TypeError) as e:
            # create_record is pure: validation/conversion errors on a
            # particular file go into the errors list; programmer
            # errors (e.g., a misshapen storage_backend table)
            # propagate per Phase 9 F1.
            errors.append({"file": ds_path, "error": str(e)})

    return records, errors


def _enumerate_via_manifest(
    dataset_id: str,
    manifest: dict,
    digested_at: str,
) -> tuple[EnumerationResult, int]:
    """Build an :class:`EnumerationResult` from a parsed manifest.json.

    Phase 8 Stage 3B extraction: this body used to live inline inside
    :func:`digest_from_manifest`. It owns the manifest-only algorithm —
    storage-base reconstruction, BIDS entity inference from paths,
    HTTP-fallback subject counting, fingerprint, per-file Record
    creation (including CTF .ds + ZIP-contents handling).

    Parameters
    ----------
    dataset_id : str
        Dataset accession.
    manifest : dict
        Parsed ``manifest.json`` content.
    digested_at : str
        ISO 8601 timestamp stamped into every Record + the Dataset.

    Returns
    -------
    (EnumerationResult, total_files)
        The result carries the Dataset metadata + Records + per-file
        errors (no montages — the manifest path produces none). The
        ``total_files`` count is the raw ``len(manifest["files"])``
        before any per-file filtering; it's surfaced in the summary
        for operational observability.
    """
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

    # Check for ZIP contents (files extracted from subject ZIPs)
    zip_contents = manifest.get("zip_contents", [])

    storage_base = _determine_manifest_storage_base(source, dataset_id, manifest)

    subjects, sessions, tasks, modalities = _collect_bids_entities_from_paths(
        files, zip_contents
    )

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

    # Fallback: try to fetch dataset_description.json / participants.tsv
    # over HTTP if counts are still missing.
    if subjects_count == 0 or not tasks:
        subjects_count = _fetch_subject_count_via_http(files, fallback=subjects_count)
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

    ctf_records, ctf_errors = _build_ctf_ds_records(
        files, dataset_id, storage_base, source, digested_at
    )
    records.extend(ctf_records)
    errors.extend(ctf_errors)

    # Dispatch each manifest entry to the right per-case helper.
    # The 4 cases (ZIP-with-contents / subject-ZIP / BIDS-data-ZIP /
    # regular file) used to be ~240 LOC of nested branches; each now
    # lives in its own helper above.
    for file_info in files:
        # Case 1: ZIP with peeked contents from the fetch step.
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

        # Case 2 + 3: ZIP files without peeked contents.
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

        # Case 4: regular neuro data file (the common case).
        rec, errs = _build_regular_manifest_record(
            file_info, dataset_id, storage_base, source, digested_at
        )
        if rec is not None:
            records.append(rec)
        errors.extend(errs)

    # Also process standalone zip_contents (from clone script).
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


def digest_from_manifest(
    dataset_id: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Digest a dataset from its manifest.json without requiring actual files.

    Used for API-only sources (OSF, Figshare, Zenodo) where we have
    file listings but no actual files on disk. Also reached as the
    fallback path from :func:`digest_dataset` when the BIDS clone is
    unparseable.

    Phase 8 Stage 3B: orchestrator only — skip-check, load manifest,
    delegate to :func:`_enumerate_via_manifest`, write outputs.

    Parameters
    ----------
    dataset_id : str
    input_dir : Path
        Directory containing cloned datasets (with manifest.json).
    output_dir : Path
        Directory for output JSON files.

    Returns
    -------
    dict
        Summary of digestion results.
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

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load manifest: {e}",
        }

    digested_at = datetime.now(timezone.utc).isoformat()
    result, total_files = _enumerate_via_manifest(dataset_id, manifest, digested_at)

    if not result.records:
        return {
            "status": "empty",
            "dataset_id": dataset_id,
            "reason": "no files in manifest"
            if total_files == 0
            else "no records extracted",
        }

    return write_dataset_outputs(
        dataset_output_dir,
        result,
        dataset_id=dataset_id,
        source=result.dataset_meta.get("source", "unknown"),
        digested_at=digested_at,
        total_files=total_files,
    )


def _attach_montage_to_record(
    record: dict[str, Any],
    bids_file: Any,
    dataset_dir: Path,
    montages: dict[str, dict[str, Any]],
    dataset_id: str,
    digested_at: str,
) -> list[dict[str, Any]]:
    """Run ``extract_layout`` for ``bids_file``; stamp the result on ``record``.

    Side effects (intentional): mutates ``record`` (sets
    ``montage_hash``) and ``montages`` (adds the layout doc on first
    sighting of a hash). Returns a list of per-file errors (empty on
    the happy path; one entry if ``extract_layout`` raises).

    Phase 8 Stage-3 follow-up: extracted from the inline body of
    :func:`_enumerate_via_bids`'s for-loop to drop it under 100 LOC.
    """
    record_datatype = (record.get("datatype") or "").lower()
    errors: list[dict[str, Any]] = []
    try:
        layout_result = extract_layout(
            Path(str(bids_file)), dataset_dir, datatype=record_datatype
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # extract_layout (electrode-coords pipeline) can fail on missing
        # electrodes.tsv / coordsystem.json, malformed numeric fields,
        # or unsupported montage variants. Best-effort; we still emit
        # the Record without a montage hash.
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
        # First time we see this hash in this dataset — stamp the
        # provenance fields that live outside the hashed content. The
        # API upsert layer uses $setOnInsert so these don't get
        # overwritten when the same hash appears in a later dataset.
        doc["first_seen"] = digested_at
        doc["representative_dataset"] = dataset_id
        subject_entity = record.get("entities", {}).get("subject")
        if subject_entity:
            doc["representative_subject"] = f"sub-{subject_entity}"
        montages[h] = doc
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
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build one Record for one BIDS file; attach montage + dedup.

    Returns ``(record, errors)``. ``record`` is ``None`` when the file
    failed extraction entirely OR was skipped because it's a split-FIF
    continuation without its companions. ``errors`` is the list of
    per-file diagnostics to append to the caller's running errors list.

    Phase 8 Stage-3 follow-up: extracted from the inline body of
    :func:`_enumerate_via_bids`'s for-loop. Brings the helper under
    the 100-LOC ceiling for the loop's orchestration.
    """
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
        # extract_record can raise on missing sidecars / malformed
        # data. Per-file failure goes into errors[]; programmer
        # errors propagate per Phase 9 F1.
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

    # Skip records for split FIF files with missing continuations.
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
            record, bids_file, dataset_dir, montages, dataset_id, digested_at
        )
    )

    # Emit successful record_built event with provenance payload (P1.1).
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
    """Build an :class:`EnumerationResult` by walking a BIDS filesystem.

    Phase 8 Stage 3 extraction: this body used to live inline inside
    :func:`digest_dataset` (lines ~2596-2763 pre-extraction). It owns
    the BIDS-filesystem algorithm — extract_dataset_metadata,
    fingerprint, the per-file loop over ``extract_record`` +
    ``extract_layout``, montage deduplication.

    Parameters
    ----------
    dataset_dir : Path
        Filesystem root of the cloned dataset.
    dataset_id : str
        Dataset accession.
    source : str
        Source identifier (``"openneuro"``, ``"nemar"``, ...).
    source_adapter : SourceAdapter
        Per-Source ingest behaviour (already built by caller).
    digested_at : str
        ISO 8601 timestamp; stamped into every Record and the Dataset.
    bids_dataset : EEGBIDSDataset
        Already-loaded BIDS dataset object. The caller handles
        construction failures and the manifest fallback decision.
    manifest_data : dict, optional
        Parsed ``manifest.json`` content for metadata enrichment.

    Returns
    -------
    EnumerationResult
        Dataset metadata, Records list, accumulated per-file errors,
        deduplicated montages. ``records`` may be empty if every file
        failed extraction or no files matched ``is_neuro_data_file``;
        the orchestrator decides whether to fall back to the manifest
        path or surface an "empty" status.
    """
    files = bids_dataset.get_files()
    if not files:
        return EnumerationResult(
            dataset_meta={"dataset_id": dataset_id, "source": source},
            digest_method="bids_filesystem",
        )

    # Extract Dataset metadata. Recoverable failures become an error
    # dataset record so the rest of the pipeline can continue.
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
        # Fingerprint is an optimisation; failures don't block ingest.
        pass

    # Per-file extraction loop. Each iteration delegates to
    # _build_one_record_from_bids which returns ``(record, errors)``.
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    montages: dict[str, dict[str, Any]] = {}

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


def _summarise_empty_or_error(
    dataset_id: str,
    result: EnumerationResult,
) -> dict[str, Any]:
    """Build the terminal "no records" summary for the orchestrator.

    When the Adapter finishes but returns ``result.records == []`` we
    can't write the per-Dataset JSON files (record_count would be zero
    and downstream tooling treats those as failures). This helper
    picks the right "empty" vs "error" status based on whether any
    per-file errors got accumulated.

    Phase 8 Stage 3D: extracted from ``digest_dataset`` so its body
    drops to an orchestrator-only shape. The error case carries the
    full ``errors`` list so the caller can see WHY nothing landed.
    """
    if result.errors:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": "No records extracted",
            "errors": result.errors,
        }
    return {
        "status": "empty",
        "dataset_id": dataset_id,
        "reason": "no neurophysiology files found",
    }


def _check_dataset_skip_conditions(
    dataset_id: str,
    dataset_dir: Path,
    dataset_output_dir: Path,
) -> dict[str, Any] | None:
    """Return a "skipped" summary if this Dataset should not be digested.

    Two conditions:
      1. Output dir already exists — already-digested guard (the
         pipeline runner relies on this to be idempotent).
      2. Input dataset dir doesn't exist — caller passed an id we
         never cloned. Surface a clear "directory not found".

    Returns ``None`` when neither skip condition fires; the caller
    proceeds with normal digestion.

    Phase 8 Stage 3D: extracted from ``digest_dataset`` so its body
    drops to an orchestrator-only shape.
    """
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
    """Digest a single dataset and generate JSON output.

    Phase 8 Stage 3D: orchestrator only. Skip-check, setup, dispatch
    via :func:`get_record_enumerator`, run the Adapter, fall back to
    manifest if the BIDS path produces nothing, write outputs.

    The algorithms themselves live in:
      - :func:`_enumerate_via_bids` — BIDS-filesystem walk
      - :func:`_enumerate_via_manifest` — manifest-only walk
      - :func:`write_dataset_outputs` — the 4-file JSON writes

    Produces:
    - {dataset_id}_dataset.json: Dataset-level metadata
    - {dataset_id}_records.json: Per-file Record metadata
    - {dataset_id}_montages.json: Deduplicated montage hashes
    - {dataset_id}_summary.json: Per-Dataset summary

    Parameters
    ----------
    dataset_id : str
    input_dir : Path
        Directory containing cloned datasets.
    output_dir : Path
        Directory for output JSON files.

    Returns
    -------
    dict
        Summary of digestion results.
    """
    dataset_dir = input_dir / dataset_id
    dataset_output_dir = output_dir / dataset_id

    skip = _check_dataset_skip_conditions(dataset_id, dataset_dir, dataset_output_dir)
    if skip is not None:
        return skip

    # Build per-Dataset state shared with the Adapter.
    source = detect_source(dataset_dir)
    digested_at = datetime.now(timezone.utc).isoformat()
    _repair_participants_tsv_ids(dataset_dir)
    source_adapter = get_source_adapter(source, dataset_id, dataset_dir)

    # Telemetry: open the per-Dataset event stream.
    get_emitter().emit(
        TelemetryEvent(
            event_kind="dataset_started",
            dataset_id=dataset_id,
            payload={"source": source, "dataset_dir": str(dataset_dir)},
        )
    )

    # Pick the Adapter via the factory. The factory's fallback rules
    # mirror what used to live as if-ladders inside this function:
    # no actual files -> ManifestEnumerator; otherwise BIDS path.
    has_manifest = (dataset_dir / "manifest.json").exists()
    enumerator = get_record_enumerator(
        dataset_id, dataset_dir, source, source_adapter, digested_at
    )

    # Run it. The BIDS Adapter raises on EEGBIDSDataset load failure
    # (per Stage 3C contract); fall back to manifest if available.
    try:
        result = enumerator.enumerate()
    except (
        OSError,
        ValueError,
        KeyError,
        FileNotFoundError,
        PermissionError,
    ) as e:
        if has_manifest and not isinstance(enumerator, ManifestEnumerator):
            return digest_from_manifest(dataset_id, input_dir, output_dir)
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": f"Failed to load BIDS dataset: {e}",
        }

    if not result.records:
        # No records — fall back to manifest if we haven't tried it yet,
        # else surface the right "empty / error" status.
        if has_manifest and not isinstance(enumerator, ManifestEnumerator):
            return digest_from_manifest(dataset_id, input_dir, output_dir)
        return _summarise_empty_or_error(dataset_id, result)

    summary = write_dataset_outputs(
        dataset_output_dir,
        result,
        dataset_id=dataset_id,
        source=source,
        digested_at=digested_at,
    )
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
            },
        )
    )
    return summary


def _json_serializer(obj):
    """Handle non-serializable objects."""
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


def _positive_float(value: str) -> float:
    """Argparse type for positive float values."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _dataset_boundary_profile(dataset_id: str, input_dir: Path) -> str:
    """Return a cheap structural profile for stall-boundary diagnostics."""
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
    """Print the datasets around the reported deterministic #287 stall point."""
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
    """Build an error result that keeps batch summary accounting unchanged."""
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
    """Run one dataset digestion and always report success or failure."""
    try:
        try:
            result = digest_dataset(dataset_id, input_dir, output_dir)
            if not isinstance(result, dict):
                result = _worker_error_result(
                    dataset_id,
                    f"digest_dataset returned {type(result).__name__}, expected dict",
                )
        except BaseException as exc:  # noqa: BLE001
            # Worker-process boundary: catch BaseException so the parent
            # process learns about KeyboardInterrupt / SystemExit /
            # MemoryError via the result_queue rather than silently
            # losing the worker. Re-raising would not help — the parent
            # is the one that needs to decide whether to retry, escalate,
            # or shut down. Deliberate broad catch.
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
    """Start a child process for one dataset."""
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
    """Release local process and queue handles without blocking indefinitely."""
    result_queue = active.get("queue")
    if result_queue is not None:
        try:
            result_queue.close()
        except (OSError, ValueError, AttributeError):
            # OSError: already closed. ValueError: queue in invalid
            # state. AttributeError: not a real Queue. All best-effort.
            pass

    process = active.get("process")
    if process is not None:
        try:
            process.close()
        except (OSError, ValueError, AttributeError):
            # Same best-effort contract as the queue close above.
            pass


def _terminate_active_process(active: dict[str, Any], reason: str) -> None:
    """Terminate a child process, escalating to kill if it ignores terminate."""
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
    """Collect one completed child result; queue reads use an explicit timeout."""
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
        # OSError: pipe closed mid-read. EOFError: worker died before
        # sending. ValueError: queue in invalid state. All map to a
        # generic "could not collect" error record.
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
    """Kill a stalled dataset worker and return an error result."""
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
    """Process datasets with per-dataset process supervision.

    Each dataset runs in its own process so a stuck parser, file read, or MNE
    call can be terminated without wedging the whole batch. Blocking process
    joins and queue operations all use explicit timeouts.
    """
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

                    # **CRITICAL FIX**: Try to drain queue FIRST (non-blocking)
                    # to prevent deadlock on full queue with blocking worker.
                    # Worker may be alive but stuck on queue.put() if main never reads.
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
                        # queue.Empty is the common case (no result yet).
                        # OSError/EOFError: pipe / process gone.
                        # ValueError: queue in invalid state.
                        print(
                            f"[QUEUE] No result yet for {dataset_id}: {type(e).__name__}",
                            flush=True,
                        )
                        pass  # No result yet or queue error, continue checking

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
    # CLI + env var parsing + validation via Pydantic-settings (C7.2 —
    # same pattern as _inject_config + _validate_config). Replaces the
    # 40 lines of argparse + ``_positive_float`` custom-type boilerplate
    # with declarative Field(gt=0, le=...) bounds.
    from pydantic import ValidationError

    from _digest_config import load_digest_config_from_argv

    try:
        cfg = load_digest_config_from_argv()
    except ValidationError as exc:
        print("Config error(s):", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", []))
            print(f"  {field}: {err.get('msg')}", file=sys.stderr)
        return 1

    # Find datasets
    dataset_ids = find_datasets(cfg.input, cfg.datasets)
    if cfg.limit:
        dataset_ids = dataset_ids[: cfg.limit]

    print(f"Found {len(dataset_ids)} datasets to digest")
    print(f"Workers: {cfg.workers}")
    print(f"Dataset timeout: {cfg.dataset_timeout:g}s")
    print_stall_boundary_diagnostics(dataset_ids, cfg.input)
    print("=" * 60)

    # Create output directory
    cfg.output.mkdir(parents=True, exist_ok=True)

    # Process datasets under a watchdog even with one worker. This keeps
    # dataset-specific parser or filesystem stalls from freezing the batch.
    results, stats = process_datasets_with_watchdog(
        dataset_ids,
        cfg.input,
        cfg.output,
        workers=cfg.workers,
        dataset_timeout=cfg.dataset_timeout,
    )

    # Save batch summary
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
