"""BIDS path Seam: pure string/path heuristics for entity parsing and classification.

Extracted from ``3_digest.py``. These are deterministic, I/O-free functions over
file paths/names: BIDS entity extraction, dataset-prefix stripping, modality
normalization, and neurophysiology-data-file classification. They depend only on
``_constants`` and the mne-bids extension table, so this is a leaf module that is
safe to import from any stage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from _constants import (
    CTF_INTERNAL_EXTENSIONS,
    MEF3_INTERNAL_DIRS,
    MEF3_INTERNAL_EXTENSIONS,
    MODALITY_CANONICAL_MAP,
    MODALITY_DETECTION_TARGETS,
    NEURO_MODALITIES,
)
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS

__all__ = [
    "detect_modality_from_path",
    "is_neuro_data_file",
    "normalize_modality",
    "parse_bids_entities_from_path",
    "strip_dataset_prefix",
]


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
