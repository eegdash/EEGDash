"""Dataset-level BIDS metadata extraction Seam.

Extracted from ``3_digest.py``. Reads README + participants.tsv demographics,
builds the dataset-level storage doc, harvests dataset_description.json extras, and
assembles the Dataset document via ``create_dataset``. Pure leaf logic — depends
only on ``_bids_path``, ``_file_utils``, ``_constants`` and ``eegdash`` — never on
``3_digest``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from _bids_path import normalize_modality
from _constants import NEURO_MODALITIES
from _file_utils import get_annex_file_size
from eegdash.dataset._source_inference import DEFAULT_STORAGE_CONFIG, STORAGE_CONFIGS
from eegdash.schemas import Storage, create_dataset
from source_adapter import SourceAdapter

logger = logging.getLogger(__name__)

__all__ = ["extract_dataset_metadata"]


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
) -> Storage | None:
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
    source_adapter: SourceAdapter | None = None,
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
