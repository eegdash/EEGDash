# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""EEGDash Record and Dataset schemas.

Two-level hierarchy:
- Dataset: per-dataset metadata (one per ds*, for discovery/filtering)
  Contains: identity, demographics, clinical, paradigm, timestamps
- Record: per-file metadata (many per dataset, optimized for fast loading)
  Contains: dataset FK, storage location, BIDS entities only
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, TypedDict


# =============================================================================
# Shared Types
# =============================================================================


class Timestamps(TypedDict, total=False):
    """Processing timestamps."""

    digested_at: str  # ISO 8601 timestamp of when digestion occurred
    dataset_modified_at: str | None  # ISO 8601 timestamp of last dataset update


# =============================================================================
# Dataset Schema (per-dataset, for discovery/filtering)
# =============================================================================


class Demographics(TypedDict, total=False):
    """Subject demographics for a dataset."""

    subjects_count: int
    ages: list[int]
    age_min: int | None
    age_max: int | None
    species: str | None  # e.g., "Human", "Mouse"


class Clinical(TypedDict, total=False):
    """Clinical classification (dataset-level)."""

    is_clinical: bool  # Whether the dataset is clinical
    purpose: str | None  # e.g., "epilepsy", "depression", "parkinson", "alzheimer", "sleep_disorder"


class Paradigm(TypedDict, total=False):
    """Experimental paradigm classification (dataset-level)."""

    modality: str | None  # e.g., "visual", "auditory", "somatosensory", "multisensory", "resting_state"
    cognitive_domain: str | None  # e.g., "attention", "memory", "learning", "motor", "language", "emotion"
    is_10_20_system: bool | None  # Whether electrodes follow the 10-20 system


class Dataset(TypedDict, total=False):
    """Dataset-level metadata (one per ds*)."""

    # Identity
    dataset_id: str  # e.g., "ds001785"
    name: str  # Dataset title
    source: str  # e.g., "openneuro", "nemar", "gin"

    # Recording info
    recording_modality: str  # Primary modality: "eeg", "meg", "ieeg"
    modalities: list[str]  # All modalities present: ["eeg", "mri", "beh"]

    # BIDS metadata
    bids_version: str | None
    license: str | None
    authors: list[str]
    funding: list[str]
    dataset_doi: str | None
    associated_paper_doi: str | None

    # Content summary
    tasks: list[str]
    sessions: list[str]
    total_files: int | None
    size_bytes: int | None
    data_processed: bool | None

    # Study classification
    study_domain: str | None  # e.g., "Perceptual consciousness", "Motor control"
    study_design: str | None

    # Demographics
    demographics: Demographics

    # Classification
    clinical: Clinical
    paradigm: Paradigm

    # Timestamps
    timestamps: Timestamps


def create_dataset(
    *,
    dataset_id: str,
    name: str | None = None,
    source: str = "openneuro",
    recording_modality: str = "eeg",
    modalities: list[str] | None = None,
    bids_version: str | None = None,
    license: str | None = None,
    authors: list[str] | None = None,
    funding: list[str] | None = None,
    dataset_doi: str | None = None,
    associated_paper_doi: str | None = None,
    tasks: list[str] | None = None,
    sessions: list[str] | None = None,
    total_files: int | None = None,
    size_bytes: int | None = None,
    data_processed: bool | None = None,
    study_domain: str | None = None,
    study_design: str | None = None,
    subjects_count: int | None = None,
    ages: list[int] | None = None,
    species: str | None = None,
    # Clinical classification
    is_clinical: bool | None = None,
    clinical_purpose: str | None = None,
    # Paradigm classification
    paradigm_modality: str | None = None,
    cognitive_domain: str | None = None,
    is_10_20_system: bool | None = None,
    # Timestamps
    digested_at: str | None = None,
    dataset_modified_at: str | None = None,
) -> Dataset:
    """Create a Dataset document.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "ds001785").
    name : str, optional
        Dataset title/name.
    source : str, default "openneuro"
        Data source ("openneuro", "nemar", "gin").
    recording_modality : str, default "eeg"
        Primary recording modality.
    modalities : list[str], optional
        All modalities present in the dataset.
    bids_version : str, optional
        BIDS version of the dataset.
    license : str, optional
        Dataset license (e.g., "CC0", "CC-BY-4.0").
    authors : list[str], optional
        Dataset authors.
    funding : list[str], optional
        Funding sources.
    dataset_doi : str, optional
        Dataset DOI.
    associated_paper_doi : str, optional
        DOI of associated publication.
    tasks : list[str], optional
        Tasks in the dataset.
    sessions : list[str], optional
        Sessions in the dataset.
    total_files : int, optional
        Total number of files.
    size_bytes : int, optional
        Total size in bytes.
    data_processed : bool, optional
        Whether data is processed.
    study_domain : str, optional
        Study domain/topic.
    study_design : str, optional
        Study design description.
    subjects_count : int, optional
        Number of subjects.
    ages : list[int], optional
        Subject ages.
    species : str, optional
        Species (e.g., "Human").
    is_clinical : bool, optional
        Whether this is clinical data.
    clinical_purpose : str, optional
        Clinical purpose (e.g., "epilepsy", "depression").
    paradigm_modality : str, optional
        Experimental modality (e.g., "visual", "auditory", "resting_state").
    cognitive_domain : str, optional
        Cognitive domain (e.g., "attention", "memory", "motor").
    is_10_20_system : bool, optional
        Whether electrodes follow the 10-20 system.
    digested_at : str, optional
        ISO 8601 timestamp. Defaults to current time.
    dataset_modified_at : str, optional
        Last modification timestamp.

    Returns
    -------
    Dataset
        A Dataset document.
    """
    if not dataset_id:
        raise ValueError("dataset_id is required")

    ages = ages or []
    ages_clean = [a for a in ages if a is not None]

    dataset = Dataset(
        dataset_id=dataset_id,
        name=name or dataset_id,
        source=source,
        recording_modality=recording_modality,
        modalities=modalities or [recording_modality],
        bids_version=bids_version,
        license=license,
        authors=authors or [],
        funding=funding or [],
        dataset_doi=dataset_doi,
        associated_paper_doi=associated_paper_doi,
        tasks=tasks or [],
        sessions=sessions or [],
        total_files=total_files,
        size_bytes=size_bytes,
        data_processed=data_processed,
        study_domain=study_domain,
        study_design=study_design,
        demographics=Demographics(
            subjects_count=subjects_count or 0,
            ages=ages_clean,
            age_min=min(ages_clean) if ages_clean else None,
            age_max=max(ages_clean) if ages_clean else None,
            species=species,
        ),
        timestamps=Timestamps(
            digested_at=digested_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            dataset_modified_at=dataset_modified_at,
        ),
    )

    # Add clinical if any field provided
    if is_clinical is not None or clinical_purpose is not None:
        dataset["clinical"] = Clinical(
            is_clinical=is_clinical if is_clinical is not None else False,
            purpose=clinical_purpose,
        )

    # Add paradigm if any field provided
    if paradigm_modality is not None or cognitive_domain is not None or is_10_20_system is not None:
        dataset["paradigm"] = Paradigm(
            modality=paradigm_modality,
            cognitive_domain=cognitive_domain,
            is_10_20_system=is_10_20_system,
        )

    return dataset


# =============================================================================
# Record Schema (per-file, for loading)
# =============================================================================


class Storage(TypedDict):
    """Remote storage location."""

    backend: Literal["s3", "https", "local"]
    base: str  # e.g., "s3://openneuro.org/ds000001"
    raw_key: str  # relative to base
    dep_keys: list[str]  # relative to base


class Entities(TypedDict, total=False):
    """BIDS entities."""

    subject: str | None
    session: str | None
    task: str | None
    run: str | None


class Record(TypedDict, total=False):
    """EEGDash record schema (per-file, optimized for fast loading).

    Minimal schema - clinical/paradigm info lives in Dataset.
    """

    dataset: str  # FK to Dataset.dataset_id
    data_name: str
    bids_relpath: str
    datatype: str
    suffix: str
    extension: str
    recording_modality: str | None  # e.g., "eeg", "meg", "ieeg"
    entities: Entities
    entities_mne: Entities  # run sanitized for MNE-BIDS (numeric or None)
    storage: Storage
    digested_at: str  # ISO 8601 timestamp


def _sanitize_run_for_mne(value: Any) -> str | None:
    """Sanitize run value for MNE-BIDS (must be numeric or None)."""
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value if value.isdigit() else None
    return None


def create_record(
    *,
    dataset: str,
    storage_base: str,
    bids_relpath: str,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    dep_keys: list[str] | None = None,
    datatype: str = "eeg",
    suffix: str = "eeg",
    storage_backend: Literal["s3", "https", "local"] = "s3",
    recording_modality: str | None = None,
    digested_at: str | None = None,
) -> Record:
    """Create an EEGDash record.

    Parameters
    ----------
    dataset : str
        Dataset identifier (e.g., "ds000001").
    storage_base : str
        Remote storage base URI (e.g., "s3://openneuro.org/ds000001").
    bids_relpath : str
        BIDS-relative path to the raw file (e.g., "sub-01/eeg/sub-01_task-rest_eeg.vhdr").
    subject, session, task, run : str, optional
        BIDS entities.
    dep_keys : list[str], optional
        Dependency paths relative to storage_base.
    datatype : str, default "eeg"
        BIDS datatype.
    suffix : str, default "eeg"
        BIDS suffix.
    storage_backend : {"s3", "https", "local"}, default "s3"
        Storage backend type.
    recording_modality : str, optional
        Recording modality (e.g., "eeg", "meg", "ieeg").
    digested_at : str, optional
        ISO 8601 timestamp. Defaults to current time.

    Returns
    -------
    Record
        A slim EEGDash record optimized for loading.

    Notes
    -----
    Clinical and paradigm info is stored at the Dataset level, not per-file.

    Examples
    --------
    >>> record = create_record(
    ...     dataset="ds000001",
    ...     storage_base="s3://openneuro.org/ds000001",
    ...     bids_relpath="sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    ...     subject="01",
    ...     task="rest",
    ... )
    """
    if not dataset:
        raise ValueError("dataset is required")
    if not storage_base:
        raise ValueError("storage_base is required")
    if not bids_relpath:
        raise ValueError("bids_relpath is required")

    dep_keys = dep_keys or []
    extension = PurePosixPath(bids_relpath).suffix

    entities: Entities = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
    }

    entities_mne: Entities = dict(entities)  # type: ignore[assignment]
    entities_mne["run"] = _sanitize_run_for_mne(run)

    return Record(
        dataset=dataset,
        data_name=f"{dataset}_{PurePosixPath(bids_relpath).name}",
        bids_relpath=bids_relpath,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        recording_modality=recording_modality or datatype,
        entities=entities,
        entities_mne=entities_mne,
        storage=Storage(
            backend=storage_backend,
            base=storage_base.rstrip("/"),
            raw_key=bids_relpath,
            dep_keys=dep_keys,
        ),
        digested_at=digested_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def validate_record(record: dict[str, Any]) -> list[str]:
    """Validate a record has required fields. Returns list of errors."""
    errors: list[str] = []

    if not record.get("dataset"):
        errors.append("missing: dataset")
    if not record.get("bids_relpath"):
        errors.append("missing: bids_relpath")

    storage = record.get("storage")
    if not storage:
        errors.append("missing: storage")
    elif not storage.get("base"):
        errors.append("missing: storage.base")

    return errors


def validate_dataset(dataset: dict[str, Any]) -> list[str]:
    """Validate a dataset has required fields. Returns list of errors."""
    errors: list[str] = []

    if not dataset.get("dataset_id"):
        errors.append("missing: dataset_id")

    return errors


__all__ = [
    # Dataset (per-dataset, for discovery)
    "Dataset",
    "Demographics",
    "Clinical",
    "Paradigm",
    "create_dataset",
    "validate_dataset",
    # Record (per-file, for loading)
    "Record",
    "Entities",
    "Storage",
    "create_record",
    "validate_record",
    # Shared
    "Timestamps",
]
