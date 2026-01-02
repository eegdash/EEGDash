# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# TODO: merge this with records.py, maybe change the name for better name, now that
# we have the dataset model.
"""Pydantic models and TypedDict schemas for EEGDash documents.

This module contains both the Pydantic models (for validation) and TypedDict
definitions (for fast loading/runtime) of EEGDash Dataset and Record objects.

Two-level hierarchy:
- Dataset: per-dataset metadata (one per ds*, for discovery/filtering)
  Contains: identity, demographics, clinical, paradigm, timestamps
- Record: per-file metadata (many per dataset, optimized for fast loading)
  Contains: dataset FK, storage location, BIDS entities only
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StorageModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(min_length=1)
    base: str = Field(min_length=1)
    raw_key: str = Field(min_length=1)
    dep_keys: list[str] = Field(default_factory=list)


class EntitiesModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str = Field(min_length=1)
    bids_relpath: str = Field(min_length=1)
    storage: StorageModel
    recording_modality: list[str] = Field(min_length=1)

    datatype: str | None = None
    suffix: str | None = None
    extension: str | None = None
    entities: EntitiesModel | dict[str, Any] | None = None


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    recording_modality: list[str] = Field(min_length=1)
    ingestion_fingerprint: str | None = None
    senior_author: str | None = None
    contact_info: list[str] | None = None
    timestamps: dict[str, Any] | None = None


class ManifestFileModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str | None = None
    name: str | None = None

    def path_or_name(self) -> str:
        return (self.path or self.name or "").strip()


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = None
    files: list[str | ManifestFileModel]


# =============================================================================
# TypedDict Schemas (Optimized for fast loading/runtime usage)
# =============================================================================

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal, TypedDict


class Timestamps(TypedDict, total=False):
    """Processing timestamps."""

    digested_at: str  # ISO 8601 timestamp of when digestion occurred
    dataset_created_at: str | None  # ISO 8601 timestamp of when dataset was created
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
    age_mean: float | None  # Mean age of subjects
    species: str | None  # e.g., "Human", "Mouse"
    sex_distribution: dict[str, int] | None  # e.g., {"m": 50, "f": 45, "o": 5}
    handedness_distribution: dict[str, int] | None  # e.g., {"r": 80, "l": 15, "a": 5}


class Clinical(TypedDict, total=False):
    """Clinical classification (dataset-level)."""

    is_clinical: bool  # Whether the dataset is clinical
    purpose: (
        str | None
    )  # e.g., "epilepsy", "depression", "parkinson", "alzheimer", "sleep_disorder"


class Paradigm(TypedDict, total=False):
    """Experimental paradigm classification (dataset-level)."""

    modality: (
        str | None
    )  # e.g., "visual", "auditory", "somatosensory", "multisensory", "resting_state"
    cognitive_domain: (
        str | None
    )  # e.g., "attention", "memory", "learning", "motor", "language", "emotion"
    is_10_20_system: bool | None  # Whether electrodes follow the 10-20 system


class ExternalLinks(TypedDict, total=False):
    """External links to the dataset."""

    source_url: str | None  # Primary URL (OpenNeuro, GIN, etc.)
    osf_url: str | None  # Open Science Framework link
    github_url: str | None  # GitHub repository
    paper_url: str | None  # Link to associated paper


class RepositoryStats(TypedDict, total=False):
    """Repository statistics (for git-based sources like GIN)."""

    stars: int
    forks: int
    watchers: int


class Dataset(TypedDict, total=False):
    """Dataset-level metadata (one per ds*)."""

    # Identity
    dataset_id: str  # e.g., "ds001785"
    name: str  # Dataset title
    source: str  # e.g., "openneuro", "nemar", "gin"
    readme: str | None  # README content from dataset

    # Recording info
    recording_modality: list[str]  # Recording types: ["eeg", "meg", "ieeg"]
    datatypes: list[str]  # BIDS datatypes present: ["eeg", "anat", "beh"]

    # Experimental info
    experimental_modalities: (
        list[str] | None
    )  # Stimulus types: ["visual", "auditory", "tactile"]

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

    # Multi-site studies
    contributing_labs: list[str] | None  # Labs that contributed data
    n_contributing_labs: int | None

    # Demographics
    demographics: Demographics

    # Classification
    clinical: Clinical
    paradigm: Paradigm

    # External resources
    external_links: ExternalLinks
    repository_stats: RepositoryStats | None  # For git-based sources

    # Contact
    senior_author: str | None
    contact_info: list[str] | None  # Admin users/emails

    # Timestamps
    timestamps: Timestamps


def create_dataset(
    *,
    dataset_id: str,
    name: str | None = None,
    source: str = "openneuro",
    readme: str | None = None,
    recording_modality: list[str] | None = None,
    datatypes: list[str] | None = None,
    modalities: list[str] | None = None,
    experimental_modalities: list[str] | None = None,
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
    # Demographics
    subjects_count: int | None = None,
    ages: list[int] | None = None,
    age_mean: float | None = None,
    species: str | None = None,
    sex_distribution: dict[str, int] | None = None,
    handedness_distribution: dict[str, int] | None = None,
    # Multi-site studies
    contributing_labs: list[str] | None = None,
    # Clinical classification
    is_clinical: bool | None = None,
    clinical_purpose: str | None = None,
    # Paradigm classification
    paradigm_modality: str | None = None,
    cognitive_domain: str | None = None,
    is_10_20_system: bool | None = None,
    # External links
    source_url: str | None = None,
    osf_url: str | None = None,
    github_url: str | None = None,
    paper_url: str | None = None,
    # Repository stats (for git-based sources)
    stars: int | None = None,
    forks: int | None = None,
    watchers: int | None = None,
    # Contact
    senior_author: str | None = None,
    contact_info: list[str] | None = None,
    # Timestamps
    digested_at: str | None = None,
    dataset_created_at: str | None = None,
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
    recording_modality : list[str], optional
        Recording types (e.g., ["eeg", "meg", "ieeg"]).
    datatypes : list[str], optional
        BIDS datatypes present in the dataset (e.g., ["eeg", "anat", "beh"]).
    experimental_modalities : list[str], optional
        Stimulus/experimental modalities (e.g., ["visual", "auditory", "tactile"]).
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
    age_mean : float, optional
        Mean age of subjects.
    species : str, optional
        Species (e.g., "Human").
    sex_distribution : dict[str, int], optional
        Sex distribution (e.g., {"m": 50, "f": 45}).
    handedness_distribution : dict[str, int], optional
        Handedness distribution (e.g., {"r": 80, "l": 15}).
    contributing_labs : list[str], optional
        Labs that contributed data (for multi-site studies).
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
    source_url : str, optional
        Primary URL to the dataset source.
    osf_url : str, optional
        Open Science Framework URL.
    github_url : str, optional
        GitHub repository URL.
    paper_url : str, optional
        URL to associated paper.
    stars : int, optional
        Repository stars count (for git-based sources).
    forks : int, optional
        Repository forks count.
    watchers : int, optional
        Repository watchers count.
    digested_at : str, optional
        ISO 8601 timestamp. If not provided, no timestamp is set (for deterministic output).
    dataset_modified_at : str, optional
        Last modification timestamp.

    Returns
    -------
    Dataset
        A Dataset document.

    """
    if not dataset_id:
        raise ValueError("dataset_id is required")

    if datatypes is None and modalities:
        datatypes = modalities

    ages = ages or []
    ages_clean = [a for a in ages if a is not None]

    # Build demographics
    demographics = Demographics(
        subjects_count=subjects_count or 0,
        ages=ages_clean,
        age_min=min(ages_clean) if ages_clean else None,
        age_max=max(ages_clean) if ages_clean else None,
        age_mean=age_mean,
        species=species,
        sex_distribution=sex_distribution,
        handedness_distribution=handedness_distribution,
    )

    dataset = Dataset(
        dataset_id=dataset_id,
        name=name or dataset_id,
        source=source,
        readme=readme,
        recording_modality=recording_modality or ["eeg"],
        datatypes=datatypes or recording_modality or ["eeg"],
        experimental_modalities=experimental_modalities,
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
        contributing_labs=contributing_labs,
        n_contributing_labs=len(contributing_labs) if contributing_labs else None,
        demographics=demographics,
        senior_author=senior_author,
        contact_info=contact_info,
        timestamps=Timestamps(
            digested_at=digested_at,
            dataset_created_at=dataset_created_at,
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
    if (
        paradigm_modality is not None
        or cognitive_domain is not None
        or is_10_20_system is not None
    ):
        dataset["paradigm"] = Paradigm(
            modality=paradigm_modality,
            cognitive_domain=cognitive_domain,
            is_10_20_system=is_10_20_system,
        )

    # Add external links if any provided
    if source_url or osf_url or github_url or paper_url:
        dataset["external_links"] = ExternalLinks(
            source_url=source_url,
            osf_url=osf_url,
            github_url=github_url,
            paper_url=paper_url,
        )

    # Add repository stats if any provided
    if stars is not None or forks is not None or watchers is not None:
        dataset["repository_stats"] = RepositoryStats(
            stars=stars,
            forks=forks,
            watchers=watchers,
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
    bidspath: str  # Legacy field, but used by tests/clients. Format: "{dataset}/{bids_relpath}"
    bids_relpath: str
    datatype: str
    suffix: str
    extension: str
    recording_modality: list[str] | None  # e.g., ["eeg", "meg", "ieeg"]
    entities: Entities
    entities_mne: Entities  # run sanitized for MNE-BIDS (numeric or None)
    storage: Storage
    ch_names: list[str] | None  # List of channel names from channels.tsv or raw file
    sampling_frequency: float | None  # Sampling rate in Hz
    nchans: int | None  # Number of channels
    ntimes: int | None  # Number of samples (duration in samples)
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
    recording_modality: list[str] | None = None,
    ch_names: list[str] | None = None,
    sampling_frequency: float | None = None,
    nchans: int | None = None,
    ntimes: int | None = None,
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
    recording_modality : list[str], optional
        Recording modalities (e.g., ["eeg", "meg", "ieeg"]).
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
        bidspath=f"{dataset}/{bids_relpath}",
        bids_relpath=bids_relpath,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        recording_modality=recording_modality or [datatype],
        entities=entities,
        entities_mne=entities_mne,
        storage=Storage(
            backend=storage_backend,
            base=storage_base.rstrip("/"),
            raw_key=bids_relpath,
            dep_keys=dep_keys,
        ),
        ch_names=ch_names,
        sampling_frequency=sampling_frequency,
        nchans=nchans,
        ntimes=ntimes,
        digested_at=digested_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def validate_record(record: dict[str, Any]) -> list[str]:
    """Validate a record has required fields. Returns list of errors."""
    errors: list[str] = []

    if not record.get("dataset"):
        errors.append("missing: dataset")
    if not record.get("bids_relpath"):
        errors.append("missing: bids_relpath")
    if not record.get("bidspath"):
        errors.append("missing: bidspath")

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
