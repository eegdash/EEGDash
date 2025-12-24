# Authors: The EEGDash contributors.
# License: BSD-3-Clause

"""EEGDash Record schema.

Records are self-contained documents describing where EEG data lives
and how to cache it locally. Each record explicitly specifies its storage base.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, TypedDict


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


class Timestamps(TypedDict, total=False):
    """Processing timestamps."""

    digested_at: str  # ISO 8601 timestamp of when digestion occurred
    dataset_modified_at: str | None  # ISO 8601 timestamp of last dataset update


class Clinical(TypedDict, total=False):
    """Clinical classification."""

    is_clinical: bool  # Whether the dataset is clinical
    purpose: str | None  # e.g., "epilepsy", "depression", "parkinson", "alzheimer", "sleep_disorder"


class Paradigm(TypedDict, total=False):
    """Experimental paradigm classification."""

    modality: str | None  # e.g., "visual", "auditory", "somatosensory", "multisensory", "resting_state"
    cognitive_domain: str | None  # e.g., "attention", "memory", "learning", "motor", "language", "emotion"
    is_10_20_system: bool | None  # Whether electrodes follow the 10-20 system


class Record(TypedDict, total=False):
    """EEGDash record schema."""

    dataset: str
    data_name: str
    bids_relpath: str
    datatype: str
    suffix: str
    extension: str
    recording_modality: str | None  # e.g., "eeg", "meg", "ieeg", "emg", "ecog"
    entities: Entities
    entities_mne: Entities  # run sanitized for MNE-BIDS (numeric or None)
    storage: Storage
    timestamps: Timestamps
    clinical: Clinical  # Clinical classification
    paradigm: Paradigm  # Experimental paradigm


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
    # Recording modality
    recording_modality: str | None = None,
    # Digestion metadata
    digested_at: str | None = None,
    dataset_modified_at: str | None = None,
    # Clinical classification
    is_clinical: bool | None = None,
    clinical_purpose: str | None = None,
    # Experimental info
    modality: str | None = None,
    cognitive_domain: str | None = None,
    is_10_20_system: bool | None = None,
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
        Recording modality (e.g., "eeg", "meg", "ieeg", "emg", "ecog").
    digested_at : str, optional
        ISO 8601 timestamp of when digestion occurred. Defaults to current time.
    dataset_modified_at : str, optional
        ISO 8601 timestamp of last dataset update.
    is_clinical : bool, optional
        Whether this is clinical data.
    clinical_purpose : str, optional
        Clinical purpose (e.g., "epilepsy", "depression", "parkinson").
    modality : str, optional
        Experimental modality (e.g., "visual", "auditory", "multisensory", "resting_state").
    cognitive_domain : str, optional
        Cognitive domain (e.g., "attention", "memory", "learning", "motor").
    is_10_20_system : bool, optional
        Whether electrodes follow the 10-20 system.

    Returns
    -------
    Record
        A complete EEGDash record.

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

    # Build timestamps (always set digested_at)
    timestamps: Timestamps = {
        "digested_at": digested_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if dataset_modified_at is not None:
        timestamps["dataset_modified_at"] = dataset_modified_at

    record = Record(
        dataset=dataset,
        data_name=f"{dataset}_{PurePosixPath(bids_relpath).name}",
        bids_relpath=bids_relpath,
        datatype=datatype,
        suffix=suffix,
        extension=extension,
        recording_modality=recording_modality or datatype,  # Default to datatype if not specified
        entities=entities,
        entities_mne=entities_mne,
        storage=Storage(
            backend=storage_backend,
            base=storage_base.rstrip("/"),
            raw_key=bids_relpath,
            dep_keys=dep_keys,
        ),
        timestamps=timestamps,
    )

    # Add clinical info if provided
    if is_clinical is not None or clinical_purpose is not None:
        record["clinical"] = Clinical(
            is_clinical=is_clinical if is_clinical is not None else False,
            purpose=clinical_purpose,
        )

    # Add paradigm info if any field is provided
    if modality is not None or cognitive_domain is not None or is_10_20_system is not None:
        record["paradigm"] = Paradigm(
            modality=modality,
            cognitive_domain=cognitive_domain,
            is_10_20_system=is_10_20_system,
        )

    return record


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


__all__ = [
    "Clinical",
    "Entities",
    "Paradigm",
    "Record",
    "Storage",
    "Timestamps",
    "create_record",
    "validate_record",
]
