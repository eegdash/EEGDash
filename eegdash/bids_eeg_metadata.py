# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""BIDS metadata processing and query building utilities.

This module provides functions for processing BIDS-formatted EEG metadata, building database
queries from user parameters, and enriching metadata records with participant information.
It handles the translation between user-friendly query parameters and MongoDB query syntax.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd
from mne_bids import BIDSPath

from .const import ALLOWED_QUERY_FIELDS
from .const import config as data_config
from .logging import logger

__all__ = [
    "build_query_from_kwargs",
    "load_eeg_attrs_from_bids_file",
    "merge_participants_fields",
    "normalize_key",
    "participants_row_for_subject",
    "participants_extras_from_tsv",
    "attach_participants_extras",
    "enrich_from_participants",
]


def build_query_from_kwargs(**kwargs) -> dict[str, Any]:
    """Build and validate a MongoDB query from keyword arguments.

    This function converts user-friendly keyword arguments into a valid
    MongoDB query dictionary. It handles scalar values as exact matches and
    list-like values as ``$in`` queries. It also performs validation to
    reject unsupported fields and empty values.

    Parameters
    ----------
    **kwargs
        Keyword arguments representing query filters. Allowed keys are defined
        in ``eegdash.const.ALLOWED_QUERY_FIELDS``.

    Returns
    -------
    dict
        A MongoDB query dictionary.

    Raises
    ------
    ValueError
        If an unsupported query field is provided, or if a value is None or
        an empty string/list.
    """
    # 1. Validate that all provided keys are allowed for querying
    unknown_fields = set(kwargs.keys()) - ALLOWED_QUERY_FIELDS
    if unknown_fields:
        raise ValueError(
            f"Unsupported query field(s): {', '.join(sorted(unknown_fields))}. "
            f"Allowed fields are: {', '.join(sorted(ALLOWED_QUERY_FIELDS))}"
        )

    # 2. Construct the query dictionary
    query = {}
    for key, value in kwargs.items():
        # None is not a valid constraint
        if value is None:
            raise ValueError(
                f"Received None for query parameter '{key}'. Provide a concrete value."
            )

        # Handle list-like values as multi-constraints
        if isinstance(value, (list, tuple, set)):
            cleaned: list[Any] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    item = item.strip()
                    if not item:
                        continue
                cleaned.append(item)
            # Deduplicate while preserving order
            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                raise ValueError(
                    f"Received an empty list for query parameter '{key}'. This is not supported."
                )
            query[key] = {"$in": cleaned}
        else:
            # Scalars: trim strings and validate
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise ValueError(
                        f"Received an empty string for query parameter '{key}'."
                    )
            query[key] = value

    return query


def load_eeg_attrs_from_bids_file(bids_dataset, bids_file: str) -> dict[str, Any]:
    """Build a metadata record for a BIDS file.

    Extracts metadata attributes from a single BIDS EEG file within a given
    BIDS dataset. The extracted attributes include BIDS entities, file paths,
    and technical metadata required for database indexing.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object containing the file.
    bids_file : str
        The path to the BIDS file to process.

    Returns
    -------
    dict
        A dictionary of metadata attributes for the file, suitable for
        insertion into the database.

    Raises
    ------
    ValueError
        If ``bids_file`` is not found in the ``bids_dataset``.
    """
    if bids_file not in bids_dataset.files:
        raise ValueError(f"{bids_file} not in {bids_dataset.dataset}")

    # Initialize attrs with None values for all expected fields
    attrs = {field: None for field in data_config["attributes"].keys()}

    file = Path(bids_file).name
    dsnumber = bids_dataset.dataset
    # extract openneuro path by finding the first occurrence of the dataset name in the filename and remove the path before that
    openneuro_path = dsnumber + bids_file.split(dsnumber)[1]

    # Update with actual values where available
    try:
        participants_tsv = bids_dataset.subject_participant_tsv(bids_file)
    except Exception as e:
        logger.error("Error getting participants_tsv: %s", str(e))
        participants_tsv = None

    try:
        eeg_json = bids_dataset.eeg_json(bids_file)
    except Exception as e:
        logger.error("Error getting eeg_json: %s", str(e))
        eeg_json = None

    bids_dependencies_files = data_config["bids_dependencies_files"]
    bidsdependencies: list[str] = []
    for extension in bids_dependencies_files:
        try:
            dep_path = bids_dataset.get_bids_metadata_files(bids_file, extension)
            dep_path = [
                str(bids_dataset.get_relative_bidspath(dep)) for dep in dep_path
            ]
            bidsdependencies.extend(dep_path)
        except Exception:
            pass

    bids_path = BIDSPath(
        subject=bids_dataset.get_bids_file_attribute("subject", bids_file),
        session=bids_dataset.get_bids_file_attribute("session", bids_file),
        task=bids_dataset.get_bids_file_attribute("task", bids_file),
        run=bids_dataset.get_bids_file_attribute("run", bids_file),
        root=bids_dataset.bidsdir,
        datatype=bids_dataset.get_bids_file_attribute("modality", bids_file),
        suffix="eeg",
        extension=Path(bids_file).suffix,
        check=False,
    )

    sidecars_map = {
        ".set": [".fdt"],
        ".vhdr": [".eeg", ".vmrk", ".dat", ".raw"],
    }
    for ext in sidecars_map.get(bids_path.extension, []):
        sidecar = bids_path.find_matching_sidecar(extension=ext, on_error="ignore")
        if sidecar is not None:
            bidsdependencies.append(str(bids_dataset._get_relative_bidspath(sidecar)))

    # Define field extraction functions with error handling
    field_extractors = {
        "data_name": lambda: f"{bids_dataset.dataset}_{file}",
        "dataset": lambda: bids_dataset.dataset,
        "bidspath": lambda: openneuro_path,
        "subject": lambda: bids_dataset.get_bids_file_attribute("subject", bids_file),
        "task": lambda: bids_dataset.get_bids_file_attribute("task", bids_file),
        "session": lambda: bids_dataset.get_bids_file_attribute("session", bids_file),
        "run": lambda: bids_dataset.get_bids_file_attribute("run", bids_file),
        "modality": lambda: bids_dataset.get_bids_file_attribute("modality", bids_file),
        "sampling_frequency": lambda: bids_dataset.get_bids_file_attribute(
            "sfreq", bids_file
        ),
        "nchans": lambda: bids_dataset.get_bids_file_attribute("nchans", bids_file),
        "ntimes": lambda: bids_dataset.get_bids_file_attribute("ntimes", bids_file),
        "participant_tsv": lambda: participants_tsv,
        "eeg_json": lambda: eeg_json,
        "bidsdependencies": lambda: bidsdependencies,
    }

    # Dynamically populate attrs with error handling
    for field, extractor in field_extractors.items():
        try:
            attrs[field] = extractor()
        except Exception as e:
            logger.error("Error extracting %s : %s", field, str(e))
            attrs[field] = None

    return attrs


def normalize_key(key: str) -> str:
    """Normalize a string key for robust matching.

    Converts the key to lowercase, replaces non-alphanumeric characters with
    underscores, and removes leading/trailing underscores. This allows for
    tolerant matching of keys that may have different capitalization or
    separators (e.g., "p-factor" becomes "p_factor").

    Parameters
    ----------
    key : str
        The key to normalize.

    Returns
    -------
    str
        The normalized key.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def merge_participants_fields(
    description: dict[str, Any],
    participants_row: dict[str, Any] | None,
    description_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Merge fields from a participants.tsv row into a description dict.

    Enriches a description dictionary with data from a subject's row in
    ``participants.tsv``. It avoids overwriting existing keys in the
    description.

    Parameters
    ----------
    description : dict
        The description dictionary to enrich.
    participants_row : dict or None
        A dictionary representing a row from ``participants.tsv``. If None,
        the original description is returned unchanged.
    description_fields : list of str, optional
        A list of specific fields to include in the description. Matching is
        done using normalized keys.

    Returns
    -------
    dict
        The enriched description dictionary.
    """
    if not isinstance(description, dict) or not isinstance(participants_row, dict):
        return description

    # Normalize participants keys and keep first non-None value per normalized key
    norm_map: dict[str, Any] = {}
    for part_key, part_value in participants_row.items():
        norm_key = normalize_key(part_key)
        if norm_key not in norm_map and part_value is not None:
            norm_map[norm_key] = part_value

    # Ensure description_fields is a list for matching
    requested = list(description_fields or [])

    # 1) Fill requested fields first using normalized matching, preserving names
    for key in requested:
        if key in description:
            continue
        requested_norm_key = normalize_key(key)
        if requested_norm_key in norm_map:
            description[key] = norm_map[requested_norm_key]

    # 2) Add remaining participants columns generically under normalized names,
    #    unless a requested field already captured them
    requested_norm = {normalize_key(k) for k in requested}
    for norm_key, part_value in norm_map.items():
        if norm_key in requested_norm:
            continue
        if norm_key not in description:
            description[norm_key] = part_value
    return description


def participants_row_for_subject(
    bids_root: str | Path,
    subject: str,
    id_columns: tuple[str, ...] = ("participant_id", "participant", "subject"),
) -> pd.Series | None:
    """Load participants.tsv and return the row for a specific subject.

    Searches for a subject's data in the ``participants.tsv`` file within a
    BIDS dataset. It can identify the subject with or without the "sub-"
    prefix.

    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
    subject : str
        The subject identifier (e.g., "01" or "sub-01").
    id_columns : tuple of str, default ("participant_id", "participant", "subject")
        A tuple of column names to search for the subject identifier.

    Returns
    -------
    pandas.Series or None
        A pandas Series containing the subject's data if found, otherwise None.
    """
    try:
        participants_tsv = Path(bids_root) / "participants.tsv"
        if not participants_tsv.exists():
            return None

        df = pd.read_csv(
            participants_tsv, sep="\t", dtype="string", keep_default_na=False
        )
        if df.empty:
            return None

        candidates = {str(subject), f"sub-{subject}"}
        present_cols = [c for c in id_columns if c in df.columns]
        if not present_cols:
            return None

        mask = pd.Series(False, index=df.index)
        for col in present_cols:
            mask |= df[col].isin(candidates)
        match = df.loc[mask]
        if match.empty:
            return None
        return match.iloc[0]
    except Exception:
        return None


def participants_extras_from_tsv(
    bids_root: str | Path,
    subject: str,
    *,
    id_columns: tuple[str, ...] = ("participant_id", "participant", "subject"),
    na_like: tuple[str, ...] = ("", "n/a", "na", "nan", "unknown", "none"),
) -> dict[str, Any]:
    """Extract additional participant information from participants.tsv.

    Retrieves all non-identifier and non-empty fields for a subject from
    the ``participants.tsv`` file.

    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
    subject : str
        The subject identifier.
    id_columns : tuple of str, default ("participant_id", "participant", "subject")
        Column names to be treated as identifiers and excluded from the
        output.
    na_like : tuple of str, default ("", "n/a", "na", "nan", "unknown", "none")
        Values to be considered as "Not Available" and excluded.

    Returns
    -------
    dict
        A dictionary of extra participant information.
    """
    row = participants_row_for_subject(bids_root, subject, id_columns=id_columns)
    if row is None:
        return {}

    # Drop identifier columns and clean values
    extras = row.drop(labels=[c for c in id_columns if c in row.index], errors="ignore")
    s = extras.astype("string").str.strip()
    valid = ~s.isna() & ~s.str.lower().isin(na_like)
    return s[valid].to_dict()


def attach_participants_extras(
    raw: Any,
    description: Any,
    extras: dict[str, Any],
) -> None:
    """Attach extra participant data to a raw object and its description.

    Updates the ``raw.info['subject_info']`` and the description object
    (dict or pandas Series) with extra data from ``participants.tsv``.
    It does not overwrite existing keys.

    Parameters
    ----------
    raw : mne.io.Raw
        The MNE Raw object to be updated.
    description : dict or pandas.Series
        The description object to be updated.
    extras : dict
        A dictionary of extra participant information to attach.
    """
    if not extras:
        return

    # Raw.info enrichment
    try:
        subject_info = raw.info.get("subject_info") or {}
        if not isinstance(subject_info, dict):
            subject_info = {}
        pe = subject_info.get("participants_extras") or {}
        if not isinstance(pe, dict):
            pe = {}
        for k, v in extras.items():
            pe.setdefault(k, v)
        subject_info["participants_extras"] = pe
        raw.info["subject_info"] = subject_info
    except Exception:
        pass

    # Description enrichment
    try:
        import pandas as _pd  # local import to avoid hard dependency at import time

        if isinstance(description, dict):
            for k, v in extras.items():
                description.setdefault(k, v)
        elif isinstance(description, _pd.Series):
            missing = [k for k in extras.keys() if k not in description.index]
            if missing:
                description.loc[missing] = [extras[m] for m in missing]
    except Exception:
        pass


def enrich_from_participants(
    bids_root: str | Path,
    bidspath: BIDSPath,
    raw: Any,
    description: Any,
) -> dict[str, Any]:
    """Read participants.tsv and attach extra info for the subject.

    This is a convenience function that finds the subject from the
    ``bidspath``, retrieves extra information from ``participants.tsv``,
    and attaches it to the raw object and its description.

    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
    bidspath : mne_bids.BIDSPath
        The BIDSPath object for the current data file.
    raw : mne.io.Raw
        The MNE Raw object to be updated.
    description : dict or pandas.Series
        The description object to be updated.

    Returns
    -------
    dict
        The dictionary of extras that were attached.
    """
    subject = getattr(bidspath, "subject", None)
    if not subject:
        return {}
    extras = participants_extras_from_tsv(bids_root, subject)
    attach_participants_extras(raw, description, extras)
    return extras


__all__ = [
    "participants_row_for_subject",
    "participants_extras_from_tsv",
    "attach_participants_extras",
    "enrich_from_participants",
]