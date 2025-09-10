import logging
import re
from pathlib import Path
from typing import Any

from .const import ALLOWED_QUERY_FIELDS
from .const import config as data_config

logger = logging.getLogger("eegdash")

__all__ = [
    "build_query_from_kwargs",
    "load_eeg_attrs_from_bids_file",
    "merge_participants_fields",
    "normalize_key",
]


def build_query_from_kwargs(**kwargs) -> dict[str, Any]:
    """Build and validate a MongoDB query from user-friendly keyword arguments.

    This function takes keyword arguments and constructs a MongoDB query dictionary.
    It validates the keys against a list of allowed fields, handles different
    value types (scalars, lists), and cleans the input data.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments representing the query filters. Allowed keys are defined
        in ``ALLOWED_QUERY_FIELDS``.

    Returns
    -------
    dict
        A MongoDB query dictionary.

    Raises
    ------
    ValueError
        If an unsupported query field is provided, or if a value is None or an
        empty string/list.

    Notes
    -----
    - Rejects None values and empty/whitespace-only strings.
    - For list/tuple/set values: strips strings, drops None/empties, deduplicates, and uses `$in`.
    - Preserves scalars as exact matches.
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


def _get_raw_extensions(bids_file: str, bids_dataset) -> list[str]:
    """Get paths to sidecar files associated with a BIDS data file.

    This helper function finds paths to additional "sidecar" files (e.g., `.fdt` for
    `.set` files) that may be associated with a given main data file in a BIDS
    dataset. The paths are returned as relative to the parent dataset path.

    Parameters
    ----------
    bids_file : str
        The path to the main BIDS data file.
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object containing the file.

    Returns
    -------
    list[str]
        A list of relative paths to the sidecar files.
    """
    bids_file = Path(bids_file)
    extensions = {
        ".set": [".set", ".fdt"],  # eeglab
        ".edf": [".edf"],  # european
        ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
        ".bdf": [".bdf"],  # biosemi
    }
    return [
        str(bids_dataset._get_relative_bidspath(bids_file.with_suffix(suffix)))
        for suffix in extensions[bids_file.suffix]
        if bids_file.with_suffix(suffix).exists()
    ]


def load_eeg_attrs_from_bids_file(bids_dataset, bids_file: str) -> dict[str, Any]:
    """Build the metadata record for a given BIDS file in a BIDS dataset.

    This function constructs a metadata record for a single recording file within a
    BIDS dataset. The attributes are based on the fields defined in the data
    configuration, with additional metadata such as paths to related files.

    Parameters
    ----------
    bids_dataset : EEGBIDSDataset
        The BIDS dataset object containing the file.
    bids_file : str
        The path to the BIDS file within the dataset.

    Returns
    -------
    dict
        A dictionary representing the metadata record for the given file. This is the
        same format as the records stored in the database.

    Raises
    ------
    ValueError
        If the specified `bids_file` is not found in the `bids_dataset`.
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
    bidsdependencies = []
    for extension in bids_dependencies_files:
        try:
            dep_path = bids_dataset.get_bids_metadata_files(bids_file, extension)
            dep_path = [
                str(bids_dataset.get_relative_bidspath(dep)) for dep in dep_path
            ]
            bidsdependencies.extend(dep_path)
        except Exception:
            pass

    bidsdependencies.extend(_get_raw_extensions(bids_file, bids_dataset))

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
    """Normalize a metadata key for robust matching.

    This function converts a key to a standardized format by making it lowercase,
    replacing non-alphanumeric characters with underscores, and stripping any
    leading or trailing underscores. This allows for tolerant matching of keys
    (e.g., "p-factor" ≈ "p_factor" ≈ "P Factor").

    Parameters
    ----------
    key : str
        The key to be normalized.

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
    """Merge participants.tsv fields into a dataset description dictionary.

    This function enriches a description dictionary with data from a
    `participants.tsv` row. It preserves existing entries in `description`
    and adds new fields from the participants data.

    Parameters
    ----------
    description : dict
        The current description dictionary to be enriched.
    participants_row : dict or None
        A dictionary representing a row from a `participants.tsv` file.
    description_fields : list of str, optional
        A list of specific fields to include in the description. Matching is
        done using normalized keys, but the original field names are preserved.

    Returns
    -------
    dict
        The enriched description dictionary.

    Notes
    -----
    - Preserves existing entries in ``description`` (no overwrites).
    - Fills requested ``description_fields`` first, preserving their original names.
    - Adds all remaining participants columns generically using normalized keys
      unless a matching requested field already captured them.
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
