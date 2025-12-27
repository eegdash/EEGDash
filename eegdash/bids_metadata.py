# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""BIDS metadata processing and query building utilities.

This module provides functions for building database queries from user parameters
and enriching metadata records with participant information from BIDS datasets.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .const import ALLOWED_QUERY_FIELDS

__all__ = [
    "build_query_from_kwargs",
    "merge_query",
    "normalize_key",
    "merge_participants_fields",
    "participants_row_for_subject",
    "participants_extras_from_tsv",
    "attach_participants_extras",
    "enrich_from_participants",
    "get_entity_from_record",
    "get_entities_from_record",
]


def get_entity_from_record(record: dict[str, Any], entity: str) -> Any:
    """Get an entity value from a record, supporting both v1 (flat) and v2 (nested) formats.

    Parameters
    ----------
    record : dict
        A record dictionary.
    entity : str
        Entity name (e.g., "subject", "task", "session", "run").

    Returns
    -------
    Any
        The entity value, or None if not found.

    Examples
    --------
    >>> # v2 record (nested)
    >>> rec = {"entities": {"subject": "01", "task": "rest"}}
    >>> get_entity_from_record(rec, "subject")
    '01'
    >>> # v1 record (flat)
    >>> rec = {"subject": "01", "task": "rest"}
    >>> get_entity_from_record(rec, "subject")
    '01'

    """
    # Try nested entities first (v2 format)
    if "entities" in record and isinstance(record["entities"], dict):
        val = record["entities"].get(entity)
        if val is not None:
            return val
    # Fall back to flat format (v1 / legacy)
    return record.get(entity)


def get_entities_from_record(
    record: dict[str, Any],
    entities: tuple[str, ...] = ("subject", "session", "run", "task"),
) -> dict[str, Any]:
    """Get multiple entity values from a record.

    Parameters
    ----------
    record : dict
        A record dictionary.
    entities : tuple of str
        Entity names to extract.

    Returns
    -------
    dict
        Dictionary of entity values (only non-None values included).

    """
    return {
        k: v for k in entities if (v := get_entity_from_record(record, k)) is not None
    }


# BIDS entity fields that should be queried in entities.* namespace
_ENTITY_FIELDS = {"subject", "task", "session", "run"}


def build_query_from_kwargs(**kwargs) -> dict[str, Any]:
    """Build and validate a MongoDB query from keyword arguments.

    Converts user-friendly keyword arguments into a valid MongoDB query dictionary.
    Scalar values become exact matches; list-like values become ``$in`` queries.

    BIDS entity fields (subject, task, session, run) are automatically mapped to
    the nested ``entities.*`` namespace for v2 records compatibility.

    Parameters
    ----------
    **kwargs
        Query filters. Allowed keys are in ``eegdash.const.ALLOWED_QUERY_FIELDS``.

    Returns
    -------
    dict
        A MongoDB query dictionary.

    Raises
    ------
    ValueError
        If an unsupported field is provided, or if a value is None/empty.

    """
    unknown_fields = set(kwargs.keys()) - ALLOWED_QUERY_FIELDS
    if unknown_fields:
        raise ValueError(
            f"Unsupported query field(s): {', '.join(sorted(unknown_fields))}. "
            f"Allowed fields are: {', '.join(sorted(ALLOWED_QUERY_FIELDS))}"
        )

    query = {}
    for key, value in kwargs.items():
        if value is None:
            raise ValueError(
                f"Received None for query parameter '{key}'. Provide a concrete value."
            )

        # Map entity fields to nested entities.* namespace
        query_key = f"entities.{key}" if key in _ENTITY_FIELDS else key

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
            cleaned = list(dict.fromkeys(cleaned))  # dedupe preserving order
            if not cleaned:
                raise ValueError(f"Received an empty list for query parameter '{key}'.")
            query[query_key] = {"$in": cleaned}
        else:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    raise ValueError(
                        f"Received an empty string for query parameter '{key}'."
                    )
            query[query_key] = value

    return query


def merge_query(
    query: dict[str, Any] | None = None,
    require_query: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Merge a raw query dict with keyword arguments into a final query.

    Parameters
    ----------
    query : dict or None
        Raw MongoDB query dictionary. Pass ``{}`` to match all documents.
    require_query : bool, default True
        If True, raise ValueError when no query or kwargs provided.
    **kwargs
        User-friendly field filters (converted via ``build_query_from_kwargs``).

    Returns
    -------
    dict
        The merged MongoDB query.

    Raises
    ------
    ValueError
        If ``require_query=True`` and neither query nor kwargs provided,
        or if conflicting constraints are detected.

    """
    raw = query if isinstance(query, dict) else None
    kw = build_query_from_kwargs(**kwargs) if kwargs else None

    if raw is not None and kw:
        for key in set(raw.keys()) & set(kw.keys()) & ALLOWED_QUERY_FIELDS:
            _check_constraint_conflict(raw, kw, key)
        return {"$and": [raw, kw]} if raw else kw
    if raw is not None:
        return raw
    if kw:
        return kw
    if require_query:
        raise ValueError(
            "Query required: pass a query dict or keyword arguments. "
            "Use query={} to match all documents."
        )
    return {}


def _check_constraint_conflict(
    q1: dict[str, Any], q2: dict[str, Any], key: str
) -> None:
    """Raise ValueError if q1 and q2 have incompatible constraints on key."""
    v1, v2 = q1.get(key), q2.get(key)
    if v1 is None or v2 is None:
        return

    s1 = set(v1["$in"]) if isinstance(v1, dict) and "$in" in v1 else {v1}
    s2 = set(v2["$in"]) if isinstance(v2, dict) and "$in" in v2 else {v2}

    if not s1 & s2:
        raise ValueError(f"Conflicting constraints for '{key}': {v1!r} vs {v2!r}")


# =============================================================================
# Participants.tsv utilities
# =============================================================================


def normalize_key(key: str) -> str:
    """Normalize a string key for robust matching.

    Converts to lowercase, replaces non-alphanumeric chars with underscores.

    """
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def participants_row_for_subject(
    bids_root: str | Path,
    subject: str,
    id_columns: tuple[str, ...] = ("participant_id", "participant", "subject"),
) -> pd.Series | None:
    """Load participants.tsv and return the row for a specific subject.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
    subject : str
        Subject identifier (e.g., "01" or "sub-01").
    id_columns : tuple of str
        Column names to search for the subject identifier.

    Returns
    -------
    pandas.Series or None
        Subject's data if found, otherwise None.

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

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
    subject : str
        Subject identifier.
    id_columns : tuple of str
        Column names treated as identifiers (excluded from output).
    na_like : tuple of str
        Values considered as "Not Available" (excluded).

    Returns
    -------
    dict
        Extra participant information.

    """
    row = participants_row_for_subject(bids_root, subject, id_columns=id_columns)
    if row is None:
        return {}

    extras = row.drop(labels=[c for c in id_columns if c in row.index], errors="ignore")
    s = extras.astype("string").str.strip()
    valid = ~s.isna() & ~s.str.lower().isin(na_like)
    return s[valid].to_dict()


def merge_participants_fields(
    description: dict[str, Any],
    participants_row: dict[str, Any] | None,
    description_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Merge fields from a participants.tsv row into a description dict.

    Parameters
    ----------
    description : dict
        The description dictionary to enrich.
    participants_row : dict or None
        A row from participants.tsv. If None, returns description unchanged.
    description_fields : list of str, optional
        Specific fields to include (matched using normalized keys).

    Returns
    -------
    dict
        The enriched description dictionary.

    """
    if not isinstance(description, dict) or not isinstance(participants_row, dict):
        return description

    norm_map: dict[str, Any] = {}
    for part_key, part_value in participants_row.items():
        norm_key = normalize_key(part_key)
        if norm_key not in norm_map and part_value is not None:
            norm_map[norm_key] = part_value

    requested = list(description_fields or [])

    for key in requested:
        if key in description:
            continue
        requested_norm_key = normalize_key(key)
        if requested_norm_key in norm_map:
            description[key] = norm_map[requested_norm_key]

    requested_norm = {normalize_key(k) for k in requested}
    for norm_key, part_value in norm_map.items():
        if norm_key in requested_norm:
            continue
        if norm_key not in description:
            description[norm_key] = part_value

    return description


def attach_participants_extras(
    raw: Any,
    description: Any,
    extras: dict[str, Any],
) -> None:
    """Attach extra participant data to a raw object and its description.

    Parameters
    ----------
    raw : mne.io.Raw
        The MNE Raw object to be updated.
    description : dict or pandas.Series
        The description object to be updated.
    extras : dict
        Extra participant information to attach.

    """
    if not extras:
        return

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

    try:
        import pandas as _pd

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
    bidspath: Any,
    raw: Any,
    description: Any,
) -> dict[str, Any]:
    """Read participants.tsv and attach extra info for the subject.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
    bidspath : mne_bids.BIDSPath
        BIDSPath object for the current data file.
    raw : mne.io.Raw
        The MNE Raw object to be updated.
    description : dict or pandas.Series
        The description object to be updated.

    Returns
    -------
    dict
        The extras that were attached.

    """
    subject = getattr(bidspath, "subject", None)
    if not subject:
        return {}
    extras = participants_extras_from_tsv(bids_root, subject)
    attach_participants_extras(raw, description, extras)
    return extras
