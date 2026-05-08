# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash database via REST API.
"""

from typing import Any, Iterable, Mapping, Sequence

from .bids_metadata import merge_query
from .http_api_client import get_client

# --- DataFrame projection ------------------------------------------------ #
# Search-style endpoints map MongoDB JSON records onto a fixed column
# layout. ``_records_to_dataframe`` is the single helper every such
# endpoint reuses; per-endpoint specs (``columns`` + ``aliases``) live as
# module constants right next to the method that consumes them.
#
# ``aliases`` lets one canonical column draw from several legacy/nested
# field paths (dotted keys are resolved via :func:`pandas.json_normalize`).
# The first non-null value across the alias list wins per row, so the
# helper survives v1/v2 record schema drift without per-endpoint glue.

_DATASET_SUMMARY_COLUMNS: Sequence[str] = (
    "dataset_id",
    "name",
    "modality",
    "task",
    "n_subjects",
    "source",
    "license",
    "dataset_doi",
)
_DATASET_FIELD_ALIASES: Mapping[str, Sequence[str]] = {
    "dataset_id": ("dataset_id", "dataset", "_id"),
    "source": ("source", "provider"),
}


def _records_to_dataframe(
    records: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    aliases: Mapping[str, Sequence[str]] | None = None,
):
    """Project a list of MongoDB JSON records onto a fixed DataFrame layout.

    Uses :func:`pandas.json_normalize` to flatten one level of nesting
    (so dotted alias paths like ``clinical.group`` resolve), then for
    each canonical column picks the first non-null value across its
    alias list. Records that are not mappings are skipped.

    Returns an empty DataFrame with the right column set when ``records``
    is empty, so callers get a stable schema regardless of result size.
    """
    import pandas as pd

    aliases = dict(aliases or {})
    rows = [r for r in records if isinstance(r, Mapping)]
    if not rows:
        return pd.DataFrame(columns=list(columns))

    flat = pd.json_normalize(rows, max_level=1)
    out = pd.DataFrame(index=flat.index)
    for col in columns:
        sources = aliases.get(col, (col,))
        present = [s for s in sources if s in flat.columns]
        if not present:
            out[col] = None
        elif len(present) == 1:
            out[col] = flat[present[0]]
        else:
            out[col] = flat[present].bfill(axis=1).iloc[:, 0]
    return out[list(columns)]


class EEGDash:
    """High-level interface to the EEGDash metadata database.

    Provides methods to query, insert, and update metadata records stored in the
    EEGDash database via REST API gateway.

    For working with collections of recordings as PyTorch datasets, prefer
    :class:`EEGDashDataset`.
    """

    def __init__(
        self,
        *,
        database: str = "eegdash",
        api_url: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        """Create a new EEGDash client.

        Parameters
        ----------
        database : str, default "eegdash"
            Name of the MongoDB database to connect to. Common values:
            ``"eegdash"`` (production), ``"eegdash_staging"`` (staging),
            ``"eegdash_v1"`` (legacy archive).
        api_url : str, optional
            Override the default API URL. If not provided, uses the default
            public endpoint or the ``EEGDASH_API_URL`` environment variable.
        auth_token : str, optional
            Authentication token for admin write operations. Not required for
            public read operations.

        Examples
        --------
        >>> eegdash = EEGDash()  # production
        >>> eegdash = EEGDash(database="eegdash_staging")  # staging
        >>> records = eegdash.find({"dataset": "ds002718"})

        """
        self._client = get_client(api_url, database, auth_token)

    def find_datasets(
        self, query: dict[str, Any] | None = None, limit: int = 1000
    ) -> list[Mapping[str, Any]]:
        """Find datasets matching query.

        Parameters
        ----------
        query : dict
            Filter query.
        limit : int
            Max number of datasets to return.

        Returns
        -------
        list of dict
            List of dataset metadata documents.

        """
        return self._client.find_datasets(query, limit=limit)

    def search_datasets(
        self,
        *,
        modality: str | None = None,
        task: str | None = None,
        clinical_group: str | None = None,
        source: str | None = None,
        n_subjects_min: int | None = None,
        license: str | None = None,
        limit: int = 100,
    ):
        """Search the dataset catalogue with friendly keyword filters.

        Convenience wrapper around :meth:`find_datasets` that translates a
        small set of human-friendly keyword arguments into a MongoDB-style
        query and returns a tidy summary :class:`pandas.DataFrame`. This is
        the metadata-only entry point used by tutorials such as
        ``plot_00_first_search``.

        Parameters
        ----------
        modality : str, optional
            Filter by recording modality (e.g., ``"eeg"``, ``"meeg"``).
            Matched case-insensitively against the ``modality`` field.
        task : str, optional
            Filter by BIDS task name (e.g., ``"rest"``, ``"FacePerception"``).
        clinical_group : str, optional
            Filter by clinical cohort label (e.g., ``"healthy"``, ``"adhd"``).
            Matched against ``clinical.group`` (nested) and falls back to the
            flat ``clinical_group`` field.
        source : str, optional
            Filter by data source (e.g., ``"openneuro"``, ``"nemar"``,
            ``"hbn"``). Matched against ``source`` and ``provider`` fields.
        n_subjects_min : int, optional
            Minimum number of subjects in the dataset. Maps to
            ``{"n_subjects": {"$gte": n_subjects_min}}``.
        license : str, optional
            Filter by data license (e.g., ``"CC0"``, ``"CC-BY-4.0"``).
            Matched against the ``license`` field.
        limit : int, default 100
            Maximum number of datasets to return.

        Returns
        -------
        pandas.DataFrame
            One row per matching dataset with summary columns:
            ``dataset_id``, ``name``, ``modality``, ``task``, ``n_subjects``,
            ``source``, ``license``, ``dataset_doi``. Missing fields surface
            as ``None``. The frame is empty (zero rows) when nothing matches.

        Notes
        -----
        ``search_datasets`` does not download any signal bytes; only small
        JSON catalogue documents are transferred. Pair with
        :class:`~eegdash.EEGDashDataset` once a candidate dataset is chosen.

        Examples
        --------
        >>> client = EEGDash()
        >>> df = client.search_datasets(modality="eeg", n_subjects_min=10)
        >>> df = client.search_datasets(task="rest", source="openneuro")

        """
        # Build a MongoDB-style query from the friendly kwargs. Fields
        # with multiple plausible storage shapes (flat vs nested) use
        # ``$or`` so this survives v1/v2 record formats.
        and_clauses: list[dict[str, Any]] = []
        if modality is not None:
            and_clauses.append(
                {"$or": [{"modality": modality}, {"modality": modality.lower()}]}
            )
        if task is not None:
            and_clauses.append({"task": task})
        if clinical_group is not None:
            and_clauses.append(
                {
                    "$or": [
                        {"clinical.group": clinical_group},
                        {"clinical_group": clinical_group},
                    ]
                }
            )
        if source is not None:
            and_clauses.append({"$or": [{"source": source}, {"provider": source}]})
        if n_subjects_min is not None:
            and_clauses.append({"n_subjects": {"$gte": int(n_subjects_min)}})
        if license is not None:
            and_clauses.append({"license": license})

        if not and_clauses:
            query = None
        elif len(and_clauses) == 1:
            query = and_clauses[0]
        else:
            query = {"$and": and_clauses}

        return _records_to_dataframe(
            self._client.find_datasets(query, limit=limit) or [],
            _DATASET_SUMMARY_COLUMNS,
            _DATASET_FIELD_ALIASES,
        )

    def find(
        self, query: dict[str, Any] = None, /, **kwargs
    ) -> list[Mapping[str, Any]]:
        """Find records in the collection.

        Examples
        --------
        >>> from eegdash import EEGDash
        >>> eegdash = EEGDash()
        >>> eegdash.find({"dataset": "ds002718", "subject": {"$in": ["012", "013"]}})  # pre-built query
        >>> eegdash.find(dataset="ds002718", subject="012")  # keyword filters
        >>> eegdash.find(dataset="ds002718", subject=["012", "013"])  # sequence -> $in
        >>> eegdash.find({})  # fetch all (use with care)
        >>> eegdash.find({"dataset": "ds002718"}, subject=["012", "013"])  # combine query + kwargs (AND)

        Parameters
        ----------
        query : dict, optional
            Complete MongoDB query dictionary. This is a positional-only
            argument.
        **kwargs
            User-friendly field filters that are converted to a MongoDB query.
            Values can be scalars (e.g., ``"sub-01"``) or sequences (translated
            to ``$in`` queries). Special parameters: ``limit`` (int) and ``skip`` (int)
            for pagination.

        Returns
        -------
        list of dict
            DB records that match the query.

        """
        limit = kwargs.pop("limit", None)
        skip = kwargs.pop("skip", None)
        final_query = merge_query(query, require_query=True, **kwargs)
        find_kwargs = {
            k: v for k, v in {"limit": limit, "skip": skip}.items() if v is not None
        }
        return list(self._client.find(final_query, **find_kwargs))

    def exists(self, query: dict[str, Any] = None, /, **kwargs) -> bool:
        """Check if at least one record matches the query.

        Parameters
        ----------
        query : dict, optional
            Complete query dictionary. This is a positional-only argument.
        **kwargs
            User-friendly field filters (same as find()).

        Returns
        -------
        bool
            True if at least one matching record exists; False otherwise.

        Examples
        --------
        >>> eeg = EEGDash()
        >>> eeg.exists(dataset="ds002718")  # check by dataset
        >>> eeg.exists({"data_name": "ds002718_sub-001_eeg.set"})  # check by data_name

        """
        return self.find_one(query, **kwargs) is not None

    def count(self, query: dict[str, Any] = None, /, **kwargs) -> int:
        """Count documents matching the query.

        Parameters
        ----------
        query : dict, optional
            Complete query dictionary. This is a positional-only argument.
        **kwargs
            User-friendly field filters (same as find()).

        Returns
        -------
        int
            Number of matching documents.

        Examples
        --------
        >>> eeg = EEGDash()
        >>> count = eeg.count({})  # count all
        >>> count = eeg.count(dataset="ds002718")  # count by dataset

        """
        kwargs.pop("limit", None)
        kwargs.pop("skip", None)
        final_query = merge_query(query, require_query=False, **kwargs)
        return self._client.count_documents(final_query)

    def find_one(
        self, query: dict[str, Any] = None, /, **kwargs
    ) -> Mapping[str, Any] | None:
        """Find a single record matching the query.

        Parameters
        ----------
        query : dict, optional
            Complete query dictionary. This is a positional-only argument.
        **kwargs
            User-friendly field filters (same as find()).

        Returns
        -------
        dict or None
            The first matching record, or None if no match.

        Examples
        --------
        >>> eeg = EEGDash()
        >>> record = eeg.find_one(data_name="ds002718_sub-001_eeg.set")

        """
        final_query = merge_query(query, require_query=True, **kwargs)
        return self._client.find_one(final_query)

    def get_dataset(self, dataset_id: str) -> Mapping[str, Any] | None:
        """Fetch metadata for a specific dataset.

        Parameters
        ----------
        dataset_id : str
            The unique identifier of the dataset (e.g., 'ds002718').

        Returns
        -------
        dict or None
            The dataset metadata document, or None if not found.

        """
        return self._client.get_dataset(dataset_id)

    def insert(self, records: dict[str, Any] | list[dict[str, Any]]) -> int:
        """Insert one or more records (requires auth_token).

        Parameters
        ----------
        records : dict or list of dict
            A single record or list of records to insert.

        Returns
        -------
        int
            Number of records inserted.

        Examples
        --------
        >>> eeg = EEGDash(auth_token="...")
        >>> eeg.insert({"dataset": "ds001", "subject": "01", ...})  # single
        >>> eeg.insert([record1, record2, record3])  # batch

        """
        if isinstance(records, dict):
            self._client.insert_one(records)
            return 1
        return self._client.insert_many(records)

    def update_field(
        self,
        query: dict[str, Any] = None,
        /,
        *,
        update: dict[str, Any],
        **kwargs,
    ) -> tuple[int, int]:
        """Update fields on records matching the query (requires auth_token).

        Use this to add or modify fields across matching records,
        e.g., after re-extracting entities with an improved algorithm.

        Parameters
        ----------
        query : dict, optional
            Filter query to match records. This is a positional-only argument.
        update : dict
            Fields to update. Keys are field names, values are new values.
        **kwargs
            User-friendly field filters (same as find()).

        Returns
        -------
        tuple of (matched_count, modified_count)
            Number of records matched and actually modified.

        Examples
        --------
        >>> eeg = EEGDash(auth_token="...")
        >>> # Update entities for all records in a dataset
        >>> eeg.update_field({"dataset": "ds002718"}, update={"entities": {"subject": "01"}})
        >>> # Using kwargs for filter
        >>> eeg.update_field(dataset="ds002718", update={"entities": new_entities})
        >>> # Combine query + kwargs
        >>> eeg.update_field({"dataset": "ds002718"}, subject="01", update={"entities": new_entities})

        """
        final_query = merge_query(query, require_query=True, **kwargs)
        return self._client.update_many(final_query, update)

    def update_dataset(self, dataset_id: str, update: dict[str, Any]) -> int:
        """Update metadata for a specific dataset (requires auth_token).

        Parameters
        ----------
        dataset_id : str
            The unique identifier of the dataset (e.g., 'ds002718').
        update : dict
            Dictionary of fields to update.

        Returns
        -------
        int
            Number of documents modified (0 or 1).

        Examples
        --------
        >>> eeg = EEGDash(auth_token="...")
        >>> eeg.update_dataset("ds002718", {"clinical.is_clinical": True})

        """
        return self._client.update_dataset(dataset_id, update)


def __getattr__(name: str):
    # Backward-compat: allow ``from eegdash.api import EEGDashDataset`` without
    # importing braindecode unless needed.
    if name == "EEGDashDataset":
        from .dataset.dataset import EEGDashDataset

        return EEGDashDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EEGDash"]
