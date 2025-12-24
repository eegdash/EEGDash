# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash database via REST API.
"""

from typing import Any, Mapping

from .bids_eeg_metadata import merge_query
from .http_api_client import get_client


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
            ``"eegdash"`` (production), ``"eegdashstaging"`` (staging).
        api_url : str, optional
            Override the default API URL. If not provided, uses the default
            public endpoint or the ``EEGDASH_API_URL`` environment variable.
        auth_token : str, optional
            Authentication token for admin write operations. Not required for
            public read operations.

        Examples
        --------
        >>> eegdash = EEGDash()  # production
        >>> eegdash = EEGDash(database="eegdashstaging")  # staging
        >>> records = eegdash.find({"dataset": "ds002718"})

        """
        self._client = get_client(api_url, database, auth_token)

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
        find_kwargs = {k: v for k, v in {"limit": limit, "skip": skip}.items() if v is not None}
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

    def find_one(self, query: dict[str, Any] = None, /, **kwargs) -> Mapping[str, Any] | None:
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


def __getattr__(name: str):
    # Backward-compat: allow ``from eegdash.api import EEGDashDataset`` without
    # importing braindecode unless needed.
    if name == "EEGDashDataset":
        from .dataset.dataset import EEGDashDataset

        return EEGDashDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EEGDash", "EEGDashDataset"]
