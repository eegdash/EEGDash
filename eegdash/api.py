# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash database via REST API.
"""

from typing import Any, Mapping

from .bids_eeg_metadata import build_query_from_kwargs
from .const import ALLOWED_QUERY_FIELDS
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
        is_staging: bool = False,
        api_url: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        """Create a new EEGDash client.

        Parameters
        ----------
        is_staging : bool, default False
            If ``True``, use the staging database (``eegdashstaging``); otherwise
            use the production database (``eegdash``).
        api_url : str, optional
            Override the default API URL. If not provided, uses the default
            public endpoint or the ``EEGDASH_API_URL`` environment variable.
        auth_token : str, optional
            Authentication token for admin write operations. Not required for
            public read operations.

        Examples
        --------
        >>> eegdash = EEGDash()
        >>> records = eegdash.find({"dataset": "ds002718"})

        """
        self._client = get_client(api_url, is_staging, auth_token)

    def find(
        self, query: dict[str, Any] = None, /, **kwargs
    ) -> list[Mapping[str, Any]]:
        """Find records in the collection.

        Examples
        --------
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
        # Extract pagination parameters before building query
        limit = kwargs.pop("limit", None)
        skip = kwargs.pop("skip", None)

        final_query: dict[str, Any] | None = None

        # Accept explicit empty dict {} to mean "match all"
        raw_query = query if isinstance(query, dict) else None
        kwargs_query = build_query_from_kwargs(**kwargs) if kwargs else None

        # Determine presence, treating {} as a valid raw query
        has_raw = isinstance(raw_query, dict)
        has_kwargs = kwargs_query is not None

        if has_raw and has_kwargs:
            # Detect conflicting constraints on the same field (e.g., task specified
            # differently in both places) and raise a clear error instead of silently
            # producing an empty result.
            self._raise_if_conflicting_constraints(raw_query, kwargs_query)
            # Merge with logical AND so both constraints apply
            if raw_query:  # non-empty dict adds constraints
                final_query = {"$and": [raw_query, kwargs_query]}
            else:  # {} adds nothing; use kwargs_query only
                final_query = kwargs_query
        elif has_raw:
            # May be {} meaning match-all, or a non-empty dict
            final_query = raw_query
        elif has_kwargs:
            final_query = kwargs_query
        else:
            # Avoid accidental full scans
            raise ValueError(
                "find() requires a query dictionary or at least one keyword argument. "
                "To find all documents, use find({})."
            )

        # Pass limit and skip to the collection's find method
        find_kwargs = {}
        if limit is not None:
            find_kwargs["limit"] = limit
        if skip is not None:
            find_kwargs["skip"] = skip

        results = self._client.find(final_query, **find_kwargs)

        return list(results)

    def exist(self, query: dict[str, Any]) -> bool:
        """Return True if at least one record matches the query, else False.

        This is a lightweight existence check that uses MongoDB's ``find_one``
        instead of fetching all matching documents (which would be wasteful in
        both time and memory for broad queries). Only a restricted set of
        fields is accepted to avoid accidental full scans caused by malformed
        or unsupported keys.

        Parameters
        ----------
        query : dict
            Mapping of allowed field(s) to value(s). Allowed keys: ``data_name``
            and ``dataset``. The query must not be empty.

        Returns
        -------
        bool
            True if at least one matching record exists; False otherwise.

        Raises
        ------
        TypeError
            If ``query`` is not a dict.
        ValueError
            If ``query`` is empty or contains unsupported field names.

        """
        if not isinstance(query, dict):
            raise TypeError("query must be a dict")
        if not query:
            raise ValueError("query cannot be empty")

        accepted_query_fields = {"data_name", "dataset"}
        unknown = set(query.keys()) - accepted_query_fields
        if unknown:
            raise ValueError(
                f"Unsupported query field(s): {', '.join(sorted(unknown))}. "
                f"Allowed: {sorted(accepted_query_fields)}"
            )

        doc = self._client.find_one(query)
        return doc is not None

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
        # Extract limit/skip if present (not used for count but kept for consistency)
        kwargs.pop("limit", None)
        kwargs.pop("skip", None)

        final_query: dict[str, Any] | None = None

        # Accept explicit empty dict {} to mean "match all"
        raw_query = query if isinstance(query, dict) else None
        kwargs_query = build_query_from_kwargs(**kwargs) if kwargs else None

        # Determine presence, treating {} as a valid raw query
        has_raw = isinstance(raw_query, dict)
        has_kwargs = kwargs_query is not None

        if has_raw and has_kwargs:
            self._raise_if_conflicting_constraints(raw_query, kwargs_query)
            if raw_query:
                final_query = {"$and": [raw_query, kwargs_query]}
            else:
                final_query = kwargs_query
        elif has_raw:
            final_query = raw_query
        elif has_kwargs:
            final_query = kwargs_query
        else:
            # For count, empty query is acceptable (count all)
            final_query = {}

        return self._client.count_documents(final_query)

    def _extract_simple_constraint(
        self, query: dict[str, Any], key: str
    ) -> tuple[str, Any] | None:
        """Extract a simple constraint for a given key from a query dict.

        Supports top-level equality (e.g., ``{'subject': '01'}``) and ``$in``
        (e.g., ``{'subject': {'$in': ['01', '02']}}``) constraints.

        Parameters
        ----------
        query : dict
            The MongoDB query dictionary.
        key : str
            The key for which to extract the constraint.

        Returns
        -------
        tuple or None
            A tuple of (kind, value) where kind is "eq" or "in", or None if the
            constraint is not present or unsupported.

        """
        if not isinstance(query, dict) or key not in query:
            return None
        val = query[key]
        if isinstance(val, dict):
            if "$in" in val and isinstance(val["$in"], (list, tuple)):
                return ("in", list(val["$in"]))
            return None  # unsupported operator shape for conflict checking
        else:
            return "eq", val

    def _raise_if_conflicting_constraints(
        self, raw_query: dict[str, Any], kwargs_query: dict[str, Any]
    ) -> None:
        """Raise ValueError if query sources have incompatible constraints.

        Checks for mutually exclusive constraints on the same field to avoid
        silent empty results.

        Parameters
        ----------
        raw_query : dict
            The raw MongoDB query dictionary.
        kwargs_query : dict
            The query dictionary built from keyword arguments.

        Raises
        ------
        ValueError
            If conflicting constraints are found.

        """
        if not raw_query or not kwargs_query:
            return

        # Only consider fields we generally allow; skip meta operators like $and
        raw_keys = set(raw_query.keys()) & ALLOWED_QUERY_FIELDS
        kw_keys = set(kwargs_query.keys()) & ALLOWED_QUERY_FIELDS
        dup_keys = raw_keys & kw_keys
        for key in dup_keys:
            rc = self._extract_simple_constraint(raw_query, key)
            kc = self._extract_simple_constraint(kwargs_query, key)
            if rc is None or kc is None:
                # If either side is non-simple, skip conflict detection for this key
                continue

            r_kind, r_val = rc
            k_kind, k_val = kc

            # Normalize to sets when appropriate for simpler checks
            if r_kind == "eq" and k_kind == "eq":
                if r_val != k_val:
                    raise ValueError(
                        f"Conflicting constraints for '{key}': query={r_val!r} vs kwargs={k_val!r}"
                    )
            elif r_kind == "in" and k_kind == "eq":
                if k_val not in r_val:
                    raise ValueError(
                        f"Conflicting constraints for '{key}': query in {r_val!r} vs kwargs={k_val!r}"
                    )
            elif r_kind == "eq" and k_kind == "in":
                if r_val not in k_val:
                    raise ValueError(
                        f"Conflicting constraints for '{key}': query={r_val!r} vs kwargs in {k_val!r}"
                    )
            elif r_kind == "in" and k_kind == "in":
                if len(set(r_val).intersection(k_val)) == 0:
                    raise ValueError(
                        f"Conflicting constraints for '{key}': disjoint sets {r_val!r} and {k_val!r}"
                    )

    def exists(self, query: dict[str, Any]) -> bool:
        """Check if at least one record matches the query.

        This is an alias for :meth:`exist`.

        Parameters
        ----------
        query : dict
            MongoDB query to check for existence.

        Returns
        -------
        bool
            True if a matching record exists, False otherwise.

        """
        return self.exist(query)

    @property
    def collection(self):
        """The underlying collection interface.

        Returns
        -------
        EEGDashAPIClient
            The API client used for database interactions via REST API.

        """
        return self._client


def __getattr__(name: str):
    # Backward-compat: allow ``from eegdash.api import EEGDashDataset`` without
    # importing braindecode unless needed.
    if name == "EEGDashDataset":
        from .dataset.dataset import EEGDashDataset

        return EEGDashDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EEGDash", "EEGDashDataset"]
