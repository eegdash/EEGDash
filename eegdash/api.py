# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash MongoDB database, and includes utilities to load
EEG data from S3 for matched records.
"""

import os
from pathlib import Path
from typing import Any, Mapping

import mne
from mne.utils import _soft_import
from pymongo import InsertOne, UpdateOne

from .bids_eeg_metadata import (
    build_query_from_kwargs,
    load_eeg_attrs_from_bids_file,
)
from .const import (
    ALLOWED_QUERY_FIELDS,
)
from .const import config as data_config
from .dataset.bids_dataset import EEGBIDSDataset
from .logging import logger
from .mongodb import MongoConnectionManager
from .utils import _init_mongo_client


class EEGDash:
    """High-level interface to the EEGDash metadata database.

    Provides methods to query, insert, and update metadata records stored in the
    EEGDash MongoDB database (public or private). Also includes utilities to load
    EEG data from S3 for matched records.

    For working with collections of
    recordings as PyTorch datasets, prefer :class:`EEGDashDataset`.
    """

    def __init__(self, *, is_public: bool = True, is_staging: bool = False) -> None:
        """Create a new EEGDash client.

        Parameters
        ----------
        is_public : bool, default True
            Connect to the public MongoDB database. If ``False``, connect to a
            private database instance using the ``DB_CONNECTION_STRING`` environment
            variable (or value from a ``.env`` file).
        is_staging : bool, default False
            If ``True``, use the staging database (``eegdashstaging``); otherwise
            use the production database (``eegdash``).

        Examples
        --------
        >>> eegdash = EEGDash()

        """
        self.config = data_config
        self.is_public = is_public
        self.is_staging = is_staging

        if self.is_public:
            DB_CONNECTION_STRING = mne.utils.get_config("EEGDASH_DB_URI")
            if not DB_CONNECTION_STRING:
                try:
                    _init_mongo_client()
                    DB_CONNECTION_STRING = mne.utils.get_config("EEGDASH_DB_URI")
                except Exception:
                    DB_CONNECTION_STRING = None
        else:
            dotenv = _soft_import("dotenv", "eegdash[full] is necessary.")
            dotenv.load_dotenv()
            DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

        # Use singleton to get MongoDB client, database, and collection
        if not DB_CONNECTION_STRING:
            raise RuntimeError(
                "No MongoDB connection string configured. Set MNE config 'EEGDASH_DB_URI' "
                "or environment variable 'DB_CONNECTION_STRING'."
            )
        self.__client, self.__db, self.__collection = MongoConnectionManager.get_client(
            DB_CONNECTION_STRING, is_staging
        )

    def find(
        self, query: dict[str, Any] = None, /, **kwargs
    ) -> list[Mapping[str, Any]]:
        """Find records in the MongoDB collection.

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
            to ``$in`` queries).

        Returns
        -------
        list of dict
            DB records that match the query.

        """
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

        results = self.__collection.find(final_query)

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

        doc = self.__collection.find_one(query, projection={"_id": 1})
        return doc is not None

    def _validate_input(self, record: dict[str, Any]) -> dict[str, Any]:
        """Validate the input record against the expected schema.

        Parameters
        ----------
        record : dict
            A dictionary representing the EEG data record to be validated.

        Returns
        -------
        dict
            The record itself on success.

        Raises
        ------
        ValueError
            If the record is missing required keys or has values of the wrong type.

        """
        input_types = {
            "data_name": str,
            "dataset": str,
            "bidspath": str,
            "subject": str,
            "task": str,
            "session": str,
            "run": str,
            "sampling_frequency": float,
            "modality": str,
            "nchans": int,
            "ntimes": int,
            "channel_types": list,
            "channel_names": list,
        }
        if "data_name" not in record:
            raise ValueError("Missing key: data_name")
        # check if args are in the keys and has correct type
        for key, value in record.items():
            if key not in input_types:
                raise ValueError(f"Invalid input: {key}")
            if not isinstance(value, input_types[key]):
                raise ValueError(f"Invalid input: {key}")

        return record

    def _build_query_from_kwargs(self, **kwargs) -> dict[str, Any]:
        """Build a validated MongoDB query from keyword arguments.

        This delegates to the module-level builder used across the package.

        Parameters
        ----------
        **kwargs
            Keyword arguments to convert into a MongoDB query.

        Returns
        -------
        dict
            A MongoDB query dictionary.

        """
        return build_query_from_kwargs(**kwargs)

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

    def add_bids_dataset(
        self, dataset: str, data_dir: str, overwrite: bool = True
    ) -> None:
        """Scan a local BIDS dataset and upsert records into MongoDB.

        Parameters
        ----------
        dataset : str
            Dataset identifier (e.g., ``"ds002718"``).
        data_dir : str
            Path to the local BIDS dataset directory.
        overwrite : bool, default True
            If ``True``, update existing records when encountered; otherwise,
            skip records that already exist.

        Raises
        ------
        ValueError
            If called on a public client ``(is_public=True)``.

        """
        if self.is_public:
            raise ValueError("This operation is not allowed for public users")

        if not overwrite and self.exist({"dataset": dataset}):
            logger.info("Dataset %s already exists in the database", dataset)
            return
        try:
            bids_dataset = EEGBIDSDataset(
                data_dir=data_dir,
                dataset=dataset,
            )
        except Exception as e:
            logger.error("Error creating bids dataset %s: %s", dataset, str(e))
            raise e
        requests = []
        for bids_file in bids_dataset.get_files():
            try:
                data_id = f"{dataset}_{Path(bids_file).name}"

                if self.exist({"data_name": data_id}):
                    if overwrite:
                        eeg_attrs = load_eeg_attrs_from_bids_file(
                            bids_dataset, bids_file
                        )
                        requests.append(self._update_request(eeg_attrs))
                else:
                    eeg_attrs = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                    requests.append(self._add_request(eeg_attrs))
            except Exception as e:
                logger.error("Error adding record %s", bids_file)
                logger.error(str(e))

        logger.info("Number of requests: %s", len(requests))

        if requests:
            result = self.__collection.bulk_write(requests, ordered=False)
            logger.info("Inserted: %s ", result.inserted_count)
            logger.info("Modified: %s ", result.modified_count)
            logger.info("Deleted: %s", result.deleted_count)
            logger.info("Upserted: %s", result.upserted_count)
            logger.info("Errors: %s ", result.bulk_api_result.get("writeErrors", []))

    def _add_request(self, record: dict) -> InsertOne:
        """Create a MongoDB insertion request for a record.

        Parameters
        ----------
        record : dict
            The record to insert.

        Returns
        -------
        InsertOne
            A PyMongo ``InsertOne`` object.

        """
        return InsertOne(record)

    def add(self, record: dict) -> None:
        """Add a single record to the MongoDB collection.

        Parameters
        ----------
        record : dict
            The record to add.

        """
        try:
            self.__collection.insert_one(record)
        except ValueError as e:
            logger.error("Validation error for record: %s ", record["data_name"])
            logger.error(e)
        except Exception as exc:
            logger.error(
                "Error adding record: %s ", record.get("data_name", "<unknown>")
            )
            logger.debug("Add operation failed", exc_info=exc)

    def _update_request(self, record: dict) -> UpdateOne:
        """Create a MongoDB update request for a record.

        Parameters
        ----------
        record : dict
            The record to update.

        Returns
        -------
        UpdateOne
            A PyMongo ``UpdateOne`` object.

        """
        return UpdateOne({"data_name": record["data_name"]}, {"$set": record})

    def update(self, record: dict) -> None:
        """Update a single record in the MongoDB collection.

        Parameters
        ----------
        record : dict
            Record content to set at the matching ``data_name``.

        """
        try:
            self.__collection.update_one(
                {"data_name": record["data_name"]}, {"$set": record}
            )
        except Exception as exc:  # log and continue
            logger.error(
                "Error updating record: %s", record.get("data_name", "<unknown>")
            )
            logger.debug("Update operation failed", exc_info=exc)

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

    def remove_field(self, record: dict, field: str) -> None:
        """Remove a field from a specific record in the MongoDB collection.

        Parameters
        ----------
        record : dict
            Record-identifying object with a ``data_name`` key.
        field : str
            The name of the field to remove.

        """
        self.__collection.update_one(
            {"data_name": record["data_name"]}, {"$unset": {field: 1}}
        )

    def remove_field_from_db(self, field: str) -> None:
        """Remove a field from all records in the database.

        .. warning::
            This is a destructive operation and cannot be undone.

        Parameters
        ----------
        field : str
            The name of the field to remove from all documents.

        """
        self.__collection.update_many({}, {"$unset": {field: 1}})

    @property
    def collection(self):
        """The underlying PyMongo ``Collection`` object.

        Returns
        -------
        pymongo.collection.Collection
            The collection object used for database interactions.

        """
        return self.__collection

    def close(self) -> None:
        """Close the MongoDB connection.

        .. deprecated:: 0.1
            Connections are now managed globally by :class:`MongoConnectionManager`.
            This method is a no-op and will be removed in a future version.
            Use :meth:`EEGDash.close_all_connections` to close all clients.
        """
        # Individual instances no longer close the shared client
        pass

    @classmethod
    def close_all_connections(cls) -> None:
        """Close all MongoDB client connections managed by the singleton manager."""
        MongoConnectionManager.close_all()

    def __del__(self) -> None:
        """Destructor; no explicit action needed due to global connection manager."""
        # No longer needed since we're using singleton pattern
        pass


__all__ = ["EEGDash"]
