# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash MongoDB database, and includes utilities to load
EEG data from S3 for matched records.
"""

import json
import os
from pathlib import Path
from typing import Any, Mapping

import mne
import numpy as np
import pandas as pd
from mne.utils import _soft_import

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
        self,
        dataset: str,
        data_dir: str,
        overwrite: bool = True,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Collect metadata for a local BIDS dataset as JSON-ready records.

        Instead of inserting records directly into MongoDB, this method scans
        ``data_dir`` and returns a JSON-serializable manifest describing every
        EEG recording that was discovered. The manifest can be written to disk
        or forwarded to the EEGDash ingestion API for persistence.

        Parameters
        ----------
        dataset : str
            Dataset identifier (e.g., ``"ds002718"``).
        data_dir : str
            Path to the local BIDS dataset directory.
        overwrite : bool, default True
            If ``False``, skip records that already exist in the database based
            on ``data_name`` lookups.
        output_path : str | Path | None, optional
            If provided, the manifest is written to the given JSON file.

        Returns
        -------
        dict
            A manifest with keys ``dataset``, ``source``, ``records`` and, when
            applicable, ``skipped`` or ``errors``.

        """
        source_dir = Path(data_dir).expanduser()
        try:
            bids_dataset = EEGBIDSDataset(
                data_dir=str(source_dir),
                dataset=dataset,
            )
        except Exception as exc:
            logger.error("Error creating BIDS dataset %s: %s", dataset, exc)
            raise exc

        records: list[dict[str, Any]] = []
        skipped: list[str] = []
        errors: list[dict[str, str]] = []

        for bids_file in bids_dataset.get_files():
            data_id = f"{dataset}_{Path(bids_file).name}"
            if not overwrite:
                try:
                    if self.exist({"data_name": data_id}):
                        skipped.append(data_id)
                        continue
                except Exception as exc:
                    logger.warning(
                        "Could not verify existing record %s due to: %s",
                        data_id,
                        exc,
                    )

            try:
                eeg_attrs = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
                records.append(eeg_attrs)
            except Exception as exc:  # log and continue collecting
                logger.error("Error extracting metadata for %s", bids_file)
                logger.error(str(exc))
                errors.append({"file": str(bids_file), "error": str(exc)})

        manifest: dict[str, Any] = {
            "dataset": dataset,
            "source": str(source_dir.resolve()),
            "record_count": len(records),
            "records": records,
        }
        if skipped:
            manifest["skipped"] = skipped
        if errors:
            manifest["errors"] = errors

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    manifest,
                    fh,
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                )
            logger.info(
                "Wrote EEGDash ingestion manifest for %s to %s",
                dataset,
                output_path,
            )

        logger.info(
            "Prepared %s records for dataset %s (skipped=%s, errors=%s)",
            len(records),
            dataset,
            len(skipped),
            len(errors),
        )

        return manifest

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
        """The underlying PyMongo ``Collection`` object.

        Returns
        -------
        pymongo.collection.Collection
            The collection object used for database interactions.

        """
        return self.__collection

    @classmethod
    def close_all_connections(cls) -> None:
        """Close all MongoDB client connections managed by the singleton manager."""
        MongoConnectionManager.close_all()


def _json_default(value: Any) -> Any:
    """Fallback serializer for complex objects when exporting ingestion JSON."""
    try:
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        if value is pd.NA:
            return None
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
        if isinstance(value, pd.Series):
            return value.to_dict()
    except Exception:
        pass

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["EEGDash"]
