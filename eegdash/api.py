import os
from pathlib import Path
from typing import Any, Mapping

import mne
import numpy as np
import xarray as xr
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from dotenv import load_dotenv
from joblib import Parallel, delayed
from mne_bids import find_matching_paths, get_bids_path_from_fname, read_raw_bids
from pymongo import InsertOne, UpdateOne
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from braindecode.datasets import BaseConcatDataset

from . import downloader
from .bids_eeg_metadata import (
    build_query_from_kwargs,
    load_eeg_attrs_from_bids_file,
    merge_participants_fields,
    normalize_key,
)
from .const import (
    ALLOWED_QUERY_FIELDS,
    RELEASE_TO_OPENNEURO_DATASET_MAP,
)
from .const import config as data_config
from .data_utils import (
    EEGBIDSDataset,
    EEGDashBaseDataset,
)
from .logging import logger
from .mongodb import MongoConnectionManager
from .paths import get_default_cache_dir


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
        else:
            load_dotenv()
            DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

        # Use singleton to get MongoDB client, database, and collection
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
        """Internal method to validate the input record against the expected schema.

        Parameters
        ----------
        record: dict
            A dictionary representing the EEG data record to be validated.

        Returns
        -------
        dict:
            Returns the record itself on success, or raises a ValueError if the record is invalid.

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
        """Internal helper to build a validated MongoDB query from keyword args.

        This delegates to the module-level builder used across the package and
        is exposed here for testing and convenience.
        """
        return build_query_from_kwargs(**kwargs)

    # --- Query merging and conflict detection helpers ---
    def _extract_simple_constraint(self, query: dict[str, Any], key: str):
        """Extract a simple constraint for a given key from a query dict.

        Supports only top-level equality (key: value) and $in (key: {"$in": [...]})
        constraints. Returns a tuple (kind, value) where kind is "eq" or "in". If the
        key is not present or uses other operators, returns None.
        """
        if not isinstance(query, dict) or key not in query:
            return None
        val = query[key]
        if isinstance(val, dict):
            if "$in" in val and isinstance(val["$in"], (list, tuple)):
                return ("in", list(val["$in"]))
            return None  # unsupported operator shape for conflict checking
        else:
            return ("eq", val)

    def _raise_if_conflicting_constraints(
        self, raw_query: dict[str, Any], kwargs_query: dict[str, Any]
    ) -> None:
        """Raise ValueError if both query sources define incompatible constraints.

        We conservatively check only top-level fields with simple equality or $in
        constraints. If a field appears in both queries and constraints are mutually
        exclusive, raise an explicit error to avoid silent empty result sets.
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

    def load_eeg_data_from_bids_file(self, bids_file: str) -> xr.DataArray:
        """Load EEG data from a local BIDS-formatted file.

        Parameters
        ----------
        bids_file : str
            Path to a BIDS-compliant EEG file (e.g., ``*_eeg.edf``, ``*_eeg.bdf``,
            ``*_eeg.vhdr``, ``*_eeg.set``).

        Returns
        -------
        xr.DataArray
            EEG data with dimensions ``("channel", "time")``.

        """
        bids_path = get_bids_path_from_fname(bids_file, verbose=False)
        raw_object = read_raw_bids(bids_path=bids_path, verbose=False)
        eeg_data = raw_object.get_data()

        fs = raw_object.info["sfreq"]
        max_time = eeg_data.shape[1] / fs
        time_steps = np.linspace(0, max_time, eeg_data.shape[1]).squeeze()  # in seconds

        channel_names = raw_object.ch_names

        eeg_xarray = xr.DataArray(
            data=eeg_data,
            dims=["channel", "time"],
            coords={"time": time_steps, "channel": channel_names},
        )
        return eeg_xarray

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

    def get(self, query: dict[str, Any]) -> list[xr.DataArray]:
        """Download and return EEG data arrays for records matching a query.

        Parameters
        ----------
        query : dict
            MongoDB query used to select records.

        Returns
        -------
        list of xr.DataArray
            EEG data for each matching record, with dimensions ``("channel", "time")``.

        Notes
        -----
        Retrieval runs in parallel. Downloaded files are read and discarded
        (no on-disk caching here).

        """
        sessions = self.find(query)
        results = []
        if sessions:
            logger.info("Found %s records", len(sessions))
            results = Parallel(
                n_jobs=-1 if len(sessions) > 1 else 1, prefer="threads", verbose=1
            )(
                delayed(downloader.load_eeg_from_s3)(
                    downloader.get_s3path("s3://openneuro.org", session["bidspath"])
                )
                for session in sessions
            )
        return results

    def _add_request(self, record: dict):
        """Internal helper method to create a MongoDB insertion request for a record."""
        return InsertOne(record)

    def add(self, record: dict):
        """Add a single record to the MongoDB collection."""
        try:
            self.__collection.insert_one(record)
        except ValueError as e:
            logger.error("Validation error for record: %s ", record["data_name"])
            logger.error(e)
        except:
            logger.error("Error adding record: %s ", record["data_name"])

    def _update_request(self, record: dict):
        """Internal helper method to create a MongoDB update request for a record."""
        return UpdateOne({"data_name": record["data_name"]}, {"$set": record})

    def update(self, record: dict):
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
        except:  # silent failure
            logger.error("Error updating record: %s", record["data_name"])

    def exists(self, query: dict[str, Any]) -> bool:
        """Alias for :meth:`exist` provided for API clarity."""
        return self.exist(query)

    def remove_field(self, record, field):
        """Remove a specific field from a record in the MongoDB collection.

        Parameters
        ----------
        record : dict
            Record identifying object with ``data_name``.
        field : str
            Field name to remove.

        """
        self.__collection.update_one(
            {"data_name": record["data_name"]}, {"$unset": {field: 1}}
        )

    def remove_field_from_db(self, field):
        """Remove a field from all records (destructive).

        Parameters
        ----------
        field : str
            Field name to remove from every document.

        """
        self.__collection.update_many({}, {"$unset": {field: 1}})

    @property
    def collection(self):
        """Return the MongoDB collection object."""
        return self.__collection

    def close(self):
        """Backward-compatibility no-op; connections are managed globally.

        Notes
        -----
        Connections are managed by :class:`MongoConnectionManager`. Use
        :meth:`close_all_connections` to explicitly close all clients.

        """
        # Individual instances no longer close the shared client
        pass

    @classmethod
    def close_all_connections(cls):
        """Close all MongoDB client connections managed by the singleton."""
        MongoConnectionManager.close_all()

    def __del__(self):
        """Destructor; no explicit action needed due to global connection manager."""
        # No longer needed since we're using singleton pattern
        pass


class EEGDashDataset(BaseConcatDataset, metaclass=NumpyDocstringInheritanceInitMeta):
    """Create a new EEGDashDataset from a given query or local BIDS dataset directory
    and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
    instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

    Examples
    --------
    # Find by single subject
    >>> ds = EEGDashDataset(dataset="ds005505", subject="NDARCA153NKE")

    # Find by a list of subjects and a specific task
    >>> subjects = ["NDARCA153NKE", "NDARXT792GY8"]
    >>> ds = EEGDashDataset(dataset="ds005505", subject=subjects, task="RestingState")

    # Use a raw MongoDB query for advanced filtering
    >>> raw_query = {"dataset": "ds005505", "subject": {"$in": subjects}}
    >>> ds = EEGDashDataset(query=raw_query)

    Parameters
    ----------
    cache_dir : str | Path
        Directory where data are cached locally. If not specified, a default
        cache directory under the user cache is used.
    query : dict | None
        Raw MongoDB query to filter records. If provided, it is merged with
        keyword filtering arguments (see ``**kwargs``) using logical AND.
        You must provide at least a ``dataset`` (either in ``query`` or
        as a keyword argument). Only fields in ``ALLOWED_QUERY_FIELDS`` are
        considered for filtering.
    dataset : str
        Dataset identifier (e.g., ``"ds002718"``). Required if ``query`` does
        not already specify a dataset.
    task : str | list[str]
        Task name(s) to filter by (e.g., ``"RestingState"``).
    subject : str | list[str]
        Subject identifier(s) to filter by (e.g., ``"NDARCA153NKE"``).
    session : str | list[str]
        Session identifier(s) to filter by (e.g., ``"1"``).
    run : str | list[str]
        Run identifier(s) to filter by (e.g., ``"1"``).
    description_fields : list[str]
        Fields to extract from each record and include in dataset descriptions
        (e.g., "subject", "session", "run", "task").
    s3_bucket : str | None
        Optional S3 bucket URI (e.g., "s3://mybucket") to use instead of the
        default OpenNeuro bucket when downloading data files.
    records : list[dict] | None
        Pre-fetched metadata records. If provided, the dataset is constructed
        directly from these records and no MongoDB query is performed.
    download : bool, default True
        If False, load from local BIDS files only. Local data are expected
        under ``cache_dir / dataset``; no DB or S3 access is attempted.
    n_jobs : int
        Number of parallel jobs to use where applicable (-1 uses all cores).
    eeg_dash_instance : EEGDash | None
        Optional existing EEGDash client to reuse for DB queries. If None,
        a new client is created on demand, not used in the case of no download.
    **kwargs : dict
        Additional keyword arguments serving two purposes:

        - Filtering: any keys present in ``ALLOWED_QUERY_FIELDS`` are treated as
          query filters (e.g., ``dataset``, ``subject``, ``task``, ...).
        - Dataset options: remaining keys are forwarded to
          ``EEGDashBaseDataset``.

    """

    def __init__(
        self,
        cache_dir: str | Path,
        query: dict[str, Any] = None,
        description_fields: list[str] = [
            "subject",
            "session",
            "run",
            "task",
            "age",
            "gender",
            "sex",
        ],
        s3_bucket: str | None = None,
        records: list[dict] | None = None,
        download: bool = True,
        n_jobs: int = -1,
        eeg_dash_instance: EEGDash | None = None,
        **kwargs,
    ):
        # Parameters that don't need validation
        _suppress_comp_warning: bool = kwargs.pop("_suppress_comp_warning", False)
        self.s3_bucket = s3_bucket
        self.records = records
        self.download = download
        self.n_jobs = n_jobs
        self.eeg_dash_instance = eeg_dash_instance or EEGDash()

        # Resolve a unified cache directory across code/tests/CI
        self.cache_dir = Path(cache_dir or get_default_cache_dir())

        if not self.cache_dir.exists():
            logger.warning(
                f"Cache directory does not exist, creating it: {self.cache_dir}"
            )
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Separate query kwargs from other kwargs passed to the BaseDataset constructor
        self.query = query or {}
        self.query.update(
            {k: v for k, v in kwargs.items() if k in ALLOWED_QUERY_FIELDS}
        )
        base_dataset_kwargs = {k: v for k, v in kwargs.items() if k not in self.query}
        if "dataset" not in self.query:
            # If explicit records are provided, infer dataset from records
            if isinstance(records, list) and records and isinstance(records[0], dict):
                inferred = records[0].get("dataset")
                if inferred:
                    self.query["dataset"] = inferred
                else:
                    raise ValueError("You must provide a 'dataset' argument")
            else:
                raise ValueError("You must provide a 'dataset' argument")

        # Decide on a dataset subfolder name for cache isolation. If using
        # challenge/preprocessed buckets (e.g., BDF, mini subsets), append
        # informative suffixes to avoid overlapping with the original dataset.
        dataset_folder = self.query["dataset"]
        if self.s3_bucket:
            suffixes: list[str] = []
            bucket_lower = str(self.s3_bucket).lower()
            if "bdf" in bucket_lower:
                suffixes.append("bdf")
            if "mini" in bucket_lower:
                suffixes.append("mini")
            if suffixes:
                dataset_folder = f"{dataset_folder}-{'-'.join(suffixes)}"

        self.data_dir = self.cache_dir / dataset_folder

        if (
            not _suppress_comp_warning
            and self.query["dataset"] in RELEASE_TO_OPENNEURO_DATASET_MAP.values()
        ):
            message_text = Text.from_markup(
                "[italic]This notice is for users who are participating in the [link=https://eeg2025.github.io/]EEG 2025 Competition[/link].[/italic]\n\n"
                "[bold]EEG 2025 Competition Data Notice[/bold]\n"
                "You are loading the raw dataset via `EEGDashDataset`.\n\n"
                "[bold red]IMPORTANT[/bold red]: This data is [u]NOT[/u] identical to the official competition data, which is accessed via `EEGChallengeDataset`. The competition data has been downsampled and filtered.\n\n"
                "[bold]If you are participating in the competition, you must use the `EEGChallengeDataset` object to ensure consistency.[/bold]"
            )
            warning_panel = Panel(
                message_text,
                title="[yellow]EEG 2025 Competition Data Notice[/yellow]",
                subtitle="[cyan]Source: EEGDashDataset[/cyan]",
                border_style="yellow",
            )

            try:
                Console().print(warning_panel)
            except Exception:
                warning_message = (
                    "\n\n"
                    "[EEGChallengeDataset] EEG 2025 Competition Data Notice:\n"
                    "-------------------------------------------------------\n"
                    "This object loads the HBN dataset that has been preprocessed for the EEG Challenge:\n"
                    "  - Downsampled from 500Hz to 100Hz\n"
                    "  - Bandpass filtered (0.5–50 Hz)\n"
                    "\n"
                    "For full preprocessing applied for competition details, see:\n"
                    "  https://github.com/eeg2025/downsample-datasets\n"
                    "\n"
                    "The HBN dataset have some preprocessing applied by the HBN team:\n"
                    "  - Re-reference (Cz Channel)\n"
                    "\n"
                    "IMPORTANT: The data accessed via `EEGChallengeDataset` is NOT identical to what you get from `EEGDashDataset` directly.\n"
                    "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data.\n"
                )

                logger.warning(warning_message)

        if records is not None:
            self.records = records
            datasets = [
                EEGDashBaseDataset(
                    record,
                    self.cache_dir,
                    self.s3_bucket,
                    **base_dataset_kwargs,
                )
                for record in self.records
            ]
        elif not download:  # only assume local data is complete if not downloading
            if not self.data_dir.exists():
                raise ValueError(
                    f"Offline mode is enabled, but local data_dir {self.data_dir} does not exist."
                )
            records = self._find_local_bids_records(self.data_dir, self.query)
            # Try to enrich from local participants.tsv to restore requested fields
            try:
                bids_ds = EEGBIDSDataset(
                    data_dir=str(self.data_dir), dataset=self.query["dataset"]
                )  # type: ignore[index]
            except Exception:
                bids_ds = None

            datasets = []
            for record in records:
                # Start with entity values from filename
                desc: dict[str, Any] = {
                    k: record.get(k)
                    for k in ("subject", "session", "run", "task")
                    if record.get(k) is not None
                }

                if bids_ds is not None:
                    try:
                        rel_from_dataset = Path(record["bidspath"]).relative_to(
                            record["dataset"]
                        )  # type: ignore[index]
                        local_file = (self.data_dir / rel_from_dataset).as_posix()
                        part_row = bids_ds.subject_participant_tsv(local_file)
                        desc = merge_participants_fields(
                            description=desc,
                            participants_row=part_row
                            if isinstance(part_row, dict)
                            else None,
                            description_fields=description_fields,
                        )
                    except Exception:
                        pass

                datasets.append(
                    EEGDashBaseDataset(
                        record=record,
                        cache_dir=self.cache_dir,
                        s3_bucket=self.s3_bucket,
                        description=desc,
                        **base_dataset_kwargs,
                    )
                )
        elif self.query:
            # This is the DB query path that we are improving
            datasets = self._find_datasets(
                query=build_query_from_kwargs(**self.query),
                description_fields=description_fields,
                base_dataset_kwargs=base_dataset_kwargs,
            )
            # We only need filesystem if we need to access S3
            self.filesystem = downloader.get_s3_filesystem()
        else:
            raise ValueError(
                "You must provide either 'records', a 'data_dir', or a query/keyword arguments for filtering."
            )

        super().__init__(datasets)

    def _find_local_bids_records(
        self, dataset_root: Path, filters: dict[str, Any]
    ) -> list[dict]:
        """Discover local BIDS EEG files and build minimal records.

        This helper enumerates EEG recordings under ``dataset_root`` via
        ``mne_bids.find_matching_paths`` and applies entity filters to produce a
        list of records suitable for ``EEGDashBaseDataset``. No network access
        is performed and files are not read.

        Parameters
        ----------
        dataset_root : Path
            Local dataset directory. May be the plain dataset folder (e.g.,
            ``ds005509``) or a suffixed cache variant (e.g.,
            ``ds005509-bdf-mini``).
        filters : dict of {str, Any}
            Query filters. Must include ``'dataset'`` with the dataset id (without
            local suffixes). May include BIDS entities ``'subject'``,
            ``'session'``, ``'task'``, and ``'run'``. Each value can be a scalar
            or a sequence of scalars.

        Returns
        -------
        records : list of dict
            One record per matched EEG file with at least:

            - ``'data_name'``
            - ``'dataset'`` (dataset id, without suffixes)
            - ``'bidspath'`` (normalized to start with the dataset id)
            - ``'subject'``, ``'session'``, ``'task'``, ``'run'`` (may be None)
            - ``'bidsdependencies'`` (empty list)
            - ``'modality'`` (``"eeg"``)
            - ``'sampling_frequency'``, ``'nchans'``, ``'ntimes'`` (minimal
              defaults for offline usage)

        Notes
        -----
        - Matching uses ``datatypes=['eeg']`` and ``suffixes=['eeg']``.
        - ``bidspath`` is constructed as
          ``<dataset_id> / <relative_path_from_dataset_root>`` to ensure the
          first path component is the dataset id (without local cache suffixes).
        - Minimal defaults are set for ``sampling_frequency``, ``nchans``, and
          ``ntimes`` to satisfy dataset length requirements offline.

        """
        dataset_id = filters["dataset"]
        arg_map = {
            "subjects": "subject",
            "sessions": "session",
            "tasks": "task",
            "runs": "run",
        }
        matching_args: dict[str, list[str]] = {}
        for finder_key, entity_key in arg_map.items():
            entity_val = filters.get(entity_key)
            if entity_val is None:
                continue
            if isinstance(entity_val, (list, tuple, set)):
                entity_vals = list(entity_val)
                if not entity_vals:
                    continue
                matching_args[finder_key] = entity_vals
            else:
                matching_args[finder_key] = [entity_val]

        matched_paths = find_matching_paths(
            root=str(dataset_root),
            datatypes=["eeg"],
            suffixes=["eeg"],
            ignore_json=True,
            **matching_args,
        )
        records_out: list[dict] = []

        for bids_path in matched_paths:
            # Build bidspath as dataset_id / relative_path_from_dataset_root (POSIX)
            rel_from_root = (
                Path(bids_path.fpath)
                .resolve()
                .relative_to(Path(bids_path.root).resolve())
            )
            bidspath = f"{dataset_id}/{rel_from_root.as_posix()}"

            rec = {
                "data_name": f"{dataset_id}_{Path(bids_path.fpath).name}",
                "dataset": dataset_id,
                "bidspath": bidspath,
                "subject": (bids_path.subject or None),
                "session": (bids_path.session or None),
                "task": (bids_path.task or None),
                "run": (bids_path.run or None),
                # minimal fields to satisfy BaseDataset from eegdash
                "bidsdependencies": [],  # not needed to just run.
                "modality": "eeg",
                # minimal numeric defaults for offline length calculation
                "sampling_frequency": None,
                "nchans": None,
                "ntimes": None,
            }
            records_out.append(rec)

        return records_out

    def _find_key_in_nested_dict(self, data: Any, target_key: str) -> Any:
        """Recursively search for target_key in nested dicts/lists with normalized matching.

        This makes lookups tolerant to naming differences like "p-factor" vs "p_factor".
        Returns the first match or None.
        """
        norm_target = normalize_key(target_key)
        if isinstance(data, dict):
            for k, v in data.items():
                if normalize_key(k) == norm_target:
                    return v
                res = self._find_key_in_nested_dict(v, target_key)
                if res is not None:
                    return res
        elif isinstance(data, list):
            for item in data:
                res = self._find_key_in_nested_dict(item, target_key)
                if res is not None:
                    return res
        return None

    def _find_datasets(
        self,
        query: dict[str, Any] | None,
        description_fields: list[str],
        base_dataset_kwargs: dict,
    ) -> list[EEGDashBaseDataset]:
        """Helper method to find datasets in the MongoDB collection that satisfy the
        given query and return them as a list of EEGDashBaseDataset objects.

        Parameters
        ----------
        query : dict
            The query object, as in EEGDash.find().
        description_fields : list[str]
            A list of fields to be extracted from the dataset records and included in
            the returned dataset description(s).
        kwargs: additional keyword arguments to be passed to the EEGDashBaseDataset
            constructor.

        Returns
        -------
        list :
            A list of EEGDashBaseDataset objects that match the query.

        """
        datasets: list[EEGDashBaseDataset] = []
        self.records = self.eeg_dash_instance.find(query)

        for record in self.records:
            description: dict[str, Any] = {}
            # Requested fields first (normalized matching)
            for field in description_fields:
                value = self._find_key_in_nested_dict(record, field)
                if value is not None:
                    description[field] = value
            # Merge all participants.tsv columns generically
            part = self._find_key_in_nested_dict(record, "participant_tsv")
            if isinstance(part, dict):
                description = merge_participants_fields(
                    description=description,
                    participants_row=part,
                    description_fields=description_fields,
                )
            datasets.append(
                EEGDashBaseDataset(
                    record,
                    cache_dir=self.cache_dir,
                    s3_bucket=self.s3_bucket,
                    description=description,
                    **base_dataset_kwargs,
                )
            )
        return datasets
