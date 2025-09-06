import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import mne
import numpy as np
import platformdirs
import xarray as xr
from dotenv import load_dotenv
from joblib import Parallel, delayed
from mne.utils import warn
from mne_bids import get_bids_path_from_fname, read_raw_bids
from pymongo import InsertOne, UpdateOne
from s3fs import S3FileSystem

from braindecode.datasets import BaseConcatDataset

from .bids_eeg_metadata import build_query_from_kwargs, load_eeg_attrs_from_bids_file
from .const import (
    ALLOWED_QUERY_FIELDS,
    RELEASE_TO_OPENNEURO_DATASET_MAP,
)
from .const import config as data_config
from .data_utils import EEGBIDSDataset, EEGDashBaseDataset
from .mongodb import MongoConnectionManager

logger = logging.getLogger("eegdash")


class EEGDash:
    """A high-level interface to the EEGDash database.

    This class is primarily used to interact with the metadata records stored in the
    EEGDash database (or a private instance of it), allowing users to find, add, and
    update EEG data records.

    While this class provides basic support for loading EEG data, please see
    the EEGDashDataset class for a more complete way to retrieve and work with full
    datasets.

    """

    def __init__(self, *, is_public: bool = True, is_staging: bool = False) -> None:
        """Create new instance of the EEGDash Database client.

        Parameters
        ----------
        is_public: bool
            Whether to connect to the public MongoDB database; if False, connect to a
            private database instance as per the DB_CONNECTION_STRING env variable
            (or .env file entry).
        is_staging: bool
            If True, use staging MongoDB database ("eegdashstaging"); otherwise use the
            production database ("eegdash").

        Example
        -------
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

        self.filesystem = S3FileSystem(
            anon=True, client_kwargs={"region_name": "us-east-2"}
        )

    def find(
        self, query: dict[str, Any] = None, /, **kwargs
    ) -> list[Mapping[str, Any]]:
        """Find records in the MongoDB collection.

        This method supports four usage patterns:
        1. With a pre-built MongoDB query dictionary (positional argument):
           >>> eegdash.find({"dataset": "ds002718", "subject": {"$in": ["012", "013"]}})
        2. With user-friendly keyword arguments for simple and multi-value queries:
           >>> eegdash.find(dataset="ds002718", subject="012")
           >>> eegdash.find(dataset="ds002718", subject=["012", "013"])
        3. With an explicit empty query to return all documents:
           >>> eegdash.find({})  # fetches all records (use with care)
        4. By combining a raw query with kwargs (merged via logical AND):
           >>> eegdash.find({"dataset": "ds002718"}, subject=["012", "013"])  # yields {"$and":[{"dataset":"ds002718"}, {"subject":{"$in":["012","013"]}}]}

        Parameters
        ----------
        query: dict, optional
            A complete MongoDB query dictionary. This is a positional-only argument.
        **kwargs:
            Keyword arguments representing field-value pairs for the query.
            Values can be single items (str, int) or lists of items for multi-search.

        Returns
        -------
        list:
            A list of DB records (string-keyed dictionaries) that match the query.

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

    def load_eeg_data_from_s3(self, s3path: str) -> xr.DataArray:
        """Load EEG data from an S3 URI and return it as an xarray DataArray.

        This method preserves the original filename, downloads necessary sidecar
        files when applicable (e.g., .fdt for EEGLAB, .vmrk/.eeg for BrainVision),
        and uses MNE's direct readers rather than ``read_raw_bids``.

        Parameters
        ----------
        s3path : str
            An S3 URI (should start with "s3://").

        Returns
        -------
        xr.DataArray
            A DataArray containing the EEG data, with dimensions "channel" and "time".

        """
        from urllib.parse import urlsplit

        # choose a temp dir so sidecars can be colocated
        with tempfile.TemporaryDirectory() as tmpdir:
            # Derive local filenames from the S3 key to keep base name consistent
            s3_key = urlsplit(s3path).path  # e.g., "/dsXXXX/sub-.../..._eeg.set"
            basename = Path(s3_key).name
            ext = Path(basename).suffix.lower()
            local_main = Path(tmpdir) / basename

            # Download main file
            with (
                self.filesystem.open(s3path, mode="rb") as fsrc,
                open(local_main, "wb") as fdst,
            ):
                fdst.write(fsrc.read())

            # Determine and fetch any required sidecars
            sidecars: list[str] = []
            if ext == ".set":  # EEGLAB
                sidecars = [".fdt"]
            elif ext == ".vhdr":  # BrainVision
                sidecars = [".vmrk", ".eeg", ".dat", ".raw"]

            for sc_ext in sidecars:
                sc_key = s3_key[: -len(ext)] + sc_ext
                sc_uri = f"s3://{urlsplit(s3path).netloc}{sc_key}"
                try:
                    # If sidecar exists, download next to the main file
                    info = self.filesystem.info(sc_uri)
                    if info:
                        sc_local = Path(tmpdir) / Path(sc_key).name
                        with (
                            self.filesystem.open(sc_uri, mode="rb") as fsrc,
                            open(sc_local, "wb") as fdst,
                        ):
                            fdst.write(fsrc.read())
                except Exception:
                    # Sidecar not present; skip silently
                    pass

            # Read using appropriate MNE reader
            if ext == ".set":
                raw = mne.io.read_raw_eeglab(
                    str(local_main), preload=True, verbose=False
                )
            elif ext in {".edf", ".bdf"}:
                raw = mne.io.read_raw_edf(str(local_main), preload=True, verbose=False)
            elif ext == ".vhdr":
                raw = mne.io.read_raw_brainvision(
                    str(local_main), preload=True, verbose=False
                )
            else:
                raise ValueError(f"Unsupported EEG file extension in S3 path: {ext}")

            data = raw.get_data()
            fs = raw.info["sfreq"]
            max_time = data.shape[1] / fs
            time_steps = np.linspace(0, max_time, data.shape[1]).squeeze()
            channel_names = raw.ch_names

            return xr.DataArray(
                data=data,
                dims=["channel", "time"],
                coords={"time": time_steps, "channel": channel_names},
            )

    def load_eeg_data_from_bids_file(self, bids_file: str) -> xr.DataArray:
        """Load EEG data from a local file and return it as a xarray DataArray.

        Parameters
        ----------
        bids_file : str
            Path to the BIDS-compliant file on the local filesystem.

        Notes
        -----
        Currently, only non-epoched .set files are supported.

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
        """Traverse the BIDS dataset at data_dir and add its records to the MongoDB database,
        under the given dataset name.

        Parameters
        ----------
        dataset : str)
            The name of the dataset to be added (e.g., "ds002718").
        data_dir : str
            The path to the BIDS dataset directory.
        overwrite : bool
            Whether to overwrite/update existing records in the database.

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
        """Retrieve a list of EEG data arrays that match the given query. See also
        the `find()` method for details on the query format.

        Parameters
        ----------
        query : dict
            A dictionary that specifies the query to be executed; this is a reference
            document that is used to match records in the MongoDB collection.

        Returns
        -------
            A list of xarray DataArray objects containing the EEG data for each matching record.

        Notes
        -----
        Retrieval is done in parallel, and the downloaded data are not cached locally.

        """
        sessions = self.find(query)
        results = []
        if sessions:
            logger.info("Found %s records", len(sessions))
            results = Parallel(
                n_jobs=-1 if len(sessions) > 1 else 1, prefer="threads", verbose=1
            )(
                delayed(self.load_eeg_data_from_s3)(self._get_s3path(session))
                for session in sessions
            )
        return results

    def _get_s3path(self, record: Mapping[str, Any] | str) -> str:
        """Internal helper to build S3 URI from a record or relative path."""
        if isinstance(record, str):
            rel = record
        else:
            rel = record.get("bidspath")
            if not rel:
                raise ValueError("Record missing 'bidspath' for S3 path resolution")
        return f"s3://openneuro.org/{rel}"

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
        """Update a single record in the MongoDB collection."""
        try:
            self.__collection.update_one(
                {"data_name": record["data_name"]}, {"$set": record}
            )
        except:  # silent failure
            logger.error("Error updating record: %s", record["data_name"])

    def exists(self, query: dict[str, Any]) -> bool:
        """Alias for exist(), provided for API clarity."""
        return self.exist(query)

    def remove_field(self, record, field):
        """Remove a specific field from a record in the MongoDB collection."""
        self.__collection.update_one(
            {"data_name": record["data_name"]}, {"$unset": {field: 1}}
        )

    def remove_field_from_db(self, field):
        """Removed all occurrences of a specific field from all records in the MongoDB
        collection. WARNING: this operation is destructive and should be used with caution.
        """
        self.__collection.update_many({}, {"$unset": {field: 1}})

    @property
    def collection(self):
        """Return the MongoDB collection object."""
        return self.__collection

    def close(self):
        """Close the MongoDB client connection.

        Note: Since MongoDB clients are now managed by a singleton,
        this method no longer closes connections. Use close_all_connections()
        class method to close all connections if needed.
        """
        # Individual instances no longer close the shared client
        pass

    @classmethod
    def close_all_connections(cls):
        """Close all MongoDB client connections managed by the singleton."""
        MongoConnectionManager.close_all()

    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        # No longer needed since we're using singleton pattern
        pass


class EEGDashDataset(BaseConcatDataset):
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
        """Create a new EEGDashDataset from a given query or local BIDS dataset directory
        and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
        instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.


        Querying Examples:
        ------------------
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
        **kwargs : dict
            Keyword arguments for filtering (e.g., `subject="X"`, `task=["T1", "T2"]`) and/or
            arguments to be passed to the EEGDashBaseDataset constructor (e.g., `subject=...`).
        query : dict | None
            Additional filtering options as a raw MongoDB query dictionary. If provided, it will be merged with keyword arguments.
        cache_dir : str
            Optional. A directory where the dataset will be cached locally. If not specified, a default cache directory will be used.
        description_fields : list[str]
            A list of fields to be extracted from the dataset records
            and included in the returned data description(s). Examples are typical
            subject metadata fields such as "subject", "session", "run", "task", etc.;
            see also data_config.description_fields for the default set of fields.
        s3_bucket : str | None
            An optional S3 bucket URI (e.g., "s3://mybucket") to use instead of the
            default OpenNeuro bucket for loading data files
        records : list[dict] | None
            Optional list of pre-fetched metadata records. If provided, the dataset is
            constructed directly from these records without querying MongoDB.
        download : bool (default: True)
            If False, EEGDash will assume that the data has already been downloaded and will not attempt to query MongoDB nor S3 and will parse the local files.
        n_jobs : int
            The number of jobs to run in parallel (default is -1, meaning using all processors).
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashBaseDataset
            constructor.

        """
        self.cache_dir = Path(cache_dir or platformdirs.user_cache_dir("EEGDash"))
        if not self.cache_dir.exists():
            warn(f"Cache directory does not exist, creating it: {self.cache_dir}")
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.s3_bucket = s3_bucket

        # Separate query kwargs from other kwargs passed to the BaseDataset constructor
        self.query = query or {}
        self.query.update(
            {k: v for k, v in kwargs.items() if k in ALLOWED_QUERY_FIELDS}
        )
        base_dataset_kwargs = {k: v for k, v in kwargs.items() if k not in self.query}
        if "dataset" not in self.query:
            raise ValueError("You must provide a 'dataset' argument")

        self.data_dir = self.cache_dir / self.query["dataset"]
        if self.query["dataset"] in RELEASE_TO_OPENNEURO_DATASET_MAP.values():
            warn(
                "If you are not participating in the competition, you can ignore this warning!"
                "\n\n"
                "EEG 2025 Competition Data Notice:\n"
                "---------------------------------\n"
                " You are loading the dataset that is used in the EEG 2025 Competition:\n"
                "IMPORTANT: The data accessed via `EEGDashDataset` is NOT identical to what you get from `EEGChallengeDataset` object directly.\n"
                "and it is not what you will use for the competition. Downsampling and filtering were applied to the data"
                "to allow more people to participate.\n"
                "\n"
                "If you are participating in the competition, always use `EEGChallengeDataset` to ensure consistency with the challenge data.\n"
                "\n",
                UserWarning,
                module="eegdash",
            )
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
            if self.data_dir.exists():
                # This path loads from a local directory and is not affected by DB query logic
                datasets = self.load_bids_dataset(
                    dataset=self.query["dataset"],
                    data_dir=self.data_dir,
                    description_fields=description_fields,
                    s3_bucket=s3_bucket,
                    n_jobs=n_jobs,
                    **base_dataset_kwargs,
                )
            else:
                raise ValueError(
                    f"Offline mode is enabled, but local data_dir {self.data_dir} does not exist."
                )
        elif self.query:
            # This is the DB query path that we are improving
            datasets = self._find_datasets(
                query=build_query_from_kwargs(**self.query),
                description_fields=description_fields,
                base_dataset_kwargs=base_dataset_kwargs,
            )
            # We only need filesystem if we need to access S3
            self.filesystem = S3FileSystem(
                anon=True, client_kwargs={"region_name": "us-east-2"}
            )
        else:
            raise ValueError(
                "You must provide either 'records', a 'data_dir', or a query/keyword arguments for filtering."
            )

        super().__init__(datasets)

    def _find_key_in_nested_dict(self, data: Any, target_key: str) -> Any:
        """Helper to recursively search for a key in a nested dictionary structure; returns
        the value associated with the first occurrence of the key, or None if not found.
        """
        if isinstance(data, dict):
            if target_key in data:
                return data[target_key]
            for value in data.values():
                result = self._find_key_in_nested_dict(value, target_key)
                if result is not None:
                    return result
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
        eegdash_instance = EEGDash()
        self.records = eegdash_instance.find(query)

        for record in self.records:
            description = {}
            for field in description_fields:
                value = self._find_key_in_nested_dict(record, field)
                if value is not None:
                    description[field] = value
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

    def load_bids_dataset(
        self,
        dataset: str,
        data_dir: str | Path,
        description_fields: list[str],
        s3_bucket: str | None = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        """Helper method to load a single local BIDS dataset and return it as a list of
        EEGDashBaseDatasets (one for each recording in the dataset).

        Parameters
        ----------
        dataset : str
            A name for the dataset to be loaded (e.g., "ds002718").
        data_dir : str
            The path to the local BIDS dataset directory.
        description_fields : list[str]
            A list of fields to be extracted from the dataset records
            and included in the returned dataset description(s).
        s3_bucket : str | None
            The S3 bucket to upload the dataset files to (if any).
        n_jobs : int
            The number of jobs to run in parallel (default is -1, meaning using all processors).

        """
        logger.info(f"Loading local BIDS dataset {dataset} from {data_dir}")
        bids_dataset = EEGBIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
        )
        datasets = Parallel(n_jobs=n_jobs, prefer="threads", verbose=1)(
            delayed(self._get_base_dataset_from_bids_file)(
                bids_dataset=bids_dataset,
                bids_file=bids_file,
                s3_bucket=s3_bucket,
                description_fields=description_fields,
                **kwargs,
            )
            for bids_file in bids_dataset.get_files()
        )
        return datasets

    def _get_base_dataset_from_bids_file(
        self,
        bids_dataset: "EEGBIDSDataset",
        bids_file: str,
        s3_bucket: str | None,
        description_fields: list[str],
        **kwargs,
    ) -> "EEGDashBaseDataset":
        """Instantiate a single EEGDashBaseDataset given a local BIDS file (metadata only)."""
        record = load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
        description = {}
        for field in description_fields:
            value = self._find_key_in_nested_dict(record, field)
            if value is not None:
                description[field] = value
        return EEGDashBaseDataset(
            record,
            self.cache_dir,
            s3_bucket,
            description=description,
            **kwargs,
        )
