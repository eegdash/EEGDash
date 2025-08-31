import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import platformdirs

import mne
import numpy as np
import xarray as xr
from dotenv import load_dotenv
from joblib import Parallel, delayed
from mne_bids import get_bids_path_from_fname, read_raw_bids
from pymongo import InsertOne, UpdateOne
from s3fs import S3FileSystem

from braindecode.datasets import BaseConcatDataset

from .data_config import config as data_config
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

    _ALLOWED_QUERY_FIELDS = {
        "data_name",
        "dataset",
        "subject",
        "task",
        "session",
        "run",
        "modality",
        "sampling_frequency",
        "nchans",
        "ntimes",
    }

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
        kwargs_query = self._build_query_from_kwargs(**kwargs) if kwargs else None

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
        """Build and validate a MongoDB query from user-friendly keyword arguments.

        Improvements:
        - Reject None values and empty/whitespace-only strings
        - For list/tuple/set values: strip strings, drop None/empties, deduplicate, and use `$in`
        - Preserve scalars as exact matches
        """
        # 1. Validate that all provided keys are allowed for querying
        unknown_fields = set(kwargs.keys()) - self._ALLOWED_QUERY_FIELDS
        if unknown_fields:
            raise ValueError(
                f"Unsupported query field(s): {', '.join(sorted(unknown_fields))}. "
                f"Allowed fields are: {', '.join(sorted(self._ALLOWED_QUERY_FIELDS))}"
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
        raw_keys = set(raw_query.keys()) & self._ALLOWED_QUERY_FIELDS
        kw_keys = set(kwargs_query.keys()) & self._ALLOWED_QUERY_FIELDS
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
        """Load an EEGLAB .set file from an AWS S3 URI and return it as an xarray DataArray.

        Parameters
        ----------
        s3path : str
            An S3 URI (should start with "s3://") for the file in question.

        Returns
        -------
        xr.DataArray
            A DataArray containing the EEG data, with dimensions "channel" and "time".

        Example
        -------
        >>> eegdash = EEGDash()
        >>> mypath = "s3://openneuro.org/path/to/your/eeg_data.set"
        >>> mydata = eegdash.load_eeg_data_from_s3(mypath)

        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".set") as tmp:
            with self.filesystem.open(s3path) as s3_file:
                tmp.write(s3_file.read())
            tmp_path = tmp.name
            eeg_data = self.load_eeg_data_from_bids_file(tmp_path)
            os.unlink(tmp_path)
            return eeg_data

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

    def get_raw_extensions(
        self, bids_file: str, bids_dataset: EEGBIDSDataset
    ) -> list[str]:
        """Helper to find paths to additional "sidecar" files that may be associated
        with a given main data file in a BIDS dataset; paths are returned as relative to
        the parent dataset path.

        For example, if the input file is a .set file, this will return the relative path
        to a corresponding .fdt file (if any).
        """
        bids_file = Path(bids_file)
        extensions = {
            ".set": [".set", ".fdt"],  # eeglab
            ".edf": [".edf"],  # european
            ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
            ".bdf": [".bdf"],  # biosemi
        }
        return [
            str(bids_dataset.get_relative_bidspath(bids_file.with_suffix(suffix)))
            for suffix in extensions[bids_file.suffix]
            if bids_file.with_suffix(suffix).exists()
        ]

    def load_eeg_attrs_from_bids_file(
        self, bids_dataset: EEGBIDSDataset, bids_file: str
    ) -> dict[str, Any]:
        """Build the metadata record for a given BIDS file (single recording) in a BIDS dataset.

        Attributes are at least the ones defined in data_config attributes (set to None if missing),
        but are typically a superset, and include, among others, the paths to relevant
        meta-data files needed to load and interpret the file in question.

        Parameters
        ----------
        bids_dataset : EEGBIDSDataset
            The BIDS dataset object containing the file.
        bids_file : str
            The path to the BIDS file within the dataset.

        Returns
        -------
        dict:
            A dictionary representing the metadata record for the given file. This is the
            same format as the records stored in the database.

        """
        if bids_file not in bids_dataset.files:
            raise ValueError(f"{bids_file} not in {bids_dataset.dataset}")

        # Initialize attrs with None values for all expected fields
        attrs = {field: None for field in self.config["attributes"].keys()}

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

        bids_dependencies_files = self.config["bids_dependencies_files"]
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

        bidsdependencies.extend(self.get_raw_extensions(bids_file, bids_dataset))

        # Define field extraction functions with error handling
        field_extractors = {
            "data_name": lambda: f"{bids_dataset.dataset}_{file}",
            "dataset": lambda: bids_dataset.dataset,
            "bidspath": lambda: openneuro_path,
            "subject": lambda: bids_dataset.get_bids_file_attribute(
                "subject", bids_file
            ),
            "task": lambda: bids_dataset.get_bids_file_attribute("task", bids_file),
            "session": lambda: bids_dataset.get_bids_file_attribute(
                "session", bids_file
            ),
            "run": lambda: bids_dataset.get_bids_file_attribute("run", bids_file),
            "modality": lambda: bids_dataset.get_bids_file_attribute(
                "modality", bids_file
            ),
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
            logger.error("Error creating bids dataset %s: $s", dataset, str(e))
            raise e
        requests = []
        for bids_file in bids_dataset.get_files():
            try:
                data_id = f"{dataset}_{Path(bids_file).name}"

                if self.exist({"data_name": data_id}):
                    if overwrite:
                        eeg_attrs = self.load_eeg_attrs_from_bids_file(
                            bids_dataset, bids_file
                        )
                        requests.append(self.update_request(eeg_attrs))
                else:
                    eeg_attrs = self.load_eeg_attrs_from_bids_file(
                        bids_dataset, bids_file
                    )
                    requests.append(self.add_request(eeg_attrs))
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
                delayed(self.load_eeg_data_from_s3)(self.get_s3path(session))
                for session in sessions
            )
        return results

    def add_request(self, record: dict):
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

    def update_request(self, record: dict):
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
        query: dict[str, Any] = None,
        cache_dir: str = None,
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
        eeg_dash_instance=None,
        records: list[dict] | None = None,
        offline_mode: bool = False,
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
        query : dict | None
            A raw MongoDB query dictionary. If provided, keyword arguments for filtering are ignored.
        **kwargs : dict
            Keyword arguments for filtering (e.g., `subject="X"`, `task=["T1", "T2"]`) and/or
            arguments to be passed to the EEGDashBaseDataset constructor (e.g., `subject=...`).
        cache_dir : str
            A directory where the dataset will be cached locally.
        data_dir : str | None
            Optionally a string specifying a local BIDS dataset directory from which to load the EEG data files. Exactly one
            of query or data_dir must be provided.
        dataset : str | None
            If data_dir is given, a name for the dataset to be loaded.
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
        offline_mode : bool
            If True, do not attempt to query MongoDB at all. This is useful if you want to
            work with a local cache only, or if you are offline.
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashBaseDataset
            constructor.

        """
        self.cache_dir = Path(cache_dir or platformdirs.user_cache_dir("EEGDash"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.eeg_dash = eeg_dash_instance

        # Separate query kwargs from other kwargs passed to the BaseDataset constructor
        self.query = query or {}
        self.query.update(
            {
                k: v for k, v in kwargs.items() if k in EEGDash._ALLOWED_QUERY_FIELDS
            }
        )
        base_dataset_kwargs = {k: v for k, v in kwargs.items() if k not in self.query}
        if "dataset" not in self.query:
            raise ValueError("You must provide a 'dataset' argument")

        self.data_dir = self.cache_dir / self.query["dataset"]

        _owns_client = False
        if self.eeg_dash is None and records is None:
            self.eeg_dash = EEGDash()
            _owns_client = True

        try:
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
            elif offline_mode: # only assume local data is complete if in offline mode
                if self.data_dir.exists():
                    # This path loads from a local directory and is not affected by DB query logic
                    datasets = self.load_bids_dataset(
                        dataset=self.query["dataset"],
                        data_dir=self.data_dir,
                        description_fields=description_fields,
                        s3_bucket=s3_bucket,
                        **base_dataset_kwargs,
                    )
                else:
                    raise ValueError(
                        f"Offline mode is enabled, but local data_dir {self.data_dir} does not exist."
                    )
            elif self.query:
                # This is the DB query path that we are improving
                datasets = self._find_datasets(
                    query=self.eeg_dash._build_query_from_kwargs(**self.query),
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
        finally:
            if _owns_client and self.eeg_dash is not None:
                self.eeg_dash.close()

        super().__init__(datasets)

    def find_key_in_nested_dict(self, data: Any, target_key: str) -> Any:
        """Helper to recursively search for a key in a nested dictionary structure; returns
        the value associated with the first occurrence of the key, or None if not found.
        """
        if isinstance(data, dict):
            if target_key in data:
                return data[target_key]
            for value in data.values():
                result = self.find_key_in_nested_dict(value, target_key)
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

        self.records = self.eeg_dash.find(query)

        for record in self.records:
            description = {}
            for field in description_fields:
                value = self.find_key_in_nested_dict(record, field)
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

        """
        bids_dataset = EEGBIDSDataset(
            data_dir=data_dir,
            dataset=dataset,
        )
        datasets = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
            delayed(self.get_base_dataset_from_bids_file)(
                bids_dataset=bids_dataset,
                bids_file=bids_file,
                s3_bucket=s3_bucket,
                description_fields=description_fields,
                **kwargs,
            )
            for bids_file in bids_dataset.get_files()
        )
        return datasets

    def get_base_dataset_from_bids_file(
        self,
        bids_dataset: "EEGBIDSDataset",
        bids_file: str,
        s3_bucket: str | None,
        description_fields: list[str],
        **kwargs,
    ) -> "EEGDashBaseDataset":
        """Instantiate a single EEGDashBaseDataset given a local BIDS file (metadata only)."""
        record = self.eeg_dash.load_eeg_attrs_from_bids_file(bids_dataset, bids_file)
        description = {}
        for field in description_fields:
            value = self.find_key_in_nested_dict(record, field)
            if value is not None:
                description[field] = value
        return EEGDashBaseDataset(
            record,
            self.cache_dir,
            s3_bucket,
            description=description,
            **kwargs,
        )
