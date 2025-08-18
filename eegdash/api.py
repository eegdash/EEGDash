import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import mne
from mne_bids import read_raw_bids
from mne_bids.utils import get_bids_path_from_fname
import numpy as np
import xarray as xr
from dotenv import load_dotenv
from joblib import Parallel, delayed
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

    AWS_BUCKET = "s3://openneuro.org"

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

    def find(self, query: dict[str, Any], *args, **kwargs) -> list[Mapping[str, Any]]:
        """Find records in the MongoDB collection that satisfy the given query.

        Parameters
        ----------
        query: dict
            A dictionary that specifies the query to be executed; this is a reference
            document that is used to match records in the MongoDB collection.
        args:
            Additional positional arguments for the MongoDB find() method; see
            https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.find
        kwargs:
            Additional keyword arguments for the MongoDB find() method.

        Returns
        -------
        list:
            A list of DB records (string-keyed dictionaries) that match the query.

        Example
        -------
        >>> eegdash = EEGDash()
        >>> eegdash.find({"dataset": "ds002718", "subject": "012"})

        """
        results = self.__collection.find(query, *args, **kwargs)

        return [result for result in results]

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
        query: dict | None = None,
        data_dir: str | list | None = None,
        dataset: str | list | None = None,
        description_fields: list[str] = [
            "subject",
            "session",
            "run",
            "task",
            "age",
            "gender",
            "sex",
        ],
        cache_dir: str = ".eegdash_cache",
        s3_bucket: str | None = None,
        **kwargs,
    ):
        """Create a new EEGDashDataset from a given query or local BIDS dataset directory
        and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
        instances (individual recordings) and is a subclass of braindecode's BaseConcatDataset.

        Parameters
        ----------
        query : dict | None
            Optionally a dictionary that specifies the query to be executed; see
            EEGDash.find() for details on the query format.
        data_dir : str | list[str] | None
            Optionally a string or a list of strings specifying one or more local
            BIDS dataset directories from which to load the EEG data files. Exactly one
            of query or data_dir must be provided.
        dataset : str | list[str] | None
            If data_dir is given, a name or list of names for for the dataset(s) to be loaded.
        description_fields : list[str]
            A list of fields to be extracted from the dataset records
            and included in the returned data description(s). Examples are typical
            subject metadata fields such as "subject", "session", "run", "task", etc.;
            see also data_config.description_fields for the default set of fields.
        cache_dir : str
            A directory where the dataset will be cached locally.
        s3_bucket : str | None
            An optional S3 bucket URI (e.g., "s3://mybucket") to use instead of the
            default OpenNeuro bucket for loading data files
        kwargs : dict
            Additional keyword arguments to be passed to the EEGDashBaseDataset
            constructor.

        """
        self.cache_dir = cache_dir
        self.s3_bucket = s3_bucket
        if query:
            datasets = self.find_datasets(query, description_fields, **kwargs)
        elif data_dir:
            if isinstance(data_dir, str):
                datasets = self.load_bids_dataset(
                    dataset, data_dir, description_fields, s3_bucket
                )
            else:
                assert len(data_dir) == len(dataset), (
                    "Number of datasets and their directories must match"
                )
                datasets = []
                for i, _ in enumerate(data_dir):
                    datasets.extend(
                        self.load_bids_dataset(
                            dataset[i], data_dir[i], description_fields, s3_bucket
                        )
                    )

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

    def find_datasets(
        self, query: dict[str, Any], description_fields: list[str], **kwargs
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
        eeg_dash_instance = EEGDash()
        try:
            datasets = []
            for record in eeg_dash_instance.find(query):
                description = {}
                for field in description_fields:
                    value = self.find_key_in_nested_dict(record, field)
                    if value is not None:
                        description[field] = value
                datasets.append(
                    EEGDashBaseDataset(
                        record,
                        self.cache_dir,
                        self.s3_bucket,
                        description=description,
                        **kwargs,
                    )
                )
            return datasets
        finally:
            eeg_dash_instance.close()

    def load_bids_dataset(
        self,
        dataset,
        data_dir,
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
        eeg_dash_instance = EEGDash()
        try:
            datasets = Parallel(n_jobs=-1, prefer="threads", verbose=1)(
                delayed(self.get_base_dataset_from_bids_file)(
                    bids_dataset=bids_dataset,
                    bids_file=bids_file,
                    eeg_dash_instance=eeg_dash_instance,
                    s3_bucket=s3_bucket,
                    description_fields=description_fields,
                )
                for bids_file in bids_dataset.get_files()
            )
            return datasets
        finally:
            eeg_dash_instance.close()

    def get_base_dataset_from_bids_file(
        self,
        bids_dataset: EEGBIDSDataset,
        bids_file: str,
        eeg_dash_instance: EEGDash,
        s3_bucket: str | None,
        description_fields: list[str],
    ) -> EEGDashBaseDataset:
        """Instantiate a single EEGDashBaseDataset given a local BIDS file. Note
        this does not actually load the data from disk, but will access the metadata.
        """
        record = eeg_dash_instance.load_eeg_attrs_from_bids_file(
            bids_dataset, bids_file
        )
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
        )
