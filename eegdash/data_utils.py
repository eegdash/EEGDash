# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.
"""

import io
import json
import os
import platform
import re
import traceback
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne._fiff.utils import _read_segments_file
from mne.io import BaseRaw
from mne_bids import BIDSPath, find_matching_paths

from braindecode.datasets import BaseDataset

from . import downloader
from .bids_eeg_metadata import enrich_from_participants
from .logging import logger
from .paths import get_default_cache_dir


class EEGDashBaseDataset(BaseDataset):
    """A single EEG recording dataset.

    Represents a single EEG recording, typically hosted on a remote server (like AWS S3)
    and cached locally upon first access. This class is a subclass of
    :class:`braindecode.datasets.BaseDataset` and can be used with braindecode's
    preprocessing and training pipelines.

    Parameters
    ----------
    record : dict
        A fully resolved metadata record for the data to load.
    cache_dir : str
        The local directory where the data will be cached.
    s3_bucket : str, optional
        The S3 bucket to download data from. If not provided, defaults to the
        OpenNeuro bucket.
    **kwargs
        Additional keyword arguments passed to the
        :class:`braindecode.datasets.BaseDataset` constructor.

    """

    _AWS_BUCKET = "s3://openneuro.org"

    def __init__(
        self,
        record: dict[str, Any],
        cache_dir: str,
        s3_bucket: str | None = None,
        **kwargs,
    ):
        super().__init__(None, **kwargs)
        self.record = record
        self.cache_dir = Path(cache_dir)
        self.bids_kwargs = self._get_raw_bids_args()

        if s3_bucket:
            self.s3_bucket = s3_bucket
            self.s3_open_neuro = False
        else:
            self.s3_bucket = self._AWS_BUCKET
            self.s3_open_neuro = True

        # Compute a dataset folder name under cache_dir that encodes preprocessing
        # (e.g., bdf, mini) to avoid overlapping with the original dataset cache.
        self.dataset_folder = record.get("dataset", "")
        # TODO: remove this hack when competition is over
        if s3_bucket:
            suffixes: list[str] = []
            bucket_lower = str(s3_bucket).lower()
            if "bdf" in bucket_lower:
                suffixes.append("bdf")
            if "mini" in bucket_lower:
                suffixes.append("mini")
            if suffixes:
                self.dataset_folder = f"{self.dataset_folder}-{'-'.join(suffixes)}"

        # Place files under the dataset-specific folder (with suffix if any)
        rel = Path(record["bidspath"])  # usually starts with dataset id
        if rel.parts and rel.parts[0] == record.get("dataset"):
            rel = Path(self.dataset_folder, *rel.parts[1:])
        else:
            rel = Path(self.dataset_folder) / rel
        self.filecache = self.cache_dir / rel
        self.bids_root = self.cache_dir / self.dataset_folder

        self.bidspath = BIDSPath(
            root=self.bids_root,
            datatype="eeg",
            suffix="eeg",
            **self.bids_kwargs,
        )

        self.s3file = downloader.get_s3path(self.s3_bucket, record["bidspath"])
        self.bids_dependencies = record["bidsdependencies"]
        self.bids_dependencies_original = record["bidsdependencies"]
        # TODO: removing temporary fix for BIDS dependencies path
        # when the competition is over and dataset is digested properly
        if not self.s3_open_neuro:
            self.bids_dependencies = [
                dep.split("/", 1)[1] for dep in self.bids_dependencies
            ]

        self._raw = None

    def _get_raw_bids_args(self) -> dict[str, Any]:
        """Extract BIDS-related arguments from the metadata record."""
        desired_fields = ["subject", "session", "task", "run"]
        return {k: self.record[k] for k in desired_fields if self.record[k]}

    def _ensure_raw(self) -> None:
        """Ensure the raw data file and its dependencies are cached locally."""
        # TO-DO: remove this once is fixed on the our side
        # for the competition
        if not self.s3_open_neuro:
            self.bidspath = self.bidspath.update(extension=".bdf")
            self.filecache = self.filecache.with_suffix(".bdf")

        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:
                downloader.download_dependencies(
                    s3_bucket=self.s3_bucket,
                    bids_dependencies=self.bids_dependencies,
                    bids_dependencies_original=self.bids_dependencies_original,
                    cache_dir=self.cache_dir,
                    dataset_folder=self.dataset_folder,
                    record=self.record,
                    s3_open_neuro=self.s3_open_neuro,
                )
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
        if self._raw is None:
            try:
                # mne-bids can emit noisy warnings to stderr; keep user logs clean
                _stderr_buffer = io.StringIO()
                with redirect_stderr(_stderr_buffer):
                    self._raw = mne_bids.read_raw_bids(
                        bids_path=self.bidspath, verbose="ERROR"
                    )
                # Enrich Raw.info and description with participants.tsv extras
                enrich_from_participants(
                    self.bids_root, self.bidspath, self._raw, self.description
                )

            except Exception as e:
                logger.error(
                    f"Error while reading BIDS file: {self.bidspath}\n"
                    "This may be due to a missing or corrupted file.\n"
                    "Please check the file and try again.\n"
                    "Usually erasing the local cache and re-downloading helps.\n"
                    f"`rm {self.bidspath}`"
                )
                logger.error(f"Exception: {e}")
                logger.error(traceback.format_exc())
                raise e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._raw is None:
            if (
                self.record["ntimes"] is None
                or self.record["sampling_frequency"] is None
            ):
                self._ensure_raw()
            else:
                # FIXME: this is a bit strange and should definitely not change as a side effect
                #  of accessing the data (which it will, since ntimes is the actual length but rounded down)
                return int(self.record["ntimes"] * self.record["sampling_frequency"])
        return len(self._raw)

    @property
    def raw(self) -> BaseRaw:
        """The MNE Raw object for this recording.

        Accessing this property triggers the download and caching of the data
        if it has not been accessed before.

        Returns
        -------
        mne.io.BaseRaw
            The loaded MNE Raw object.

        """
        if self._raw is None:
            self._ensure_raw()
        return self._raw

    @raw.setter
    def raw(self, raw: BaseRaw):
        self._raw = raw


class EEGDashBaseRaw(BaseRaw):
    """MNE BaseRaw wrapper for automatic S3 data fetching.

    This class extends :class:`mne.io.BaseRaw` to automatically fetch data
    from an S3 bucket and cache it locally when data is first accessed.
    It is intended for internal use within the EEGDash ecosystem.

    Parameters
    ----------
    input_fname : str
        The path to the file on the S3 bucket (relative to the bucket root).
    metadata : dict
        The metadata record for the recording, containing information like
        sampling frequency, channel names, etc.
    preload : bool, default False
        If True, preload the data into memory.
    cache_dir : str, optional
        Local directory for caching data. If None, a default directory is used.
    bids_dependencies : list of str, default []
        A list of BIDS metadata files to download alongside the main recording.
    verbose : str, int, or None, default None
        The MNE verbosity level.

    See Also
    --------
    mne.io.Raw : The base class for Raw objects in MNE.

    """

    _AWS_BUCKET = "s3://openneuro.org"

    def __init__(
        self,
        input_fname: str,
        metadata: dict[str, Any],
        preload: bool = False,
        *,
        cache_dir: str | None = None,
        bids_dependencies: list[str] = [],
        verbose: Any = None,
    ):
        # Create a simple RawArray
        sfreq = metadata["sfreq"]  # Sampling frequency
        n_times = metadata["n_times"]
        ch_names = metadata["ch_names"]
        ch_types = []
        for ch in metadata["ch_types"]:
            chtype = ch.lower()
            if chtype == "heog" or chtype == "veog":
                chtype = "eog"
            ch_types.append(chtype)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        self.s3file = downloader.get_s3path(self._AWS_BUCKET, input_fname)
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        self.filecache = self.cache_dir / input_fname
        self.bids_dependencies = bids_dependencies

        if preload and not os.path.exists(self.filecache):
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
            preload = self.filecache

        super().__init__(
            info,
            preload,
            last_samps=[n_times - 1],
            orig_format="single",
            verbose=verbose,
        )

    def _read_segment(
        self, start=0, stop=None, sel=None, data_buffer=None, *, verbose=None
    ):
        """Read a segment of data, downloading if necessary."""
        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:  # this is use only to sidecars for now
                downloader.download_dependencies(
                    s3_bucket=self._AWS_BUCKET,
                    bids_dependencies=self.bids_dependencies,
                    bids_dependencies_original=None,
                    cache_dir=self.cache_dir,
                    dataset_folder=self.filecache,
                    record={},
                    s3_open_neuro=self.s3_open_neuro,
                )
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
        else:  # not preload and file is not cached
            self.filenames = [self.filecache]
        return super()._read_segment(start, stop, sel, data_buffer, verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data from a local file."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")


class EEGBIDSDataset:
    """An interface to a local BIDS dataset containing EEG recordings.

    This class centralizes interactions with a BIDS dataset on the local
    filesystem, providing methods to parse metadata, find files, and
    retrieve BIDS-related information.

    Parameters
    ----------
    data_dir : str or Path
        The path to the local BIDS dataset directory.
    dataset : str
        A name for the dataset (e.g., "ds002718").

    """

    ALLOWED_FILE_FORMAT = ["eeglab", "brainvision", "biosemi", "european"]
    RAW_EXTENSIONS = {
        ".set": [".set", ".fdt"],  # eeglab
        ".edf": [".edf"],  # european
        ".vhdr": [".eeg", ".vhdr", ".vmrk", ".dat", ".raw"],  # brainvision
        ".bdf": [".bdf"],  # biosemi
    }
    METADATA_FILE_EXTENSIONS = [
        "eeg.json",
        "channels.tsv",
        "electrodes.tsv",
        "events.tsv",
        "events.json",
    ]

    def __init__(
        self,
        data_dir=None,  # location of bids dataset
        dataset="",  # dataset name
    ):
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError("data_dir must be specified and must exist")

        self.bidsdir = Path(data_dir)
        self.dataset = dataset
        self.data_dir = data_dir

        # Accept exact dataset folder or a variant with informative suffixes
        # (e.g., dsXXXXX-bdf, dsXXXXX-bdf-mini) to avoid collisions.
        dir_name = self.bidsdir.name
        if not (dir_name == self.dataset or dir_name.startswith(self.dataset + "-")):
            raise AssertionError(
                f"BIDS directory '{dir_name}' does not correspond to dataset '{self.dataset}'"
            )

        # Initialize BIDS paths using fast mne_bids approach instead of pybids
        self._init_bids_paths()

        # get all recording files in the bids directory
        assert len(self.files) > 0, ValueError(
            "Unable to construct EEG dataset. No EEG recordings found."
        )
        assert self.check_eeg_dataset(), ValueError("Dataset is not an EEG dataset.")

    def check_eeg_dataset(self) -> bool:
        """Check if the BIDS dataset contains EEG data.

        Returns
        -------
        bool
            True if the dataset's modality is EEG, False otherwise.

        """
        return self.get_bids_file_attribute("modality", self.files[0]).lower() == "eeg"

    def _init_bids_paths(self) -> None:
        """Initialize BIDS file paths using mne_bids for fast discovery.

        Uses mne_bids.find_matching_paths() for efficient pattern-based file
        discovery instead of heavy pybids BIDSLayout indexing.
        """
        # Initialize cache for BIDSPath objects
        self._bids_path_cache = {}

        # Find all EEG recordings using pattern matching (fast!)
        self.files = []
        for ext in self.RAW_EXTENSIONS.keys():
            # find_matching_paths returns BIDSPath objects
            paths = find_matching_paths(self.bidsdir, datatypes="eeg", extensions=ext)
            if paths:
                # Convert BIDSPath objects to filename strings
                self.files = [str(p.fpath) for p in paths]
                break

    def _get_bids_path_from_file(self, data_filepath: str):
        """Get a BIDSPath object for a data file with caching.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        BIDSPath
            The BIDSPath object for the file.

        """
        from mne_bids import BIDSPath

        if data_filepath not in self._bids_path_cache:
            # Parse the filename to extract BIDS entities
            filepath = Path(data_filepath)
            filename = filepath.name

            # Extract entities from filename using BIDS pattern
            # Expected format: sub-<label>[_ses-<label>][_task-<label>][_run-<label>]_eeg.<ext>
            subject = re.search(r"sub-([^_]*)", filename)
            session = re.search(r"ses-([^_]*)", filename)
            task = re.search(r"task-([^_]*)", filename)
            run = re.search(r"run-([^_]*)", filename)

            bids_path = BIDSPath(
                subject=subject.group(1) if subject else None,
                session=session.group(1) if session else None,
                task=task.group(1) if task else None,
                run=int(run.group(1)) if run else None,
                datatype="eeg",
                extension=filepath.suffix,
                root=self.bidsdir,
            )
            self._bids_path_cache[data_filepath] = bids_path

        return self._bids_path_cache[data_filepath]

    def _get_json_with_inheritance(
        self, data_filepath: str, json_filename: str
    ) -> dict:
        """Get JSON metadata with BIDS inheritance handling.

        Walks up the directory tree to find and merge JSON files following
        BIDS inheritance principles.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.
        json_filename : str
            The name of the JSON file to find (e.g., "eeg.json").

        Returns
        -------
        dict
            The merged JSON metadata.

        """
        json_dict = {}
        current_dir = Path(data_filepath).parent
        root_dir = self.bidsdir

        # Walk up from file directory to root, collecting JSON files
        while current_dir >= root_dir:
            json_path = current_dir / json_filename
            if json_path.exists():
                with open(json_path) as f:
                    json_dict.update(json.load(f))

            # Stop at BIDS root (contains dataset_description.json)
            if (current_dir / "dataset_description.json").exists():
                break

            current_dir = current_dir.parent

        return json_dict

    def _get_property_from_filename(self, property: str, filename: str) -> str:
        """Parse a BIDS entity from a filename."""
        if platform.system() == "Windows":
            lookup = re.search(rf"{property}-(.*?)[_\\]", filename)
        else:
            lookup = re.search(rf"{property}-(.*?)[_\/]", filename)
        return lookup.group(1) if lookup else ""

    def _merge_json_inheritance(self, json_files: list[str | Path]) -> dict:
        """Merge a list of JSON files according to BIDS inheritance."""
        json_files.reverse()
        json_dict = {}
        for f in json_files:
            with open(f) as fp:
                json_dict.update(json.load(fp))
        return json_dict

    def _get_bids_file_inheritance(
        self, path: str | Path, basename: str, extension: str
    ) -> list[Path]:
        """Find all applicable metadata files using BIDS inheritance."""
        top_level_files = ["README", "dataset_description.json", "participants.tsv"]
        bids_files = []

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"path {path} does not exist")

        for file in os.listdir(path):
            if os.path.isfile(path / file) and file.endswith(extension):
                bids_files.append(path / file)

        if any(file in os.listdir(path) for file in top_level_files):
            return bids_files
        else:
            bids_files.extend(
                self._get_bids_file_inheritance(path.parent, basename, extension)
            )
            return bids_files

    def get_bids_metadata_files(
        self, filepath: str | Path, metadata_file_extension: str
    ) -> list[Path]:
        """Retrieve all metadata files that apply to a given data file.

        Follows the BIDS inheritance principle to find all relevant metadata
        files (e.g., ``channels.tsv``, ``eeg.json``) for a specific recording.

        Parameters
        ----------
        filepath : str or Path
            The path to the data file.
        metadata_file_extension : str
            The extension of the metadata file to search for (e.g., "channels.tsv").

        Returns
        -------
        list of Path
            A list of paths to the matching metadata files.

        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError(f"filepath {filepath} does not exist")
        path, filename = os.path.split(filepath)
        basename = filename[: filename.rfind("_")]
        meta_files = self._get_bids_file_inheritance(
            path, basename, metadata_file_extension
        )
        return meta_files

    def _scan_directory(self, directory: str, extension: str) -> list[Path]:
        """Scan a directory for files with a given extension."""
        result_files = []
        directory_to_ignore = [".git", ".datalad", "derivatives", "code"]
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(extension):
                    result_files.append(Path(entry.path))
                elif entry.is_dir() and not any(
                    name in entry.name for name in directory_to_ignore
                ):
                    result_files.append(Path(entry.path))
        return result_files

    def _get_files_with_extension_parallel(
        self, directory: str, extension: str = ".set", max_workers: int = -1
    ) -> list[Path]:
        """Scan a directory tree in parallel for files with a given extension."""
        result_files = []
        dirs_to_scan = [directory]

        while dirs_to_scan:
            logger.info(
                f"Directories to scan: {len(dirs_to_scan)}, files: {dirs_to_scan}"
            )
            results = Parallel(n_jobs=max_workers, prefer="threads", verbose=1)(
                delayed(self._scan_directory)(d, extension) for d in dirs_to_scan
            )

            dirs_to_scan = []
            for res in results:
                for path in res:
                    if os.path.isdir(path):
                        dirs_to_scan.append(path)
                    else:
                        result_files.append(path)
            logger.info(f"Found {len(result_files)} files.")

        return result_files

    def load_and_preprocess_raw(
        self, raw_file: str, preprocess: bool = False
    ) -> np.ndarray:
        """Load and optionally preprocess a raw data file.

        This is a utility function for testing or debugging, not for general use.

        Parameters
        ----------
        raw_file : str
            Path to the raw EEGLAB file (.set).
        preprocess : bool, default False
            If True, apply a high-pass filter, notch filter, and resample the data.

        Returns
        -------
        numpy.ndarray
            The loaded and processed data as a NumPy array.

        """
        logger.info(f"Loading raw data from {raw_file}")
        EEG = mne.io.read_raw_eeglab(raw_file, preload=True, verbose="error")

        if preprocess:
            EEG = EEG.filter(l_freq=0.25, h_freq=25, verbose=False)
            EEG = EEG.notch_filter(freqs=(60), verbose=False)
            sfreq = 128
            if EEG.info["sfreq"] != sfreq:
                EEG = EEG.resample(sfreq)

        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError("Expect raw data to be CxT dimension")
        return mat_data

    def get_files(self) -> list[str]:
        """Get all EEG recording file paths in the BIDS dataset.

        Returns
        -------
        list of str
            A list of file paths for all valid EEG recordings.

        """
        return self.files

    def resolve_bids_json(self, json_files: list[str]) -> dict:
        """Resolve BIDS JSON inheritance and merge files.

        Parameters
        ----------
        json_files : list of str
            A list of JSON file paths, ordered from the lowest (most specific)
            to highest level of the BIDS hierarchy.

        Returns
        -------
        dict
            A dictionary containing the merged JSON data.

        """
        if not json_files:
            raise ValueError("No JSON files provided")
        json_files.reverse()

        json_dict = {}
        for json_file in json_files:
            with open(json_file) as f:
                json_dict.update(json.load(f))
        return json_dict

    def get_bids_file_attribute(self, attribute: str, data_filepath: str) -> Any:
        """Retrieve a specific attribute from BIDS metadata.

        Parameters
        ----------
        attribute : str
            The name of the attribute to retrieve (e.g., "sfreq", "subject").
        data_filepath : str
            The path to the data file.

        Returns
        -------
        Any
            The value of the requested attribute, or None if not found.

        """
        bids_path = self._get_bids_path_from_file(data_filepath)

        # Direct BIDSPath properties for entities
        direct_attrs = {
            "subject": bids_path.subject,
            "session": bids_path.session,
            "task": bids_path.task,
            "run": bids_path.run,
            "modality": bids_path.datatype,
        }

        if attribute in direct_attrs:
            return direct_attrs[attribute]

        # For JSON-based attributes, read and cache eeg.json
        eeg_json = self._get_json_with_inheritance(data_filepath, "eeg.json")

        json_attrs = {
            "sfreq": eeg_json.get("SamplingFrequency"),
            "ntimes": eeg_json.get("RecordingDuration"),
            "nchans": eeg_json.get("EEGChannelCount"),
        }

        return json_attrs.get(attribute)

    def channel_labels(self, data_filepath: str) -> list[str]:
        """Get a list of channel labels from channels.tsv.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        list of str
            A list of channel names.

        """
        # Find channels.tsv in the same directory as the data file
        # It can be named either "channels.tsv" or "*_channels.tsv"
        filepath = Path(data_filepath)
        parent_dir = filepath.parent

        # Try the standard channels.tsv first
        channels_tsv_path = parent_dir / "channels.tsv"
        if not channels_tsv_path.exists():
            # Try to find *_channels.tsv matching the filename prefix
            base_name = filepath.stem  # filename without extension
            for tsv_file in parent_dir.glob("*_channels.tsv"):
                # Check if it matches by looking at task/run components
                tsv_name = tsv_file.stem.replace("_channels", "")
                if base_name.startswith(tsv_name):
                    channels_tsv_path = tsv_file
                    break

        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")

        channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
        return channels_tsv["name"].tolist()

    def channel_types(self, data_filepath: str) -> list[str]:
        """Get a list of channel types from channels.tsv.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        list of str
            A list of channel types.

        """
        # Find channels.tsv in the same directory as the data file
        # It can be named either "channels.tsv" or "*_channels.tsv"
        filepath = Path(data_filepath)
        parent_dir = filepath.parent

        # Try the standard channels.tsv first
        channels_tsv_path = parent_dir / "channels.tsv"
        if not channels_tsv_path.exists():
            # Try to find *_channels.tsv matching the filename prefix
            base_name = filepath.stem  # filename without extension
            for tsv_file in parent_dir.glob("*_channels.tsv"):
                # Check if it matches by looking at task/run components
                tsv_name = tsv_file.stem.replace("_channels", "")
                if base_name.startswith(tsv_name):
                    channels_tsv_path = tsv_file
                    break

        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")

        channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
        return channels_tsv["type"].tolist()

    def num_times(self, data_filepath: str) -> int:
        """Get the number of time points in the recording.

        Calculated from ``SamplingFrequency`` and ``RecordingDuration`` in eeg.json.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        int
            The approximate number of time points.

        """
        eeg_json_dict = self._get_json_with_inheritance(data_filepath, "eeg.json")
        return int(
            eeg_json_dict.get("SamplingFrequency", 0)
            * eeg_json_dict.get("RecordingDuration", 0)
        )

    def subject_participant_tsv(self, data_filepath: str) -> dict[str, Any]:
        """Get the participants.tsv record for a subject.

        Parameters
        ----------
        data_filepath : str
            The path to a data file belonging to the subject.

        Returns
        -------
        dict
            A dictionary of the subject's information from participants.tsv.

        """
        participants_tsv_path = self.get_bids_metadata_files(
            data_filepath, "participants.tsv"
        )[0]
        participants_tsv = pd.read_csv(participants_tsv_path, sep="\t")
        if participants_tsv.empty:
            return {}
        participants_tsv.set_index("participant_id", inplace=True)
        subject = f"sub-{self.get_bids_file_attribute('subject', data_filepath)}"
        return participants_tsv.loc[subject].to_dict()

    def eeg_json(self, data_filepath: str) -> dict[str, Any]:
        """Get the merged eeg.json metadata for a data file.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        dict
            The merged eeg.json metadata.

        """
        return self._get_json_with_inheritance(data_filepath, "eeg.json")

    def channel_tsv(self, data_filepath: str) -> dict[str, Any]:
        """Get the channels.tsv metadata as a dictionary.

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        dict
            The channels.tsv data, with columns as keys.

        """
        # Find channels.tsv in the same directory as the data file
        # It can be named either "channels.tsv" or "*_channels.tsv"
        filepath = Path(data_filepath)
        parent_dir = filepath.parent

        # Try the standard channels.tsv first
        channels_tsv_path = parent_dir / "channels.tsv"
        if not channels_tsv_path.exists():
            # Try to find *_channels.tsv matching the filename prefix
            base_name = filepath.stem  # filename without extension
            for tsv_file in parent_dir.glob("*_channels.tsv"):
                # Check if it matches by looking at task/run components
                tsv_name = tsv_file.stem.replace("_channels", "")
                if base_name.startswith(tsv_name):
                    channels_tsv_path = tsv_file
                    break

        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")

        channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
        channel_tsv_dict = channels_tsv.to_dict()
        for list_field in ["name", "type", "units"]:
            if list_field in channel_tsv_dict:
                channel_tsv_dict[list_field] = list(
                    channel_tsv_dict[list_field].values()
                )
        return channel_tsv_dict


__all__ = ["EEGDashBaseDataset", "EEGBIDSDataset", "EEGDashBaseRaw"]
