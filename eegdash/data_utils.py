import io
import json
import logging
import os
import re
import traceback
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pandas as pd
from bids import BIDSLayout
from joblib import Parallel, delayed
from mne._fiff.utils import _read_segments_file
from mne.io import BaseRaw
from mne_bids import BIDSPath

from braindecode.datasets import BaseDataset

from . import downloader
from .bids_eeg_metadata import enrich_from_participants
from .paths import get_default_cache_dir

logger = logging.getLogger("eegdash")


class EEGDashBaseDataset(BaseDataset):
    """A single EEG recording hosted on AWS S3 and cached locally upon first access.

    This is a subclass of braindecode's BaseDataset, which can consequently be used in
    conjunction with the preprocessing and training pipelines of braindecode.
    """

    _AWS_BUCKET = "s3://openneuro.org"

    def __init__(
        self,
        record: dict[str, Any],
        cache_dir: str,
        s3_bucket: str | None = None,
        **kwargs,
    ):
        """Create a new EEGDashBaseDataset instance. Users do not usually need to call this
        directly -- instead use the EEGDashDataset class to load a collection of these
        recordings from a local BIDS folder or using a database query.

        Parameters
        ----------
        record : dict
            A fully resolved metadata record for the data to load.
        cache_dir : str
            A local directory where the data will be cached.
        kwargs : dict
            Additional keyword arguments to pass to the BaseDataset constructor.

        """
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
        """Helper to restrict the metadata record to the fields needed to locate a BIDS
        recording.
        """
        desired_fields = ["subject", "session", "task", "run"]
        return {k: self.record[k] for k in desired_fields if self.record[k]}

    def _ensure_raw(self) -> None:
        """Download the S3 file and BIDS dependencies if not already cached."""
        # TO-DO: remove this once is fixed on the our side
        # for the competition
        if not self.s3_open_neuro:
            self.bidspath = self.bidspath.update(extension=".bdf")
            self.filecache = self.filecache.with_suffix(".bdf")

        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:
                downloader.download_dependencies(
                    self.s3_bucket,
                    self.bids_dependencies,
                    self.bids_dependencies_original,
                    self.cache_dir,
                    self.dataset_folder,
                    self.record,
                    self.s3_open_neuro,
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
                    "Please check the file and try again."
                )
                logger.error(f"Exception: {e}")
                logger.error(traceback.format_exc())
                raise e

    # === BaseDataset and PyTorch Dataset interface ===
    def __getitem__(self, index):
        """Main function to access a sample from the dataset."""
        X = self.raw[:, index][0]
        y = None
        if self.target_name is not None:
            y = self.description[self.target_name]
        if isinstance(y, pd.Series):
            y = y.to_list()
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def load(self):
        if self.raw is None:
            self.raw = self._load_data()
        return self

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
    def raw(self):
        """Return the MNE Raw object for this recording. This will perform the actual
        retrieval if not yet done so.
        """
        if self._raw is None:
            self._ensure_raw()
        return self._raw

    @raw.setter
    def raw(self, raw):
        self._raw = raw


class EEGDashBaseRaw(BaseRaw):
    """Wrapper around the MNE BaseRaw class that automatically fetches the data from S3
    (when _read_segment is called) and caches it locally. Currently for internal use.

    Parameters
    ----------
    input_fname : path-like
        Path to the S3 file
    metadata : dict
        The metadata record for the recording (e.g., from the database).
    preload : bool
        Whether to pre-loaded the data before the first access.
    cache_dir : str
        Local path under which the data will be cached.
    bids_dependencies : list
        List of additional BIDS metadata files that should be downloaded and cached
        alongside the main recording file.
    verbose : str | int | None
        Optionally the verbosity level for MNE logging (see MNE documentation for possible values).

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.

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
        """Get to work with S3 endpoint first, no caching"""
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
        if not os.path.exists(self.filecache):  # not preload
            if self.bids_dependencies:
                downloader.download_dependencies(
                    self.s3_bucket,
                    self.bids_dependencies,
                    None,
                    self.cache_dir,
                    self.dataset_folder,
                    self.record,
                    self.s3_open_neuro,
                )
            self.filecache = downloader.download_s3_file(
                self.s3file, self.filecache, self.s3_open_neuro
            )
            self.filenames = [self.filecache]
        else:  # not preload and file is not cached
            self.filenames = [self.filecache]
        return super()._read_segment(start, stop, sel, data_buffer, verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data from the file."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<f4")


class EEGBIDSDataset:
    """A one-stop shop interface to a local BIDS dataset containing EEG recordings.

    This is mainly tailored to the needs of EEGDash application and is used to centralize
    interactions with the BIDS dataset, such as parsing the metadata.

    Parameters
    ----------
    data_dir : str | Path
        The path to the local BIDS dataset directory.
    dataset : str
        A name for the dataset.

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
        # Accept exact dataset folder or a variant with informative suffixes
        # (e.g., dsXXXXX-bdf, dsXXXXX-bdf-mini) to avoid collisions.
        dir_name = self.bidsdir.name
        if not (dir_name == self.dataset or dir_name.startswith(self.dataset + "-")):
            raise AssertionError(
                f"BIDS directory '{dir_name}' does not correspond to dataset '{self.dataset}'"
            )
        self.layout = BIDSLayout(data_dir)

        # get all recording files in the bids directory
        self.files = self._get_recordings(self.layout)
        assert len(self.files) > 0, ValueError(
            "Unable to construct EEG dataset. No EEG recordings found."
        )
        assert self.check_eeg_dataset(), ValueError("Dataset is not an EEG dataset.")

    def check_eeg_dataset(self) -> bool:
        """Check if the dataset is EEG."""
        return self.get_bids_file_attribute("modality", self.files[0]).lower() == "eeg"

    def _get_recordings(self, layout: BIDSLayout) -> list[str]:
        """Get a list of all EEG recording files in the BIDS layout."""
        files = []
        for ext, exts in self.RAW_EXTENSIONS.items():
            files = layout.get(extension=ext, return_type="filename")
            if files:
                break
        return files

    def _get_relative_bidspath(self, filename: str) -> str:
        """Make the given file path relative to the BIDS directory."""
        bids_parent_dir = self.bidsdir.parent.absolute()
        return str(Path(filename).relative_to(bids_parent_dir))

    def _get_property_from_filename(self, property: str, filename: str) -> str:
        """Parse a property out of a BIDS-compliant filename. Returns an empty string
        if not found.
        """
        import platform

        if platform.system() == "Windows":
            lookup = re.search(rf"{property}-(.*?)[_\\]", filename)
        else:
            lookup = re.search(rf"{property}-(.*?)[_\/]", filename)
        return lookup.group(1) if lookup else ""

    def _merge_json_inheritance(self, json_files: list[str | Path]) -> dict:
        """Internal helper to merge list of json files found by get_bids_file_inheritance,
        expecting the order (from left to right) is from lowest
        level to highest level, and return a merged dictionary
        """
        json_files.reverse()
        json_dict = {}
        for f in json_files:
            json_dict.update(json.load(open(f)))  # FIXME: should close file
        return json_dict

    def _get_bids_file_inheritance(
        self, path: str | Path, basename: str, extension: str
    ) -> list[Path]:
        """Get all file paths that apply to the basename file in the specified directory
        and that end with the specified suffix, recursively searching parent directories
        (following the BIDS inheritance principle in the order of lowest level first).

        Parameters
        ----------
        path : str | Path
            The directory path to search for files.
        basename : str
            BIDS file basename without _eeg.set extension for example
        extension : str
            Only consider files that end with the specified suffix; e.g. channels.tsv

        Returns
        -------
        list[Path]
            A list of file paths that match the given basename and extension.

        """
        top_level_files = ["README", "dataset_description.json", "participants.tsv"]
        bids_files = []

        # check if path is str object
        if isinstance(path, str):
            path = Path(path)
        if not path.exists:
            raise ValueError("path {path} does not exist")

        # check if file is in current path
        for file in os.listdir(path):
            # target_file = path / f"{cur_file_basename}_{extension}"
            if os.path.isfile(path / file):
                # check if file has extension extension
                # check if file basename has extension
                if file.endswith(extension):
                    filepath = path / file
                    bids_files.append(filepath)

        # check if file is in top level directory
        if any(file in os.listdir(path) for file in top_level_files):
            return bids_files
        else:
            # call get_bids_file_inheritance recursively with parent directory
            bids_files.extend(
                self._get_bids_file_inheritance(path.parent, basename, extension)
            )
            return bids_files

    def get_bids_metadata_files(
        self, filepath: str | Path, metadata_file_extension: list[str]
    ) -> list[Path]:
        """Retrieve all metadata file paths that apply to a given data file path and that
        end with a specific suffix (following the BIDS inheritance principle).

        Parameters
        ----------
        filepath: str | Path
            The filepath to get the associated metadata files for.
        metadata_file_extension : str
            Consider only metadata files that end with the specified suffix,
            e.g., channels.tsv or eeg.json

        Returns
        -------
        list[Path]:
            A list of filepaths for all matching metadata files

        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists:
            raise ValueError("filepath {filepath} does not exist")
        path, filename = os.path.split(filepath)
        basename = filename[: filename.rfind("_")]
        # metadata files
        meta_files = self._get_bids_file_inheritance(
            path, basename, metadata_file_extension
        )
        return meta_files

    def _scan_directory(self, directory: str, extension: str) -> list[Path]:
        """Return a list of file paths that end with the given extension in the specified
        directory. Ignores certain special directories like .git, .datalad, derivatives,
        and code.
        """
        result_files = []
        directory_to_ignore = [".git", ".datalad", "derivatives", "code"]
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(extension):
                    result_files.append(entry.path)
                elif entry.is_dir():
                    # check that entry path doesn't contain any name in ignore list
                    if not any(name in entry.name for name in directory_to_ignore):
                        result_files.append(entry.path)  # Add directory to scan later
        return result_files

    def _get_files_with_extension_parallel(
        self, directory: str, extension: str = ".set", max_workers: int = -1
    ) -> list[Path]:
        """Efficiently scan a directory and its subdirectories for files that end with
        the given extension.

        Parameters
        ----------
        directory : str
            The root directory to scan for files.
        extension : str
            Only consider files that end with this suffix, e.g. '.set'.
        max_workers : int
            Optionally specify the maximum number of worker threads to use for parallel scanning.
            Defaults to all available CPU cores if set to -1.

        Returns
        -------
        list[Path]:
            A list of filepaths for all matching metadata files

        """
        result_files = []
        dirs_to_scan = [directory]

        # Use joblib.Parallel and delayed to parallelize directory scanning
        while dirs_to_scan:
            logger.info(
                f"Directories to scan: {len(dirs_to_scan)}, files: {dirs_to_scan}"
            )
            # Run the scan_directory function in parallel across directories
            results = Parallel(n_jobs=max_workers, prefer="threads", verbose=1)(
                delayed(self._scan_directory)(d, extension) for d in dirs_to_scan
            )

            # Reset the directories to scan and process the results
            dirs_to_scan = []
            for res in results:
                for path in res:
                    if os.path.isdir(path):
                        dirs_to_scan.append(path)  # Queue up subdirectories to scan
                    else:
                        result_files.append(path)  # Add files to the final result
            logger.info(f"Found {len(result_files)} files.")

        return result_files

    def load_and_preprocess_raw(
        self, raw_file: str, preprocess: bool = False
    ) -> np.ndarray:
        """Utility function to load a raw data file with MNE and apply some simple
        (hardcoded) preprocessing and return as a numpy array. Not meant for purposes
        other than testing or debugging.
        """
        logger.info(f"Loading raw data from {raw_file}")
        EEG = mne.io.read_raw_eeglab(raw_file, preload=True, verbose="error")

        if preprocess:
            # highpass filter
            EEG = EEG.filter(l_freq=0.25, h_freq=25, verbose=False)
            # remove 60Hz line noise
            EEG = EEG.notch_filter(freqs=(60), verbose=False)
            # bring to common sampling rate
            sfreq = 128
            if EEG.info["sfreq"] != sfreq:
                EEG = EEG.resample(sfreq)

        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError("Expect raw data to be CxT dimension")
        return mat_data

    def get_files(self) -> list[Path]:
        """Get all EEG recording file paths (with valid extensions) in the BIDS folder."""
        return self.files

    def resolve_bids_json(self, json_files: list[str]) -> dict:
        """Resolve the BIDS JSON files and return a dictionary of the resolved values.

        Parameters
        ----------
        json_files : list
            A list of JSON file paths to resolve in order of leaf level first.

        Returns
        -------
            dict: A dictionary of the resolved values.

        """
        if len(json_files) == 0:
            raise ValueError("No JSON files provided")
        json_files.reverse()  # TODO undeterministic

        json_dict = {}
        for json_file in json_files:
            with open(json_file) as f:
                json_dict.update(json.load(f))
        return json_dict

    def get_bids_file_attribute(self, attribute: str, data_filepath: str) -> Any:
        """Retrieve a specific attribute from the BIDS file metadata applicable
        to the provided recording file path.
        """
        entities = self.layout.parse_file_entities(data_filepath)
        bidsfile = self.layout.get(**entities)[0]
        attributes = bidsfile.get_entities(metadata="all")
        attribute_mapping = {
            "sfreq": "SamplingFrequency",
            "modality": "datatype",
            "task": "task",
            "session": "session",
            "run": "run",
            "subject": "subject",
            "ntimes": "RecordingDuration",
            "nchans": "EEGChannelCount",
        }
        attribute_value = attributes.get(attribute_mapping.get(attribute), None)
        return attribute_value

    def channel_labels(self, data_filepath: str) -> list[str]:
        """Get a list of channel labels for the given data file path."""
        channels_tsv = pd.read_csv(
            self.get_bids_metadata_files(data_filepath, "channels.tsv")[0], sep="\t"
        )
        return channels_tsv["name"].tolist()

    def channel_types(self, data_filepath: str) -> list[str]:
        """Get a list of channel types for the given data file path."""
        channels_tsv = pd.read_csv(
            self.get_bids_metadata_files(data_filepath, "channels.tsv")[0], sep="\t"
        )
        return channels_tsv["type"].tolist()

    def num_times(self, data_filepath: str) -> int:
        """Get the approximate number of time points in the EEG recording based on the BIDS metadata."""
        eeg_jsons = self.get_bids_metadata_files(data_filepath, "eeg.json")
        eeg_json_dict = self._merge_json_inheritance(eeg_jsons)
        return int(
            eeg_json_dict["SamplingFrequency"] * eeg_json_dict["RecordingDuration"]
        )

    def subject_participant_tsv(self, data_filepath: str) -> dict[str, Any]:
        """Get BIDS participants.tsv record for the subject to which the given file
        path corresponds, as a dictionary.
        """
        participants_tsv = pd.read_csv(
            self.get_bids_metadata_files(data_filepath, "participants.tsv")[0], sep="\t"
        )
        # if participants_tsv is not empty
        if participants_tsv.empty:
            return {}
        # set 'participant_id' as index
        participants_tsv.set_index("participant_id", inplace=True)
        subject = f"sub-{self.get_bids_file_attribute('subject', data_filepath)}"
        return participants_tsv.loc[subject].to_dict()

    def eeg_json(self, data_filepath: str) -> dict[str, Any]:
        """Get BIDS eeg.json metadata for the given data file path."""
        eeg_jsons = self.get_bids_metadata_files(data_filepath, "eeg.json")
        eeg_json_dict = self._merge_json_inheritance(eeg_jsons)
        return eeg_json_dict

    def channel_tsv(self, data_filepath: str) -> dict[str, Any]:
        """Get BIDS channels.tsv metadata for the given data file path, as a dictionary
        of lists and/or single values.
        """
        channels_tsv = pd.read_csv(
            self.get_bids_metadata_files(data_filepath, "channels.tsv")[0], sep="\t"
        )
        channel_tsv = channels_tsv.to_dict()
        # 'name' and 'type' now have a dictionary of index-value. Convert them to list
        for list_field in ["name", "type", "units"]:
            channel_tsv[list_field] = list(channel_tsv[list_field].values())
        return channel_tsv
