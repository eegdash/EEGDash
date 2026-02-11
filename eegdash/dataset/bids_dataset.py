# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Local BIDS dataset interface for EEGDash.

This module provides the EEGBIDSDataset class for interfacing with local BIDS
datasets on the filesystem, parsing metadata, and retrieving BIDS-related information.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from mne_bids import BIDSPath, find_matching_paths, get_entities_from_fname
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS, EPHY_ALLOWED_DATATYPES, reader

# Known companion/sidecar files for specific formats (BIDS spec requirement)
# These files must be downloaded together with the primary file
_COMPANION_FILES = {
    ".set": [".fdt"],  # EEGLAB: data file
    ".vhdr": [".eeg", ".vmrk"],  # BrainVision: data + marker files
}

# Modality directory aliases - different sources use different naming conventions
# BIDS spec uses 'nirs' but some sources (NEMAR, older datasets) use 'fnirs'
_MODALITY_DIR_ALIASES = {
    "nirs": ["nirs", "fnirs"],
    "fnirs": ["nirs", "fnirs"],
}

_TSV_ENCODINGS = ("utf-8", "cp1252", "latin-1")


def _read_tsv_column(tsv_path: Path, column: str) -> list[str]:
    """Read a column from a TSV file, trying multiple encodings."""
    for encoding in _TSV_ENCODINGS:
        try:
            return pd.read_csv(tsv_path, sep="\t", encoding=encoding)[column].tolist()
        except UnicodeDecodeError:
            continue
        except Exception:
            return []
    return []


class EEGBIDSDataset:
    """An interface to a local BIDS dataset containing electrophysiology recordings.

    This class centralizes interactions with a BIDS dataset on the local
    filesystem, providing methods to parse metadata, find files, and
    retrieve BIDS-related information. Supports multiple modalities including
    EEG, MEG, iEEG, and NIRS.

    The class uses MNE-BIDS constants to stay synchronized with the BIDS
    specification and automatically supports all file formats recognized by MNE.

    Parameters
    ----------
    data_dir : str or Path
        The path to the local BIDS dataset directory.
    dataset : str
        A name for the dataset (e.g., "ds002718").
    allow_symlinks : bool, default False
        If True, accept broken symlinks (e.g., git-annex) for metadata extraction.
        If False, require actual readable files for data loading.
        Set to True when doing metadata digestion without loading raw data.
    modalities : list of str or None, default None
        List of modalities to search for (e.g., ["eeg", "meg"]).
        If None, defaults to all electrophysiology modalities from MNE-BIDS:
        ['meg', 'eeg', 'ieeg', 'nirs'].

    Attributes
    ----------
    RAW_EXTENSIONS : dict
        Mapping of file extensions to their companion files, dynamically
        built from mne_bids.config.reader.
    files : list of str
        List of all recording file paths found in the dataset.
    detected_modality : str
        The modality of the first file found (e.g., 'eeg', 'meg').

    Examples
    --------
    >>> # Load EEG-only dataset
    >>> dataset = EEGBIDSDataset(
    ...     data_dir="/path/to/ds002718",
    ...     dataset="ds002718",
    ...     modalities=["eeg"]
    ... )

    >>> # Load dataset with multiple modalities
    >>> dataset = EEGBIDSDataset(
    ...     data_dir="/path/to/ds005810",
    ...     dataset="ds005810",
    ...     modalities=["meg", "eeg"]
    ... )

    >>> # Metadata extraction from git-annex (symlinks)
    >>> dataset = EEGBIDSDataset(
    ...     data_dir="/path/to/dataset",
    ...     dataset="ds000001",
    ...     allow_symlinks=True
    ... )

    """

    # Dynamically build from MNE-BIDS constants (mne_bids.config.reader)
    # reader dict maps file extensions to MNE read functions
    # This ensures compatibility with the latest BIDS specification

    # Primary extension + companions = files that must be downloaded together
    RAW_EXTENSIONS = {
        ext: [ext] + _COMPANION_FILES.get(ext, []) for ext in reader.keys()
    }

    def __init__(
        self,
        data_dir=None,  # location of bids dataset
        dataset="",  # dataset name
        allow_symlinks=False,  # allow broken symlinks for digestion
        modalities=None,
    ):
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError("data_dir must be specified and must exist")

        self.bidsdir = Path(data_dir)
        self.dataset = dataset
        self.data_dir = data_dir
        self.allow_symlinks = allow_symlinks

        # Set modalities to search for (default: all electrophysiology modalities from MNE-BIDS)
        if modalities is None:
            self.modalities = EPHY_ALLOWED_DATATYPES  # ['meg', 'eeg', 'ieeg', 'nirs']
        else:
            self.modalities = (
                modalities if isinstance(modalities, list) else [modalities]
            )

        # Accept exact dataset folder or a variant with informative suffixes
        # (e.g., dsXXXXX-bdf, dsXXXXX-bdf-mini) to avoid collisions.
        dir_name = self.bidsdir.name
        if not (dir_name == self.dataset or dir_name.startswith(self.dataset + "-")):
            raise ValueError(
                f"BIDS directory '{dir_name}' does not correspond to dataset '{self.dataset}'"
            )

        # Initialize BIDS paths using fast mne_bids approach instead of pybids
        self._init_bids_paths()

        # get all recording files in the bids directory
        assert len(self.files) > 0, ValueError(
            f"Unable to construct dataset. No recordings found for modalities: {self.modalities}"
        )
        # Store the detected modality for later use
        self.detected_modality = self.get_bids_file_attribute(
            "modality", self.files[0]
        ).lower()

    def check_eeg_dataset(self) -> bool:
        """Check if the BIDS dataset contains EEG data.

        Returns
        -------
        bool
            True if the dataset's modality is EEG, False otherwise.

        """
        return self.detected_modality == "eeg"

    def _init_bids_paths(self) -> None:
        """Initialize BIDS file paths using mne_bids for fast discovery.

        Uses mne_bids.find_matching_paths() for efficient pattern-based file
        discovery. Falls back to manual glob search if needed.

        When allow_symlinks=True, includes broken symlinks (e.g., git-annex)
        for metadata extraction without requiring actual data files.

        Searches across multiple modalities (eeg, meg, ieeg) based on self.modalities.
        """
        # Initialize cache for BIDSPath objects
        self._bids_path_cache = {}
        self._bids_entity_cache = {}

        # Find all recordings across specified modalities
        # Use MNE-BIDS constants to get valid extensions per modality
        self.files = []
        for modality in self.modalities:
            for ext in ALLOWED_DATATYPE_EXTENSIONS.get(modality, []):
                found_files = _find_bids_files(
                    self.bidsdir,
                    ext,
                    modalities=[modality],
                    allow_symlinks=self.allow_symlinks,
                )
                if found_files:
                    self.files.extend(found_files)
                    # Continue searching for other extensions and modalities to ensure
                    # we capture all data (e.g., mixed EEG/MEG datasets).

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
        if data_filepath not in self._bids_path_cache:
            filepath = Path(data_filepath)

            # Detect modality from the directory path
            # BIDS structure: .../sub-XX/[ses-YY/]<modality>/sub-XX_...
            modality = "eeg"  # default
            for part in filepath.parts:
                if part in ["eeg", "meg", "ieeg", "emg", "nirs", "fnirs"]:
                    # Normalize fnirs -> nirs for MNE-BIDS compatibility
                    modality = "nirs" if part == "fnirs" else part
                    break

            # Use MNE-BIDS to parse all standard entities from the filename.
            # on_error="warn" tolerates non-standard entity keys that appear
            # in some real-world datasets.
            entities = get_entities_from_fname(filepath.name, on_error="warn")

            task_val = entities.get("task")
            run_val = entities.get("run")

            # Sanitize task if it incorrectly absorbed 'run' due to missing
            # underscore separator in the original filename.
            # get_entities_from_fname("task-ECONrun-1_eeg.set") -> task="ECONrun"
            # We want task="ECON", run="1".
            if task_val and re.search(r"run\d*$", task_val):
                task_val = re.split(r"run\d*$", task_val)[0]
                if run_val is None:
                    run_match = re.search(r"run-([^_/.]*)", filepath.name)
                    if run_match:
                        run_val = run_match.group(1)

            # BIDSPath enforces "run" to be an index; accept numeric strings,
            # but drop non-numeric runs (e.g., "5F") while preserving them
            # in the entity cache.
            run_for_bidspath = None
            if run_val is not None and str(run_val).isdigit():
                run_for_bidspath = run_val

            bids_path = BIDSPath(
                subject=entities.get("subject"),
                session=entities.get("session"),
                task=task_val,
                acquisition=entities.get("acquisition"),
                run=run_for_bidspath,
                processing=entities.get("processing"),
                recording=entities.get("recording"),
                space=entities.get("space"),
                split=entities.get("split"),
                description=entities.get("description"),
                datatype=modality,
                extension=filepath.suffix,
                root=self.bidsdir,
            )
            self._bids_path_cache[data_filepath] = bids_path
            self._bids_entity_cache[data_filepath] = {
                "subject": entities.get("subject"),
                "session": entities.get("session"),
                "task": task_val,
                "acquisition": entities.get("acquisition"),
                "run": run_val,
                "processing": entities.get("processing"),
                "recording": entities.get("recording"),
                "space": entities.get("space"),
                "split": entities.get("split"),
                "description": entities.get("description"),
                "modality": modality,
            }

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
            # Try exact match first (e.g., "eeg.json" at root level)
            json_path = current_dir / json_filename
            if json_path.exists():
                with open(json_path) as f:
                    json_dict.update(json.load(f))
            else:
                # Look for BIDS-specific JSON files (e.g., "sub-001_task-rest_eeg.json")
                # Match files ending with the json_filename pattern
                for json_file in current_dir.glob(f"*_{json_filename}"):
                    # Check if this JSON corresponds to the data file
                    data_basename = Path(data_filepath).stem
                    json_basename = json_file.stem
                    # They should share the same BIDS entities prefix
                    if data_basename.split("_eeg")[0] == json_basename.split("_eeg")[0]:
                        with open(json_file) as f:
                            json_dict.update(json.load(f))
                        break

            # Stop at BIDS root (contains dataset_description.json)
            if (current_dir / "dataset_description.json").exists():
                break

            current_dir = current_dir.parent

        return json_dict

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

        # Validate file based on current mode
        if not _is_valid_eeg_file(filepath, allow_symlinks=self.allow_symlinks):
            raise ValueError(
                f"filepath {filepath} does not exist. "
                f"If doing metadata extraction from git-annex, set allow_symlinks=True"
            )

        path, filename = os.path.split(filepath)
        basename = filename[: filename.rfind("_")]
        meta_files = self._get_bids_file_inheritance(
            path, basename, metadata_file_extension
        )
        return meta_files

    def get_files(self) -> list[str]:
        """Get all EEG recording file paths in the BIDS dataset.

        Returns
        -------
        list of str
            A list of file paths for all valid EEG recordings.

        """
        return self.files

    def get_relative_bidspath(self, filepath: str | Path) -> str:
        """Get the dataset-relative path for a file.

        Parameters
        ----------
        filepath : str or Path
            The absolute or relative path to a file in the BIDS dataset.

        Returns
        -------
        str
            The path relative to the dataset root, prefixed with the dataset name.
            e.g., "ds004477/sub-001/eeg/sub-001_task-PES_eeg.json"

        """
        return self._get_relative_bidspath(filepath)

    def _get_relative_bidspath(self, filepath: str | Path) -> str:
        """Internal method to get the dataset-relative path for a file.

        Parameters
        ----------
        filepath : str or Path
            The absolute or relative path to a file in the BIDS dataset.

        Returns
        -------
        str
            The path relative to the dataset root, prefixed with the dataset name.

        """
        filepath = Path(filepath)
        try:
            rel_path = filepath.relative_to(self.bidsdir)
        except ValueError:
            # If filepath is not under bidsdir, just use the filename
            rel_path = Path(filepath.name)
        return f"{self.dataset}/{rel_path.as_posix()}"

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
        entities = self._bids_entity_cache.get(data_filepath, {})

        # Direct BIDSPath properties for entities
        direct_attrs = {
            "subject": entities.get("subject", bids_path.subject),
            "session": entities.get("session", bids_path.session),
            "task": entities.get("task", bids_path.task),
            "run": entities.get("run", bids_path.run),
            "modality": entities.get("modality", bids_path.datatype),
        }

        if attribute in direct_attrs:
            return direct_attrs[attribute]

        # For JSON-based attributes, read the modality-specific JSON file
        # (eeg.json for EEG, meg.json for MEG, ieeg.json for iEEG, nirs.json for NIRS)
        modality = bids_path.datatype or "eeg"
        json_filename = f"{modality}.json"
        modality_json = self._get_json_with_inheritance(data_filepath, json_filename)

        json_attrs = {
            "sfreq": modality_json.get("SamplingFrequency"),
            "duration": modality_json.get("RecordingDuration"),
            "nchans": modality_json.get("EEGChannelCount")
            or modality_json.get("MEGChannelCount")
            or modality_json.get("iEEGChannelCount")
            or modality_json.get("NIRSChannelCount"),
        }

        if attribute == "ntimes":
            sfreq = json_attrs.get("sfreq")
            duration = json_attrs.get("duration")
            if sfreq is None or duration is None:
                return None
            try:
                return int(float(sfreq) * float(duration))
            except (TypeError, ValueError):
                return None

        return json_attrs.get(attribute)

    def _find_channels_tsv(self, data_filepath: str) -> Path:
        """Find channels.tsv for a data file."""
        filepath = Path(data_filepath)
        channels_tsv_path = filepath.parent / "channels.tsv"
        if not channels_tsv_path.exists():
            for tsv_file in filepath.parent.glob("*_channels.tsv"):
                if filepath.stem.startswith(tsv_file.stem.replace("_channels", "")):
                    return tsv_file
        return channels_tsv_path

    def channel_labels(self, data_filepath: str) -> list[str]:
        """Get a list of channel labels from channels.tsv."""
        channels_tsv_path = self._find_channels_tsv(data_filepath)
        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")
        return _read_tsv_column(channels_tsv_path, "name")

    def channel_types(self, data_filepath: str) -> list[str]:
        """Get a list of channel types from channels.tsv."""
        channels_tsv_path = self._find_channels_tsv(data_filepath)
        if not channels_tsv_path.exists():
            raise FileNotFoundError(f"No channels.tsv found for {data_filepath}")
        return _read_tsv_column(channels_tsv_path, "type")

    def num_times(self, data_filepath: str) -> int:
        """Get the number of time points in the recording.

        Calculated from ``SamplingFrequency`` and ``RecordingDuration`` in the
        modality-specific JSON sidecar (e.g., ``eeg.json`` or ``meg.json``).

        Parameters
        ----------
        data_filepath : str
            The path to the data file.

        Returns
        -------
        int
            The approximate number of time points.

        """
        ntimes = self.get_bids_file_attribute("ntimes", data_filepath)
        return int(ntimes or 0)

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
        participants_tsv_files = self.get_bids_metadata_files(
            data_filepath, "participants.tsv"
        )
        if not participants_tsv_files:
            return {}
        # the participant first column is always 'participant_id' or
        # some variation like 'participantID'
        participants_tsv_path = participants_tsv_files[0]
        # Use dtype=str to avoid pandas auto-converting values (e.g., '5F' being interpreted as hex)
        try:
            participants_tsv = pd.read_csv(participants_tsv_path, sep="\t", dtype=str)
        except Exception:
            return {}

        if participants_tsv.empty:
            return {}
        participants_tsv.set_index("participant_id", inplace=True)

        subj_val = self.get_bids_file_attribute("subject", data_filepath)
        if subj_val is None:
            return {}
        # Ensure 'sub-' prefix is handled cleanly
        if not subj_val.startswith("sub-"):
            subject = f"sub-{subj_val}"
        else:
            subject = subj_val

        # Handle case where subject not found in participants.tsv
        if subject not in participants_tsv.index:
            return {}

        row_dict = participants_tsv.loc[subject].to_dict()
        # Convert NaN values to None for JSON compatibility
        return {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}

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


def _is_valid_eeg_file(filepath: Path, allow_symlinks: bool = False) -> bool:
    """Check if a file path is valid for EEG processing.

    Parameters
    ----------
    filepath : Path
        The file path to check.
    allow_symlinks : bool, default False
        If True, accept broken symlinks (e.g., git-annex pointers).
        If False, only accept files that actually exist and can be read.

    Returns
    -------
    bool
        True if the file is valid for the current mode.

    """
    if filepath.exists():
        return True
    if allow_symlinks and filepath.is_symlink():
        return True
    return False


def _find_bids_files(
    bidsdir: Path,
    extension: str,
    modalities: list[str] = None,
    allow_symlinks: bool = False,
) -> list[str]:
    """Find BIDS files in a BIDS directory across multiple modalities.

    Parameters
    ----------
    bidsdir : Path
        The BIDS dataset root directory.
    extension : str
        File extension to search for (e.g., '.set', '.bdf', '.fif').
    modalities : list of str, optional
        List of modalities to search (e.g., ["eeg", "meg", "ieeg"]).
        If None, defaults to EPHY_ALLOWED_DATATYPES from mne_bids.config.
    allow_symlinks : bool, default False
        If True, include broken symlinks in results (for metadata extraction).
        If False, only return files that can be read (for data loading).

    Returns
    -------
    list of str
        List of file paths found.

    """
    if modalities is None:
        modalities = EPHY_ALLOWED_DATATYPES

    all_files = []

    for modality in modalities:
        # First try mne_bids (fast, but skips broken symlinks)
        if not allow_symlinks:
            try:
                paths = find_matching_paths(
                    bidsdir, datatypes=modality, extensions=extension
                )
                if paths:
                    all_files.extend([str(p.fpath) for p in paths])
            except Exception:
                pass  # Continue to fallback search

        # Fallback: manual glob search (finds symlinks too)
        # Search for both the modality name and its aliases (e.g., nirs and fnirs)
        dir_names = _MODALITY_DIR_ALIASES.get(modality, [modality])
        found = []
        for dir_name in dir_names:
            pattern = f"**/{dir_name}/*{extension}"
            found.extend(bidsdir.glob(pattern))

        # Filter based on validation mode and exclude derivatives
        valid_files = [
            str(f)
            for f in found
            if _is_valid_eeg_file(f, allow_symlinks=allow_symlinks)
            and "derivatives" not in f.parts
        ]
        all_files.extend(valid_files)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


__all__ = ["EEGBIDSDataset"]
