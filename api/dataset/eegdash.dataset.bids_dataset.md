# eegdash.dataset.bids_dataset module

Local BIDS dataset interface for EEGDash.

This module provides the EEGBIDSDataset class for interfacing with local BIDS
datasets on the filesystem, parsing metadata, and retrieving BIDS-related information.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.dataset.bids_dataset.EEGBIDSDataset(data_dir=None, dataset='', allow_symlinks=False, modalities=None)

Bases: `object`

An interface to a local BIDS dataset containing electrophysiology recordings.

This class centralizes interactions with a BIDS dataset on the local
filesystem, providing methods to parse metadata, find files, and
retrieve BIDS-related information. Supports multiple modalities including
EEG, MEG, iEEG, and NIRS.

The class uses MNE-BIDS constants to stay synchronized with the BIDS
specification and automatically supports all file formats recognized by MNE.

* **Parameters:**
  * **data_dir** (*str* *or* *Path*) – The path to the local BIDS dataset directory.
  * **dataset** (*str*) – A name for the dataset (e.g., “ds002718”).
  * **allow_symlinks** (*bool* *,* *default False*) – If True, accept broken symlinks (e.g., git-annex) for metadata extraction.
    If False, require actual readable files for data loading.
    Set to True when doing metadata digestion without loading raw data.
  * **modalities** (*list* *of* *str* *or* *None* *,* *default None*) – List of modalities to search for (e.g., [“eeg”, “meg”]).
    If None, defaults to all electrophysiology modalities from MNE-BIDS:
    [‘meg’, ‘eeg’, ‘ieeg’, ‘nirs’].

#### RAW_EXTENSIONS

Mapping of file extensions to their companion files, dynamically
built from mne_bids.config.reader.

* **Type:**
  dict

#### files

List of all recording file paths found in the dataset.

* **Type:**
  list of str

#### detected_modality

The modality of the first file found (e.g., ‘eeg’, ‘meg’).

* **Type:**
  str

### Examples

```pycon
>>> # Load EEG-only dataset
>>> dataset = EEGBIDSDataset(
...     data_dir="/path/to/ds002718",
...     dataset="ds002718",
...     modalities=["eeg"]
... )
```

```pycon
>>> # Load dataset with multiple modalities
>>> dataset = EEGBIDSDataset(
...     data_dir="/path/to/ds005810",
...     dataset="ds005810",
...     modalities=["meg", "eeg"]
... )
```

```pycon
>>> # Metadata extraction from git-annex (symlinks)
>>> dataset = EEGBIDSDataset(
...     data_dir="/path/to/dataset",
...     dataset="ds000001",
...     allow_symlinks=True
... )
```

<!-- !! processed by numpydoc !! -->

#### RAW_EXTENSIONS *= {'.CNT': ['.CNT'], '.EDF': ['.EDF'], '.EEG': ['.EEG'], '.bdf': ['.bdf'], '.bin': ['.bin'], '.cdt': ['.cdt'], '.cnt': ['.cnt'], '.con': ['.con'], '.ds': ['.ds'], '.edf': ['.edf'], '.fif': ['.fif'], '.lay': ['.lay'], '.pdf': ['.pdf'], '.set': ['.set', '.fdt'], '.snirf': ['.snirf'], '.sqd': ['.sqd'], '.vhdr': ['.vhdr', '.eeg', '.vmrk', '.dat']}*

#### channel_labels(data_filepath: str) → list[str]

Get a list of channel labels from channels.tsv.

<!-- !! processed by numpydoc !! -->

#### channel_types(data_filepath: str) → list[str]

Get a list of channel types from channels.tsv.

<!-- !! processed by numpydoc !! -->

#### check_eeg_dataset() → bool

Check if the BIDS dataset contains EEG data.

* **Returns:**
  True if the dataset’s modality is EEG, False otherwise.
* **Return type:**
  bool

<!-- !! processed by numpydoc !! -->

#### eeg_json(data_filepath: str) → dict[str, Any]

Get the merged eeg.json metadata for a data file.

* **Parameters:**
  **data_filepath** (*str*) – The path to the data file.
* **Returns:**
  The merged eeg.json metadata.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

#### get_all_participants_tsv() → dict[str, dict[str, Any]]

Get all rows from participants.tsv as a dictionary.

* **Returns:**
  A dictionary mapping participant_id to a dict of column values.
  Returns `{}` if no participants.tsv exists or it is empty.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

#### get_bids_file_attribute(attribute: str, data_filepath: str) → Any

Retrieve a specific attribute from BIDS metadata.

* **Parameters:**
  * **attribute** (*str*) – The name of the attribute to retrieve (e.g., “sfreq”, “subject”).
  * **data_filepath** (*str*) – The path to the data file.
* **Returns:**
  The value of the requested attribute, or None if not found.
* **Return type:**
  Any

<!-- !! processed by numpydoc !! -->

#### get_bids_metadata_files(filepath: str | Path, metadata_file_extension: str) → list[Path]

Retrieve all metadata files that apply to a given data file.

Follows the BIDS inheritance principle to find all relevant metadata
files (e.g., `channels.tsv`, `eeg.json`) for a specific recording.

* **Parameters:**
  * **filepath** (*str* *or* *Path*) – The path to the data file.
  * **metadata_file_extension** (*str*) – The extension of the metadata file to search for (e.g., “channels.tsv”).
* **Returns:**
  A list of paths to the matching metadata files.
* **Return type:**
  list of Path

<!-- !! processed by numpydoc !! -->

#### get_files() → list[str]

Get all EEG recording file paths in the BIDS dataset.

* **Returns:**
  A list of file paths for all valid EEG recordings.
* **Return type:**
  list of str

<!-- !! processed by numpydoc !! -->

#### get_orphan_participants() → dict[str, dict[str, Any]]

Get participant rows that have no matching file in the dataset.

Identifies subjects present in `participants.tsv` but with no
corresponding recording file in `self.files`.

* **Returns:**
  A dictionary mapping orphan participant_id to their TSV data.
  Returns `{}` if there are no orphans, no TSV, or no files.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

#### get_relative_bidspath(filepath: str | Path) → str

Get the dataset-relative path for a file.

* **Parameters:**
  **filepath** (*str* *or* *Path*) – The absolute or relative path to a file in the BIDS dataset.
* **Returns:**
  The path relative to the dataset root, prefixed with the dataset name.
  e.g., “ds004477/sub-001/eeg/sub-001_task-PES_eeg.json”
* **Return type:**
  str

<!-- !! processed by numpydoc !! -->

#### num_times(data_filepath: str) → int

Get the number of time points in the recording.

Calculated from `SamplingFrequency` and `RecordingDuration` in the
modality-specific JSON sidecar (e.g., `eeg.json` or `meg.json`).

* **Parameters:**
  **data_filepath** (*str*) – The path to the data file.
* **Returns:**
  The approximate number of time points.
* **Return type:**
  int

<!-- !! processed by numpydoc !! -->

#### subject_participant_tsv(data_filepath: str) → dict[str, Any]

Get the participants.tsv record for a subject.

* **Parameters:**
  **data_filepath** (*str*) – The path to a data file belonging to the subject.
* **Returns:**
  A dictionary of the subject’s information from participants.tsv.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->
