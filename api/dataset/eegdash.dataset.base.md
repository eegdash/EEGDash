# eegdash.dataset.base module

Data utilities and dataset classes for EEG data handling.

This module provides core dataset classes for working with EEG data in the EEGDash ecosystem,
including classes for individual recordings and collections of datasets. It integrates with
braindecode for machine learning workflows and handles data loading from both local and remote sources.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.dataset.base.EEGDashRaw(record: dict[str, Any], cache_dir: str, \*\*kwargs)

Bases: `RawDataset`

A single EEG recording dataset.

Represents a single EEG recording, typically hosted on a remote server (like AWS S3)
and cached locally upon first access. This class is a subclass of
`braindecode.datasets.BaseDataset` and can be used with braindecode’s
preprocessing and training pipelines.

* **Parameters:**
  * **record** (*dict*) – A v2 record containing all metadata and storage information.
    Must have schema_version=2 and include storage.base (no default bucket).
  * **cache_dir** (*str*) – The local directory where the data will be cached.
  * **on_error** (*str* *,* *default "raise"*) – 

    How to handle `DataIntegrityError` when accessing `.raw`:
    - `"raise"` (default): propagate the exception.
    - `"warn"`: log the error as a warning and set `.raw` to `None`.
    - `"skip"`: silently set `.raw` to `None`.
  * **\*\*kwargs** – Additional keyword arguments passed to the
    `braindecode.datasets.BaseDataset` constructor.
* **Raises:**
  **ValueError** – If the record is not a valid v2 record or is missing required fields.

<!-- !! processed by numpydoc !! -->

#### *property* raw *: BaseRaw | None*

The MNE Raw object for this recording.

Accessing this property triggers the download and caching of the data
if it has not been accessed before.

Returns `None` when `on_error` is `"warn"` or `"skip"` and
the record could not be loaded due to a
`DataIntegrityError`.

* **Returns:**
  The loaded MNE Raw object, or `None` for skipped records.
* **Return type:**
  mne.io.BaseRaw | None

<!-- !! processed by numpydoc !! -->
