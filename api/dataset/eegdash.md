# eegdash package

## Subpackages

* [eegdash.dataset package](eegdash.dataset.md)
  * [Submodules](eegdash.dataset.md#submodules)
    * [eegdash.dataset.base module](eegdash.dataset.base.md)
    * [eegdash.dataset.bids_dataset module](eegdash.dataset.bids_dataset.md)
    * [eegdash.dataset.dataset module](eegdash.dataset.dataset.md)
    * [eegdash.dataset.exceptions module](eegdash.dataset.exceptions.md)
    * [eegdash.dataset.io module](eegdash.dataset.io.md)
    * [eegdash.dataset.registry module](eegdash.dataset.registry.md)
  * [Module contents](eegdash.dataset.md#module-contents)
* [eegdash.hbn package](eegdash.hbn.md)
  * [Submodules](eegdash.hbn.md#submodules)
    * [eegdash.hbn.preprocessing module](eegdash.hbn.preprocessing.md)
    * [eegdash.hbn.windows module](eegdash.hbn.windows.md)
  * [Module contents](eegdash.hbn.md#module-contents)

## Submodules

* [eegdash.api module](eegdash.api.md)
* [eegdash.bids_metadata module](eegdash.bids_metadata.md)
* [eegdash.const module](eegdash.const.md)
* [eegdash.downloader module](eegdash.downloader.md)
* [eegdash.http_api_client module](eegdash.http_api_client.md)
* [eegdash.local_bids module](eegdash.local_bids.md)
* [eegdash.logging module](eegdash.logging.md)
* [eegdash.paths module](eegdash.paths.md)
* [eegdash.schemas module](eegdash.schemas.md)
  * [EEGDash Data Schemas](eegdash.schemas.md#eegdash-data-schemas)
    * [Core Concepts](eegdash.schemas.md#core-concepts)
    * [Usage](eegdash.schemas.md#usage)

## Module contents

EEGDash: A comprehensive platform for EEG data management and analysis.

EEGDash provides a unified interface for accessing, querying, and analyzing large-scale
EEG datasets. It integrates with cloud storage and REST APIs to streamline EEG research
workflows.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.DataIntegrityError(message: str, record: dict[str, Any] | None = None, issues: list[str] | None = None, authors: list[str] | None = None, contact_info: list[str] | None = None, source_url: str | None = None)

Bases: `EEGDashError`

Raised when a dataset record has known data integrity issues.

This exception is raised when attempting to load a record that has been
flagged during ingestion as having missing companion files or other
integrity problems.

#### record

The problematic record metadata.

* **Type:**
  dict

#### issues

List of specific integrity issues found.

* **Type:**
  list[str]

#### authors

Dataset authors who can be contacted about the issue.

* **Type:**
  list[str]

#### contact_info

Contact information for reporting the issue.

* **Type:**
  list[str] | None

#### source_url

URL to the dataset source for reporting issues.

* **Type:**
  str | None

### Examples

```pycon
>>> try:
...     dataset.raw  # Attempt to load data
... except DataIntegrityError as e:
...     print(f"Cannot load: {e.issues}")
...     print(f"Contact authors: {e.authors}")
```

<!-- !! processed by numpydoc !! -->

#### *classmethod* from_record(record: dict[str, Any]) → DataIntegrityError

Create a DataIntegrityError from a record with integrity issues.

* **Parameters:**
  **record** (*dict*) – Record containing `_data_integrity_issues` and optionally
  `_dataset_authors`, `_dataset_contact`, `_source_url`.
* **Returns:**
  Exception with all relevant context.
* **Return type:**
  DataIntegrityError

<!-- !! processed by numpydoc !! -->

#### log_error() → None

Log the error using the EEGDash logger with rich formatting.

<!-- !! processed by numpydoc !! -->

#### log_warning() → None

Log the integrity issues as warnings (non-blocking).

<!-- !! processed by numpydoc !! -->

#### print_rich(console: Console | None = None) → None

Print a rich formatted version of the error to the console.

* **Parameters:**
  **console** (*Console* *,* *optional*) – Rich console to print to. If None, creates a new one.

<!-- !! processed by numpydoc !! -->

#### *classmethod* warn_from_record(record: dict[str, Any]) → None

Log a warning about data integrity issues without raising an exception.

Use this when you want to warn about issues but still allow loading.

* **Parameters:**
  **record** (*dict*) – Record containing `_data_integrity_issues` and optionally
  `_dataset_authors`, `_dataset_contact`, `_source_url`.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.EEGChallengeDataset(release: str, cache_dir: str, mini: bool = True, query: dict | None = None, s3_bucket: str | None = None, \*\*kwargs)

Bases: [`EEGDashDataset`](eegdash.EEGDashDataset.md#eegdash.EEGDashDataset)

A dataset helper for the EEG 2025 Challenge.

This class simplifies access to the EEG 2025 Challenge datasets. It is a
specialized version of `EEGDashDataset` that is
pre-configured for the challenge’s data releases. It automatically maps a
release name (e.g., “R1”) to the corresponding OpenNeuro dataset and handles
the selection of subject subsets (e.g., “mini” release).

* **Parameters:**
  * **release** (*str*) – The name of the challenge release to load. Must be one of the keys in
    `RELEASE_TO_OPENNEURO_DATASET_MAP`
    (e.g., “R1”, “R2”, …, “R11”).
  * **cache_dir** (*str*) – The local directory where the dataset will be downloaded and cached.
  * **mini** (*bool* *,* *default True*) – If True, the dataset is restricted to the official “mini” subset of
    subjects for the specified release. If False, all subjects for the
    release are included.
  * **query** (*dict* *,* *optional*) – An additional MongoDB-style query to apply as a filter. This query is
    combined with the release and subject filters using a logical AND.
    The query must not contain the `dataset` key, as this is determined
    by the `release` parameter.
  * **s3_bucket** (*str* *,* *optional*) – The base S3 bucket URI where the challenge data is stored. Defaults to
    the official challenge bucket.
  * **\*\*kwargs** – Additional keyword arguments that are passed directly to the
    `EEGDashDataset` constructor.
* **Raises:**
  **ValueError** – If the specified `release` is unknown, or if the `query` argument
  contains a `dataset` key. Also raised if `mini` is True and a
  requested subject is not part of the official mini-release subset.

#### SEE ALSO
[`EEGDashDataset`](eegdash.EEGDashDataset.md#eegdash.EEGDashDataset)
: The base class for creating datasets from queries.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.EEGDash(, database: str = 'eegdash', api_url: str | None = None, auth_token: str | None = None)

Bases: `object`

High-level interface to the EEGDash metadata database.

Provides methods to query, insert, and update metadata records stored in the
EEGDash database via REST API gateway.

For working with collections of recordings as PyTorch datasets, prefer
[`EEGDashDataset`](eegdash.EEGDashDataset.md#eegdash.EEGDashDataset).

<!-- !! processed by numpydoc !! -->

Create a new EEGDash client.

* **Parameters:**
  * **database** (*str* *,* *default "eegdash"*) – Name of the MongoDB database to connect to. Common values:
    `"eegdash"` (production), `"eegdash_staging"` (staging),
    `"eegdash_v1"` (legacy archive).
  * **api_url** (*str* *,* *optional*) – Override the default API URL. If not provided, uses the default
    public endpoint or the `EEGDASH_API_URL` environment variable.
  * **auth_token** (*str* *,* *optional*) – Authentication token for admin write operations. Not required for
    public read operations.

### Examples

```pycon
>>> eegdash = EEGDash()  # production
>>> eegdash = EEGDash(database="eegdash_staging")  # staging
>>> records = eegdash.find({"dataset": "ds002718"})
```

<!-- !! processed by numpydoc !! -->

#### count(query: dict[str, Any] = None, , \*\*kwargs) → int

Count documents matching the query.

* **Parameters:**
  * **query** (*dict* *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  Number of matching documents.
* **Return type:**
  int

### Examples

```pycon
>>> eeg = EEGDash()
>>> count = eeg.count({})  # count all
>>> count = eeg.count(dataset="ds002718")  # count by dataset
```

<!-- !! processed by numpydoc !! -->

#### exists(query: dict[str, Any] = None, , \*\*kwargs) → bool

Check if at least one record matches the query.

* **Parameters:**
  * **query** (*dict* *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  True if at least one matching record exists; False otherwise.
* **Return type:**
  bool

### Examples

```pycon
>>> eeg = EEGDash()
>>> eeg.exists(dataset="ds002718")  # check by dataset
>>> eeg.exists({"data_name": "ds002718_sub-001_eeg.set"})  # check by data_name
```

<!-- !! processed by numpydoc !! -->

#### find(query: dict[str, Any] = None, , \*\*kwargs) → list[Mapping[str, Any]]

Find records in the collection.

### Examples

```pycon
>>> from eegdash import EEGDash
>>> eegdash = EEGDash()
>>> eegdash.find({"dataset": "ds002718", "subject": {"$in": ["012", "013"]}})  # pre-built query
>>> eegdash.find(dataset="ds002718", subject="012")  # keyword filters
>>> eegdash.find(dataset="ds002718", subject=["012", "013"])  # sequence -> $in
>>> eegdash.find({})  # fetch all (use with care)
>>> eegdash.find({"dataset": "ds002718"}, subject=["012", "013"])  # combine query + kwargs (AND)
```

* **Parameters:**
  * **query** (*dict* *,* *optional*) – Complete MongoDB query dictionary. This is a positional-only
    argument.
  * **\*\*kwargs** – User-friendly field filters that are converted to a MongoDB query.
    Values can be scalars (e.g., `"sub-01"`) or sequences (translated
    to `$in` queries). Special parameters: `limit` (int) and `skip` (int)
    for pagination.
* **Returns:**
  DB records that match the query.
* **Return type:**
  list of dict

<!-- !! processed by numpydoc !! -->

#### find_datasets(query: dict[str, Any] | None = None, limit: int = 1000) → list[Mapping[str, Any]]

Find datasets matching query.

* **Parameters:**
  * **query** (*dict*) – Filter query.
  * **limit** (*int*) – Max number of datasets to return.
* **Returns:**
  List of dataset metadata documents.
* **Return type:**
  list of dict

<!-- !! processed by numpydoc !! -->

#### find_one(query: dict[str, Any] = None, , \*\*kwargs) → Mapping[str, Any] | None

Find a single record matching the query.

* **Parameters:**
  * **query** (*dict* *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  The first matching record, or None if no match.
* **Return type:**
  dict or None

### Examples

```pycon
>>> eeg = EEGDash()
>>> record = eeg.find_one(data_name="ds002718_sub-001_eeg.set")
```

<!-- !! processed by numpydoc !! -->

#### get_dataset(dataset_id: str) → Mapping[str, Any] | None

Fetch metadata for a specific dataset.

* **Parameters:**
  **dataset_id** (*str*) – The unique identifier of the dataset (e.g., ‘ds002718’).
* **Returns:**
  The dataset metadata document, or None if not found.
* **Return type:**
  dict or None

<!-- !! processed by numpydoc !! -->

#### insert(records: dict[str, Any] | list[dict[str, Any]]) → int

Insert one or more records (requires auth_token).

* **Parameters:**
  **records** (*dict* *or* *list* *of* *dict*) – A single record or list of records to insert.
* **Returns:**
  Number of records inserted.
* **Return type:**
  int

### Examples

```pycon
>>> eeg = EEGDash(auth_token="...")
>>> eeg.insert({"dataset": "ds001", "subject": "01", ...})  # single
>>> eeg.insert([record1, record2, record3])  # batch
```

<!-- !! processed by numpydoc !! -->

#### update_dataset(dataset_id: str, update: dict[str, Any]) → int

Update metadata for a specific dataset (requires auth_token).

* **Parameters:**
  * **dataset_id** (*str*) – The unique identifier of the dataset (e.g., ‘ds002718’).
  * **update** (*dict*) – Dictionary of fields to update.
* **Returns:**
  Number of documents modified (0 or 1).
* **Return type:**
  int

### Examples

```pycon
>>> eeg = EEGDash(auth_token="...")
>>> eeg.update_dataset("ds002718", {"clinical.is_clinical": True})
```

<!-- !! processed by numpydoc !! -->

#### update_field(query: dict[str, Any] = None, , , update: dict[str, Any], \*\*kwargs) → tuple[int, int]

Update fields on records matching the query (requires auth_token).

Use this to add or modify fields across matching records,
e.g., after re-extracting entities with an improved algorithm.

* **Parameters:**
  * **query** (*dict* *,* *optional*) – Filter query to match records. This is a positional-only argument.
  * **update** (*dict*) – Fields to update. Keys are field names, values are new values.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  Number of records matched and actually modified.
* **Return type:**
  tuple of (matched_count, modified_count)

### Examples

```pycon
>>> eeg = EEGDash(auth_token="...")
>>> # Update entities for all records in a dataset
>>> eeg.update_field({"dataset": "ds002718"}, update={"entities": {"subject": "01"}})
>>> # Using kwargs for filter
>>> eeg.update_field(dataset="ds002718", update={"entities": new_entities})
>>> # Combine query + kwargs
>>> eeg.update_field({"dataset": "ds002718"}, subject="01", update={"entities": new_entities})
```

<!-- !! processed by numpydoc !! -->

### *class* eegdash.EEGDashDataset(cache_dir: str | Path, query: dict[str, Any] = None, description_fields: list[str] | None = None, s3_bucket: str | None = None, records: list[dict] | None = None, download: bool = True, n_jobs: int = -1, eeg_dash_instance: Any = None, database: str | None = None, auth_token: str | None = None, on_error: str = 'raise', max_concurrency: int = 20, \*\*kwargs)

Bases: `BaseConcatDataset`

Create a new EEGDashDataset from a given query or local BIDS dataset directory
and dataset name. An EEGDashDataset is pooled collection of EEGDashBaseDataset
instances (individual recordings) and is a subclass of braindecode’s BaseConcatDataset.

### Examples

Basic usage with dataset and subject filtering:

```pycon
>>> from eegdash import EEGDashDataset
>>> dataset = EEGDashDataset(
...     cache_dir="./data",
...     dataset="ds002718",
...     subject="012"
... )
>>> print(f"Number of recordings: {len(dataset)}")
```

Filter by multiple subjects and specific task:

```pycon
>>> subjects = ["012", "013", "014"]
>>> dataset = EEGDashDataset(
...     cache_dir="./data",
...     dataset="ds002718",
...     subject=subjects,
...     task="RestingState"
... )
```

Load and inspect EEG data from recordings:

```pycon
>>> if len(dataset) > 0:
...     recording = dataset[0]
...     raw = recording.load()
...     print(f"Sampling rate: {raw.info['sfreq']} Hz")
...     print(f"Number of channels: {len(raw.ch_names)}")
...     print(f"Duration: {raw.times[-1]:.1f} seconds")
```

Advanced filtering with raw MongoDB queries:

```pycon
>>> from eegdash import EEGDashDataset
>>> query = {
...     "dataset": "ds002718",
...     "subject": {"$in": ["012", "013"]},
...     "task": "RestingState"
... }
>>> dataset = EEGDashDataset(cache_dir="./data", query=query)
```

Working with dataset collections and braindecode integration:

```pycon
>>> # EEGDashDataset is a braindecode BaseConcatDataset
>>> for i, recording in enumerate(dataset):
...     if i >= 2:  # limit output
...         break
...     print(f"Recording {i}: {recording.description}")
...     raw = recording.load()
...     print(f"  Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s")
```

* **Parameters:**
  * **cache_dir** (*str* *|* *Path*) – Directory where data are cached locally.
  * **query** (*dict* *|* *None*) – Raw MongoDB query to filter records. If provided, it is merged with
    keyword filtering arguments (see `**kwargs`) using logical AND.
    You must provide at least a `dataset` (either in `query` or
    as a keyword argument). Only fields in `ALLOWED_QUERY_FIELDS` are
    considered for filtering.
  * **dataset** (*str*) – Dataset identifier (e.g., `"ds002718"`). Required if `query` does
    not already specify a dataset.
  * **task** (*str* *|* *list* *[**str* *]*) – Task name(s) to filter by (e.g., `"RestingState"`).
  * **subject** (*str* *|* *list* *[**str* *]*) – Subject identifier(s) to filter by (e.g., `"NDARCA153NKE"`).
  * **session** (*str* *|* *list* *[**str* *]*) – Session identifier(s) to filter by (e.g., `"1"`).
  * **run** (*str* *|* *list* *[**str* *]*) – Run identifier(s) to filter by (e.g., `"1"`).
  * **description_fields** (*list* *[**str* *]*) – Fields to extract from each record and include in dataset descriptions
    (e.g., “subject”, “session”, “run”, “task”).
  * **s3_bucket** (*str* *|* *None*) – Optional S3 bucket URI (e.g., “s3://mybucket”) to use instead of the
    default OpenNeuro bucket when downloading data files.
  * **records** (*list* *[**dict* *]*  *|* *None*) – Pre-fetched metadata records. If provided, the dataset is constructed
    directly from these records and no MongoDB query is performed.
  * **download** (*bool* *,* *default True*) – If False, load from local BIDS files only. Local data are expected
    under `cache_dir / dataset`; no DB or S3 access is attempted.
  * **n_jobs** (*int*) – Number of parallel jobs to use where applicable (-1 uses all cores).
  * **eeg_dash_instance** (*EEGDash* *|* *None*) – Optional existing EEGDash client to reuse for DB queries. If None,
    a new client is created on demand, not used in the case of no download.
  * **database** (*str* *|* *None*) – Database name to use (e.g., “eegdash”, “eegdash_staging”). If None,
    uses the default database.
  * **auth_token** (*str* *|* *None*) – Authentication token for accessing protected databases. Required for
    staging or admin operations.
  * **max_concurrency** (*int* *,* *default 20*) – Maximum number of parallel S3 transfer connections used when
    downloading data.  Higher values speed up large/multi-file
    downloads but consume more bandwidth.
  * **on_error** (*str* *,* *default "raise"*) – 

    How to handle `DataIntegrityError` when accessing `.raw`
    on individual recordings:
    - `"raise"` (default): propagate the exception.
    - `"warn"`: log the error as a warning and set `.raw` to `None`.
    - `"skip"`: silently set `.raw` to `None`.

    Use [`drop_bad()`](eegdash.EEGDashDataset.md#eegdash.EEGDashDataset.drop_bad) after iteration to remove skipped recordings.
  * **\*\*kwargs** (*dict*) – 

    Additional keyword arguments serving two purposes:
    - Filtering: any keys present in `ALLOWED_QUERY_FIELDS` are treated as
      query filters (e.g., `dataset`, `subject`, `task`, …).
    - Dataset options: remaining keys are forwarded to
      `EEGDashRaw`.

<!-- !! processed by numpydoc !! -->

#### *property* cumulative_sizes *: list[int]*

Recompute cumulative sizes from current dataset lengths.

Overrides the cached version from BaseConcatDataset because individual
dataset lengths can change after lazy raw loading (estimated ntimes
from JSON metadata may differ from actual n_times in the raw file).

<!-- !! processed by numpydoc !! -->

#### download_all(n_jobs: int | None = None) → None

Download missing remote files in parallel.

* **Parameters:**
  **n_jobs** (*int* *|* *None*) – Number of parallel workers to use. If None, defaults to `self.n_jobs`.

<!-- !! processed by numpydoc !! -->

#### drop_bad() → list[dict]

Remove skipped datasets and return their records.

Call after accessing `.raw` on all datasets (e.g. after iteration
or preprocessing) to clean up the dataset list.

* **Returns:**
  Records that were removed because loading failed.
* **Return type:**
  list of dict

<!-- !! processed by numpydoc !! -->

#### drop_short(min_samples: int) → list[dict]

Remove recordings shorter than *min_samples* and return their records.

This is useful when downstream processing (e.g., fixed-length
windowing) requires a minimum number of samples per recording.
Recordings whose `.raw` is `None` (failed to load) are also
dropped.

* **Parameters:**
  **min_samples** (*int*) – Minimum number of time-domain samples a recording must have
  to be kept.
* **Returns:**
  Records that were removed.
* **Return type:**
  list of dict

<!-- !! processed by numpydoc !! -->

#### save(path, overwrite=False)

Save the dataset to disk.

* **Parameters:**
  * **path** (*str* *or* *Path*) – Destination file path.
  * **overwrite** (*bool* *,* *default False*) – If True, overwrite existing file.
* **Return type:**
  None

<!-- !! processed by numpydoc !! -->
