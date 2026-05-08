# eegdash.api module

High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash database via REST API.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.api.EEGDash(, database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash', api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, auth_token: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

High-level interface to the EEGDash metadata database.

Provides methods to query, insert, and update metadata records stored in the
EEGDash database via REST API gateway.

For working with collections of recordings as PyTorch datasets, prefer
`EEGDashDataset`.

<!-- !! processed by numpydoc !! -->

Create a new EEGDash client.

* **Parameters:**
  * **database** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *default "eegdash"*) – Name of the MongoDB database to connect to. Common values:
    `"eegdash"` (production), `"eegdash_staging"` (staging),
    `"eegdash_v1"` (legacy archive).
  * **api_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Override the default API URL. If not provided, uses the default
    public endpoint or the `EEGDASH_API_URL` environment variable.
  * **auth_token** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Authentication token for admin write operations. Not required for
    public read operations.

### Examples

```pycon
>>> eegdash = EEGDash()  # production
>>> eegdash = EEGDash(database="eegdash_staging")  # staging
>>> records = eegdash.find({"dataset": "ds002718"})
```

<!-- !! processed by numpydoc !! -->

#### count(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None, , \*\*kwargs) → [int](https://docs.python.org/3/library/functions.html#int)

Count documents matching the query.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  Number of matching documents.
* **Return type:**
  [int](https://docs.python.org/3/library/functions.html#int)

### Examples

```pycon
>>> eeg = EEGDash()
>>> count = eeg.count({})  # count all
>>> count = eeg.count(dataset="ds002718")  # count by dataset
```

<!-- !! processed by numpydoc !! -->

#### exists(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None, , \*\*kwargs) → [bool](https://docs.python.org/3/library/functions.html#bool)

Check if at least one record matches the query.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  True if at least one matching record exists; False otherwise.
* **Return type:**
  [bool](https://docs.python.org/3/library/functions.html#bool)

### Examples

```pycon
>>> eeg = EEGDash()
>>> eeg.exists(dataset="ds002718")  # check by dataset
>>> eeg.exists({"data_name": "ds002718_sub-001_eeg.set"})  # check by data_name
```

<!-- !! processed by numpydoc !! -->

#### find(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None, , \*\*kwargs) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

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
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Complete MongoDB query dictionary. This is a positional-only
    argument.
  * **\*\*kwargs** – User-friendly field filters that are converted to a MongoDB query.
    Values can be scalars (e.g., `"sub-01"`) or sequences (translated
    to `$in` queries). Special parameters: `limit` (int) and `skip` (int)
    for pagination.
* **Returns:**
  DB records that match the query.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

#### find_datasets(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, limit: [int](https://docs.python.org/3/library/functions.html#int) = 1000) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Find datasets matching query.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Filter query.
  * **limit** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Max number of datasets to return.
* **Returns:**
  List of dataset metadata documents.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

#### find_one(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None, , \*\*kwargs) → [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)

Find a single record matching the query.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Complete query dictionary. This is a positional-only argument.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  The first matching record, or None if no match.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict) or None

### Examples

```pycon
>>> eeg = EEGDash()
>>> record = eeg.find_one(data_name="ds002718_sub-001_eeg.set")
```

<!-- !! processed by numpydoc !! -->

#### get_dataset(dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)

Fetch metadata for a specific dataset.

* **Parameters:**
  **dataset_id** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The unique identifier of the dataset (e.g., ‘ds002718’).
* **Returns:**
  The dataset metadata document, or None if not found.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict) or None

<!-- !! processed by numpydoc !! -->

#### insert(records: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]) → [int](https://docs.python.org/3/library/functions.html#int)

Insert one or more records (requires auth_token).

* **Parameters:**
  **records** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A single record or list of records to insert.
* **Returns:**
  Number of records inserted.
* **Return type:**
  [int](https://docs.python.org/3/library/functions.html#int)

### Examples

```pycon
>>> eeg = EEGDash(auth_token="...")
>>> eeg.insert({"dataset": "ds001", "subject": "01", ...})  # single
>>> eeg.insert([record1, record2, record3])  # batch
```

<!-- !! processed by numpydoc !! -->

#### search_datasets(, modality: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, task: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, clinical_group: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, source: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, n_subjects_min: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, license: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, limit: [int](https://docs.python.org/3/library/functions.html#int) = 100)

Search the dataset catalogue with friendly keyword filters.

Convenience wrapper around `find_datasets()` that translates a
small set of human-friendly keyword arguments into a MongoDB-style
query and returns a tidy summary [`pandas.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame). This is
the metadata-only entry point used by tutorials such as
`plot_00_first_search`.

* **Parameters:**
  * **modality** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Filter by recording modality (e.g., `"eeg"`, `"meeg"`).
    Matched case-insensitively against the `modality` field.
  * **task** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Filter by BIDS task name (e.g., `"rest"`, `"FacePerception"`).
  * **clinical_group** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Filter by clinical cohort label (e.g., `"healthy"`, `"adhd"`).
    Matched against `clinical.group` (nested) and falls back to the
    flat `clinical_group` field.
  * **source** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Filter by data source (e.g., `"openneuro"`, `"nemar"`,
    `"hbn"`). Matched against `source` and `provider` fields.
  * **n_subjects_min** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Minimum number of subjects in the dataset. Maps to
    `{"n_subjects": {"$gte": n_subjects_min}}`.
  * **license** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Filter by data license (e.g., `"CC0"`, `"CC-BY-4.0"`).
    Matched against the `license` field.
  * **limit** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *default 100*) – Maximum number of datasets to return.
* **Returns:**
  One row per matching dataset with summary columns:
  `dataset_id`, `name`, `modality`, `task`, `n_subjects`,
  `source`, `license`, `dataset_doi`. Missing fields surface
  as `None`. The frame is empty (zero rows) when nothing matches.
* **Return type:**
  [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)

### Notes

`search_datasets` does not download any signal bytes; only small
JSON catalogue documents are transferred. Pair with
[`EEGDashDataset`](eegdash.EEGDashDataset.md#eegdash.EEGDashDataset) once a candidate dataset is chosen.

### Examples

```pycon
>>> client = EEGDash()
>>> df = client.search_datasets(modality="eeg", n_subjects_min=10)
>>> df = client.search_datasets(task="rest", source="openneuro")
```

<!-- !! processed by numpydoc !! -->

#### update_dataset(dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str), update: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [int](https://docs.python.org/3/library/functions.html#int)

Update metadata for a specific dataset (requires auth_token).

* **Parameters:**
  * **dataset_id** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The unique identifier of the dataset (e.g., ‘ds002718’).
  * **update** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Dictionary of fields to update.
* **Returns:**
  Number of documents modified (0 or 1).
* **Return type:**
  [int](https://docs.python.org/3/library/functions.html#int)

### Examples

```pycon
>>> eeg = EEGDash(auth_token="...")
>>> eeg.update_dataset("ds002718", {"clinical.is_clinical": True})
```

<!-- !! processed by numpydoc !! -->

#### update_field(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None, , , update: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], \*\*kwargs) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

Update fields on records matching the query (requires auth_token).

Use this to add or modify fields across matching records,
e.g., after re-extracting entities with an improved algorithm.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Filter query to match records. This is a positional-only argument.
  * **update** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Fields to update. Keys are field names, values are new values.
  * **\*\*kwargs** – User-friendly field filters (same as find()).
* **Returns:**
  Number of records matched and actually modified.
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of (matched_count, modified_count)

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
