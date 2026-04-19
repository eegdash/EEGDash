# eegdash.api

High-level interface to the EEGDash metadata database.

This module provides the main EEGDash class which serves as the primary entry point for
interacting with the EEGDash ecosystem. It offers methods to query, insert, and update
metadata records stored in the EEGDash database via REST API.

<!-- !! processed by numpydoc !! -->

### Classes

| `EEGDash`(\*[, database, api_url, auth_token])   | High-level interface to the EEGDash metadata database.   |
|--------------------------------------------------|----------------------------------------------------------|

### *class* eegdash.api.EEGDash(, database: str = 'eegdash', api_url: str | None = None, auth_token: str | None = None)

Bases: `object`

High-level interface to the EEGDash metadata database.

Provides methods to query, insert, and update metadata records stored in the
EEGDash database via REST API gateway.

For working with collections of recordings as PyTorch datasets, prefer
`EEGDashDataset`.

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
