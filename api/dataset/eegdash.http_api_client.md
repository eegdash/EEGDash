# eegdash.http_api_client module

HTTP API client for EEGDash REST API.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.http_api_client.EEGDashAPIClient(api_url: str | None = None, database: str = 'eegdash', auth_token: str | None = None)

Bases: `object`

HTTP client for EEGDash API.

* **Parameters:**
  * **api_url** (*str* *,* *optional*) – Base API URL. Default: [https://data.eegdash.org](https://data.eegdash.org)
  * **database** (*str* *,* *default "eegdash"*) – Database name (“eegdash”, “eegdash_staging”, or “eegdash_v1”).
  * **auth_token** (*str* *,* *optional*) – Auth token for admin write operations.

<!-- !! processed by numpydoc !! -->

#### count_documents(query: dict[str, Any] | None = None, \*\*kwargs) → int

Count documents matching query.

<!-- !! processed by numpydoc !! -->

#### find(query: dict[str, Any] | None = None, limit: int | None = None, skip: int | None = None, \*\*kwargs) → list[dict[str, Any]]

Query records. Auto-paginates if no limit specified.

<!-- !! processed by numpydoc !! -->

#### find_datasets(query: dict[str, Any] | None = None, limit: int = 1000) → list[dict[str, Any]]

Find datasets matching query.

<!-- !! processed by numpydoc !! -->

#### find_one(query: dict[str, Any] | None = None, \*\*kwargs) → dict[str, Any] | None

Find a single record.

<!-- !! processed by numpydoc !! -->

#### get_dataset(dataset_id: str) → dict[str, Any] | None

Fetch a dataset document by ID.

<!-- !! processed by numpydoc !! -->

#### insert_many(records: list[dict[str, Any]]) → int

Insert multiple records (requires auth).

<!-- !! processed by numpydoc !! -->

#### insert_one(record: dict[str, Any]) → str

Insert single record (requires auth).

<!-- !! processed by numpydoc !! -->

#### update_dataset(dataset_id: str, update: dict[str, Any]) → int

Update dataset metadata (requires auth).

* **Parameters:**
  * **dataset_id** (*str*) – The dataset identifier.
  * **update** (*dict*) – Fields to update (will be wrapped in $set automatically).
* **Returns:**
  Modified count (1 or 0).
* **Return type:**
  int

<!-- !! processed by numpydoc !! -->

#### update_many(query: dict[str, Any], update: dict[str, Any]) → tuple[int, int]

Update records matching query (requires auth).

* **Parameters:**
  * **query** (*dict*) – Filter query to match records.
  * **update** (*dict*) – Fields to set (wrapped in $set automatically).
* **Return type:**
  tuple of (matched_count, modified_count)

<!-- !! processed by numpydoc !! -->

#### upsert_many(records: list[dict[str, Any]]) → dict[str, int]

Upsert multiple records (requires auth).

New endpoint that uses bulk upsert based on dataset+bidspath.

<!-- !! processed by numpydoc !! -->

### eegdash.http_api_client.get_client(api_url: str | None = None, database: str = 'eegdash', auth_token: str | None = None) → EEGDashAPIClient

Get an API client instance.

<!-- !! processed by numpydoc !! -->
