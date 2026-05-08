# eegdash.http_api_client

HTTP API client for EEGDash REST API.

<!-- !! processed by numpydoc !! -->

### Functions

| `get_client`([api_url, database, auth_token])   | Get an API client instance.   |
|-------------------------------------------------|-------------------------------|

### Classes

| `EEGDashAPIClient`([api_url, database, auth_token])   | HTTP client for EEGDash API.   |
|-------------------------------------------------------|--------------------------------|

### *class* eegdash.http_api_client.EEGDashAPIClient(api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash', auth_token: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

HTTP client for EEGDash API.

* **Parameters:**
  * **api_url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Base API URL. Default: [https://data.eegdash.org](https://data.eegdash.org)
  * **database** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *default "eegdash"*) – Database name (“eegdash”, “eegdash_staging”, or “eegdash_v1”).
  * **auth_token** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Auth token for admin write operations.

<!-- !! processed by numpydoc !! -->

#### find(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, limit: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, skip: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Query records. Auto-paginates if no limit specified.

<!-- !! processed by numpydoc !! -->

#### find_one(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)

Find a single record.

<!-- !! processed by numpydoc !! -->

#### get_dataset(dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None)

Fetch a dataset document by ID.

<!-- !! processed by numpydoc !! -->

#### find_datasets(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, limit: [int](https://docs.python.org/3/library/functions.html#int) = 1000) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Find datasets matching query.

<!-- !! processed by numpydoc !! -->

#### count_documents(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs) → [int](https://docs.python.org/3/library/functions.html#int)

Count documents matching query.

<!-- !! processed by numpydoc !! -->

#### insert_one(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Insert single record (requires auth).

<!-- !! processed by numpydoc !! -->

#### insert_many(records: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]) → [int](https://docs.python.org/3/library/functions.html#int)

Insert multiple records (requires auth).

<!-- !! processed by numpydoc !! -->

#### update_many(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], update: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]

Update records matching query (requires auth).

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Filter query to match records.
  * **update** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Fields to set (wrapped in $set automatically).
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of (matched_count, modified_count)

<!-- !! processed by numpydoc !! -->

#### update_dataset(dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str), update: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [int](https://docs.python.org/3/library/functions.html#int)

Update dataset metadata (requires auth).

* **Parameters:**
  * **dataset_id** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The dataset identifier.
  * **update** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Fields to update (will be wrapped in $set automatically).
* **Returns:**
  Modified count (1 or 0).
* **Return type:**
  [int](https://docs.python.org/3/library/functions.html#int)

<!-- !! processed by numpydoc !! -->

#### upsert_many(records: [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]

Upsert multiple records (requires auth).

New endpoint that uses bulk upsert based on dataset+bidspath.

<!-- !! processed by numpydoc !! -->

### eegdash.http_api_client.get_client(api_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, database: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'eegdash', auth_token: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None) → EEGDashAPIClient

Get an API client instance.

<!-- !! processed by numpydoc !! -->
