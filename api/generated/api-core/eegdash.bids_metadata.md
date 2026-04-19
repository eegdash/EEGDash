# eegdash.bids_metadata

BIDS metadata processing and query building utilities.

This module provides functions for building database queries from user parameters
and enriching metadata records with participant information from BIDS datasets.

<!-- !! processed by numpydoc !! -->

### Functions

| `build_query_from_kwargs`(\*\*kwargs)               | Build and validate a MongoDB query from keyword arguments.                            |
|-----------------------------------------------------|---------------------------------------------------------------------------------------|
| `merge_query`([query, require_query])               | Merge a raw query dict with keyword arguments into a final query.                     |
| `normalize_key`(key)                                | Normalize a string key for robust matching.                                           |
| `merge_participants_fields`(description, ...)       | Merge fields from a participants.tsv row into a description dict.                     |
| `participants_row_for_subject`(bids_root, subject)  | Load participants.tsv and return the row for a specific subject.                      |
| `participants_extras_from_tsv`(bids_root, ...)      | Extract additional participant information from participants.tsv.                     |
| `attach_participants_extras`(raw, description, ...) | Attach extra participant data to a raw object and its description.                    |
| `enrich_from_participants`(bids_root, ...)          | Read participants.tsv and attach extra info for the subject.                          |
| `get_entity_from_record`(record, entity)            | Get an entity value from a record, supporting both v1 (flat) and v2 (nested) formats. |
| `get_entities_from_record`(record[, entities])      | Get multiple entity values from a record.                                             |

### eegdash.bids_metadata.build_query_from_kwargs(\*\*kwargs) → dict[str, Any]

Build and validate a MongoDB query from keyword arguments.

Converts user-friendly keyword arguments into a valid MongoDB query dictionary.
Scalar values become exact matches; list-like values become `$in` queries.

Entity fields (subject, task, session, run) are queried at the top level
since the inject script flattens these from nested entities.

* **Parameters:**
  **\*\*kwargs** – Query filters. Allowed keys are in `eegdash.const.ALLOWED_QUERY_FIELDS`.
* **Returns:**
  A MongoDB query dictionary.
* **Return type:**
  dict
* **Raises:**
  **ValueError** – If an unsupported field is provided, or if a value is None/empty.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.merge_query(query: dict[str, Any] | None = None, require_query: bool = True, \*\*kwargs) → dict[str, Any]

Merge a raw query dict with keyword arguments into a final query.

* **Parameters:**
  * **query** (*dict* *or* *None*) – Raw MongoDB query dictionary. Pass `{}` to match all documents.
  * **require_query** (*bool* *,* *default True*) – If True, raise ValueError when no query or kwargs provided.
  * **\*\*kwargs** – User-friendly field filters (converted via `build_query_from_kwargs`).
* **Returns:**
  The merged MongoDB query.
* **Return type:**
  dict
* **Raises:**
  **ValueError** – If `require_query=True` and neither query nor kwargs provided,
  or if conflicting constraints are detected.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.normalize_key(key: str) → str

Normalize a string key for robust matching.

Converts to lowercase, replaces non-alphanumeric chars with underscores.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.merge_participants_fields(description: dict[str, Any], participants_row: dict[str, Any] | None, description_fields: list[str] | None = None) → dict[str, Any]

Merge fields from a participants.tsv row into a description dict.

* **Parameters:**
  * **description** (*dict*) – The description dictionary to enrich.
  * **participants_row** (*dict* *or* *None*) – A row from participants.tsv. If None, returns description unchanged.
  * **description_fields** (*list* *of* *str* *,* *optional*) – Specific fields to include (matched using normalized keys).
* **Returns:**
  The enriched description dictionary.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.participants_row_for_subject(bids_root: str | Path, subject: str, id_columns: tuple[str, ...] = ('participant_id', 'participant', 'subject')) → Series | None

Load participants.tsv and return the row for a specific subject.

* **Parameters:**
  * **bids_root** (*str* *or* *Path*) – Root directory of the BIDS dataset.
  * **subject** (*str*) – Subject identifier (e.g., “01” or “sub-01”).
  * **id_columns** (*tuple* *of* *str*) – Column names to search for the subject identifier.
* **Returns:**
  Subject’s data if found, otherwise None.
* **Return type:**
  pandas.Series or None

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.participants_extras_from_tsv(bids_root: str | Path, subject: str, , id_columns: tuple[str, ...] = ('participant_id', 'participant', 'subject'), na_like: tuple[str, ...] = ('', 'n/a', 'na', 'nan', 'unknown', 'none')) → dict[str, Any]

Extract additional participant information from participants.tsv.

* **Parameters:**
  * **bids_root** (*str* *or* *Path*) – Root directory of the BIDS dataset.
  * **subject** (*str*) – Subject identifier.
  * **id_columns** (*tuple* *of* *str*) – Column names treated as identifiers (excluded from output).
  * **na_like** (*tuple* *of* *str*) – Values considered as “Not Available” (excluded).
* **Returns:**
  Extra participant information.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.attach_participants_extras(raw: Any, description: Any, extras: dict[str, Any]) → None

Attach extra participant data to a raw object and its description.

* **Parameters:**
  * **raw** (*mne.io.Raw*) – The MNE Raw object to be updated.
  * **description** (*dict* *or* *pandas.Series*) – The description object to be updated.
  * **extras** (*dict*) – Extra participant information to attach.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.enrich_from_participants(bids_root: str | Path, bidspath: Any, raw: Any, description: Any) → dict[str, Any]

Read participants.tsv and attach extra info for the subject.

* **Parameters:**
  * **bids_root** (*str* *or* *Path*) – Root directory of the BIDS dataset.
  * **bidspath** (*mne_bids.BIDSPath*) – BIDSPath object for the current data file.
  * **raw** (*mne.io.Raw*) – The MNE Raw object to be updated.
  * **description** (*dict* *or* *pandas.Series*) – The description object to be updated.
* **Returns:**
  The extras that were attached.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.get_entity_from_record(record: dict[str, Any], entity: str) → Any

Get an entity value from a record, supporting both v1 (flat) and v2 (nested) formats.

* **Parameters:**
  * **record** (*dict*) – A record dictionary.
  * **entity** (*str*) – Entity name (e.g., “subject”, “task”, “session”, “run”).
* **Returns:**
  The entity value, or None if not found.
* **Return type:**
  Any

### Examples

```pycon
>>> # v2 record (nested)
>>> rec = {"entities": {"subject": "01", "task": "rest"}}
>>> get_entity_from_record(rec, "subject")
'01'
>>> # v1 record (flat)
>>> rec = {"subject": "01", "task": "rest"}
>>> get_entity_from_record(rec, "subject")
'01'
```

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.get_entities_from_record(record: dict[str, Any], entities: tuple[str, ...] = ('subject', 'session', 'run', 'task')) → dict[str, Any]

Get multiple entity values from a record.

* **Parameters:**
  * **record** (*dict*) – A record dictionary.
  * **entities** (*tuple* *of* *str*) – Entity names to extract.
* **Returns:**
  Dictionary of entity values (only non-None values included).
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->
