# eegdash.bids_metadata module

BIDS metadata processing and query building utilities.

This module provides functions for building database queries from user parameters
and enriching metadata records with participant information from BIDS datasets.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.attach_participants_extras(raw: [Any](https://docs.python.org/3/library/typing.html#typing.Any), description: [Any](https://docs.python.org/3/library/typing.html#typing.Any), extras: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [None](https://docs.python.org/3/library/constants.html#None)

Attach extra participant data to a raw object and its description.

* **Parameters:**
  * **raw** ([*mne.io.Raw*](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)) – The MNE Raw object to be updated.
  * **description** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* [*pandas.Series*](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html#pandas.Series)) – The description object to be updated.
  * **extras** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Extra participant information to attach.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.build_query_from_kwargs(, allowed_fields: [AbstractSet](https://docs.python.org/3/library/typing.html#typing.AbstractSet)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, field_spec: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Build and validate a MongoDB query from keyword arguments.

Converts user-friendly keyword arguments into a valid MongoDB query
dictionary. Scalar values become exact matches; list-like values
become `$in` queries.

Entity fields (subject, task, session, run) are queried at the top
level since the inject script flattens these from nested entities.

* **Parameters:**
  * **allowed_fields** ([*set*](https://docs.python.org/3/library/stdtypes.html#set) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Override the default `eegdash.const.ALLOWED_QUERY_FIELDS`
    whitelist. Useful when querying a different collection (e.g.
    the `datasets` collection from
    `search_datasets()`).
  * **field_spec** (*mapping* *of* *str to mapping* *,* *optional*) – 

    Per-field rule map describing how a friendly key translates to a
    MongoDB filter. Each rule is a dict with optional keys:
    - `"paths"` (sequence of str): DB field paths the key resolves
      to. When more than one path is given, the rule emits an
      `$or` over `{path: value}` per path. Default: `[<key>]`.
    - `"operator"` (str, e.g. `"$gte"`): wrap the value in
      `{operator: value}` (range operators). Default: exact match.
    - `"value_aliases"` (callable): `v -> list` returns the full
      list of values to OR-match (e.g. `lambda v: [v, v.lower()]`
      for case-insensitive fallback, or `lambda v: [int(v)]` to
      coerce). Default: just `[v]`. Result is deduplicated.

    Keys without a spec follow the legacy scalar / `$in` rules.
  * **\*\*kwargs** – Query filters. Allowed keys are constrained by `allowed_fields`.
* **Returns:**
  A MongoDB query dictionary. Multiple OR-flavoured fields collect
  under a top-level `$and`; single-path single-value fields stay
  flat.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If an unsupported field is provided, or if a value is None/empty.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.enrich_from_participants(bids_root: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), bidspath: [Any](https://docs.python.org/3/library/typing.html#typing.Any), raw: [Any](https://docs.python.org/3/library/typing.html#typing.Any), description: [Any](https://docs.python.org/3/library/typing.html#typing.Any)) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Read participants.tsv and attach extra info for the subject.

* **Parameters:**
  * **bids_root** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *Path*) – Root directory of the BIDS dataset.
  * **bidspath** (*mne_bids.BIDSPath*) – BIDSPath object for the current data file.
  * **raw** ([*mne.io.Raw*](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)) – The MNE Raw object to be updated.
  * **description** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* [*pandas.Series*](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html#pandas.Series)) – The description object to be updated.
* **Returns:**
  The extras that were attached.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.get_entities_from_record(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], entities: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ('subject', 'session', 'run', 'task')) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Get multiple entity values from a record.

* **Parameters:**
  * **record** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A record dictionary.
  * **entities** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Entity names to extract.
* **Returns:**
  Dictionary of entity values (only non-None values included).
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.get_entity_from_record(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], entity: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Get an entity value from a record, supporting both v1 (flat) and v2 (nested) formats.

* **Parameters:**
  * **record** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A record dictionary.
  * **entity** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Entity name (e.g., “subject”, “task”, “session”, “run”).
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

### eegdash.bids_metadata.merge_participants_fields(description: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], participants_row: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None), description_fields: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Merge fields from a participants.tsv row into a description dict.

* **Parameters:**
  * **description** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – The description dictionary to enrich.
  * **participants_row** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* *None*) – A row from participants.tsv. If None, returns description unchanged.
  * **description_fields** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Specific fields to include (matched using normalized keys).
* **Returns:**
  The enriched description dictionary.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.merge_query(query: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, require_query: [bool](https://docs.python.org/3/library/functions.html#bool) = True, \*\*kwargs) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Merge a raw query dict with keyword arguments into a final query.

* **Parameters:**
  * **query** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* *None*) – Raw MongoDB query dictionary. Pass `{}` to match all documents.
  * **require_query** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *default True*) – If True, raise ValueError when no query or kwargs provided.
  * **\*\*kwargs** – User-friendly field filters (converted via `build_query_from_kwargs`).
* **Returns:**
  The merged MongoDB query.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If `require_query=True` and neither query nor kwargs provided,
  or if conflicting constraints are detected.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.normalize_key(key: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [str](https://docs.python.org/3/library/stdtypes.html#str)

Normalize a string key for robust matching.

Converts to lowercase, replaces non-alphanumeric chars with underscores.

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.participants_extras_from_tsv(bids_root: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), subject: [str](https://docs.python.org/3/library/stdtypes.html#str), , id_columns: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ('participant_id', 'participant', 'subject'), na_like: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ('', 'n/a', 'na', 'nan', 'unknown', 'none')) → [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Extract additional participant information from participants.tsv.

* **Parameters:**
  * **bids_root** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *Path*) – Root directory of the BIDS dataset.
  * **subject** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Subject identifier.
  * **id_columns** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Column names treated as identifiers (excluded from output).
  * **na_like** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Values considered as “Not Available” (excluded).
* **Returns:**
  Extra participant information.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.participants_row_for_subject(bids_root: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), subject: [str](https://docs.python.org/3/library/stdtypes.html#str), id_columns: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), ...] = ('participant_id', 'participant', 'subject')) → [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html#pandas.Series) | [None](https://docs.python.org/3/library/constants.html#None)

Load participants.tsv and return the row for a specific subject.

* **Parameters:**
  * **bids_root** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* *Path*) – Root directory of the BIDS dataset.
  * **subject** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Subject identifier (e.g., “01” or “sub-01”).
  * **id_columns** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Column names to search for the subject identifier.
* **Returns:**
  Subject’s data if found, otherwise None.
* **Return type:**
  [pandas.Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html#pandas.Series) or None

<!-- !! processed by numpydoc !! -->

### eegdash.bids_metadata.records_to_dataframe(records: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]], columns: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/library/stdtypes.html#str)], aliases: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)

Project a list of MongoDB JSON records onto a fixed DataFrame layout.

Uses [`pandas.json_normalize()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html#pandas.json_normalize) to flatten one level of nesting (so
dotted alias paths like `"clinical.group"` resolve), then for each
canonical column picks the first non-null value across its alias list.
Records that are not mappings are skipped.

* **Parameters:**
  * **records** (*iterable* *of* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Raw JSON records (e.g., from `EEGDash.find_datasets()`).
  * **columns** (*sequence* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Canonical column names in the order they should appear in the
    returned DataFrame.
  * **aliases** (*mapping* *of* *str to sequence* *of* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – For each canonical column, the ordered list of source field paths
    to look at (back-fill). Dotted paths supported via `json_normalize`.
    When omitted, the canonical column name itself is the only source.
* **Returns:**
  One row per mapping record, with exactly `columns`. Missing
  fields surface as `None`/`NaN`. Empty input returns an empty
  DataFrame with the right column set, so callers get a stable
  schema regardless of result size.
* **Return type:**
  [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame)

<!-- !! processed by numpydoc !! -->
