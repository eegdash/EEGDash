# eegdash.local_bids module

Local BIDS discovery helpers.

These utilities support offline workflows (no DB/S3) by discovering BIDS
recordings on the filesystem and returning EEGDash v2 records.

<!-- !! processed by numpydoc !! -->

### eegdash.local_bids.discover_local_bids_records(dataset_root: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), filters: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]

Discover local BIDS recordings and build EEGDash v2 records.

* **Parameters:**
  * **dataset_root** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* *Path*) – Local dataset directory (e.g., `/path/to/ds005509`).
  * **filters** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Filters dict. Must include `'dataset'` and may include BIDS entities
    like `'subject'`, `'session'`, `'task'`, `'run'`, plus
    `'modality'` (default: `'eeg'`).
* **Returns:**
  A list of v2 records, one for each matched recording file.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), Any]]

### Notes

Matching is performed via `mne_bids.find_matching_paths()` using
datatypes/suffixes derived from the `'modality'` filter. The returned
records use `storage.backend='local'` and point `storage.base` at
`dataset_root`.

<!-- !! processed by numpydoc !! -->
