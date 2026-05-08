# eegdash.dataset.exceptions module

Custom exceptions for EEGDash.

This module defines exceptions used throughout the EEGDash library to provide
informative error messages for common issues.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.dataset.exceptions.DataIntegrityError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, issues: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, authors: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, contact_info: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, source_url: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `EEGDashError`

Raised when a dataset record has known data integrity issues.

This exception is raised when attempting to load a record that has been
flagged during ingestion as having missing companion files or other
integrity problems.

#### record

The problematic record metadata.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### issues

List of specific integrity issues found.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### authors

Dataset authors who can be contacted about the issue.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

#### contact_info

Contact information for reporting the issue.

* **Type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | None

#### source_url

URL to the dataset source for reporting issues.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

### Examples

```pycon
>>> try:
...     dataset.raw  # Attempt to load data
... except DataIntegrityError as e:
...     print(f"Cannot load: {e.issues}")
...     print(f"Contact authors: {e.authors}")
```

<!-- !! processed by numpydoc !! -->

#### *classmethod* from_record(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → DataIntegrityError

Create a DataIntegrityError from a record with integrity issues.

* **Parameters:**
  **record** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Record containing `_data_integrity_issues` and optionally
  `_dataset_authors`, `_dataset_contact`, `_source_url`.
* **Returns:**
  Exception with all relevant context.
* **Return type:**
  DataIntegrityError

<!-- !! processed by numpydoc !! -->

#### log_error() → [None](https://docs.python.org/3/library/constants.html#None)

Log the error using the EEGDash logger with rich formatting.

<!-- !! processed by numpydoc !! -->

#### log_warning() → [None](https://docs.python.org/3/library/constants.html#None)

Log the integrity issues as warnings (non-blocking).

<!-- !! processed by numpydoc !! -->

#### print_rich(console: Console | [None](https://docs.python.org/3/library/constants.html#None) = None) → [None](https://docs.python.org/3/library/constants.html#None)

Print a rich formatted version of the error to the console.

* **Parameters:**
  **console** (*Console* *,* *optional*) – Rich console to print to. If None, creates a new one.

<!-- !! processed by numpydoc !! -->

#### *classmethod* warn_from_record(record: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]) → [None](https://docs.python.org/3/library/constants.html#None)

Log a warning about data integrity issues without raising an exception.

Use this when you want to warn about issues but still allow loading.

* **Parameters:**
  **record** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Record containing `_data_integrity_issues` and optionally
  `_dataset_authors`, `_dataset_contact`, `_source_url`.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.dataset.exceptions.EEGDashError

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

Base exception for all EEGDash errors.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.dataset.exceptions.StorageAccessError(message: [str](https://docs.python.org/3/library/stdtypes.html#str), , dataset_id: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, backend: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, logical_uri: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, cache_path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `EEGDashError`

Raised when a record’s storage backend cannot be reached.

### dataset_id, backend, logical_uri, cache_path

Optional context attached to the error for downstream handlers.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

<!-- !! processed by numpydoc !! -->
