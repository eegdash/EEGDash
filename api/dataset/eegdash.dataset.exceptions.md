# eegdash.dataset.exceptions module

Custom exceptions for EEGDash.

This module defines exceptions used throughout the EEGDash library to provide
informative error messages for common issues.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.dataset.exceptions.DataIntegrityError(message: str, record: dict[str, Any] | None = None, issues: list[str] | None = None, authors: list[str] | None = None, contact_info: list[str] | None = None, source_url: str | None = None)

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

### *exception* eegdash.dataset.exceptions.EEGDashError

Bases: `Exception`

Base exception for all EEGDash errors.

<!-- !! processed by numpydoc !! -->

### *exception* eegdash.dataset.exceptions.StorageAccessError(message: str, , dataset_id: str | None = None, backend: str | None = None, logical_uri: str | None = None, cache_path: str | None = None)

Bases: `EEGDashError`

Raised when a record’s storage backend cannot be reached.

### dataset_id, backend, logical_uri, cache_path

Optional context attached to the error for downstream handlers.

* **Type:**
  str | None

<!-- !! processed by numpydoc !! -->
