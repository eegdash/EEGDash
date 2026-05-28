# eegdash.testing module

Lazy fetch of the eegdash binary test corpus.

The raw signal fixtures (BDF, EDF, SET, VHDR, SNIRF, FIF, MEF3 …) live in
the separate [eegdash/eegdash-testing-data](https://github.com/eegdash/eegdash-testing-data) repository, modeled
after `mne-testing-data`. The first time a test asks for one we download
the pinned tarball, verify its SHA-256, and unpack into a per-user cache.

Pin (bump both lines when re-tagging the upstream repo):

* `VERSION` — git tag on `eegdash-testing-data`
* `SHA256` — sha256 of the codeload tarball for that tag

## Environment overrides

`EEGDASH_TESTING_DATA_DIR`
: Cache directory (default: `~/.cache/eegdash/testing-data`).

`EEGDASH_SKIP_TESTING_DATA=true`
: Skip every `@requires_testing_data` test; used by air-gapped CI.

### Examples

```pycon
>>> from eegdash.testing import data_path
>>> bdf = data_path() / "eeg" / "sub-001_ses-01_task-meditation_eeg.bdf"
```

<!-- !! processed by numpydoc !! -->

### eegdash.testing.data_file(relpath: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Convenience: `data_path() / relpath` as a single call.

<!-- !! processed by numpydoc !! -->

### eegdash.testing.data_path() → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Return the root of the test corpus, fetching on first use.

* **Returns:**
  The unpacked `eegdash-testing-data-{VERSION}/` directory.
* **Return type:**
  Path
* **Raises:**
  [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError) – If the download is required but `EEGDASH_SKIP_TESTING_DATA=true`
  is set, or if pooch fails to retrieve the tarball.

<!-- !! processed by numpydoc !! -->

### eegdash.testing.has_testing_data() → [bool](https://docs.python.org/3/library/functions.html#bool)

Return True if the corpus is already unpacked in the cache.

<!-- !! processed by numpydoc !! -->

### eegdash.testing.requires_testing_data(func)

Pytest decorator: skip if the corpus is unavailable.

Skips when `EEGDASH_SKIP_TESTING_DATA=true` or the corpus cannot
be fetched (e.g. offline CI without a cache hit). The decorator
triggers the fetch at collection time so tests that depend on the
corpus all share a single download.

<!-- !! processed by numpydoc !! -->
