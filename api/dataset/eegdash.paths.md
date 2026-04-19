# eegdash.paths module

Path utilities and cache directory management.

This module provides functions for resolving consistent cache directories and path
management throughout the EEGDash package, with integration to MNE-Python’s
configuration system.

<!-- !! processed by numpydoc !! -->

### eegdash.paths.get_default_cache_dir() → Path

Resolve the default cache directory for EEGDash data.

The function determines the cache directory based on the following
priority order:

> 1. The path specified by the `EEGDASH_CACHE_DIR` environment variable.
> 2. A hidden directory named `.eegdash_cache` in the current working
>    : directory.
> 3. The path specified by the `MNE_DATA` configuration in the MNE-Python
>    : config file (fallback).
* **Returns:**
  The resolved, absolute path to the default cache directory.
* **Return type:**
  pathlib.Path

<!-- !! processed by numpydoc !! -->
