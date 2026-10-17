# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Path utilities and cache directory management.

This module provides functions for resolving consistent cache directories and path
management throughout the EEGDash package, with integration to MNE-Python's
configuration system.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mne.utils import get_config as mne_get_config
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

_cache_dir_logged = False


def _is_writable_dir(path: Path) -> bool:
    """Return True when *path* is an existing, writable directory."""
    try:
        return path.is_dir() and os.access(path, os.W_OK)
    except OSError:
        return False


def _create_and_check_writable(path: Path) -> bool:
    """Create *path* including parents and return True when it is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK)


def _log_resolved(path: Path) -> Path:
    """Log the resolved cache directory once per process and return it."""
    global _cache_dir_logged
    if not _cache_dir_logged:
        logger.info("EEGDash cache directory: %s", path)
        _cache_dir_logged = True
    return path


def get_default_cache_dir() -> Path:
    """Resolve the default cache directory for EEGDash data.

    The function determines the cache directory based on the following
    priority order:

     1. The path specified by the ``EEGDASH_CACHE_DIR`` environment variable.
     2. An ``eegdash`` directory inside ``$SCRATCH`` when it points to a
        writable directory (common on HPC clusters).
     3. An ``eegdash`` directory inside ``$TMPDIR`` when it points to a
        writable directory.
     4. The path specified by the ``MNE_DATA`` configuration in the
        MNE-Python config file.
     5. The platform user-cache directory
        (:func:`platformdirs.user_cache_dir`).
     6. A hidden directory named ``.eegdash_cache`` in the current working
        directory (last resort).

    The resolved path is logged once per process at first use.

    Returns
    -------
    pathlib.Path
        The resolved, absolute path to the default cache directory.

    """
    # 1) Explicit env var wins
    env_dir = os.environ.get("EEGDASH_CACHE_DIR")
    if env_dir:
        return _log_resolved(Path(env_dir).expanduser().resolve())

    # 2) HPC scratch space if usable
    scratch = os.environ.get("SCRATCH")
    if scratch and _is_writable_dir(Path(scratch).expanduser()):
        return _log_resolved(Path(scratch).expanduser().resolve() / "eegdash")

    # 3) TMPDIR if usable
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and _is_writable_dir(Path(tmpdir).expanduser()):
        return _log_resolved(Path(tmpdir).expanduser().resolve() / "eegdash")

    # 4) MNE's configured data directory
    mne_data = mne_get_config("MNE_DATA")
    if mne_data:
        return _log_resolved(Path(mne_data).expanduser().resolve())

    # 5) Platform user-cache directory
    platform_cache = Path(user_cache_dir("eegdash", appauthor=False))
    if _create_and_check_writable(platform_cache):
        return _log_resolved(platform_cache)

    # 6) Last resort: a hidden folder in the current working directory
    local = Path.cwd() / ".eegdash_cache"
    try:
        local.mkdir(exist_ok=True, parents=True)
    except Exception:
        pass
    return _log_resolved(local)


__all__ = ["get_default_cache_dir"]
