# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Path utilities and cache directory management.

This module provides functions for resolving consistent cache directories and path
management throughout the EEGDash package, with integration to MNE-Python's
configuration system.
"""

from __future__ import annotations

import os
from pathlib import Path

from mne.utils import get_config as mne_get_config


def get_default_cache_dir() -> Path:
    """Resolve the default cache directory for EEGDash data.

    The function determines the cache directory based on the following
    priority order:
    1. The path specified by the ``EEGDASH_CACHE_DIR`` environment variable.
    2. The path specified by the ``MNE_DATA`` configuration in the MNE-Python
       config file.
    3. A hidden directory named ``.eegdash_cache`` in the current working
       directory.

    Returns
    -------
    pathlib.Path
        The resolved, absolute path to the default cache directory.

    """
    # 1) Explicit env var wins
    env_dir = os.environ.get("EEGDASH_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    # 2) Reuse MNE's data cache location if configured
    mne_data = mne_get_config("MNE_DATA")
    if mne_data:
        return Path(mne_data).expanduser().resolve()

    # 3) Default to a project-local hidden folder
    return Path.cwd() / ".eegdash_cache"


__all__ = ["get_default_cache_dir"]
