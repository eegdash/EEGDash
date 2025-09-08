from __future__ import annotations

import os
from pathlib import Path

from mne.utils import get_config as mne_get_config


def get_default_cache_dir() -> Path:
    """Resolve a consistent default cache directory for EEGDash.

    Priority order:
    1) Environment variable ``EEGDASH_CACHE_DIR`` if set.
    2) MNE config ``MNE_DATA`` if set (aligns with tests and ecosystem caches).
    3) ``.eegdash_cache`` under the current working directory.
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
