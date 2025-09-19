from pathlib import Path

import pytest

from eegdash.paths import get_default_cache_dir


@pytest.fixture(scope="session")
def cache_dir():
    """Provide a shared cache directory for tests that need to cache datasets."""
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    return cache_dir
