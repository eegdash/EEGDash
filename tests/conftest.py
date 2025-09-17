from pathlib import Path

import pytest

from eegdash.paths import get_default_cache_dir


@pytest.fixture(scope="session")
def download_dir(tmp_path_factory) -> Path:
    """Provide a shared download directory for tests that need S3 assets."""
    return tmp_path_factory.mktemp("downloader")


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """Provide a shared cache directory for tests that need to cache datasets."""
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
