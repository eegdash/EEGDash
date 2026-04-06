from pathlib import Path

import pytest

from eegdash.paths import get_default_cache_dir


@pytest.fixture(scope="session")
def cache_dir():
    """Provide a shared cache directory for dataset tests.

    Re-exports the same fixture from tests/conftest.py so that tests
    under unit_tests/dataset/ can be collected with ``--noconftest``
    or when run in isolation.
    """
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
