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


@pytest.fixture(autouse=True)
def _reset_dataset_snapshot_cache(tmp_path_factory):
    """Wipe the process-wide :class:`DatasetSnapshot` cache between tests.

    The snapshot module memoises builds keyed by
    ``(api_base, database, limit)`` so production callers don't pay for
    repeated fetches. Tests that mock the network and then call into
    the registry shim would otherwise see *the previous test's*
    mocked data on a cache hit — leading to misleading failures.

    The on-disk parquet cache (``.eegdash_cache/snapshot_*.parquet``)
    has the same staleness risk, so we redirect ``get_default_cache_dir``
    to a per-test temp dir for the snapshot module. Tests that monkey-
    patch the cache directory at the registry layer keep working
    because we resolve the snapshot's path lookup independently.
    """
    import eegdash.dataset.snapshot as snapshot_mod
    from eegdash.dataset.snapshot import _reset_instance_cache_for_testing

    tmp_cache = tmp_path_factory.mktemp("snapshot_cache")
    original_get_cache = snapshot_mod.get_default_cache_dir
    snapshot_mod.get_default_cache_dir = lambda: tmp_cache

    _reset_instance_cache_for_testing()
    try:
        yield
    finally:
        _reset_instance_cache_for_testing()
        snapshot_mod.get_default_cache_dir = original_get_cache
