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

    tmp_cache = tmp_path_factory.mktemp("snapshot_cache")
    original_get_cache = snapshot_mod.get_default_cache_dir
    snapshot_mod.get_default_cache_dir = lambda: tmp_cache

    snapshot_mod._INSTANCE_CACHE.clear()
    try:
        yield
    finally:
        snapshot_mod._INSTANCE_CACHE.clear()
        snapshot_mod.get_default_cache_dir = original_get_cache


# --- snapshot test fixtures (lifted from tests/test_dataset_snapshot.py) -----
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from eegdash.dataset.snapshot import DatasetSnapshot


@pytest.fixture
def chart_data_payload():
    """Factory: a minimal well-formed ``/datasets/chart-data`` response."""

    def _make(dataset_id: str = "ds_live_1", *, metadata: dict | None = None) -> dict:
        dataset = {
            "dataset_id": dataset_id,
            "name": f"{dataset_id} dataset",
            "demographics": {"subjects_count": 12},
            "total_files": 30,
            "tasks": ["rest"],
            "sessions": ["s1"],
            "recording_modality": ["eeg"],
            "tags": {"modality": ["visual"]},
            "size_bytes": 1024,
            "source": "openneuro",
        }
        if metadata is not None:
            dataset["metadata"] = metadata
        # Server-shaped docs row (chart-data?include=rows). The client now
        # reads ``d["row"]`` instead of mapping raw fields, so the stub
        # carries the row the way the live server would.
        dataset["row"] = {
            "dataset": dataset_id,
            "dataset_title": f"{dataset_id} dataset",
            "n_subjects": 12,
            "n_records": 30,
            "recording_modality": "eeg",
            "modality of exp": "visual",
            "Type Subject": "Unknown",
            "source": "openneuro",
            "size_bytes": 1024,
        }
        return {
            "success": True,
            "datasets": [dataset],
            "aggregations": {
                "totals": {"datasets": 1, "subjects": 12},
                "modality_counts": {"eeg": 1},
                "source_counts": {"openneuro": 1},
            },
        }

    return _make


@pytest.fixture
def make_urlopen_response():
    """Factory: wrap a payload dict as a context-manager urlopen mock."""

    def _make(payload: dict) -> MagicMock:
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    return _make


@pytest.fixture
def server_manifest():
    """Factory: a minimal well-formed ``/build-manifest`` response."""

    def _make(
        dataset_count: int = 1, schema_version: str = "2.1.0", **overrides
    ) -> dict:
        base = {
            "dataset_count": dataset_count,
            "last_ingested_at": "2026-04-18T16:10:52.827000Z",
            "last_stats_computed_at": "2026-05-10T19:09:03.501782Z",
            "schema_version": schema_version,
            "git_sha": "unknown",
            "name_coverage": 0.03,
        }
        base.update(overrides)
        return base

    return _make


@pytest.fixture
def routing_urlopen(make_urlopen_response):
    """Factory: a urlopen side_effect that selects payloads by URL substring.

    Earliest-matching key wins (insertion order), so put more-specific
    paths first. Raises AssertionError on an unrouted URL.
    """

    def _make(routes: dict[str, dict]):
        def urlopen_side_effect(url, *_, **__):
            for needle, payload in routes.items():
                if needle in url:
                    return make_urlopen_response(payload)
            raise AssertionError(f"unexpected URL: {url}")

        return urlopen_side_effect

    return _make


@pytest.fixture
def gate_snapshot():
    """Factory: a snapshot carrying only the provenance flags the CI gate reads."""

    def _make(*, source="live", dataset_count=1000, api_errors=()) -> DatasetSnapshot:
        rows = pd.DataFrame([{"dataset": f"ds_{i}"} for i in range(dataset_count)])
        return DatasetSnapshot(
            rows=rows,
            aggregations={},
            montages={},
            source=source,
            fetched_at=datetime.now(timezone.utc),
            api_errors=list(api_errors),
        )

    return _make
