"""Tests for the HTTP API connection manager."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from eegdash.api import EEGDash
from eegdash.http_api_client import HTTPAPIConnectionManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Clean the singleton registry before/after each test."""
    HTTPAPIConnectionManager._instances.clear()
    yield
    HTTPAPIConnectionManager.close_all()
    HTTPAPIConnectionManager._instances.clear()


@pytest.fixture
def http_api_mocks():
    """Mock the HTTP API client creation."""
    with patch("eegdash.http_api_client.HTTPAPIClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.close = MagicMock()

        # Track how many times HTTPAPIClient is constructed
        count = {"count": 0, "clients": []}

        def side_effect(*args, **kwargs):
            count["count"] += 1
            # Create a new mock for each call but track them
            new_client = MagicMock()
            new_client.close = MagicMock()
            # Mock the database/collection chain
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_db.__getitem__ = MagicMock(return_value=mock_collection)
            new_client.__getitem__ = MagicMock(return_value=mock_db)
            count["clients"].append(new_client)
            return new_client

        mock_client_cls.side_effect = side_effect
        yield count


def test_uses_singleton(http_api_mocks):
    """EEGDash instances with same flags share the same underlying connection."""
    e1 = EEGDash(is_staging=False)
    e2 = EEGDash(is_staging=False)

    assert e1._EEGDash__client is e2._EEGDash__client
    assert e1._EEGDash__db is e2._EEGDash__db
    assert e1._EEGDash__collection is e2._EEGDash__collection

    assert len(HTTPAPIConnectionManager._instances) == 1
    assert http_api_mocks["count"] == 1  # only one HTTPAPIClient() constructed


def test_different_staging_flags_create_separate_instances(http_api_mocks):
    """Changing is_staging should produce a distinct singleton entry."""
    _prod = EEGDash(is_staging=False)
    _stg = EEGDash(is_staging=True)

    # Different staging flags should create separate instances
    assert len(HTTPAPIConnectionManager._instances) == 2
    assert http_api_mocks["count"] == 2


def test_close_all_connections_clears_registry(http_api_mocks):
    """EEGDash.close_all_connections() closes clients and clears the registry."""
    inst = EEGDash(is_staging=False)
    client = inst._EEGDash__client

    EEGDash.close_all_connections()

    client.close.assert_called_once()
    assert len(HTTPAPIConnectionManager._instances) == 0


def test_concurrent_creation_shares_singleton(http_api_mocks):
    """Concurrent EEGDash() calls should converge to one singleton instance."""
    results = []

    def make_instance():
        results.append(EEGDash(is_staging=False))

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(make_instance) for _ in range(10)]
        for f in as_completed(futures):
            f.result()

    assert len(results) == 10
    first_client = results[0]._EEGDash__client
    for inst in results[1:]:
        assert inst._EEGDash__client is first_client

    # Requires thread-safe singleton creation in production code (lock + double-check).
    assert len(HTTPAPIConnectionManager._instances) == 1
    assert http_api_mocks["count"] == 1  # only one HTTPAPIClient() constructed
