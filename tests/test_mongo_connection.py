from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from eegdash.api import EEGDash
from eegdash.mongodb import MongoConnectionManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Clean the singleton registry before/after each test."""
    MongoConnectionManager._instances.clear()
    yield
    MongoConnectionManager.close_all()
    MongoConnectionManager._instances.clear()


@pytest.fixture
def mongo_mocks():
    with patch("eegdash.mongodb.MongoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.close = MagicMock()
        mock_client_cls.side_effect = lambda *args, **kwargs: mock_client
        # Track how many times MongoClient is constructed
        count = {"count": 0, "clients": []}

        def side_effect(*args, **kwargs):
            count["count"] += 1
            count["clients"].append(mock_client)
            return mock_client

        mock_client_cls.side_effect = side_effect
        yield count


def test_fields_live_db():
    expected = ["dataset", "subject"]
    eegdash = EEGDash(is_public=True)
    or_query = [{f: {"$exists": False}} for f in expected]
    missing = eegdash.collection.count_documents({"$or": or_query})
    assert missing == 0, f"Missing fields in {missing} records"


def test_uses_singleton(mongo_mocks):
    """EEGDash instances with same flags share the same underlying connection."""
    e1 = EEGDash(is_public=True, is_staging=False)
    e2 = EEGDash(is_public=True, is_staging=False)

    assert e1._EEGDash__client is e2._EEGDash__client
    assert e1._EEGDash__db is e2._EEGDash__db
    assert e1._EEGDash__collection is e2._EEGDash__collection

    assert len(MongoConnectionManager._instances) == 1
    assert mongo_mocks["count"] == 1  # only one MongoClient() constructed


def test_different_staging_flags_use_different_connections(mongo_mocks):
    """Changing is_staging should produce a distinct singleton entry."""
    prod = EEGDash(is_public=True, is_staging=False)
    stg = EEGDash(is_public=True, is_staging=True)

    assert prod._EEGDash__client is stg._EEGDash__client
    assert prod._EEGDash__db is stg._EEGDash__db
    assert prod._EEGDash__collection is stg._EEGDash__collection

    assert len(MongoConnectionManager._instances) == 2
    assert (
        mongo_mocks["count"] == 2
    )  # two distinct MongoClient() calls for the same computer


def test_close_does_not_close_singleton(mongo_mocks):
    """EEGDash.close() should not close the shared Mongo client."""
    e1 = EEGDash(is_public=True, is_staging=False)
    client = e1._EEGDash__client  # grab the underlying client
    e1.close()

    client.close.assert_not_called()

    e2 = EEGDash(is_public=True, is_staging=False)
    assert e2._EEGDash__client is client


def test_close_all_connections_closes_clients(mongo_mocks):
    """EEGDash.close_all_connections() closes clients and clears the registry."""
    inst = EEGDash(is_public=True, is_staging=False)
    client = inst._EEGDash__client

    EEGDash.close_all_connections()

    client.close.assert_called_once()
    assert len(MongoConnectionManager._instances) == 0


def test_concurrent_creation_shares_singleton(mongo_mocks):
    """Concurrent EEGDash() calls should converge to one singleton instance."""
    results = []

    def make_instance():
        results.append(EEGDash(is_public=True, is_staging=False))

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(make_instance) for _ in range(10)]
        for f in as_completed(futures):
            f.result()

    assert len(results) == 10
    first_client = results[0]._EEGDash__client
    for inst in results[1:]:
        assert inst._EEGDash__client is first_client

    # Requires thread-safe singleton creation in production code (lock + double-check).
    assert len(MongoConnectionManager._instances) == 1
    assert mongo_mocks["count"] == 1  # only one MongoClient() constructed
