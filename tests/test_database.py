from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from eegdash.api import EEGDash, MongoConnectionManager

# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Clean the singleton registry before/after each test."""
    MongoConnectionManager._instances.clear()
    yield
    MongoConnectionManager.close_all()
    MongoConnectionManager._instances.clear()


@pytest.fixture
def mongo_mocks(monkeypatch):
    """Patch mne.utils.get_config and eegdash.api.MongoClient.
    Return a structure tracking how many clients were constructed and references to them.
    """
    # Always return a test URI (so code path doesn't branch on env)
    monkeypatch.setattr(
        "mne.utils.get_config",
        lambda *a, **k: "mongodb://test_connection",
        raising=True,
    )

    created = {"count": 0, "clients": []}  # list of {"client","db","coll"}

    def fake_mongo_client(*args, **kwargs):
        created["count"] += 1
        client = MagicMock(name=f"MongoClient[{created['count']}]")
        db = MagicMock(name=f"DB[{created['count']}]")
        coll = MagicMock(name=f"Coll[{created['count']}]")
        client.__getitem__.return_value = db
        db.__getitem__.return_value = coll
        created["clients"].append({"client": client, "db": db, "coll": coll})
        return client

    monkeypatch.setattr("eegdash.api.MongoClient", fake_mongo_client, raising=True)
    return created


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

    assert prod._EEGDash__client is not stg._EEGDash__client
    assert prod._EEGDash__db is not stg._EEGDash__db
    assert prod._EEGDash__collection is not stg._EEGDash__collection

    assert len(MongoConnectionManager._instances) == 2
    assert mongo_mocks["count"] == 2  # two distinct MongoClient() calls


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
