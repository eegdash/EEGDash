from unittest.mock import MagicMock

import pytest

from eegdash.api import EEGDash


class DummyCollection:
    def __init__(self, docs):
        self._docs = docs
        self.find_one_calls = []

    def find_one(self, query, projection=None):
        self.find_one_calls.append((query, projection))
        for d in self._docs:
            # naive exact match on allowed keys
            if all(d.get(k) == v for k, v in query.items()):
                return {"_id": d.get("_id", "dummy")}
        return None


class DummyMongoManager:
    @staticmethod
    def get_client(conn, is_staging):
        # Return (client, db, collection)
        client = MagicMock(name="client")
        db = MagicMock(name="db")
        # Small dummy docs for testing
        docs = [
            {"_id": 1, "data_name": "ds1_file1", "dataset": "ds1"},
            {"_id": 2, "data_name": "ds2_file2", "dataset": "ds2"},
        ]
        collection = DummyCollection(docs)
        return client, db, collection


@pytest.fixture(autouse=True)
def patch_manager(monkeypatch):
    # Patch the MongoConnectionManager used inside EEGDash
    from eegdash import api as api_module

    monkeypatch.setattr(api_module, "MongoConnectionManager", DummyMongoManager)
    yield


@pytest.fixture
def eegdash_instance():
    return EEGDash(is_public=True)


def test_exist_returns_true_for_matching_data_name(eegdash_instance):
    assert eegdash_instance.exist({"data_name": "ds1_file1"}) is True


def test_exist_returns_false_for_non_matching_data_name(eegdash_instance):
    assert eegdash_instance.exist({"data_name": "nonexistent"}) is False


def test_exist_returns_true_for_matching_dataset(eegdash_instance):
    assert eegdash_instance.exist({"dataset": "ds2"}) is True


def test_exist_invalid_field_raises(eegdash_instance):
    with pytest.raises(ValueError) as exc:
        eegdash_instance.exist({"subject": "01"})
    assert "Unsupported" in str(exc.value)


def test_exist_empty_query_raises(eegdash_instance):
    with pytest.raises(ValueError):
        eegdash_instance.exist({})


def test_exist_non_dict_raises(eegdash_instance):
    with pytest.raises(TypeError):
        eegdash_instance.exist([("dataset", "ds1")])  # type: ignore


def test_exist_uses_minimal_projection(eegdash_instance, monkeypatch):
    # Access the underlying dummy collection to verify projection call
    collection = eegdash_instance.collection
    eegdash_instance.exist({"dataset": "ds1"})
    assert collection.find_one_calls[-1][1] == {"_id": 1}
