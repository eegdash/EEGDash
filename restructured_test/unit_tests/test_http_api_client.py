import os
from unittest.mock import MagicMock, patch

from eegdash.http_api_client import DEFAULT_API_URL, EEGDashAPIClient


def test_client_init_defaults():
    client = EEGDashAPIClient()
    assert client.api_url == DEFAULT_API_URL
    assert client.database == "eegdash"


def test_client_auth_injection():
    # Mock session
    with patch("requests.Session") as mock_session_cls:
        session = mock_session_cls.return_value

        # 1. Constructor auth
        _client = EEGDashAPIClient(auth_token="secret")
        # Session created, headers updated?
        # The code does session.headers.update(...)
        session.headers.update.assert_any_call({"Authorization": "Bearer secret"})

        # 2. Admin token env (for dev/test)
        os.environ["EEGDASH_ADMIN_TOKEN"] = "admin_secret"
        _client2 = EEGDashAPIClient()
        session.headers.update.assert_any_call({"X-EEGDASH-TOKEN": "admin_secret"})
        del os.environ["EEGDASH_ADMIN_TOKEN"]


def test_find_pagination():
    with patch("requests.Session") as mock_session_cls:
        session = mock_session_cls.return_value
        client = EEGDashAPIClient()

        # Mock 2 pages response
        # Page 1
        r1 = MagicMock()
        r1.json.return_value = {"data": [{"id": i} for i in range(1000)]}
        # Page 2
        r2 = MagicMock()
        r2.json.return_value = {"data": [{"id": i} for i in range(1000, 1100)]}

        session.get.side_effect = [r1, r2]

        # Should fetch all
        results = client.find(dataset="ds1")
        assert len(results) == 1100
        assert results[0]["id"] == 0
        assert results[-1]["id"] == 1099

        assert session.get.call_count == 2


def test_find_pagination_break_empty():
    with patch("requests.Session") as mock_session_cls:
        session = mock_session_cls.return_value
        client = EEGDashAPIClient()

        r1 = MagicMock()
        r1.json.return_value = {"data": []}
        session.get.return_value = r1

        results = client.find()
        assert len(results) == 0


def test_insert_methods():
    with patch("requests.Session") as mock_session_cls:
        session = mock_session_cls.return_value
        client = EEGDashAPIClient()

        # Insert One
        r_one = MagicMock()
        r_one.json.return_value = {"insertedId": "abc"}
        session.post.return_value = r_one
        assert client.insert_one({"a": 1}) == "abc"

        # Insert Many
        r_many = MagicMock()
        r_many.json.return_value = {"insertedCount": 5}
        session.post.return_value = r_many
        assert client.insert_many([{"a": 1}]) == 5


def test_update_many():
    with patch("requests.Session") as mock_session_cls:
        session = mock_session_cls.return_value
        client = EEGDashAPIClient()

        r_patch = MagicMock()
        r_patch.json.return_value = {"matched_count": 2, "modified_count": 1}
        session.patch.return_value = r_patch

        m, mod = client.update_many({"a": 1}, {"$set": {"b": 2}})
        assert m == 2
        assert mod == 1
