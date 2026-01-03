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


def test_find_with_skip_parameter():
    """Test find method with skip parameter."""
    from eegdash.http_api_client import EEGDashAPIClient
    from unittest.mock import patch, MagicMock

    with patch("requests.Session") as MockSession:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": 1}]}
        mock_response.raise_for_status = MagicMock()
        MockSession.return_value.get.return_value = mock_response

        client = EEGDashAPIClient()
        # Use skip parameter (line 73)
        client.find({"test": "value"}, limit=10, skip=5)

        # Verify skip was passed
        call_args = MockSession.return_value.get.call_args
        assert call_args[1]["params"]["skip"] == 5


def test_find_none_returns_empty():
    """Test find_one when no results found."""
    from eegdash.http_api_client import EEGDashAPIClient
    from unittest.mock import patch, MagicMock

    with patch("requests.Session") as MockSession:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        MockSession.return_value.get.return_value = mock_response

        client = EEGDashAPIClient()
        # Lines 104-105: find_one returning None
        result = client.find_one({"nonexistent": "query"})
        assert result is None


def test_find_with_skip_parameter_percent():
    """Test find with skip parameter (line 73)."""
    from eegdash.http_api_client import EEGDashAPIClient
    from unittest.mock import patch, MagicMock

    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": 1}]}
        mock_session.get.return_value = mock_response

        client = EEGDashAPIClient()
        client.find(query={"test": 1}, limit=10, skip=5)

        # Check skip was passed
        call_args = mock_session.get.call_args
        assert call_args[1]["params"]["skip"] == 5


def test_insert_many():
    """Test insert_many method (lines 104-105)."""
    from eegdash.http_api_client import EEGDashAPIClient
    from unittest.mock import patch, MagicMock

    with patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {"insertedCount": 2}
        mock_session.post.return_value = mock_response

        client = EEGDashAPIClient()
        result = client.insert_many([{"a": 1}, {"b": 2}])
        assert result == 2


def test_api_client_session_auth():
    from eegdash.http_api_client import get_client
    from unittest.mock import patch
    import os

    with patch.dict(os.environ, {"EEGDASH_ADMIN_TOKEN": "adm"}):
        client = get_client(auth_token="usr")
        assert client._session.headers["Authorization"] == "Bearer usr"
        assert client._session.headers["X-EEGDASH-TOKEN"] == "adm"


def test_api_pagination():
    from eegdash.http_api_client import get_client
    from unittest.mock import MagicMock

    client = get_client()
    mock_resp_1 = MagicMock()
    mock_resp_1.json.return_value = {"data": [{"id": i} for i in range(1000)]}
    mock_resp_2 = MagicMock()
    mock_resp_2.json.return_value = {"data": [{"id": 1001}]}

    client._session.get = MagicMock(side_effect=[mock_resp_1, mock_resp_2])

    res = client.find(query={})
    assert len(res) == 1001
    assert client._session.get.call_count == 2


def test_api_methods():
    from eegdash.http_api_client import get_client
    from unittest.mock import MagicMock

    client = get_client()
    client._session.post = MagicMock()
    client._session.patch = MagicMock()

    client.insert_one({"a": 1})
    client.insert_many([{"a": 1}])
    client.update_many({"a": 1}, {"b": 2})

    assert client._session.post.call_count == 2
    assert client._session.patch.call_count == 1
