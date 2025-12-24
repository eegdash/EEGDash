from unittest.mock import MagicMock, patch

import pytest

from eegdash import EEGDash
from eegdash.bids_eeg_metadata import build_query_from_kwargs


# Mock get_client to prevent actual API connections during tests
@pytest.fixture(autouse=True)
def mock_http_api_connection():
    """Automatically mocks the HTTP API connection for all tests."""
    with patch("eegdash.api.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_mongo_connection(mock_http_api_connection):
    """Alias for the mocked client for tests that need to interact with it."""
    return mock_http_api_connection


@pytest.fixture
def eegdash_instance(mock_http_api_connection):
    """Provides a clean instance of EEGDash for each test."""
    return EEGDash()


def test_build_query_with_single_values():
    """Test 1: Validates that the query builder correctly handles simple
    key-value pairs.
    """
    kwargs = {"dataset": "ds001", "subject": "sub-01"}
    expected_query = {"dataset": "ds001", "subject": "sub-01"}

    query = build_query_from_kwargs(**kwargs)

    assert query == expected_query


def test_build_query_with_list_value():
    """Test 2: Validates that the query builder correctly translates a list
    of values into a MongoDB `$in` operator.
    """
    kwargs = {"dataset": "ds002", "subject": ["sub-01", "sub-02", "sub-03"]}
    expected_query = {
        "dataset": "ds002",
        "subject": {"$in": ["sub-01", "sub-02", "sub-03"]},
    }

    query = build_query_from_kwargs(**kwargs)

    assert query == expected_query


def test_build_query_with_invalid_field():
    """Test 3: Ensures the query builder raises a ValueError when an unsupported
    query field is provided.
    """
    kwargs = {"dataset": "ds003", "invalid_field": "some_value"}

    with pytest.raises(
        ValueError, match="Unsupported query field\\(s\\): invalid_field"
    ):
        build_query_from_kwargs(**kwargs)


def test_find_method_with_kwargs(eegdash_instance, mock_mongo_connection):
    """Test 4: Verifies that the `find` method correctly uses the query builder
    and calls the underlying database collection with the constructed query.
    """
    # Mock the return value of the collection's find method
    mock_mongo_connection.find.return_value = [{"_id": "123", "dataset": "ds004"}]

    # Call the method with user-friendly kwargs
    results = eegdash_instance.find(dataset="ds004", subject=["sub-05", "sub-06"])

    # Define the query we expect to be built and passed to the DB
    expected_db_query = {"dataset": "ds004", "subject": {"$in": ["sub-05", "sub-06"]}}

    # Assert that the collection's find method was called once with the correct query
    mock_mongo_connection.find.assert_called_once_with(expected_db_query)

    # Assert that the method returned the mocked data
    assert len(results) == 1
    assert results[0]["dataset"] == "ds004"


def test_find_method_with_query_and_kwargs_merging(
    eegdash_instance, mock_mongo_connection
):
    """When both a raw query and kwargs are provided, they should be merged with $and."""
    mock_mongo_connection.find.return_value = [{"_id": "xyz", "dataset": "ds010"}]

    raw_query = {"dataset": "ds010", "subject": {"$in": ["sub-01", "sub-02"]}}
    _ = eegdash_instance.find(raw_query, task="RestingState")

    # Expect the final query to be an $and of raw_query and the built kwargs query
    expected_kwargs_query = {"task": "RestingState"}
    # We can't easily inspect the internal builder here, but we can check the structural call
    called_with = mock_mongo_connection.find.call_args[0][0]
    assert "$and" in called_with
    assert raw_query in called_with["$and"]
    assert expected_kwargs_query in called_with["$and"]


def test_find_conflict_on_duplicate_task_raises(
    eegdash_instance, mock_mongo_connection
):
    """If the same field is given in both raw query and kwargs with conflicting values, raise."""
    raw_query = {"task": "RestingState"}
    with pytest.raises(ValueError, match="Conflicting constraints for 'task'"):
        _ = eegdash_instance.find(raw_query, task="DespicableMe")
    mock_mongo_connection.find.assert_not_called()


def test_find_duplicate_task_consistent_ok(eegdash_instance, mock_mongo_connection):
    """No error when duplicate field constraints are compatible (e.g., $in contains scalar)."""
    mock_mongo_connection.find.return_value = []
    raw_query = {"task": {"$in": ["RestingState", "DespicableMe"]}}
    _ = eegdash_instance.find(raw_query, task="RestingState")
    called_with = mock_mongo_connection.find.call_args[0][0]
    assert "$and" in called_with


def test_find_all_documents_with_empty_query(eegdash_instance, mock_mongo_connection):
    mock_mongo_connection.find.return_value = [{"_id": "1"}]
    res = eegdash_instance.find({})
    mock_mongo_connection.find.assert_called_once_with({})
    assert len(res) == 1


def test_build_query_rejects_none_and_empty():
    with pytest.raises(ValueError, match="None for query parameter 'task'"):
        build_query_from_kwargs(task=None)
    with pytest.raises(ValueError, match="empty string for query parameter 'task'"):
        build_query_from_kwargs(task="   ")


def test_build_query_cleans_list_values():
    q = build_query_from_kwargs(
        task=[" A ", None, "B", "", "A"]
    )  # dedupe + strip
    assert q == {"task": {"$in": ["A", "B"]}}
