from unittest.mock import MagicMock, call, patch

import pytest

import eegdash.api as api_module
from eegdash.api import EEGDash


@pytest.fixture
def mocked_client(monkeypatch):
    client = MagicMock()
    get_client = MagicMock(return_value=client)
    monkeypatch.setattr("eegdash.api.get_client", get_client)
    eeg = EEGDash(database="db", api_url="https://api.test", auth_token="token")
    get_client.assert_called_once_with("https://api.test", "db", "token")
    return eeg, client


@pytest.mark.parametrize(
    "limit,skip,expected_forwarded",
    [
        (None, None, {}),
        (10, None, {"limit": 10}),
        (None, 5, {"skip": 5}),
        (10, 5, {"limit": 10, "skip": 5}),
    ],
)
def test_find_builds_query_and_forwards_pagination(
    mocked_client, limit, skip, expected_forwarded
):
    eeg, client = mocked_client
    client.find.return_value = ({"id": 1}, {"id": 2})

    with patch("eegdash.api.merge_query", return_value={"merged": True}) as merge_query:
        result = eeg.find({"dataset": "ds1"}, subject="01", limit=limit, skip=skip)

    assert result == [{"id": 1}, {"id": 2}]
    merge_query.assert_called_once_with(
        {"dataset": "ds1"}, require_query=True, subject="01"
    )
    client.find.assert_called_once_with({"merged": True}, **expected_forwarded)


def test_count_ignores_limit_skip(mocked_client):
    eeg, client = mocked_client
    client.count_documents.return_value = 12

    with patch("eegdash.api.merge_query", return_value={"merged": True}) as merge_query:
        result = eeg.count({"dataset": "ds1"}, subject="01", limit=100, skip=50)

    assert result == 12
    merge_query.assert_called_once_with(
        {"dataset": "ds1"}, require_query=False, subject="01"
    )
    client.count_documents.assert_called_once_with({"merged": True})


@pytest.mark.parametrize("find_one_result,expected", [({"id": 1}, True), (None, False)])
def test_exists_delegates_to_find_one(mocked_client, find_one_result, expected):
    eeg, _ = mocked_client
    with patch.object(eeg, "find_one", return_value=find_one_result) as find_one:
        assert eeg.exists(dataset="ds1") is expected
    find_one.assert_called_once_with(None, dataset="ds1")


@pytest.mark.parametrize(
    "records,insert_one_result,insert_many_result,expected,one_calls,many_calls",
    [
        ({"a": 1}, "id1", 0, 1, 1, 0),
        ([{"a": 1}, {"a": 2}], None, 2, 2, 0, 1),
    ],
)
def test_insert_dispatch(
    mocked_client,
    records,
    insert_one_result,
    insert_many_result,
    expected,
    one_calls,
    many_calls,
):
    eeg, client = mocked_client
    client.insert_one.return_value = insert_one_result
    client.insert_many.return_value = insert_many_result

    assert eeg.insert(records) == expected
    assert client.insert_one.call_count == one_calls
    assert client.insert_many.call_count == many_calls


def test_find_one_and_update_field_use_merged_query(mocked_client):
    eeg, client = mocked_client
    client.find_one.return_value = {"id": 1}
    client.update_many.return_value = (4, 2)

    with patch(
        "eegdash.api.merge_query", side_effect=[{"q": 1}, {"q": 2}]
    ) as merge_query:
        assert eeg.find_one({"dataset": "ds1"}, subject="01") == {"id": 1}
        assert eeg.update_field({"dataset": "ds1"}, subject="01", update={"x": 1}) == (
            4,
            2,
        )

    assert merge_query.call_args_list == [
        call({"dataset": "ds1"}, require_query=True, subject="01"),
        call({"dataset": "ds1"}, require_query=True, subject="01"),
    ]
    client.find_one.assert_called_once_with({"q": 1})
    client.update_many.assert_called_once_with({"q": 2}, {"x": 1})


@pytest.mark.parametrize(
    "method,args,expected_return",
    [
        ("find_datasets", ({"dataset": "ds1"},), [{"dataset": "ds1"}]),
        ("get_dataset", ("ds1",), {"dataset_id": "ds1"}),
        ("update_dataset", ("ds1", {"x": 1}), 1),
    ],
)
def test_simple_passthrough_methods(mocked_client, method, args, expected_return):
    eeg, client = mocked_client
    getattr(client, method).return_value = expected_return

    assert getattr(eeg, method)(*args) == expected_return


def test_module_getattr_invalid_name_raises():
    with pytest.raises(AttributeError, match="has no attribute"):
        api_module.__getattr__("does_not_exist")


def test_search_datasets_no_filters_returns_dataframe(mocked_client):
    """No filters -> query is None and find_datasets is called with limit only."""
    eeg, client = mocked_client
    client.find_datasets.return_value = [
        {
            "dataset_id": "ds002718",
            "name": "Face perception",
            "modality": "eeg",
            "task": "FacePerception",
            "n_subjects": 18,
            "source": "openneuro",
            "license": "CC0",
            "dataset_doi": "10.1038/sdata.2015.1",
        }
    ]

    df = eeg.search_datasets()

    client.find_datasets.assert_called_once_with(None, limit=100)
    assert df is not None
    assert len(df) == 1
    # Summary columns are present even when records omit some keys.
    for col in (
        "dataset_id",
        "name",
        "modality",
        "task",
        "n_subjects",
        "source",
        "license",
        "dataset_doi",
    ):
        assert col in df.columns
    assert df.iloc[0]["dataset_id"] == "ds002718"
    assert df.iloc[0]["task"] == "FacePerception"


def test_search_datasets_combines_filters_into_and_query(mocked_client):
    """Multiple filters compose as $and; modality wraps in $or for casing."""
    eeg, client = mocked_client
    client.find_datasets.return_value = []

    eeg.search_datasets(
        modality="EEG",
        task="rest",
        clinical_group="adhd",
        source="openneuro",
        n_subjects_min=10,
        license="CC0",
        limit=25,
    )

    client.find_datasets.assert_called_once()
    args, kwargs = client.find_datasets.call_args
    query = args[0]
    assert kwargs == {"limit": 25}
    assert "$and" in query
    clauses = query["$and"]
    # modality clause uses $or to handle case variants.
    assert {"$or": [{"modality": "EEG"}, {"modality": "eeg"}]} in clauses
    assert {"task": "rest"} in clauses
    assert {"$or": [{"clinical.group": "adhd"}, {"clinical_group": "adhd"}]} in clauses
    assert {"$or": [{"source": "openneuro"}, {"provider": "openneuro"}]} in clauses
    assert {"n_subjects": {"$gte": 10}} in clauses
    assert {"license": "CC0"} in clauses


def test_search_datasets_handles_legacy_schemas(mocked_client):
    """Records using ``dataset`` or ``provider`` instead of ``dataset_id``/``source``."""
    eeg, client = mocked_client
    client.find_datasets.return_value = [
        {"dataset": "ds_legacy", "provider": "nemar", "name": "legacy entry"},
    ]

    df = eeg.search_datasets(source="nemar")

    assert df.iloc[0]["dataset_id"] == "ds_legacy"
    assert df.iloc[0]["source"] == "nemar"
    # missing fields surface as None, not KeyError.
    assert df.iloc[0]["modality"] is None


def test_search_datasets_returns_empty_dataframe_when_no_match(mocked_client):
    eeg, client = mocked_client
    client.find_datasets.return_value = []

    df = eeg.search_datasets(task="nonexistent_task")

    assert len(df) == 0
    assert list(df.columns) == [
        "dataset_id",
        "name",
        "modality",
        "task",
        "n_subjects",
        "source",
        "license",
        "dataset_doi",
    ]
