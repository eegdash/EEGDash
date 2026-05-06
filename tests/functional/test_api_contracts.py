from unittest.mock import MagicMock

import pytest

from eegdash.api import EEGDash


def _make_eegdash(monkeypatch: pytest.MonkeyPatch) -> tuple[EEGDash, MagicMock]:
    client = MagicMock()
    monkeypatch.setattr("eegdash.api.get_client", lambda *_args, **_kwargs: client)
    return EEGDash(), client


@pytest.mark.parametrize(
    ("query", "kwargs", "expected_query", "expected_find_kwargs"),
    [
        (
            None,
            {"dataset": "ds001"},
            {"dataset": "ds001"},
            {},
        ),
        (
            None,
            {"dataset": "ds001", "subject": [" 01 ", None, "02", "", "01"], "limit": 3},
            {"dataset": "ds001", "subject": {"$in": ["01", "02"]}},
            {"limit": 3},
        ),
        (
            {"dataset": "ds001"},
            {"task": "RestingState", "skip": 7},
            {"$and": [{"dataset": "ds001"}, {"task": "RestingState"}]},
            {"skip": 7},
        ),
    ],
)
def test_find_contract_normalizes_and_merges_inputs(
    monkeypatch: pytest.MonkeyPatch,
    query: dict | None,
    kwargs: dict,
    expected_query: dict,
    expected_find_kwargs: dict,
):
    eeg, client = _make_eegdash(monkeypatch)
    client.find.return_value = [{"id": "row-1"}]

    if query is None:
        result = eeg.find(**kwargs)
    else:
        result = eeg.find(query, **kwargs)

    assert result == [{"id": "row-1"}]
    client.find.assert_called_once_with(expected_query, **expected_find_kwargs)


@pytest.mark.parametrize(
    ("method_name", "call_kwargs"),
    [
        ("find", {}),
        ("find_one", {}),
        ("exists", {}),
        ("update_field", {"update": {"task": "new-value"}}),
    ],
)
def test_query_required_contract_raises_meaningful_error(
    monkeypatch: pytest.MonkeyPatch, method_name: str, call_kwargs: dict
):
    eeg, client = _make_eegdash(monkeypatch)

    with pytest.raises(ValueError, match="Query required"):
        getattr(eeg, method_name)(**call_kwargs)

    client.find.assert_not_called()
    client.find_one.assert_not_called()
    client.update_many.assert_not_called()


def test_count_contract_ignores_pagination_kwargs(monkeypatch: pytest.MonkeyPatch):
    eeg, client = _make_eegdash(monkeypatch)
    client.count_documents.return_value = 12

    count = eeg.count(dataset="ds001", limit=2, skip=99)

    assert count == 12
    client.count_documents.assert_called_once_with({"dataset": "ds001"})


@pytest.mark.parametrize(
    ("records", "expected_count", "client_method"),
    [
        ({"dataset": "ds001", "subject": "01"}, 1, "insert_one"),
        ([{"dataset": "ds001"}, {"dataset": "ds001"}], 2, "insert_many"),
    ],
)
def test_insert_contract_switches_single_vs_bulk(
    monkeypatch: pytest.MonkeyPatch,
    records: dict | list[dict],
    expected_count: int,
    client_method: str,
):
    eeg, client = _make_eegdash(monkeypatch)
    client.insert_many.return_value = 2

    inserted = eeg.insert(records)

    assert inserted == expected_count
    assert getattr(client, client_method).call_count == 1
