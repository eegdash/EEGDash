"""Offline contract tests: never fetch signals or use the live catalog."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from eegdash.api import EEGDash
from eegdash.http_api_client import EEGDashAPIClient, get_client
from eegdash.nemar_backend import (
    NemarContractError,
    NemarMetadataClient,
    NemarUnsupportedOperation,
)


def page(start=0, size=200, total=201):
    rows = [
        {"dataset_id": f"nm{i:06d}"} for i in range(start, min(start + size, total))
    ]
    return dict(
        datasets=rows, count=len(rows), total_count=total, limit=size, offset=start
    )


def rich():
    return {
        "dataset_id": "nm000132",
        "doc_type": "dataset",
        "schema_version": "0.4.1",
        "source": "nemar",
        "license": "CC-BY-4.0",
        "provenance": {"latest_snapshot": "v1.1.1"},
        "external_links": {"dataset_doi": "concept"},
        "extensions": {
            "nemar": {"versions": [{"version": "v1.1.1", "doi": "version"}]}
        },
        "demographics": {"age_min": 18, "age_max": 30},
        "citation": "upstream citation",
    }


def test_selection(monkeypatch):
    monkeypatch.setenv("EEGDASH_API_URL", "https://example.invalid")
    monkeypatch.setenv("EEGDASH_API_TOKEN", "secret")
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN", "secret")
    assert isinstance(get_client(), EEGDashAPIClient)
    client = EEGDash(backend="nemar")._client
    assert isinstance(client, NemarMetadataClient)
    assert not client._session.trust_env
    assert "Authorization" not in client._session.headers
    assert "X-EEGDASH-TOKEN" not in client._session.headers
    with pytest.raises(ValueError):
        get_client(backend="other")


@pytest.mark.parametrize(
    "options",
    [{"api_url": "https://example.com"}, {"database": "staging"}, {"auth_token": "x"}],
)
def test_incompatible_options(options):
    with pytest.raises(ValueError):
        get_client(backend="nemar", **options)


@pytest.mark.parametrize(
    "query",
    [
        {"license": "CC0"},
        {"dataset_id": {"$in": ["nm000132"]}},
        {"dataset_id": "ds000132"},
        {"$or": []},
        [],
        {"dataset_id": "../../x"},
    ],
)
def test_filters_reject_before_network(query):
    client = NemarMetadataClient()
    client._get = MagicMock()
    with pytest.raises(NemarUnsupportedOperation):
        client.find_datasets(query)
    client._get.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 1001, None, True, 1.2])
def test_limit(limit):
    with pytest.raises(ValueError):
        NemarMetadataClient().find_datasets(limit=limit)


@pytest.mark.parametrize(
    "method",
    [
        "find",
        "find_one",
        "count_documents",
        "insert_one",
        "insert_many",
        "update_many",
        "update_dataset",
        "upsert_many",
    ],
)
def test_unsupported(method):
    with pytest.raises(NemarUnsupportedOperation):
        getattr(NemarMetadataClient(), method)({})


def test_rich_preserves_unknowns_and_citation():
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=rich())
    doc = client.find_datasets({"dataset_id": "nm000132"})[0]
    assert doc["dataset_doi"] == "concept"
    assert doc["version_doi"] == "version"
    assert doc["version"] == "v1.1.1"
    assert doc["license"] == "CC-BY-4.0"
    assert doc["citation"] == "upstream citation"
    assert doc["n_subjects"] is None and doc["modality"] is None
    assert "ages" not in doc and "species" not in doc
    assert doc["demographics"] == {"age_min": 18, "age_max": 30}
    assert doc["metadata_scope"] == "current"
    client._get.return_value = None
    assert client.get_dataset("nm000132") is None


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_id": "nm000133"},
        {"schema_version": "1.0"},
        {"doc_type": "record"},
        {"demographics": [], "provenance": "bad"},
        {"provenance": {"latest_snapshot": "latest"}},
    ],
)
def test_bad_rich(change):
    client = NemarMetadataClient()
    client._get = MagicMock(return_value={**rich(), **change})
    with pytest.raises(NemarContractError):
        client.get_dataset("nm000132")


def test_pagination_and_unknown_catalog():
    client = NemarMetadataClient()
    client._get = MagicMock(side_effect=[page(), page(200, 1)])
    docs = client.find_datasets(limit=201)
    assert len(docs) == 201
    assert client._get.call_args.kwargs["params"] == {"limit": 1, "offset": 200}
    assert docs[0]["n_subjects"] is None
    assert docs[0]["source"] is None
    assert docs[0]["citation"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"count": 0},
        {"offset": 1},
        {"limit": 999},
        {"total_count": None},
        {"datasets": []},
    ],
)
def test_bad_pagination(change):
    client = NemarMetadataClient()
    client._get = MagicMock(return_value={**page(), **change})
    with pytest.raises(NemarContractError):
        client.find_datasets()


def test_duplicate_and_changed_total():
    client = NemarMetadataClient()
    second = page(200, 200)
    second["datasets"][0]["dataset_id"] = "nm000000"
    client._get = MagicMock(side_effect=[page(), second])
    with pytest.raises(NemarContractError):
        client.find_datasets()
    client._get = MagicMock(side_effect=[page(), page(200, 200, 202)])
    with pytest.raises(NemarContractError):
        client.find_datasets()


def response_client(body=b"{}", status=200):
    client = NemarMetadataClient()
    response = MagicMock(status_code=status)
    response.iter_content.return_value = [body]
    response.__enter__.return_value = response
    client._session = MagicMock()
    client._session.get.return_value = response
    return client, response


@pytest.mark.parametrize(
    "body",
    [
        b"invalid",
        b"[]",
        b'{"fallback":true}',
        b'{"degraded":true}',
        b'{"error":"oops"}',
    ],
)
def test_bad_response(body):
    client, _ = response_client(body)
    with pytest.raises(NemarContractError):
        client._get("https://api.nemar.org/datasets")


def test_transport_budgets_and_errors(monkeypatch):
    client, response = response_client(json.dumps(rich()).encode())
    assert client._get("url")["dataset_id"] == "nm000132"
    assert client._session.get.call_args.kwargs == dict(
        params=None, timeout=(5, 15), stream=True, allow_redirects=False
    )
    response.iter_content.return_value = [b"x" * (client.MAX_BYTES + 1)]
    with pytest.raises(NemarContractError):
        client._get("url")
    response.status_code = 302
    with pytest.raises(NemarContractError):
        client._get("url")
    response.status_code = 404
    assert client._get("url", missing_ok=True) is None
    response.status_code = 429
    response.raise_for_status.side_effect = requests.HTTPError("429")
    with pytest.raises(requests.HTTPError):
        client._get("url")
    response.raise_for_status.side_effect = None
    response.status_code = 200
    response.iter_content.return_value = [b"{}"]
    ticks = iter([0, 31])
    monkeypatch.setattr("eegdash.nemar_backend.time.monotonic", lambda: next(ticks))
    with pytest.raises(NemarContractError):
        client._get("url")


def test_high_level_summary_and_rejections():
    api = EEGDash(backend="nemar")
    api._client._get = MagicMock(return_value=page(0, 1, 1))
    assert api.search_datasets(limit=1).iloc[0]["dataset_id"] == "nm000000"
    with pytest.raises(NemarUnsupportedOperation):
        api.search_datasets(modality="eeg")
    with pytest.raises(NemarUnsupportedOperation):
        api.count()
