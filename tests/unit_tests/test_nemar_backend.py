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


@pytest.mark.parametrize("flag", ["truncated", "partial"])
def test_partial_response_is_not_success(flag):
    client, _ = response_client(json.dumps({flag: True}).encode())
    with pytest.raises(NemarContractError):
        client._get("url")


@pytest.mark.parametrize(
    "field", ["provenance", "external_links", "demographics", "extensions"]
)
@pytest.mark.parametrize("value", [[], "", False, 0])
def test_malformed_optional_objects_are_not_unknown(field, value):
    client = NemarMetadataClient()
    client._get = MagicMock(return_value={**rich(), field: value})
    with pytest.raises(NemarContractError):
        client.get_dataset("nm000132")


@pytest.mark.parametrize("version", ["0.4.", "0.4.bad", "0.4.1unexpected"])
def test_schema_version_is_validated(version):
    client = NemarMetadataClient()
    client._get = MagicMock(return_value={**rich(), "schema_version": version})
    with pytest.raises(NemarContractError):
        client.get_dataset("nm000132")


@pytest.mark.parametrize(
    "field,value", [("offset", False), ("limit", True), ("count", True)]
)
def test_page_integer_fields_reject_booleans(field, value):
    client = NemarMetadataClient()
    client._get = MagicMock(return_value={**page(0, 1, 1), field: value})
    with pytest.raises(NemarContractError):
        client.find_datasets(limit=1)


def test_missing_snapshot_does_not_match_unversioned_doi():
    doc = rich()
    doc["provenance"] = {}
    doc["extensions"]["nemar"]["versions"] = [{"doi": "unversioned"}]
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=doc)
    with pytest.raises(NemarContractError):
        client.get_dataset("nm000132")


def test_metadata_includes_acquisition_provenance():
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=rich())
    doc = client.get_dataset("nm000132")
    assert doc["metadata_retrieved_at"].endswith("+00:00")
    client._get.return_value = page(0, 1, 1)
    row = client.find_datasets(limit=1)[0]
    assert row["metadata_retrieved_at"].endswith("+00:00")
    assert row["metadata_url"] == "https://api.nemar.org/datasets"


@pytest.mark.parametrize(
    "query",
    [
        {"license": "CC-BY-4.0"},
        {"demographics.sex_distribution.female": {"$gte": 1}},
        {"has_doi": False},
        {"dataset_id": {"$regex": "nm.*"}},
        {"dataset_id": {"$in": ["nm000001", "ds000002"]}},
        *[{"dataset_id": alias} for alias in ("ds000001", "ds000002", "ds000004")],
    ],
)
def test_independent_query_hazards_reject(query):
    api = EEGDash(backend="nemar")
    api._client._get = MagicMock()
    with pytest.raises(NemarUnsupportedOperation):
        api.find_datasets(query)
    api._client._get.assert_not_called()


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("find", ({},), {}),
        ("find_one", ({},), {}),
        ("exists", ({},), {}),
        ("count", (), {}),
        ("insert", ({},), {}),
        ("insert", ([],), {}),
        ("update_field", ({},), {"update": {}}),
        ("update_dataset", ("nm000001", {}), {}),
        ("find", ({"dataset": "nm000001", "version": "v999.0.0"},), {}),
        ("find", ({"participants": {}},), {}),
    ],
)
def test_all_public_record_and_write_entrypoints(method, args, kwargs):
    api = EEGDash(backend="nemar")
    api._client._get = MagicMock()
    with pytest.raises(NemarUnsupportedOperation):
        getattr(api, method)(*args, **kwargs)
    api._client._get.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"license": "CC-BY-4.0"},
        {"modality": "eeg"},
        {"task": "rest"},
        {"source": "openneuro"},
        {"clinical_group": "healthy"},
        {"n_subjects_min": 0},
    ],
)
def test_friendly_filters_do_not_broaden(kwargs):
    api = EEGDash(backend="nemar")
    api._client._get = MagicMock()
    with pytest.raises(NemarUnsupportedOperation):
        api.search_datasets(**kwargs)
    api._client._get.assert_not_called()


def test_imported_version_provenance_is_not_source_version():
    doc = rich()
    doc.update(dataset_id="on000117", recording_modality=["MEG"], source="nemar")
    doc["provenance"]["latest_snapshot"] = "v1.0.0"
    doc["external_links"]["dataset_doi"] = "10.82901/nemar.on000117"
    doc["extensions"]["nemar"]["versions"] = [
        {"version": "v1.0.0", "doi": "10.82901/nemar.on000117.v1.0.0"}
    ]
    doc["related_identifiers"] = [
        {
            "identifier": "10.18112/openneuro.ds000117.v1.1.0",
            "relation_type": "IsDerivedFrom",
            "identifier_type": "DOI",
        }
    ]
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=doc)
    actual = client.get_dataset("on000117")
    assert actual["version"] == "v1.0.0"
    assert actual["version_doi"].endswith("on000117.v1.0.0")
    assert actual["related_identifiers"] == doc["related_identifiers"]
    assert actual["dataset_doi"] != actual["version_doi"]
    assert actual["modality"] == ["MEG"]
    assert "storage" not in actual


@pytest.mark.parametrize("limit", [1, 200, 205, 1000])
def test_paging_bounds_and_order(limit):
    client = NemarMetadataClient()
    client._get = MagicMock(
        side_effect=lambda url, params: page(params["offset"], params["limit"], 1200)
    )
    docs = client.find_datasets(limit=limit)
    assert [doc["dataset_id"] for doc in docs] == [f"nm{i:06d}" for i in range(limit)]
    assert client._get.call_count == (limit + 199) // 200
    assert all(
        call.kwargs["params"]["limit"] <= 200 for call in client._get.call_args_list
    )


def test_empty_catalog_and_zero_not_unknown():
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=page(0, 1, 0))
    assert client.find_datasets(limit=1) == []
    result = page(0, 2, 2)
    result["datasets"][0].update(subject_count=None, participants=0)
    result["datasets"][1].update(subject_count=0)
    client._get.return_value = result
    docs = client.find_datasets(limit=2)
    assert docs[0]["n_subjects"] is None
    assert docs[1]["n_subjects"] == 0


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout,
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ],
)
def test_transport_failures_propagate(error):
    client, response = response_client()
    response.iter_content.side_effect = error("upstream failed")
    with pytest.raises(error):
        client.find_datasets(limit=1)
    assert client._session.get.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 404, 429, 503])
def test_http_failure_is_not_empty_or_retried(status):
    client, response = response_client(status=status)
    response.headers = {"Retry-After": "2"}
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    with pytest.raises(requests.HTTPError) as caught:
        client.find_datasets(limit=1)
    assert caught.value.response.headers["Retry-After"] == "2"
    assert client._session.get.call_count == 1


def test_exact_404_only_is_missing():
    client, response = response_client(status=404)
    assert client.get_dataset("nm088888") is None
    assert client.find_datasets({"dataset_id": "nm088888"}) == []
    response.status_code = 503
    response.raise_for_status.side_effect = requests.HTTPError("503")
    with pytest.raises(requests.HTTPError):
        client.get_dataset("nm088888")


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        False,
        [{"version": "latest"}],
        [{"version": "v1.0.0"}, {"version": "v1.0.0"}],
    ],
)
def test_malformed_version_inventory(value):
    doc = rich()
    doc["extensions"]["nemar"]["versions"] = value
    client = NemarMetadataClient()
    client._get = MagicMock(return_value=doc)
    with pytest.raises(NemarContractError):
        client.get_dataset("nm000132")


def test_documented_usage(monkeypatch):
    from pathlib import Path

    text = (Path(__file__).parents[2] / "docs" / "nemar_backend.md").read_text()
    snippet = text.split("```python\n", 1)[1].split("```", 1)[0]
    get = MagicMock(side_effect=[page(0, 20, 20), rich(), page(0, 20, 20)])
    monkeypatch.setattr(NemarMetadataClient, "_get", get)
    scope = {}
    exec(snippet, scope)
    assert len(scope["first_page"]) == len(scope["summary"]) == 20
    assert scope["metadata"]["dataset_id"] == "nm000132"
    assert get.call_count == 3


def test_unsupported_keyword_surfaces_do_not_query():
    api = EEGDash(backend="nemar")
    api._client._get = MagicMock()
    with pytest.raises(TypeError):
        api.find_datasets(skip=-1)
    with pytest.raises(TypeError):
        api.get_dataset("nm000132", version="v999.0.0")
    with pytest.raises(TypeError):
        api.search_datasets(q="rest", skip=1)
    for method in ("participants", "aggregate"):
        with pytest.raises(AttributeError):
            getattr(api, method)()
    api._client._get.assert_not_called()
