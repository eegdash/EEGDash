import os
from unittest.mock import MagicMock, patch

import pytest

from eegdash.http_api_client import (
    DEFAULT_API_URL,
    EEGDashAPIClient,
    _make_session,
    _twin_value,
    get_client,
)


def _resp(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


@pytest.mark.parametrize(
    "auth_token,admin_token,expected_header_updates",
    [
        ("user-token", None, [{"Authorization": "Bearer user-token"}]),
        (None, "admin-token", [{"X-EEGDASH-TOKEN": "admin-token"}]),
        (
            "user-token",
            "admin-token",
            [
                {"Authorization": "Bearer user-token"},
                {"X-EEGDASH-TOKEN": "admin-token"},
            ],
        ),
        (None, None, []),
    ],
)
def test_make_session_header_injection(
    auth_token, admin_token, expected_header_updates
):
    with patch("eegdash.http_api_client.requests.Session") as session_cls:
        session = session_cls.return_value
        session.headers = MagicMock()

        with patch.dict(os.environ, {}, clear=False):
            if admin_token is None:
                os.environ.pop("EEGDASH_ADMIN_TOKEN", None)
            else:
                os.environ["EEGDASH_ADMIN_TOKEN"] = admin_token

            _make_session(auth_token)

        assert [
            args.args[0] for args in session.headers.update.call_args_list
        ] == expected_header_updates
        assert [args.args[0] for args in session.mount.call_args_list] == [
            "https://",
            "http://",
        ]


@pytest.mark.parametrize(
    "api_url_arg,env_api_url,env_auth_token,auth_token_arg,expected_url,expected_auth",
    [
        (
            "https://explicit.test/",
            "https://env.test/",
            "env-token",
            None,
            "https://explicit.test",
            "env-token",
        ),
        (None, "https://env.test///", None, None, "https://env.test", None),
        (None, None, None, None, DEFAULT_API_URL, None),
        (None, None, "env-token", "direct-token", DEFAULT_API_URL, "direct-token"),
    ],
)
def test_client_init_url_and_auth_precedence(
    monkeypatch,
    api_url_arg,
    env_api_url,
    env_auth_token,
    auth_token_arg,
    expected_url,
    expected_auth,
):
    if env_api_url is None:
        monkeypatch.delenv("EEGDASH_API_URL", raising=False)
    else:
        monkeypatch.setenv("EEGDASH_API_URL", env_api_url)

    if env_auth_token is None:
        monkeypatch.delenv("EEGDASH_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EEGDASH_API_TOKEN", env_auth_token)

    with patch(
        "eegdash.http_api_client._make_session", return_value=MagicMock()
    ) as make_session:
        client = EEGDashAPIClient(
            api_url=api_url_arg, auth_token=auth_token_arg, database="db"
        )

    assert client.api_url == expected_url
    assert client.database == "db"
    make_session.assert_called_once_with(expected_auth)


@pytest.fixture
def client_and_session():
    with patch("eegdash.http_api_client.requests.Session") as session_cls:
        session = session_cls.return_value
        client = EEGDashAPIClient(api_url="https://api.test")
        yield client, session


@pytest.mark.parametrize(
    "skip,expected_params",
    [
        (0, {"filter": '{"dataset": "ds1"}', "limit": 10}),
        (5, {"filter": '{"dataset": "ds1"}', "skip": 5, "limit": 10}),
    ],
)
def test_find_with_limit_builds_expected_params(
    client_and_session, skip, expected_params
):
    client, session = client_and_session
    response = _resp({"data": [{"id": 1}]})
    session.get.return_value = response

    records = client.find(query={"dataset": "ds1"}, limit=10, skip=skip)

    assert records == [{"id": 1}]
    session.get.assert_called_once_with(
        "https://api.test/api/eegdash/records", params=expected_params, timeout=30
    )


@pytest.mark.parametrize(
    "page_sizes,start_skip,expected_call_skips,expected_total",
    [
        ([1000, 1000, 50], 0, [0, 1000, 2000], 2050),
        ([1000, 0], 25, [25, 1025], 1000),
        ([0], 0, [0], 0),
    ],
)
def test_find_autopagination(
    client_and_session, page_sizes, start_skip, expected_call_skips, expected_total
):
    client, session = client_and_session
    responses = [
        _resp({"data": [{"i": idx} for idx in range(size)]}) for size in page_sizes
    ]
    session.get.side_effect = responses

    records = client.find(query={"k": "v"}, skip=start_skip)

    assert len(records) == expected_total
    observed_skips = [
        kwargs["params"]["skip"] for _, kwargs in session.get.call_args_list
    ]
    assert observed_skips == expected_call_skips


@pytest.mark.parametrize(
    "status_code,payload,expect_none,raises",
    [
        (404, {"data": {"dataset_id": "x"}}, True, False),
        (200, {"data": {"dataset_id": "x"}}, False, False),
        (500, {}, False, True),
    ],
)
def test_get_dataset_status_handling(
    client_and_session, status_code, payload, expect_none, raises
):
    client, session = client_and_session
    response = _resp(payload, status_code=status_code)
    if raises:
        response.raise_for_status.side_effect = RuntimeError("boom")
    session.get.return_value = response

    if raises:
        with pytest.raises(RuntimeError, match="boom"):
            client.get_dataset("ds1")
        return

    result = client.get_dataset("ds1")
    assert (result is None) is expect_none
    if not expect_none:
        assert result == {"dataset_id": "x"}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"data": [{"dataset": "ds1"}]}, [{"dataset": "ds1"}]),
        ([{"dataset": "ds1"}], [{"dataset": "ds1"}]),
    ],
)
def test_find_datasets_supports_dict_or_list_payload(
    client_and_session, payload, expected
):
    client, session = client_and_session
    session.get.return_value = _resp(payload)

    result = client.find_datasets(query={"dataset": "ds1"}, limit=50)

    assert result == expected
    session.get.assert_called_once_with(
        "https://api.test/api/eegdash/datasets",
        params={"limit": 50, "filter": '{"dataset": "ds1"}'},
        timeout=60,
    )


@pytest.mark.parametrize(
    "query,expected_params",
    [
        (None, {}),
        ({"dataset": "ds1"}, {"filter": '{"dataset": "ds1"}'}),
    ],
)
def test_count_documents_optional_filter(client_and_session, query, expected_params):
    client, session = client_and_session
    session.get.return_value = _resp({"count": 7})

    assert client.count_documents(query) == 7
    session.get.assert_called_once_with(
        "https://api.test/api/eegdash/count", params=expected_params, timeout=30
    )


@pytest.mark.parametrize(
    "method_name,http_method,args,response_payload,expected_result,expected_timeout,expected_path",
    [
        (
            "insert_one",
            "post",
            ({"a": 1},),
            {"insertedId": "abc"},
            "abc",
            30,
            "/admin/eegdash/records",
        ),
        (
            "insert_many",
            "post",
            ([{"a": 1}],),
            {"insertedCount": 3},
            3,
            60,
            "/admin/eegdash/records/bulk",
        ),
        (
            "update_many",
            "patch",
            ({"a": 1}, {"b": 2}),
            {"matched_count": 4, "modified_count": 1},
            (4, 1),
            60,
            "/admin/eegdash/records",
        ),
        (
            "update_dataset",
            "patch",
            ("ds1", {"a": 1}),
            {"modified_count": 1},
            1,
            30,
            "/admin/eegdash/datasets/ds1",
        ),
        (
            "upsert_many",
            "post",
            ([{"a": 1}],),
            {"inserted_count": 2, "updated_count": 5},
            {"inserted_count": 2, "updated_count": 5},
            60,
            "/admin/eegdash/records/upsert",
        ),
    ],
)
def test_write_methods(
    client_and_session,
    method_name,
    http_method,
    args,
    response_payload,
    expected_result,
    expected_timeout,
    expected_path,
):
    client, session = client_and_session
    mocked_method = getattr(session, http_method)
    mocked_method.return_value = _resp(response_payload)

    result = getattr(client, method_name)(*args)

    assert result == expected_result
    request_call = mocked_method.call_args
    assert request_call.args[0] == f"https://api.test{expected_path}"
    assert request_call.kwargs["timeout"] == expected_timeout


def test_find_one_and_get_client(client_and_session):
    client, session = client_and_session
    session.get.return_value = _resp({"data": [{"id": "first"}]})
    assert client.find_one({"dataset": "ds1"}) == {"id": "first"}

    session.get.return_value = _resp({"data": []})
    assert client.find_one({"dataset": "missing"}) is None

    wrapped = get_client(api_url="https://api.test", database="db", auth_token="token")
    assert isinstance(wrapped, EEGDashAPIClient)


# --- NEMAR twin fallback -------------------------------------------------
# NEMAR re-hosts OpenNeuro datasets as on<NNNNNN>. Once the ds* twin is retired,
# code naming the OpenNeuro id must keep working -- without rewriting the 71
# ds* datasets that have no twin and still resolve on their own.


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ds005506", "on005506"),
        ("on005506", None),  # already a NEMAR id
        ("nm000281", None),  # NEMAR-native, no OpenNeuro twin
        ("dsABCDEF", None),  # non-numeric body
        (None, None),
        ({"$in": ["ds005506", "nm1"]}, {"$in": ["on005506", "nm1"]}),
        ({"$in": ["on1"]}, None),  # nothing to swap
        ({"$regex": "^ds00"}, None),  # other operator forms left alone
    ],
)
def test_twin_value_mapping(value, expected):
    assert _twin_value(value) == expected


@pytest.mark.parametrize(
    "method,key,kwargs,empty,found,expected",
    [
        (
            "find",
            "dataset",
            {"limit": 10},
            {"data": []},
            {"data": [{"dataset": "on005506"}]},
            [{"dataset": "on005506"}],
        ),
        (
            "find_datasets",
            "dataset_id",
            {},
            {"data": []},
            {"data": [{"dataset_id": "on005506"}]},
            [{"dataset_id": "on005506"}],
        ),
        ("count_documents", "dataset", {}, {"count": 0}, {"count": 42}, 42),
    ],
)
def test_query_methods_retry_against_twin(
    client_and_session, method, key, kwargs, empty, found, expected
):
    client, session = client_and_session
    session.get.side_effect = [_resp(empty), _resp(found)]

    assert getattr(client, method)({key: "ds005506"}, **kwargs) == expected
    assert session.get.call_count == 2
    assert "on005506" in session.get.call_args_list[1].kwargs["params"]["filter"]


def test_no_retry_when_the_id_still_resolves(client_and_session):
    """The 71 OpenNeuro datasets without a NEMAR twin must never be rewritten."""
    client, session = client_and_session
    session.get.side_effect = [_resp({"data": [{"dataset": "ds008730"}]})]

    assert client.find({"dataset": "ds008730"}, limit=10) == [{"dataset": "ds008730"}]
    assert session.get.call_count == 1


def test_get_dataset_falls_back_to_twin_on_404(client_and_session):
    """get_dataset takes the id as a URL path segment, not a query filter."""
    client, session = client_and_session
    session.get.side_effect = [
        _resp(None, status_code=404),
        _resp({"data": {"dataset_id": "on005506"}}),
    ]

    assert client.get_dataset("ds005506") == {"dataset_id": "on005506"}
    assert session.get.call_count == 2
