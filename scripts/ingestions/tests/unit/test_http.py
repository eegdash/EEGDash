"""Tests for the shared HTTP client (``_http``). All calls intercepted via respx."""

from __future__ import annotations

import warnings

import httpx
import pytest
import respx

import _http
from _http import (
    build_headers,
    close_client,
    get_client,
    make_authed_client,
    make_retry_client,
    request_json,
    request_response,
    request_text,
)

# A representative external URL — the host doesn't matter, only that
# respx intercepts the request before httpx makes it.
API = "https://example.invalid/api"


@pytest.fixture
def _no_cache_env(monkeypatch):
    """Disable caching and reset the module-level client singleton."""
    monkeypatch.setenv("EEGDASH_HTTP_CACHE", "0")
    monkeypatch.setenv("EEGDASH_HTTP_CACHE_DISABLED", "1")
    close_client()

    _http._client = None
    yield
    _http._client = None
    close_client()


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_200_returns_payload_and_response():
    route = respx.get(API).mock(
        return_value=httpx.Response(200, json={"key": "value", "n": 42})
    )
    payload, response = request_json("GET", API)
    assert payload == {"key": "value", "n": 42}
    assert response is not None
    assert response.status_code == 200
    assert route.call_count == 1


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_404_returns_none_payload_with_response():
    """404 is not in DEFAULT_RETRY_STATUSES; returns (None, response) without raising."""
    route = respx.get(API).mock(return_value=httpx.Response(404))
    payload, response = request_json("GET", API)
    assert payload is None
    assert response is not None
    assert response.status_code == 404
    assert route.call_count == 1, "404 must not be retried"


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_502_retries_then_succeeds_on_3rd_attempt():
    route = respx.get(API).mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    payload, _ = request_json("GET", API, retries=3, backoff_factor=0.0)
    assert payload == {"ok": True}
    assert route.call_count == 3, "expected 2 retries before the 200"


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_persistent_500_returns_none_payload():
    """Exhausted retries return (None, last_5xx_response)."""
    route = respx.get(API).mock(return_value=httpx.Response(503))
    payload, response = request_json("GET", API, retries=2, backoff_factor=0.0)
    assert payload is None
    assert response is not None
    assert response.status_code == 503
    assert route.call_count == 2  # tenacity counts attempts, not retries


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_timeout_returns_none_payload():
    """Persistent timeouts return (None, None); no response object is available."""
    respx.get(API).mock(side_effect=httpx.TimeoutException("upstream"))
    payload, response = request_json("GET", API, retries=2, backoff_factor=0.0)
    assert payload is None
    assert response is None


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_raise_for_request_surfaces_network_error():
    respx.get(API).mock(side_effect=httpx.TimeoutException("upstream"))
    with pytest.raises(httpx.TimeoutException):
        request_json(
            "GET",
            API,
            retries=1,
            backoff_factor=0.0,
            raise_for_request=True,
        )


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_malformed_json_returns_none_payload():
    route = respx.get(API).mock(
        return_value=httpx.Response(
            200,
            content=b"not valid json {{ ... ]]",
            headers={"content-type": "text/plain"},
        )
    )
    payload, response = request_json("GET", API)
    assert payload is None
    assert response is not None
    assert response.status_code == 200
    assert route.call_count == 1


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_retry_disabled_with_retries_zero():
    route = respx.get(API).mock(return_value=httpx.Response(503))
    payload, _ = request_json("GET", API, retries=1, backoff_factor=0.0)
    assert payload is None
    assert route.call_count == 1


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_408_request_timeout_is_retried_by_default():
    """408 is in DEFAULT_RETRY_STATUSES — some upstreams use it instead of 504."""
    route = respx.get(API).mock(
        side_effect=[
            httpx.Response(408),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    payload, _ = request_json("GET", API, retries=2, backoff_factor=0.0)
    assert payload == {"ok": True}
    assert route.call_count == 2, "408 must be in DEFAULT_RETRY_STATUSES"


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_custom_retry_statuses():
    route = respx.get(API).mock(
        side_effect=[
            httpx.Response(418),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    payload, _ = request_json(
        "GET",
        API,
        retries=2,
        backoff_factor=0.0,
        retry_statuses={418},
    )
    assert payload == {"ok": True}
    assert route.call_count == 2


def test_make_retry_client_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        client = make_retry_client("dummy_token")
        client.close()

    deprecation = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "expected DeprecationWarning"
    assert "make_authed_client" in str(deprecation[0].message)


def test_make_retry_client_returns_same_shape_as_authed_client():
    """Deprecation alias produces identical headers and timeout to make_authed_client."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        deprecated_client = make_retry_client("dummy_token")
        new_client = make_authed_client("dummy_token")

    try:
        assert deprecated_client.headers.get("Authorization") == new_client.headers.get(
            "Authorization"
        )
        assert deprecated_client.timeout == new_client.timeout
    finally:
        deprecated_client.close()
        new_client.close()


@pytest.mark.parametrize(
    ("kwargs", "expected_key_values"),
    [
        pytest.param(
            {"accept": "application/json"},
            {"Accept": "application/json"},
            id="includes_accept",
        ),
        pytest.param(
            {},
            {"User-Agent": True},  # sentinel: key present and non-empty
            id="includes_user_agent_by_default",
        ),
        pytest.param(
            {
                "accept": "application/json",
                "extra": {"Authorization": "Bearer token", "X-Custom": "value"},
            },
            {
                "Authorization": "Bearer token",
                "X-Custom": "value",
                "Accept": "application/json",
            },
            id="merges_extra",
        ),
    ],
)
def test_build_headers_matrix(kwargs, expected_key_values):
    """Extra headers merge into defaults; User-Agent is always present."""
    out = build_headers(**kwargs)
    for key, value in expected_key_values.items():
        if value is True:
            assert key in out
            assert out[key]
        else:
            assert out[key] == value


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
@pytest.mark.parametrize(
    ("mock_kwargs", "call_kwargs", "expected_text", "expected_status"),
    [
        pytest.param(
            {"return_value": httpx.Response(200, text="hello")},
            {},
            "hello",
            200,
            id="200_returns_text",
        ),
        pytest.param(
            {"return_value": httpx.Response(200, text="hello world")},
            {"retries": 1, "backoff_factor": 0.0},
            "hello world",
            200,
            id="returns_text_and_response",
        ),
        pytest.param(
            {"return_value": httpx.Response(404, text="not found")},
            {"retries": 1, "backoff_factor": 0.0},
            None,  # caller inspects response, text not asserted
            404,
            id="404_returns_text_with_response",
        ),
        pytest.param(
            {"side_effect": httpx.TimeoutException("upstream")},
            {"retries": 1, "backoff_factor": 0.0},
            None,
            None,  # no response on timeout
            id="timeout_returns_none_pair",
        ),
    ],
)
def test_request_text_status_matrix(
    mock_kwargs, call_kwargs, expected_text, expected_status
):
    """4xx returns (text, response); persistent network failure returns (None, None)."""
    respx.get(API).mock(**mock_kwargs)
    text, response = request_text("GET", API, **call_kwargs)
    if expected_status is None:
        assert text is None
        assert response is None
    else:
        assert response is not None
        assert response.status_code == expected_status
        if expected_text is not None:
            assert text == expected_text


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_raise_for_status_propagates_4xx():
    respx.get(API).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        request_text("GET", API, retries=1, backoff_factor=0.0, raise_for_status=True)


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_raise_for_request_propagates_network():
    respx.get(API).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        request_text("GET", API, retries=1, backoff_factor=0.0, raise_for_request=True)


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_response_returns_raw_response():
    respx.get(API).mock(return_value=httpx.Response(200, text="body"))
    response = request_response("GET", API, retries=1, backoff_factor=0.0)
    assert response is not None
    assert response.status_code == 200
    assert response.text == "body"


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_response_returns_none_on_timeout():
    respx.get(API).mock(side_effect=httpx.TimeoutException("up"))
    response = request_response("GET", API, retries=1, backoff_factor=0.0)
    assert response is None


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_response_post_with_json_body():
    route = respx.post(API).mock(return_value=httpx.Response(200, text="ok"))
    response = request_response(
        "POST",
        API,
        json_body={"key": "value"},
        retries=1,
        backoff_factor=0.0,
    )
    assert response is not None
    assert response.status_code == 200
    assert route.call_count == 1


def test_get_client_returns_singleton():
    close_client()
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2
    close_client()


def test_get_client_recreates_after_close():
    close_client()
    c1 = get_client()
    close_client()
    c2 = get_client()
    assert c1 is not c2
    close_client()


def test_make_authed_client_sets_authorization_header():
    client = make_authed_client("my_token_abc")
    try:
        assert "Authorization" in client.headers
        assert "my_token_abc" in client.headers["Authorization"]
        assert client.headers["Authorization"].startswith("Bearer ")
    finally:
        client.close()


def test_make_retry_client_includes_authorization():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client = make_retry_client("retry_token_xyz")
    try:
        assert "Authorization" in client.headers
        assert "retry_token_xyz" in client.headers["Authorization"]
    finally:
        client.close()


def test_make_retry_client_has_transport_with_retries():
    """make_retry_client sets a non-default transport on the httpx.Client."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client = make_retry_client("token")
    try:
        assert client._transport is not None
    finally:
        client.close()
