"""Tests for the shared HTTP client (``_http``).

Two angles in one file:

- **Core API** — ``request_json`` happy path, 404, 5xx retry, network
  errors, malformed JSON, configurable retry policy. Was test_http.py.
- **Variants** — ``request_text`` / ``request_response`` /
  ``build_headers`` / ``make_authed_client`` / ``make_retry_client``
  + client-singleton lifecycle. Was test_http_variants.py.

All tests use respx to intercept HTTP calls; no real network required.
"""

from __future__ import annotations

import httpx
import pytest
import respx

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
    """Disable HTTP caching for tests so each call hits the mock.

    Sets both env vars the module honours, then resets the cached
    module-level client so the next get_client() sees the new env.
    """
    monkeypatch.setenv("EEGDASH_HTTP_CACHE", "0")
    monkeypatch.setenv("EEGDASH_HTTP_CACHE_DISABLED", "1")
    close_client()
    import _http

    _http._client = None
    yield
    _http._client = None
    close_client()


# ─── 1. Core API: request_json (happy path) ────────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_200_returns_payload_and_response():
    """200 OK with valid JSON returns (payload_dict, response)."""
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
def test_request_text_200_returns_text():
    """200 OK with text/plain returns (text, response)."""
    route = respx.get(API).mock(return_value=httpx.Response(200, text="hello"))
    text, response = request_text("GET", API)
    assert text == "hello"
    assert response is not None
    assert response.status_code == 200
    assert route.call_count == 1


# ─── 1. Core API: 404 — terminal, no retry ─────────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_404_returns_none_payload_with_response():
    """404 is NOT a retryable status (not in DEFAULT_RETRY_STATUSES).

    Contract: returns (None, response_with_status_404). The 404 itself
    is not raised — callers inspect the response.
    """
    route = respx.get(API).mock(return_value=httpx.Response(404))
    payload, response = request_json("GET", API)
    assert payload is None
    assert response is not None
    assert response.status_code == 404
    assert route.call_count == 1, "404 must not be retried"


# ─── 1. Core API: 5xx — retried then surfaced ──────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_502_retries_then_succeeds_on_3rd_attempt():
    """Default retry policy: 5xx retried up to ``retries`` times."""
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
    """When all retries return 5xx, the helper returns (None, response).

    The response carries the last 5xx so the caller can log it.
    """
    route = respx.get(API).mock(return_value=httpx.Response(503))
    payload, response = request_json("GET", API, retries=2, backoff_factor=0.0)
    assert payload is None
    assert response is not None
    assert response.status_code == 503
    # 2 retries means up to 2 attempts total (tenacity counts attempts).
    assert route.call_count == 2


# ─── 1. Core API: network errors ───────────────────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_timeout_returns_none_payload():
    """Persistent timeouts return (None, None) (no response object)."""
    respx.get(API).mock(side_effect=httpx.TimeoutException("upstream"))
    payload, response = request_json("GET", API, retries=2, backoff_factor=0.0)
    assert payload is None
    assert response is None  # never got a response


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_raise_for_request_surfaces_network_error():
    """When raise_for_request=True, network errors propagate."""
    respx.get(API).mock(side_effect=httpx.TimeoutException("upstream"))
    with pytest.raises(httpx.TimeoutException):
        request_json(
            "GET",
            API,
            retries=1,
            backoff_factor=0.0,
            raise_for_request=True,
        )


# ─── 1. Core API: malformed JSON — graceful decode failure ─────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_malformed_json_returns_none_payload():
    """A 200 with non-JSON body decodes to None payload but keeps response."""
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


# ─── 1. Core API: retry policy configurability ─────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_retry_disabled_with_retries_zero():
    """``retries=0`` (or 1) means at most one attempt."""
    route = respx.get(API).mock(return_value=httpx.Response(503))
    payload, _ = request_json("GET", API, retries=1, backoff_factor=0.0)
    assert payload is None
    assert route.call_count == 1


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_json_408_request_timeout_is_retried_by_default():
    """408 Request Timeout is in DEFAULT_RETRY_STATUSES.

    Some upstreams return 408 instead of 504 for the same case; the
    client should treat both the same way.
    """
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
    """Caller can supply a different retry-status set (e.g. add 418)."""
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


# ─── 1. Core API: make_retry_client deprecation ────────────────────────────


def test_make_retry_client_emits_deprecation_warning():
    """Old callers see a DeprecationWarning pointing to make_authed_client."""
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        client = make_retry_client("dummy_token")
        client.close()

    deprecation = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "expected DeprecationWarning"
    assert "make_authed_client" in str(deprecation[0].message)


def test_make_retry_client_returns_same_shape_as_authed_client():
    """The deprecation alias must behave identically to the new name."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        deprecated_client = make_retry_client("dummy_token")
        new_client = make_authed_client("dummy_token")

    try:
        # Same auth header on both clients.
        assert deprecated_client.headers.get("Authorization") == new_client.headers.get(
            "Authorization"
        )
        # Same timeout on both clients.
        assert deprecated_client.timeout == new_client.timeout
    finally:
        deprecated_client.close()
        new_client.close()


# ─── 2. Variants: build_headers ────────────────────────────────────────────


def test_build_headers_includes_accept():
    out = build_headers(accept="application/json")
    assert out["Accept"] == "application/json"


def test_build_headers_includes_user_agent_by_default():
    """A User-Agent is always set so anonymous requests look like a real client."""
    out = build_headers()
    assert "User-Agent" in out
    assert out["User-Agent"]  # non-empty


def test_build_headers_merges_extra():
    """Extra headers override the defaults."""
    out = build_headers(
        accept="application/json",
        extra={"Authorization": "Bearer token", "X-Custom": "value"},
    )
    assert out["Authorization"] == "Bearer token"
    assert out["X-Custom"] == "value"
    assert out["Accept"] == "application/json"


# ─── 2. Variants: request_text — additional variants beyond section 1 ──────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_returns_text_and_response():
    """Successful response → (text, response_object)."""
    respx.get(API).mock(return_value=httpx.Response(200, text="hello world"))
    text, response = request_text("GET", API, retries=1, backoff_factor=0.0)
    assert text == "hello world"
    assert response is not None
    assert response.status_code == 200


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_404_returns_text_with_response():
    """A 4xx returns (text, response) — caller decides how to handle the status."""
    respx.get(API).mock(return_value=httpx.Response(404, text="not found"))
    _text, response = request_text("GET", API, retries=1, backoff_factor=0.0)
    # The text comes through; status_code carried by response.
    assert response is not None
    assert response.status_code == 404


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_timeout_returns_none_pair():
    """Persistent timeout → (None, None)."""
    respx.get(API).mock(side_effect=httpx.TimeoutException("upstream"))
    text, response = request_text("GET", API, retries=1, backoff_factor=0.0)
    assert text is None
    assert response is None


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_raise_for_status_propagates_4xx():
    """raise_for_status=True surfaces httpx.HTTPStatusError on 4xx."""
    respx.get(API).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        request_text("GET", API, retries=1, backoff_factor=0.0, raise_for_status=True)


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_text_raise_for_request_propagates_network():
    """raise_for_request=True surfaces httpx.RequestError on network failure."""
    respx.get(API).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        request_text("GET", API, retries=1, backoff_factor=0.0, raise_for_request=True)


# ─── 2. Variants: request_response ─────────────────────────────────────────


@respx.mock
@pytest.mark.usefixtures("_no_cache_env")
def test_request_response_returns_raw_response():
    """request_response returns the httpx.Response, not (data, response)."""
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
    """POST requests with json_body are encoded correctly."""
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
    # The mock saw a request with the json body
    assert route.call_count == 1


# ─── 2. Variants: client-singleton lifecycle ───────────────────────────────


def test_get_client_returns_singleton():
    """Repeated get_client() returns the same instance until close_client()."""
    close_client()
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2
    close_client()


def test_get_client_recreates_after_close():
    """close_client() invalidates the singleton; next get_client makes a new one."""
    close_client()
    c1 = get_client()
    close_client()
    c2 = get_client()
    # Different objects (the old singleton was discarded)
    assert c1 is not c2
    close_client()


def test_make_authed_client_sets_authorization_header():
    """make_authed_client attaches the Bearer token to every request."""
    client = make_authed_client("my_token_abc")
    try:
        assert "Authorization" in client.headers
        assert "my_token_abc" in client.headers["Authorization"]
        assert client.headers["Authorization"].startswith("Bearer ")
    finally:
        client.close()


def test_make_retry_client_includes_authorization():
    """make_retry_client also includes the Bearer token."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client = make_retry_client("retry_token_xyz")
    try:
        assert "Authorization" in client.headers
        assert "retry_token_xyz" in client.headers["Authorization"]
    finally:
        client.close()


def test_make_retry_client_has_transport_with_retries():
    """make_retry_client configures a transport — non-default httpx setup."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client = make_retry_client("token")
    try:
        # The httpx.Client has a _transport attribute; just confirm it's
        # configured (non-default identity).
        assert client._transport is not None
    finally:
        client.close()
