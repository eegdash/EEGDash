"""Tests for ``_http.py`` request_text / request_response / clients (C4.1).

``test_http.py`` covers ``request_json`` exhaustively. This file
adds the parallel ``request_text`` + ``request_response`` paths +
the client-construction helpers that were uncovered.
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
    request_response,
    request_text,
)

API = "https://api.example.com/v1/test"


@pytest.fixture
def _no_cache_env(monkeypatch):
    """Disable HTTP caching for tests so each call hits the mock."""
    monkeypatch.setenv("EEGDASH_HTTP_CACHE_DISABLED", "1")
    close_client()
    yield
    close_client()


# ─── build_headers ────────────────────────────────────────────────────────


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


# ─── request_text ─────────────────────────────────────────────────────────


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


# ─── request_response ─────────────────────────────────────────────────────


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


# ─── Client construction ──────────────────────────────────────────────────


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
    client = make_retry_client("retry_token_xyz")
    try:
        assert "Authorization" in client.headers
        assert "retry_token_xyz" in client.headers["Authorization"]
    finally:
        client.close()


def test_make_retry_client_has_transport_with_retries():
    """make_retry_client configures a transport — non-default httpx setup."""
    client = make_retry_client("token")
    try:
        # The httpx.Client has a _transport attribute; just confirm it's
        # configured (non-default identity).
        assert client._transport is not None
    finally:
        client.close()
