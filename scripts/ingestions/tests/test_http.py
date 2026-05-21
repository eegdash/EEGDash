"""Contract tests for the shared ``_http`` retry/transport helper.

The ingestion pipeline depends on stable behaviour from ``_http``:
- HTTP 5xx responses retry up to ``retries`` times then surface or
  return ``None``.
- HTTP 404 returns ``(None, response_with_404)`` without retrying.
- Network errors (timeouts, DNS failures) retry then surface.
- ``request_json`` decodes JSON; malformed JSON returns ``(None, response)``.

Mocked via ``respx`` — no real network access required.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from _http import request_json, request_text

# A representative external URL — the host doesn't matter, only that
# respx intercepts the request before httpx makes it.
API = "https://example.invalid/api"


@pytest.fixture
def _no_cache_env(monkeypatch):
    """Disable the hishel cache transport for predictable timing.

    Without this, retries can collide with cached responses and the
    test becomes time-sensitive.
    """
    monkeypatch.setenv("EEGDASH_HTTP_CACHE", "0")
    # Force a fresh client per test — the module-level cached _client
    # was built under different env.
    import _http

    _http._client = None
    yield
    _http._client = None


# ─── Happy path ────────────────────────────────────────────────────────────


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


# ─── 404 — terminal, no retry ──────────────────────────────────────────────


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


# ─── 5xx — retried then surfaced ───────────────────────────────────────────


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


# ─── Network errors — retried then surfaced ────────────────────────────────


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


# ─── Malformed JSON — graceful decode failure ─────────────────────────────


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


# ─── Retry policy is configurable ─────────────────────────────────────────


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
    """408 Request Timeout is in DEFAULT_RETRY_STATUSES per Phase 9 audit-2 F2.

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
