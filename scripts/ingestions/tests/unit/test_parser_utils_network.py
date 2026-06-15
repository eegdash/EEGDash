"""Tests for ``fetch_bytes_from_s3``, ``head_content_length``, and ``fetch_from_s3``."""

from __future__ import annotations

import httpx
import pytest
import respx

import _parser_utils
from _parser_utils import (
    fetch_bytes_from_s3,
    fetch_from_s3,
    head_content_length,
    reset_http_client_for_testing,
)


@pytest.fixture(autouse=True)
def _fresh_http_client():
    """Each test gets a fresh pooled client so respx hooks attach cleanly."""
    reset_http_client_for_testing()
    yield
    reset_http_client_for_testing()


# ─── fetch_bytes_from_s3 ──────────────────────────────────────────────────


@respx.mock
def test_fetch_bytes_from_s3_returns_requested_range():
    """Happy path: server returns 1024 bytes; fetch returns them."""
    payload = b"x" * 1024
    respx.get("https://s3.example.com/file").mock(
        return_value=httpx.Response(206, content=payload)
    )
    out = fetch_bytes_from_s3("https://s3.example.com/file", max_bytes=1024)
    assert out == payload


@respx.mock
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        pytest.param({"return_value": httpx.Response(404)}, id="http_error_404"),
        pytest.param(
            {"side_effect": httpx.ConnectError("DNS lookup failed")},
            id="connect_error",
        ),
        pytest.param(
            {"side_effect": httpx.TimeoutException("slow")},
            id="timeout",
        ),
    ],
)
def test_fetch_bytes_from_s3_failure_returns_none(mock_kwargs):
    """Any HTTP / network failure surfaces as ``None`` rather than a raise."""
    respx.get("https://s3.example.com/file").mock(**mock_kwargs)
    assert (
        fetch_bytes_from_s3("https://s3.example.com/file", max_bytes=1024, timeout=0.1)
        is None
    )


@respx.mock
def test_fetch_bytes_caps_response_at_max_bytes_when_server_ignores_range():
    """Response is capped at ``max_bytes`` even when the server returns the full body."""
    big_payload = b"x" * 100_000  # server returned 100 KB
    respx.get("https://lazy-server.example.com/file").mock(
        return_value=httpx.Response(200, content=big_payload)
    )
    out = fetch_bytes_from_s3("https://lazy-server.example.com/file", max_bytes=1024)
    assert out is not None
    assert len(out) == 1024


@respx.mock
def test_fetch_bytes_sends_correct_range_header():
    """Verify the Range header is built correctly (bytes=start-end inclusive)."""
    route = respx.get("https://s3.example.com/file").mock(
        return_value=httpx.Response(206, content=b"y" * 256)
    )
    fetch_bytes_from_s3("https://s3.example.com/file", start=128, max_bytes=256)
    assert route.called
    sent = route.calls.last.request
    # bytes=128-(128+256-1) → bytes=128-383
    assert sent.headers["Range"] == "bytes=128-383"


# ─── head_content_length ──────────────────────────────────────────────────


@respx.mock
def test_head_content_length_parses_int():
    """Content-Length header returned as int."""
    respx.head("https://s3.example.com/file").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "12345"})
    )
    assert head_content_length("https://s3.example.com/file") == 12345


@respx.mock
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        # No Content-Length header
        pytest.param(
            {"return_value": httpx.Response(200, headers={})},
            id="header_missing",
        ),
        # 4xx response
        pytest.param(
            {"return_value": httpx.Response(404)},
            id="http_error",
        ),
        # Non-numeric Content-Length
        pytest.param(
            {
                "return_value": httpx.Response(
                    200, headers={"Content-Length": "not_a_number"}
                )
            },
            id="non_numeric_value",
        ),
    ],
)
def test_head_content_length_returns_none_on_unparseable(mock_kwargs):
    """Missing header, HTTP error, and non-numeric value all surface as None."""
    respx.head("https://s3.example.com/file").mock(**mock_kwargs)
    assert head_content_length("https://s3.example.com/file") is None


# ─── fetch_from_s3 (text variant) ─────────────────────────────────────────


@respx.mock
def test_fetch_from_s3_decodes_utf8():
    """UTF-8 response body decoded cleanly."""
    respx.get("https://s3.example.com/file.txt").mock(
        return_value=httpx.Response(200, content=b"Hello, world!\nLine 2")
    )
    out = fetch_from_s3("https://s3.example.com/file.txt")
    assert out is not None
    assert "Hello, world!" in out


@respx.mock
def test_fetch_from_s3_falls_back_to_latin1():
    """When UTF-8 decoding fails, tries latin-1."""
    respx.get("https://s3.example.com/file.txt").mock(
        # 0xe9 is invalid UTF-8 mid-byte but valid latin-1 ('é')
        return_value=httpx.Response(200, content=b"foo \xe9 bar")
    )
    out = fetch_from_s3("https://s3.example.com/file.txt")
    assert out is not None
    assert "foo" in out
    assert "bar" in out


@respx.mock
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        pytest.param({"return_value": httpx.Response(404)}, id="http_error"),
        pytest.param(
            {"side_effect": httpx.ConnectError("DNS lookup failed")},
            id="connect_error",
        ),
    ],
)
def test_fetch_from_s3_returns_none_on_failure(mock_kwargs):
    """HTTP error and network error both surface as ``None``."""
    respx.get("https://s3.example.com/missing").mock(**mock_kwargs)
    assert fetch_from_s3("https://s3.example.com/missing") is None


# ─── Pooled-client behaviour ──────────────────────────────────────────────


@respx.mock
def test_http_client_is_a_singleton():
    """Repeated calls to ``_http_client`` return the same client instance."""
    c1 = _parser_utils._http_client()
    c2 = _parser_utils._http_client()
    c3 = _parser_utils._http_client()
    assert c1 is c2 is c3


@respx.mock
def test_reset_http_client_for_testing_actually_rebuilds():
    """``reset_http_client_for_testing`` drops the cached client; next call rebuilds it."""
    c1 = _parser_utils._http_client()
    reset_http_client_for_testing()
    c2 = _parser_utils._http_client()
    assert c1 is not c2


@respx.mock
def test_repeated_head_calls_reuse_pooled_client():
    """All HEAD calls during a session go through the same httpx.Client."""
    respx.head("https://s3.example.com/a.fif").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "100"})
    )
    respx.head("https://s3.example.com/b.fif").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "200"})
    )
    respx.head("https://s3.example.com/c.fif").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "300"})
    )

    client_before = _parser_utils._http_client()
    assert head_content_length("https://s3.example.com/a.fif") == 100
    assert head_content_length("https://s3.example.com/b.fif") == 200
    assert head_content_length("https://s3.example.com/c.fif") == 300
    client_after = _parser_utils._http_client()

    assert client_before is client_after, (
        "head_content_length must reuse the pooled client across "
        "calls — rebuilding loses keep-alive benefit"
    )


@respx.mock
def test_mixed_helpers_share_one_client():
    """All three public helpers route through the same pooled client."""
    respx.get("https://s3.example.com/range").mock(
        return_value=httpx.Response(206, content=b"x" * 32)
    )
    respx.head("https://s3.example.com/head").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "42"})
    )
    respx.get("https://s3.example.com/text").mock(
        return_value=httpx.Response(200, content=b"hello")
    )

    c0 = _parser_utils._http_client()
    fetch_bytes_from_s3("https://s3.example.com/range", max_bytes=32)
    head_content_length("https://s3.example.com/head")
    fetch_from_s3("https://s3.example.com/text")
    c1 = _parser_utils._http_client()

    assert c0 is c1


def test_http_client_enables_http2():
    """Shared client is constructed with ``http2=True``."""
    client = _parser_utils._http_client()
    transport = client._transport
    pool = getattr(transport, "_pool", None)
    assert pool is not None, (
        "client._transport._pool not found — httpx internal layout may "
        "have changed; update the test"
    )
    assert getattr(pool, "_http2", False) is True, (
        "Shared httpx.Client must be constructed with http2=True so "
        "concurrent HEAD requests multiplex over one connection"
    )
