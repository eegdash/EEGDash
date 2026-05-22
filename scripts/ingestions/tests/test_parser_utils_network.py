"""Network-helper tests for ``_parser_utils.py`` (re-shaped 2026-05-22).

These cover ``fetch_bytes_from_s3``, ``head_content_length``, and
``fetch_from_s3`` after the migration from ``urllib.request`` to a
pooled ``httpx.Client``. The mocking strategy mirrors
``tests/test_inject_config.py`` (respx for httpx).

Why respx vs the previous urllib monkeypatch:
- The three helpers now share a module-level pooled client. urllib
  patching no longer reaches the call path.
- respx intercepts at httpx's transport layer, so the pooled client
  is preserved (which is the WHOLE POINT of the migration — we want
  to verify connection reuse).
- A dedicated ``test_helper_reuses_pooled_connection`` asserts the
  same client (= same TLS handshake) services repeated calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

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
def test_fetch_bytes_from_s3_http_error_returns_none():
    """A 404 / 500 → httpx.HTTPStatusError (via raise_for_status) → None."""
    respx.get("https://s3.example.com/missing").mock(return_value=httpx.Response(404))
    assert fetch_bytes_from_s3("https://s3.example.com/missing", max_bytes=1024) is None


@respx.mock
def test_fetch_bytes_from_s3_connect_error_returns_none():
    """DNS / connection failure → httpx.ConnectError → None."""
    respx.get("https://nowhere.example.com/file").mock(
        side_effect=httpx.ConnectError("DNS lookup failed")
    )
    assert (
        fetch_bytes_from_s3("https://nowhere.example.com/file", max_bytes=1024) is None
    )


@respx.mock
def test_fetch_bytes_from_s3_timeout_returns_none():
    """Network timeout → httpx.TimeoutException → None."""
    respx.get("https://s3.example.com/file").mock(
        side_effect=httpx.TimeoutException("slow")
    )
    assert (
        fetch_bytes_from_s3("https://s3.example.com/file", max_bytes=1024, timeout=0.1)
        is None
    )


@respx.mock
def test_fetch_bytes_handles_server_returning_more_than_requested():
    """Some servers ignore Range and return the full body. The fetcher
    accepts whatever it gets (caller can re-slice if needed)."""
    big_payload = b"x" * 100_000  # server returned 100 KB
    respx.get("https://lazy-server.example.com/file").mock(
        return_value=httpx.Response(200, content=big_payload)
    )
    out = fetch_bytes_from_s3("https://lazy-server.example.com/file", max_bytes=1024)
    assert out is not None
    assert len(out) == 100_000


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
def test_head_content_length_returns_none_when_header_missing():
    """A response without Content-Length → None."""
    respx.head("https://s3.example.com/file").mock(
        return_value=httpx.Response(200, headers={})
    )
    assert head_content_length("https://s3.example.com/file") is None


@respx.mock
def test_head_content_length_http_error_returns_none():
    """A 404 / 5xx → None."""
    respx.head("https://s3.example.com/missing").mock(return_value=httpx.Response(404))
    assert head_content_length("https://s3.example.com/missing") is None


@respx.mock
def test_head_content_length_non_numeric_returns_none():
    """A Content-Length value that's not an integer → None (ValueError caught)."""
    respx.head("https://s3.example.com/file").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "not_a_number"})
    )
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
def test_fetch_from_s3_returns_none_on_http_error():
    respx.get("https://s3.example.com/missing").mock(return_value=httpx.Response(404))
    assert fetch_from_s3("https://s3.example.com/missing") is None


@respx.mock
def test_fetch_from_s3_returns_none_on_connect_error():
    respx.get("https://nowhere.example.com/file").mock(
        side_effect=httpx.ConnectError("DNS lookup failed")
    )
    assert fetch_from_s3("https://nowhere.example.com/file") is None


# ─── Pooled-client behaviour (the WHOLE POINT of the migration) ──────────


def test_http_client_is_a_singleton():
    """Repeated calls to ``_http_client`` return the same client instance.

    This is the load-bearing property: the Stage-3 MEG montage path
    issues ~100 HEAD requests per dataset; if each call rebuilt the
    client, the connection-pool benefit would evaporate.
    """
    c1 = _parser_utils._http_client()
    c2 = _parser_utils._http_client()
    c3 = _parser_utils._http_client()
    assert c1 is c2 is c3


def test_reset_http_client_for_testing_actually_rebuilds():
    """``reset_http_client_for_testing`` drops the cached client so the
    next call rebuilds it. Required for respx-based tests."""
    c1 = _parser_utils._http_client()
    reset_http_client_for_testing()
    c2 = _parser_utils._http_client()
    assert c1 is not c2


@respx.mock
def test_repeated_head_calls_reuse_pooled_client():
    """All HEAD calls during a session go through the same httpx.Client.

    This is the indirect evidence of connection reuse — respx hooks
    intercept at the transport, and a single transport / client means
    the keep-alive pool is shared. (We can't directly assert "1 TLS
    handshake" because respx mocks below the transport, but a stable
    client identity is the necessary condition.)
    """
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
