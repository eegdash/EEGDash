"""Network-helper tests for _parser_utils.py (C4.2 — extends C2.3).

The pure helpers were covered in C2.3 (test_parser_utils.py). This
file mocks urllib to cover ``fetch_bytes_from_s3``,
``head_content_length``, and ``fetch_from_s3``.

The strategy: monkeypatch ``urllib.request.urlopen`` to return a
fake context-manager object. Avoids pytest-httpserver (heavier
fixture) since the urllib paths don't go through httpx.
"""

from __future__ import annotations

import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from urllib.error import HTTPError, URLError

import _parser_utils  # imported as module so we can monkeypatch its urllib
from _parser_utils import (
    fetch_bytes_from_s3,
    fetch_from_s3,
    head_content_length,
)


class _FakeResponse:
    """Minimal stand-in for urllib's response object (context manager)."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self._body = body
        self.headers = headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ─── Module-level _raises helpers (avoids no-nested-functions lint) ───────


def _raises_http_404(*_args, **_kwargs):
    raise HTTPError(
        url="https://s3.example.com/missing",
        code=404,
        msg="not found",
        hdrs=None,
        fp=None,
    )


def _raises_url_error(*_args, **_kwargs):
    raise URLError("DNS lookup failed")


def _raises_timeout(*_args, **_kwargs):
    raise TimeoutError("slow")


# ─── fetch_bytes_from_s3 ──────────────────────────────────────────────────


def test_fetch_bytes_from_s3_returns_requested_range(monkeypatch):
    """Happy path: urlopen returns 1024 bytes; fetch returns them."""
    payload = b"x" * 1024
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(payload),
    )
    out = fetch_bytes_from_s3("https://s3.example.com/file", max_bytes=1024)
    assert out == payload


def test_fetch_bytes_from_s3_http_error_returns_none(monkeypatch):
    """A 404 / 500 → HTTPError → None."""
    monkeypatch.setattr(_parser_utils.urllib.request, "urlopen", _raises_http_404)
    assert fetch_bytes_from_s3("https://s3.example.com/missing", max_bytes=1024) is None


def test_fetch_bytes_from_s3_url_error_returns_none(monkeypatch):
    """DNS / connection failure → URLError → None."""
    monkeypatch.setattr(_parser_utils.urllib.request, "urlopen", _raises_url_error)
    assert (
        fetch_bytes_from_s3("https://nowhere.example.com/file", max_bytes=1024) is None
    )


def test_fetch_bytes_from_s3_timeout_returns_none(monkeypatch):
    """Network timeout → None."""
    monkeypatch.setattr(_parser_utils.urllib.request, "urlopen", _raises_timeout)
    assert (
        fetch_bytes_from_s3("https://s3.example.com/file", max_bytes=1024, timeout=0.1)
        is None
    )


def test_fetch_bytes_handles_server_returning_more_than_requested(monkeypatch):
    """Some servers ignore Range and return the full body. The fetcher
    accepts whatever it gets (caller can re-slice if needed)."""
    big_payload = b"x" * 100_000  # server returned 100KB
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(big_payload),
    )
    out = fetch_bytes_from_s3("https://lazy-server.example.com/file", max_bytes=1024)
    # Got more than asked; that's accepted (caller slices)
    assert out is not None
    assert len(out) == 100_000


# ─── head_content_length ──────────────────────────────────────────────────


def test_head_content_length_parses_int(monkeypatch):
    """Content-Length header returned as int."""
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(b"", headers={"Content-Length": "12345"}),
    )
    assert head_content_length("https://s3.example.com/file") == 12345


def test_head_content_length_returns_none_when_header_missing(monkeypatch):
    """A response without Content-Length → None."""
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(b"", headers={}),
    )
    assert head_content_length("https://s3.example.com/file") is None


def test_head_content_length_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(_parser_utils.urllib.request, "urlopen", _raises_http_404)
    assert head_content_length("https://s3.example.com/missing") is None


def test_head_content_length_non_numeric_returns_none(monkeypatch):
    """A Content-Length value that's not an integer → None (ValueError caught)."""
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(
            b"", headers={"Content-Length": "not_a_number"}
        ),
    )
    assert head_content_length("https://s3.example.com/file") is None


# ─── fetch_from_s3 (text variant) ─────────────────────────────────────────


def test_fetch_from_s3_decodes_utf8(monkeypatch):
    """UTF-8 response body decoded cleanly."""
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(b"Hello, world!\nLine 2"),
    )
    out = fetch_from_s3("https://s3.example.com/file.txt")
    assert out is not None
    assert "Hello, world!" in out


def test_fetch_from_s3_falls_back_to_latin1(monkeypatch):
    """When UTF-8 decoding fails, tries latin-1."""
    monkeypatch.setattr(
        _parser_utils.urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResponse(b"foo \xe9 bar"),  # 0xe9 → invalid UTF-8
    )
    out = fetch_from_s3("https://s3.example.com/file.txt")
    assert out is not None
    assert "foo" in out
    assert "bar" in out


def test_fetch_from_s3_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(_parser_utils.urllib.request, "urlopen", _raises_http_404)
    assert fetch_from_s3("https://s3.example.com/missing") is None
