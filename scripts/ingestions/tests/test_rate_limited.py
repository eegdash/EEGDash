"""Regression tests for the tenacity-backed ``rate_limited`` decorator.

The decorator in ``_file_utils.py``
previously used a hand-rolled try/except retry loop. The rewrite uses
``tenacity`` to share semantics with ``_http.request_json``.

Contract preserved:

- Retries on HTTP 429 and ``httpx.RequestError`` (network errors).
- Surfaces non-429 ``HTTPStatusError`` immediately.
- Returns ``None`` when retries exhaust (legacy behaviour).
- Spaces calls at least ``min_interval`` seconds apart.

The decorated test targets are built via module-level factory helpers
to satisfy the project's no-nested-functions lint rule (which allows
the factory pattern when the inner function is returned).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _file_utils import rate_limited


def _make_response(status_code: int) -> httpx.Response:
    """Construct a Response object suitable for raising as HTTPStatusError."""
    return httpx.Response(status_code, request=httpx.Request("GET", "http://x"))


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    """Construct an HTTPStatusError carrying the given status code."""
    return httpx.HTTPStatusError(
        str(status_code),
        request=httpx.Request("GET", "http://x"),
        response=_make_response(status_code),
    )


def _make_request_error(msg: str = "network") -> httpx.RequestError:
    """Construct a RequestError (network-tier failure)."""
    return httpx.RequestError(msg, request=httpx.Request("GET", "http://x"))


# ─── Module-level factory helpers (allowed by no-nested-functions) ────────


class _Counter:
    """Mutable call counter shared between a test and its factory-made fn."""

    def __init__(self) -> None:
        self.n = 0

    def bump(self) -> None:
        self.n += 1


def make_always_succeeds(counter: _Counter) -> Callable[[], str]:
    """Factory: returns a fn that always returns 'ok'."""

    def fn() -> str:
        counter.bump()
        return "ok"

    return fn


def make_429_then_success(counter: _Counter) -> Callable[[], str]:
    """Factory: 429 on first call, success on the rest."""

    def fn() -> str:
        counter.bump()
        if counter.n == 1:
            raise _make_http_error(429)
        return "recovered"

    return fn


def make_network_error_then_success(counter: _Counter) -> Callable[[], str]:
    """Factory: RequestError on first call, success after."""

    def fn() -> str:
        counter.bump()
        if counter.n == 1:
            raise _make_request_error("connection reset")
        return "recovered"

    return fn


def make_always_raises_http(status_code: int, counter: _Counter) -> Callable[[], Any]:
    """Factory: always raises HTTPStatusError with the given code."""

    def fn() -> Any:
        counter.bump()
        raise _make_http_error(status_code)

    return fn


def make_always_raises_network(counter: _Counter) -> Callable[[], Any]:
    """Factory: always raises RequestError."""

    def fn() -> Any:
        counter.bump()
        raise _make_request_error("DNS fail")

    return fn


def make_always_raises_attribute_error(counter: _Counter) -> Callable[[], Any]:
    """Factory: always raises AttributeError (programmer-error class)."""

    def fn() -> Any:
        counter.bump()
        raise AttributeError("'NoneType' object has no attribute 'foo'")

    return fn


def make_timestamp_fn() -> Callable[[], float]:
    """Factory: returns a fn that yields the current monotonic timestamp."""

    def fn() -> float:
        return time.monotonic()

    return fn


# ─── Happy path ────────────────────────────────────────────────────────────


def test_rate_limited_passes_through_on_success():
    """If the wrapped function returns normally, the result passes through."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(make_always_succeeds(counter))
    assert fn() == "ok"
    assert counter.n == 1


def test_rate_limited_retries_on_429_then_succeeds():
    """HTTP 429 is retried; success on the 2nd attempt is returned."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(make_429_then_success(counter))
    assert fn() == "recovered"
    assert counter.n == 2


def test_rate_limited_retries_on_network_error_then_succeeds():
    """httpx.RequestError is retried; success on the 2nd attempt returned."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(
        make_network_error_then_success(counter)
    )
    assert fn() == "recovered"
    assert counter.n == 2


# ─── Non-retryable: surface immediately ──────────────────────────────────


def test_rate_limited_surfaces_404_without_retry():
    """A 404 (non-retryable) must not be retried; the exception escapes."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(
        make_always_raises_http(404, counter)
    )
    with pytest.raises(httpx.HTTPStatusError):
        fn()
    assert counter.n == 1, "404 must be raised on the first attempt"


def test_rate_limited_surfaces_500_without_retry():
    """5xx is non-retryable by this decorator (the _http helper handles those)."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(
        make_always_raises_http(500, counter)
    )
    with pytest.raises(httpx.HTTPStatusError):
        fn()
    assert counter.n == 1


# ─── Exhausted retries: legacy contract returns None ──────────────────────


def test_rate_limited_returns_none_after_persistent_429():
    """When all retries fail on 429, the decorator returns None."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=2)(
        make_always_raises_http(429, counter)
    )
    assert fn() is None
    assert counter.n == 2


def test_rate_limited_returns_none_after_persistent_network_error():
    """Persistent network errors return None per legacy contract."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=2)(
        make_always_raises_network(counter)
    )
    assert fn() is None
    assert counter.n == 2


# ─── Rate-limit spacing ──────────────────────────────────────────────────


def test_rate_limited_enforces_min_interval():
    """Consecutive calls are spaced at least min_interval seconds apart."""
    fn = rate_limited(min_interval=0.1, max_retries=1)(make_timestamp_fn())
    t1 = fn()
    t2 = fn()
    # Allow a 5 % timing-jitter margin on slow CI runners.
    assert (t2 - t1) >= 0.095, f"second call too soon ({t2 - t1:.3f}s)"


# ─── Programmer errors propagate (Phase 9 F1 contract) ───────────────────


def test_rate_limited_propagates_attribute_error():
    """Non-httpx exceptions (programmer errors) are NOT retried."""
    counter = _Counter()
    fn = rate_limited(min_interval=0.0, max_retries=3)(
        make_always_raises_attribute_error(counter)
    )
    with pytest.raises(AttributeError):
        fn()
    assert counter.n == 1, "programmer errors must not be retried"
