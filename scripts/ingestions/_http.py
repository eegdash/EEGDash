"""Shared HTTP helpers for ingestion scripts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import httpx

try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        retry_if_result,
        stop_after_attempt,
        wait_exponential,
    )

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _TENACITY_AVAILABLE = False

    class RetryError(Exception):  # type: ignore[no-redef]
        """Fallback when tenacity is missing; never raised in that case."""


try:
    import hishel
except ImportError:  # pragma: no cover - optional dependency
    hishel = None

DEFAULT_USER_AGENT = "EEGDash-DataHarvester/1.0"
# RFC 9110 § 15.5.9: 408 Request Timeout SHOULD be retried by clients
# (server saw the connection open but did not receive a complete request
# in time — same root cause as a 504, just observed from the upstream
# side). Most servers return 504 instead, but including 408 is hygiene
# and survives upstreams that don't.
DEFAULT_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
DEFAULT_TIMEOUT = 30.0

RequestError = httpx.RequestError
HTTPStatusError = httpx.HTTPStatusError
TimeoutException = httpx.TimeoutException


def build_headers(
    *,
    user_agent: str | None = DEFAULT_USER_AGENT,
    accept: str | None = "application/json",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build headers with consistent defaults."""
    headers: dict[str, str] = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    if accept:
        headers["Accept"] = accept
    if extra:
        headers.update(extra)
    return headers


def _cache_enabled() -> bool:
    """Return True if HTTP caching is enabled."""
    return os.getenv("EEGDASH_HTTP_CACHE", "1").lower() not in {"0", "false", "no"}


def _build_transport() -> httpx.BaseTransport:
    """Build a transport with optional caching."""
    if not _cache_enabled() or hishel is None:
        return httpx.HTTPTransport(retries=0)

    try:
        cache_dir = os.getenv("EEGDASH_HTTP_CACHE_DIR")
        if cache_dir:
            try:
                storage = hishel.FileStorage(base_path=cache_dir)
                return hishel.CacheTransport(storage=storage)
            except (OSError, PermissionError, ValueError):
                # OSError covers ENOENT/EACCES on the cache dir; ValueError
                # covers hishel's "invalid base_path" check. Cache disabled
                # but the request path still works → silent fall-through.
                pass
        return hishel.CacheTransport()
    except (OSError, ImportError, AttributeError):
        # ImportError if hishel is half-installed; AttributeError if its
        # API changed between releases. Fall back to plain HTTPTransport.
        return httpx.HTTPTransport(retries=0)


_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """Return a shared HTTP client."""
    global _client
    if _client is None:
        _client = httpx.Client(
            transport=_build_transport(),
            headers=build_headers(),
            timeout=DEFAULT_TIMEOUT,
        )
    return _client


def close_client() -> None:
    """Close the shared HTTP client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def make_authed_client(auth_token: str) -> httpx.Client:
    """Create an httpx.Client with Bearer auth headers and no retries.

    Retries are injected at the call site by ``request_json`` /
    ``request_text`` via tenacity (see ``_request_with_retry``). This
    function only configures the auth header and timeout — the name
    reflects that.

    Parameters
    ----------
    auth_token : str
        Bearer token to attach as ``Authorization: Bearer <token>``.

    Returns
    -------
    httpx.Client
        Configured client. The caller is responsible for closing it,
        either explicitly or via the ``with`` statement.

    Notes
    -----
    Renamed from ``make_retry_client`` — the old name implied that
    retries were baked into the client itself, which they are not.
    The old name remains as a deprecated alias for one release.
    """
    headers = build_headers(
        extra={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
    )
    return httpx.Client(
        transport=httpx.HTTPTransport(retries=0),
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )


def make_retry_client(auth_token: str) -> httpx.Client:
    """Deprecated alias for :func:`make_authed_client`.

    .. deprecated:: 0.1
        ``make_retry_client`` is misleadingly named — the client it
        returns has ``retries=0``; retries happen at call sites via
        tenacity. Use :func:`make_authed_client` instead. Will be
        removed in 0.2.
    """
    import warnings

    warnings.warn(
        "make_retry_client is deprecated; use make_authed_client. "
        "The returned client does NOT have retries baked in; retries "
        "are injected by request_json / request_text. "
        "Will be removed in v0.2.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_authed_client(auth_token)


def _should_retry_response(
    response: httpx.Response | None, retry_statuses: set[int]
) -> bool:
    if response is None:
        return True
    return response.status_code in retry_statuses


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None,
    params: dict[str, Any] | None,
    json_body: Any | None,
    data: Any | None,
    timeout: float,
    follow_redirects: bool,
    retries: int,
    backoff_factor: float,
    retry_statuses: set[int],
    stream: bool,
) -> httpx.Response:
    def _do_request() -> httpx.Response:
        request = client.build_request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
        )
        return client.send(
            request,
            follow_redirects=follow_redirects,
            stream=stream,
        )

    if not _TENACITY_AVAILABLE:
        return _do_request()

    @retry(
        reraise=True,
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
        retry=(
            retry_if_exception_type(httpx.RequestError)
            | retry_if_result(lambda resp: _should_retry_response(resp, retry_statuses))
        ),
    )
    def _do_request_with_retry() -> httpx.Response:
        return _do_request()

    return _do_request_with_retry()


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
    retries: int = 3,
    backoff_factor: float = 1.0,
    retry_statuses: Iterable[int] | None = None,
    client: httpx.Client | None = None,
) -> tuple[Any | None, httpx.Response | None]:
    """Perform an HTTP request and decode JSON response.

    Returns (payload, response). Payload is None if decoding fails.
    Response is None if the request raises an exception.
    """
    retry_set = set(retry_statuses or DEFAULT_RETRY_STATUSES)
    http_client = client or get_client()

    response: httpx.Response | None = None
    try:
        response = _request_with_retry(
            http_client,
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            data=data,
            timeout=timeout,
            follow_redirects=allow_redirects,
            retries=retries,
            backoff_factor=backoff_factor,
            retry_statuses=retry_set,
            stream=False,
        )
        if raise_for_status:
            response.raise_for_status()
        try:
            return response.json(), response
        except ValueError:
            return None, response
    except RetryError as e:
        # tenacity raises this when retries exhaust on a RETRY_RESULT
        # condition (e.g., persistent 5xx — no exception, just a bad
        # status). The last_attempt's result IS an httpx.Response that
        # the caller may want to inspect for the final status code.
        # (Bug found by tests/test_http.py: persistent 503 used to leak
        # tenacity.RetryError to callers, breaking the documented
        # "(payload, response)" contract.)
        try:
            last_response = e.last_attempt.result()
        except Exception:  # noqa: BLE001 — last_attempt internals are tenacity-private
            last_response = None
        return None, last_response
    except httpx.RequestError:
        if raise_for_request:
            raise
        return None, None
    except httpx.HTTPStatusError:
        if raise_for_status:
            raise
        return None, response


def request_text(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
    retries: int = 3,
    backoff_factor: float = 1.0,
    retry_statuses: Iterable[int] | None = None,
    client: httpx.Client | None = None,
) -> tuple[str | None, httpx.Response | None]:
    """Perform an HTTP request and return response text."""
    retry_set = set(retry_statuses or DEFAULT_RETRY_STATUSES)
    http_client = client or get_client()

    try:
        response = _request_with_retry(
            http_client,
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            data=data,
            timeout=timeout,
            follow_redirects=allow_redirects,
            retries=retries,
            backoff_factor=backoff_factor,
            retry_statuses=retry_set,
            stream=False,
        )
        if raise_for_status:
            response.raise_for_status()
        return response.text, response
    except httpx.RequestError:
        if raise_for_request:
            raise
        return None, None
    except httpx.HTTPStatusError:
        if raise_for_status:
            raise
        return None, response


def request_response(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
    stream: bool = False,
    retries: int = 3,
    backoff_factor: float = 1.0,
    retry_statuses: Iterable[int] | None = None,
    client: httpx.Client | None = None,
) -> httpx.Response | None:
    """Perform an HTTP request and return the raw response."""
    retry_set = set(retry_statuses or DEFAULT_RETRY_STATUSES)
    http_client = client or get_client()

    try:
        response = _request_with_retry(
            http_client,
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            data=data,
            timeout=timeout,
            follow_redirects=allow_redirects,
            retries=retries,
            backoff_factor=backoff_factor,
            retry_statuses=retry_set,
            stream=stream,
        )
        if raise_for_status:
            response.raise_for_status()
        return response
    except httpx.RequestError:
        if raise_for_request:
            raise
        return None
    except httpx.HTTPStatusError:
        if raise_for_status:
            raise
        return None
