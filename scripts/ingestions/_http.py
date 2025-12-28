"""Shared HTTP helpers for ingestion scripts."""

from __future__ import annotations

import os
from typing import Any, Iterable

import httpx

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        retry_if_result,
        stop_after_attempt,
        wait_exponential,
    )

    _TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _TENACITY_AVAILABLE = False

try:
    import hishel
except Exception:  # pragma: no cover - optional dependency
    hishel = None

DEFAULT_USER_AGENT = "EEGDash-DataHarvester/1.0"
DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}
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
            except Exception:
                pass
        return hishel.CacheTransport()
    except Exception:
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


def make_retry_client(auth_token: str) -> httpx.Client:
    """Create an HTTP client with auth headers for ingestion injection."""
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
        return client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
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
