"""Shared HTTP helpers for ingestion scripts."""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "EEGDash-DataHarvester/1.0"


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


def make_retry_session(
    auth_token: str,
    *,
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: list[int] | None = None,
) -> requests.Session:
    """Create a requests session with retry strategy and auth headers."""
    if status_forcelist is None:
        status_forcelist = [500, 502, 503, 504]

    session = requests.Session()
    retry = Retry(
        total=retries,
        status_forcelist=status_forcelist,
        backoff_factor=backoff_factor,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers["Authorization"] = f"Bearer {auth_token}"
    session.headers["Content-Type"] = "application/json"
    return session


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = 30.0,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
) -> tuple[Any | None, requests.Response | None]:
    """Perform an HTTP request and decode JSON response.

    Returns (payload, response). Payload is None if decoding fails.
    Response is None if the request raises an exception.
    """
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        if raise_for_status:
            response.raise_for_status()
        try:
            return response.json(), response
        except ValueError:
            return None, response
    except requests.RequestException:
        if raise_for_request:
            raise
        return None, None


def request_text(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = 30.0,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
) -> tuple[str | None, requests.Response | None]:
    """Perform an HTTP request and return response text."""
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        if raise_for_status:
            response.raise_for_status()
        return response.text, response
    except requests.RequestException:
        if raise_for_request:
            raise
        return None, None


def request_response(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    data: Any | None = None,
    timeout: float = 30.0,
    allow_redirects: bool = True,
    raise_for_status: bool = False,
    raise_for_request: bool = False,
    stream: bool = False,
) -> requests.Response | None:
    """Perform an HTTP request and return the raw response."""
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
            allow_redirects=allow_redirects,
            stream=stream,
        )
        if raise_for_status:
            response.raise_for_status()
        return response
    except requests.RequestException:
        if raise_for_request:
            raise
        return None
