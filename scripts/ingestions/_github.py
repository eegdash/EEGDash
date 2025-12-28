"""Shared GitHub helpers for ingestion scripts.

Prefers PyGithub (popular, stable) when available, with a fallback to the GitHub
REST API via our `_http` helper to keep scripts runnable in minimal environments.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from _http import (
    HTTPStatusError,
    RequestError,
    build_headers,
    request_json,
    request_text,
)

GITHUB_API_URL = "https://api.github.com"
GITHUB_RAW_URL = "https://raw.githubusercontent.com"


def get_github_token(token: str | None = None) -> str | None:
    """Return a GitHub token from args or env (GITHUB_TOKEN/GH_TOKEN)."""
    return token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _pygithub_client(
    *,
    token: str | None,
    per_page: int,
    timeout: float,
) -> Any | None:
    try:
        from github import Github  # type: ignore[import-not-found]
    except Exception:
        return None
    return Github(
        login_or_token=token or None,
        per_page=min(per_page, 100),
        timeout=timeout,
    )


def iter_org_repos(
    organization: str,
    *,
    per_page: int = 100,
    timeout: float = 30.0,
    retries: int = 5,
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate repositories in a GitHub organization.

    Yields dicts with a stable subset of repo metadata needed by ingestion.
    """
    token = get_github_token(token)
    gh = _pygithub_client(token=token, per_page=per_page, timeout=timeout)
    if gh is not None:
        try:
            org_obj = gh.get_organization(organization)
            for repo in org_obj.get_repos(type="all"):
                yield {
                    "name": repo.name,
                    "description": repo.description,
                    "default_branch": repo.default_branch,
                    "pushed_at": _to_iso(getattr(repo, "pushed_at", None)),
                    "html_url": getattr(repo, "html_url", None),
                }
            return
        except Exception:
            pass

    headers_extra = {"Authorization": f"Bearer {token}"} if token else None
    headers = build_headers(
        accept="application/vnd.github.v3+json",
        extra=headers_extra,
    )

    page = 1
    while True:
        url = f"{GITHUB_API_URL}/orgs/{organization}/repos"
        params = {
            "per_page": min(per_page, 100),
            "page": page,
            "type": "all",
            "sort": "created",
            "direction": "asc",
        }

        try:
            data, response = request_json(
                "get",
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                retries=retries,
                raise_for_status=True,
                raise_for_request=True,
            )
        except (RequestError, HTTPStatusError):
            return

        if not response or not isinstance(data, list) or not data:
            return

        for repo in data:
            if not isinstance(repo, dict):
                continue
            yield {
                "name": repo.get("name"),
                "description": repo.get("description"),
                "default_branch": repo.get("default_branch"),
                "pushed_at": repo.get("pushed_at"),
                "html_url": repo.get("html_url"),
            }

        link_header = response.headers.get("Link", "")
        if 'rel="next"' not in link_header:
            return
        page += 1


def fetch_repo_file_text(
    organization: str,
    repo: str,
    path: str,
    *,
    ref: str,
    timeout: float = 10.0,
    retries: int = 3,
    token: str | None = None,
) -> str | None:
    """Fetch a file's content from a GitHub repo."""
    token = get_github_token(token)
    gh = _pygithub_client(token=token, per_page=100, timeout=timeout)
    if gh is not None:
        try:
            repo_obj = gh.get_repo(f"{organization}/{repo}")
            content = repo_obj.get_contents(path, ref=ref)
            if isinstance(content, list):
                return None
            decoded = getattr(content, "decoded_content", None)
            if not decoded:
                return None
            return decoded.decode("utf-8", errors="replace")
        except Exception:
            pass

    url = f"{GITHUB_RAW_URL}/{organization}/{repo}/{ref}/{path}"
    text, response = request_text(
        "get",
        url,
        timeout=timeout,
        retries=retries,
    )
    if (
        response
        and response.status_code == 200
        and text
        and not text.startswith("<!DOCTYPE")
    ):
        return text
    return None


def fetch_first_repo_file_text(
    organization: str,
    repo: str,
    paths: list[str],
    *,
    ref: str,
    timeout: float = 10.0,
    retries: int = 3,
    token: str | None = None,
) -> str | None:
    """Fetch first matching file content from candidate paths."""
    for candidate in paths:
        text = fetch_repo_file_text(
            organization,
            repo,
            candidate,
            ref=ref,
            timeout=timeout,
            retries=retries,
            token=token,
        )
        if text:
            return text
    return None


def fetch_repo_file_json(
    organization: str,
    repo: str,
    path: str,
    *,
    ref: str,
    timeout: float = 10.0,
    retries: int = 3,
    token: str | None = None,
) -> dict[str, Any] | None:
    """Fetch and parse a JSON file from a GitHub repo."""
    text = fetch_repo_file_text(
        organization,
        repo,
        path,
        ref=ref,
        timeout=timeout,
        retries=retries,
        token=token,
    )
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
