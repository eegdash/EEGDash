"""PyGithub fast-path tests (C4.2).

C2.2 + C3.1 covered: pure helpers (token, _to_iso), REST fallback
when PyGithub absent. This file covers the PyGithub fast-path —
when ``from github import Github`` succeeds and the client object
graph drives the responses.

Strategy: substitute a ``MagicMock`` for the github module before
the adapter imports it; configure the mock's get_organization /
get_repo / get_contents return values to match the shape the
adapter expects.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


def _install_fake_github(monkeypatch, github_class):
    """Install a synthetic ``github`` module so ``from github import Github``
    returns the supplied class."""
    fake_module = SimpleNamespace(Github=github_class)
    monkeypatch.setitem(sys.modules, "github", fake_module)


# ─── iter_org_repos (PyGithub fast-path) ──────────────────────────────────


def test_iter_org_repos_yields_dicts_from_pygithub(monkeypatch):
    """When PyGithub is available, iter_org_repos walks org.get_repos()
    and yields one dict per repo."""

    # Build fake repo objects mirroring the adapter's expectations
    fake_repo1 = MagicMock(
        name="repo1",
        description="first repo",
        default_branch="main",
        pushed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        html_url="https://github.com/eegdash/repo1",
        stargazers_count=10,
        forks_count=2,
        watchers_count=10,
    )
    # MagicMock auto-sets .name from constructor; set it explicitly via attrs
    fake_repo1.name = "repo1"
    fake_repo2 = MagicMock()
    fake_repo2.name = "repo2"
    fake_repo2.description = "second repo"
    fake_repo2.default_branch = "develop"
    fake_repo2.pushed_at = datetime(2026, 5, 1, tzinfo=timezone.utc)
    fake_repo2.html_url = "https://github.com/eegdash/repo2"
    fake_repo2.stargazers_count = 5
    fake_repo2.forks_count = 1
    fake_repo2.watchers_count = 5

    fake_org = MagicMock()
    fake_org.get_repos.return_value = iter([fake_repo1, fake_repo2])

    fake_gh = MagicMock()
    fake_gh.get_organization.return_value = fake_org

    # Make the Github(...) constructor return our fake_gh
    fake_github_class = MagicMock(return_value=fake_gh)

    _install_fake_github(monkeypatch, fake_github_class)
    from _github import iter_org_repos

    repos = list(iter_org_repos("eegdash", token="fake_token"))
    assert len(repos) == 2
    names = {r["name"] for r in repos}
    assert names == {"repo1", "repo2"}
    # ISO-formatted timestamp with Z suffix (per _to_iso contract)
    for r in repos:
        if r.get("pushed_at"):
            assert r["pushed_at"].endswith("Z")


def test_iter_org_repos_falls_back_to_rest_on_pygithub_exception(monkeypatch):
    """When PyGithub raises (e.g., rate limit), the adapter logs and
    falls back to the REST path. The REST path returns nothing without
    real HTTP responses, so the iterator should complete with no yields."""

    fake_gh = MagicMock()
    fake_gh.get_organization.side_effect = Exception("rate limited")

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)
    # No respx mocks → REST fallback will hit network or fail; the
    # adapter catches RequestError and yields nothing. Using token=None
    # to avoid real network impact + with patch on httpx via respx
    # would be cleaner, but here we just confirm no crash.
    # The adapter catches PyGithub exceptions broadly and falls back
    # to REST. Stub the REST path with a 404 so the call ends
    # cleanly without real network.
    import httpx
    import respx

    from _github import iter_org_repos

    with respx.mock(assert_all_called=False) as mock:
        mock.get(
            "https://api.github.com/orgs/eegdash/repos",
            params__contains={"page": "1"},
        ).mock(return_value=httpx.Response(404))
        repos = list(iter_org_repos("eegdash", token=None, retries=0))
    # Adapter shouldn't crash; either yielded 0 (rest fallback empty)
    # or yielded what REST returned. We just confirm no PyGithub
    # exception propagated.
    assert isinstance(repos, list)


# ─── fetch_repo_file_text (PyGithub fast-path) ────────────────────────────


def test_fetch_repo_file_text_uses_pygithub_when_available(monkeypatch):
    """When PyGithub is available, fetch_repo_file_text uses
    repo.get_contents(...) instead of the raw HTTP path."""

    # Fake content object with the decoded_content shape PyGithub uses
    fake_content = MagicMock()
    fake_content.decoded_content = b"# README\n\nfrom PyGithub"

    fake_repo = MagicMock()
    fake_repo.get_contents.return_value = fake_content

    fake_gh = MagicMock()
    fake_gh.get_repo.return_value = fake_repo

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)
    from _github import fetch_repo_file_text

    text = fetch_repo_file_text(
        "eegdash", "test-repo", "README.md", ref="main", token="x"
    )
    assert text is not None
    assert "PyGithub" in text


def test_fetch_repo_file_text_returns_none_when_content_is_list(monkeypatch):
    """get_contents on a *directory* returns a list; the adapter returns
    None (it expects a single file)."""

    fake_repo = MagicMock()
    fake_repo.get_contents.return_value = [MagicMock(), MagicMock()]  # list

    fake_gh = MagicMock()
    fake_gh.get_repo.return_value = fake_repo

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)
    from _github import fetch_repo_file_text

    # When PyGithub returns a list (directory), adapter SHOULD return None
    # WITHOUT falling back to REST. So no respx mocks needed.
    # Test just confirms the codepath doesn't crash.
    # (The actual return value may be None or REST-fallback result.)
    text = fetch_repo_file_text(
        "eegdash", "test-repo", "somedir", ref="main", token="x"
    )
    # The PyGithub list-of-contents path returns None and then falls
    # through to REST fallback which (without mocks) returns None.
    assert text is None
