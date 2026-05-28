"""PyGithub fast-path tests — covers the path where ``from github import Github`` succeeds."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import respx

from _github import fetch_repo_file_text, iter_org_repos


def _install_fake_github(monkeypatch, github_class):
    """Inject a synthetic ``github`` module so ``from github import Github`` returns ``github_class``."""
    fake_module = SimpleNamespace(Github=github_class)
    monkeypatch.setitem(sys.modules, "github", fake_module)


# ─── iter_org_repos (PyGithub fast-path) ──────────────────────────────────


def test_iter_org_repos_yields_dicts_from_pygithub(monkeypatch):
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

    fake_github_class = MagicMock(return_value=fake_gh)

    _install_fake_github(monkeypatch, fake_github_class)

    repos = list(iter_org_repos("eegdash", token="fake_token"))
    assert len(repos) == 2
    names = {r["name"] for r in repos}
    assert names == {"repo1", "repo2"}
    for r in repos:
        if r.get("pushed_at"):
            assert r["pushed_at"].endswith("Z")


def test_iter_org_repos_falls_back_to_rest_on_pygithub_exception(monkeypatch):
    """PyGithub raises → falls back to REST path without propagating the exception."""
    fake_gh = MagicMock()
    fake_gh.get_organization.side_effect = Exception("rate limited")

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(
            "https://api.github.com/orgs/eegdash/repos",
            params__contains={"page": "1"},
        ).mock(return_value=httpx.Response(404))
        repos = list(iter_org_repos("eegdash", token=None, retries=0))
    assert isinstance(repos, list)


# ─── fetch_repo_file_text (PyGithub fast-path) ────────────────────────────


def test_fetch_repo_file_text_uses_pygithub_when_available(monkeypatch):
    fake_content = MagicMock()
    fake_content.decoded_content = b"# README\n\nfrom PyGithub"

    fake_repo = MagicMock()
    fake_repo.get_contents.return_value = fake_content

    fake_gh = MagicMock()
    fake_gh.get_repo.return_value = fake_repo

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)

    text = fetch_repo_file_text(
        "eegdash", "test-repo", "README.md", ref="main", token="x"
    )
    assert text is not None
    assert "PyGithub" in text


def test_fetch_repo_file_text_returns_none_when_content_is_list(monkeypatch):
    """get_contents on a directory returns a list; adapter must return None."""
    fake_repo = MagicMock()
    fake_repo.get_contents.return_value = [MagicMock(), MagicMock()]  # list

    fake_gh = MagicMock()
    fake_gh.get_repo.return_value = fake_repo

    fake_github_class = MagicMock(return_value=fake_gh)
    _install_fake_github(monkeypatch, fake_github_class)

    text = fetch_repo_file_text(
        "eegdash", "test-repo", "somedir", ref="main", token="x"
    )
    assert text is None
