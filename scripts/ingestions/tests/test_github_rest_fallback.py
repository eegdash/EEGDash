"""GitHub adapter tests via the REST fallback path (ROADMAP-C3).

C2.2 covered the pure helpers (token, _to_iso, _pygithub_client
import-error path). This file covers the **REST fallback** that
fires when PyGithub is absent — using respx to mock the raw GitHub
API.

Strategy: force the PyGithub client to return None (via the
sys.modules trick from C2.2), then mock the raw GitHub REST URLs
that the fallback code path uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import respx

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))


def _hide_pygithub(monkeypatch) -> None:
    """Force ``from github import Github`` to raise ImportError.

    Idiom is the same as test_github_helpers.py's
    test_pygithub_client_returns_none_when_pygithub_missing.
    """
    monkeypatch.setitem(sys.modules, "github", None)


# ─── fetch_repo_file_text (REST fallback) ─────────────────────────────────


@respx.mock
def test_fetch_repo_file_text_rest_fallback_success(monkeypatch):
    """When PyGithub is unavailable, falls back to raw.githubusercontent.com."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_text

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/README.md"
    respx.get(url).mock(
        return_value=httpx.Response(200, text="# Test repo\n\nReadme content")
    )
    text = fetch_repo_file_text("eegdash", "test-repo", "README.md", ref="main")
    assert text is not None
    assert "Test repo" in text


@respx.mock
def test_fetch_repo_file_text_rest_fallback_404_returns_none(monkeypatch):
    """A 404 from raw.githubusercontent.com → None."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_text

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/missing.txt"
    respx.get(url).mock(return_value=httpx.Response(404))
    assert (
        fetch_repo_file_text("eegdash", "test-repo", "missing.txt", ref="main") is None
    )


@respx.mock
def test_fetch_repo_file_text_rejects_html_doctype(monkeypatch):
    """If GitHub returns a 200 with an HTML page (login redirect etc.),
    the adapter must NOT treat that as a real file's content."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_text

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/file.txt"
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            text="<!DOCTYPE html><html><body>Please login</body></html>",
        )
    )
    assert fetch_repo_file_text("eegdash", "test-repo", "file.txt", ref="main") is None


# ─── fetch_first_repo_file_text ───────────────────────────────────────────


@respx.mock
def test_fetch_first_returns_earliest_match(monkeypatch):
    """The adapter walks candidate paths in order and returns the first
    one with content."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_first_repo_file_text

    # First candidate 404s; second succeeds.
    respx.get(
        "https://raw.githubusercontent.com/eegdash/test-repo/main/README.md"
    ).mock(return_value=httpx.Response(404))
    respx.get("https://raw.githubusercontent.com/eegdash/test-repo/main/README").mock(
        return_value=httpx.Response(200, text="readme content")
    )

    text = fetch_first_repo_file_text(
        "eegdash", "test-repo", ["README.md", "README"], ref="main"
    )
    assert text == "readme content"


@respx.mock
def test_fetch_first_returns_none_when_all_404(monkeypatch):
    """All candidate paths 404 → None."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_first_repo_file_text

    for cand in ("a.txt", "b.txt", "c.txt"):
        respx.get(
            f"https://raw.githubusercontent.com/eegdash/test-repo/main/{cand}"
        ).mock(return_value=httpx.Response(404))

    text = fetch_first_repo_file_text(
        "eegdash", "test-repo", ["a.txt", "b.txt", "c.txt"], ref="main"
    )
    assert text is None


# ─── fetch_repo_file_json ─────────────────────────────────────────────────


@respx.mock
def test_fetch_repo_file_json_decodes_response(monkeypatch):
    """A valid JSON file response → parsed dict."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_json

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/config.json"
    respx.get(url).mock(
        return_value=httpx.Response(200, json={"key": "value", "n": 42})
    )
    out = fetch_repo_file_json("eegdash", "test-repo", "config.json", ref="main")
    assert out == {"key": "value", "n": 42}


@respx.mock
def test_fetch_repo_file_json_returns_none_on_404(monkeypatch):
    """A 404 → None (consistent with the text variant)."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_json

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/missing.json"
    respx.get(url).mock(return_value=httpx.Response(404))
    assert (
        fetch_repo_file_json("eegdash", "test-repo", "missing.json", ref="main") is None
    )


@respx.mock
def test_fetch_repo_file_json_returns_none_on_malformed_json(monkeypatch):
    """Malformed JSON in the response body → None, no exception."""
    _hide_pygithub(monkeypatch)
    from _github import fetch_repo_file_json

    url = "https://raw.githubusercontent.com/eegdash/test-repo/main/bad.json"
    respx.get(url).mock(return_value=httpx.Response(200, text="not actually json"))
    assert fetch_repo_file_json("eegdash", "test-repo", "bad.json", ref="main") is None
