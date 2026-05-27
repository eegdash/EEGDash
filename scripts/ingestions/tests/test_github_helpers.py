"""Tests for ``_github.py`` — GitHub API helpers (ROADMAP-C2 C2.2).

Was at 0% coverage. The full PyGithub-driven repo iterator is hard to
test without mocking the entire pygithub object graph, so this focuses
on the pure helpers (token resolution, ISO timestamp formatting,
error path) — the parts where regressions are most likely.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _github import _to_iso, get_github_token

# ─── get_github_token ──────────────────────────────────────────────────────


def test_token_from_explicit_arg_wins_over_env(monkeypatch):
    """Explicit ``token=`` argument wins over env var."""
    monkeypatch.setenv("GITHUB_TOKEN", "env_token")
    monkeypatch.setenv("GH_TOKEN", "gh_token")
    assert get_github_token("explicit") == "explicit"


def test_token_from_GITHUB_TOKEN_env(monkeypatch):
    """GITHUB_TOKEN is the canonical name; preferred over GH_TOKEN."""
    monkeypatch.setenv("GITHUB_TOKEN", "from_github_token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token() == "from_github_token"


def test_token_falls_back_to_GH_TOKEN_env(monkeypatch):
    """If GITHUB_TOKEN unset, fall back to GH_TOKEN (gh-cli convention)."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "from_gh_token")
    assert get_github_token() == "from_gh_token"


def test_token_returns_none_when_nothing_configured(monkeypatch):
    """No arg + no env → None (anonymous access)."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token() is None


def test_token_empty_string_arg_falls_through_to_env(monkeypatch):
    """Falsy explicit token (``''`` or None) falls through to env."""
    monkeypatch.setenv("GITHUB_TOKEN", "from_env")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token("") == "from_env"


# ─── _to_iso ──────────────────────────────────────────────────────────────


def test_to_iso_returns_none_for_none():
    assert _to_iso(None) is None


def test_to_iso_formats_utc_datetime_with_Z_suffix():
    """tz-aware UTC datetimes use 'Z' suffix (canonical GitHub timestamp)."""
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    out = _to_iso(dt)
    assert out is not None
    assert out.endswith("Z")
    assert "2026-05-22T12:00:00" in out


def test_to_iso_assigns_utc_when_naive():
    """A naive (no-tzinfo) datetime is treated as UTC."""
    dt = datetime(2026, 5, 22, 12, 0, 0)  # naive
    out = _to_iso(dt)
    assert out is not None
    assert out.endswith("Z")


def test_to_iso_preserves_non_utc_offsets():
    """If a non-UTC offset is supplied, it's preserved verbatim."""
    from datetime import timedelta

    tz_offset = timezone(timedelta(hours=5, minutes=30))  # India Std Time
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=tz_offset)
    out = _to_iso(dt)
    assert out is not None
    # Non-UTC offsets keep their +HH:MM format (only +00:00 → Z)
    assert "+05:30" in out


def test_to_iso_round_trips_via_fromisoformat():
    """Output is parseable back via datetime.fromisoformat."""
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    out = _to_iso(dt)
    # Z → +00:00 for round-trip
    parsed = datetime.fromisoformat(out.replace("Z", "+00:00"))
    assert parsed == dt


# ─── _pygithub_client (graceful import-error path) ────────────────────────


def test_pygithub_client_returns_none_when_pygithub_missing(monkeypatch):
    """If ``from github import Github`` fails, return None.

    The function must catch the ImportError and let the caller fall
    back to REST. Simulated by hiding ``github`` from sys.modules
    *and* monkey-patching the module attribute (handles the case
    where pygithub is already imported in the venv).
    """
    import sys as _sys

    from _github import _pygithub_client

    # Hide an already-imported 'github' module to force ImportError on
    # re-import inside _pygithub_client. Save + restore around the call.
    saved = _sys.modules.pop("github", None)
    monkeypatch.setitem(_sys.modules, "github", None)  # mark as un-importable
    try:
        result = _pygithub_client(token=None, per_page=30, timeout=10.0)
        assert result is None
    finally:
        if saved is not None:
            _sys.modules["github"] = saved
        else:
            _sys.modules.pop("github", None)
