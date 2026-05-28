"""Tests for ``_github.py`` — token resolution, ISO timestamp formatting, error path."""

from __future__ import annotations

import sys as _sys
from datetime import datetime, timedelta, timezone

from _github import _pygithub_client, _to_iso, get_github_token

# ─── get_github_token ──────────────────────────────────────────────────────


def test_token_from_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env_token")
    monkeypatch.setenv("GH_TOKEN", "gh_token")
    assert get_github_token("explicit") == "explicit"


def test_token_from_GITHUB_TOKEN_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from_github_token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token() == "from_github_token"


def test_token_falls_back_to_GH_TOKEN_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "from_gh_token")
    assert get_github_token() == "from_gh_token"


def test_token_returns_none_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token() is None


def test_token_empty_string_arg_falls_through_to_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "from_env")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert get_github_token("") == "from_env"


# ─── _to_iso ──────────────────────────────────────────────────────────────


def test_to_iso_returns_none_for_none():
    assert _to_iso(None) is None


def test_to_iso_formats_utc_datetime_with_Z_suffix():
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    out = _to_iso(dt)
    assert out is not None
    assert out.endswith("Z")
    assert "2026-05-22T12:00:00" in out


def test_to_iso_assigns_utc_when_naive():
    dt = datetime(2026, 5, 22, 12, 0, 0)  # naive
    out = _to_iso(dt)
    assert out is not None
    assert out.endswith("Z")


def test_to_iso_preserves_non_utc_offsets():
    tz_offset = timezone(timedelta(hours=5, minutes=30))  # IST
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=tz_offset)
    out = _to_iso(dt)
    assert out is not None
    assert "+05:30" in out


def test_to_iso_round_trips_via_fromisoformat():
    dt = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    out = _to_iso(dt)
    parsed = datetime.fromisoformat(out.replace("Z", "+00:00"))
    assert parsed == dt


# ─── _pygithub_client (graceful import-error path) ────────────────────────


def test_pygithub_client_returns_none_when_pygithub_missing(monkeypatch):
    """ImportError on ``from github import Github`` → return None."""
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
