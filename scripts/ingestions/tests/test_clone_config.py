"""Tests for the Stage 2 clone config (C8).

Same pattern as test_inject_config.py / test_stage_configs.py — direct
construction + argv parsing, no subprocess harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _clone_config import (
    KNOWN_SOURCES,
    CloneConfig,
    load_clone_config_from_argv,
)

# ─── Defaults + required fields ───────────────────────────────────────────


def test_clone_config_defaults(tmp_path: Path):
    c = CloneConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.output == Path("data/cloned")
    assert c.sources is None
    assert c.timeout == 300
    assert c.workers == 8
    assert c.limit is None
    assert c.limit_per_source is None
    assert c.datasets is None
    assert c.manifest_only is False


def test_clone_config_rejects_missing_input(tmp_path: Path):
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path / "does_not_exist")


# ─── Bounds ───────────────────────────────────────────────────────────────


def test_clone_config_workers_bounds(tmp_path: Path):
    """--workers must be in [1, 128]."""
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, workers=0)
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, workers=9999)
    c = CloneConfig(input=tmp_path, workers=64)
    assert c.workers == 64


def test_clone_config_timeout_bounds(tmp_path: Path):
    """--timeout must be in [1, 21600] (6h)."""
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, timeout=0)
    with pytest.raises(ValidationError):
        # > 6 hours = misconfig
        CloneConfig(input=tmp_path, timeout=60 * 60 * 24)


def test_clone_config_limit_bounds(tmp_path: Path):
    """--limit and --limit-per-source must be >= 1 when set."""
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, limit=0)
    with pytest.raises(ValidationError):
        CloneConfig(input=tmp_path, limit_per_source=0)


# ─── Sources validation ───────────────────────────────────────────────────


def test_clone_config_accepts_known_sources(tmp_path: Path):
    """All HANDLERS keys are valid source names."""
    c = CloneConfig(
        input=tmp_path,
        sources=["openneuro", "nemar", "zenodo"],
    )
    assert c.sources == ["openneuro", "nemar", "zenodo"]


def test_clone_config_rejects_unknown_source(tmp_path: Path):
    """A typo in --sources is caught before any handler runs."""
    with pytest.raises(ValidationError) as exc:
        CloneConfig(input=tmp_path, sources=["openneuro", "made_up_source"])
    msg = str(exc.value)
    assert "made_up_source" in msg


def test_clone_config_accepts_every_known_source_individually(tmp_path: Path):
    """Every entry in KNOWN_SOURCES is accepted."""
    for s in KNOWN_SOURCES:
        c = CloneConfig(input=tmp_path, sources=[s])
        assert c.sources == [s]


# ─── CLI parsing ──────────────────────────────────────────────────────────


def test_clone_argv_parses_all_flags(tmp_path: Path):
    out_dir = tmp_path / "out"
    c = load_clone_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--sources",
            "openneuro",
            "nemar",
            "--timeout",
            "600",
            "--workers",
            "4",
            "--limit",
            "10",
            "--limit-per-source",
            "5",
            "--datasets",
            "ds-001",
            "--manifest-only",
        ]
    )
    assert c.input == tmp_path
    assert c.output == out_dir
    assert c.sources == ["openneuro", "nemar"]
    assert c.timeout == 600
    assert c.workers == 4
    assert c.limit == 10
    assert c.limit_per_source == 5
    assert c.datasets == ["ds-001"]
    assert c.manifest_only is True


def test_clone_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_CLONE_WORKERS=16 → workers via env."""
    monkeypatch.setenv("EEGDASH_CLONE_WORKERS", "16")
    c = load_clone_config_from_argv(["--input", str(tmp_path)])
    assert c.workers == 16


def test_clone_argv_validation_error_for_unknown_source_via_argparse(
    tmp_path: Path,
):
    """Argparse's choices=KNOWN_SOURCES rejects unknown source via SystemExit
    BEFORE pydantic sees it (the more-helpful first-line-of-defence)."""
    with pytest.raises(SystemExit):
        load_clone_config_from_argv(
            [
                "--input",
                str(tmp_path),
                "--sources",
                "made_up_source",
            ]
        )
