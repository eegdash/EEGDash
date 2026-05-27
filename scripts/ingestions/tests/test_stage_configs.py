"""Tests for the Stage 3 + Stage 4 Pydantic-settings configs (C7).

Same pattern as test_inject_config.py — direct construction + argv
parsing, no subprocess harness needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from _digest_config import (
    DEFAULT_DATASET_TIMEOUT_SECONDS,
    DigestConfig,
    load_digest_config_from_argv,
)
from _validate_config import (
    ValidateConfig,
    load_validate_config_from_argv,
)

# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — ValidateConfig
# ═══════════════════════════════════════════════════════════════════════════


def test_validate_config_defaults(tmp_path: Path):
    """Defaults match the documented behaviour."""
    c = ValidateConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.pre_check is False
    assert c.verbose is False
    assert c.strict is False
    assert c.json_output is False


def test_validate_config_rejects_missing_input(tmp_path: Path):
    """The stage IS the validation — input dir must exist."""
    with pytest.raises(ValidationError) as exc:
        ValidateConfig(input=tmp_path / "does_not_exist")
    assert "input" in str(exc.value).lower()


def test_validate_config_accepts_real_input(tmp_path: Path):
    c = ValidateConfig(input=tmp_path)
    assert c.input == tmp_path


def test_validate_argv_parses_all_flags(tmp_path: Path):
    c = load_validate_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--pre-check",
            "--verbose",
            "--strict",
            "--json",
        ]
    )
    assert c.input == tmp_path
    assert c.pre_check is True
    assert c.verbose is True
    assert c.strict is True
    assert c.json_output is True


def test_validate_argv_short_flag_for_input(tmp_path: Path):
    """``-i`` is an alias for ``--input`` (preserved from the original)."""
    c = load_validate_config_from_argv(["-i", str(tmp_path)])
    assert c.input == tmp_path


def test_validate_argv_short_flag_for_verbose(tmp_path: Path):
    """``-v`` is an alias for ``--verbose``."""
    c = load_validate_config_from_argv(["-i", str(tmp_path), "-v"])
    assert c.verbose is True


def test_validate_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_VALIDATE_STRICT=1 → strict via env."""
    monkeypatch.setenv("EEGDASH_VALIDATE_STRICT", "1")
    c = load_validate_config_from_argv(["-i", str(tmp_path)])
    assert c.strict is True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — DigestConfig
# ═══════════════════════════════════════════════════════════════════════════


def test_digest_config_defaults(tmp_path: Path):
    """Defaults match the original argparse layout."""
    c = DigestConfig(input=tmp_path)
    assert c.input == tmp_path
    assert c.output == Path("digestion_output")
    assert c.datasets is None
    assert c.workers == 1
    assert c.limit is None
    assert c.dataset_timeout == float(DEFAULT_DATASET_TIMEOUT_SECONDS)


def test_digest_config_rejects_missing_input(tmp_path: Path):
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path / "does_not_exist")


def test_digest_config_workers_lower_bound(tmp_path: Path):
    """``--workers 0`` is rejected (replaces the ad-hoc default-of-1)."""
    with pytest.raises(ValidationError) as exc:
        DigestConfig(input=tmp_path, workers=0)
    assert "workers" in str(exc.value).lower()


def test_digest_config_workers_upper_bound(tmp_path: Path):
    """Sanity cap: ``--workers 9999`` is a misconfig."""
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, workers=9999)


def test_digest_config_limit_lower_bound(tmp_path: Path):
    """``--limit 0`` makes no sense (no datasets to process)."""
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, limit=0)


def test_digest_config_dataset_timeout_must_be_positive(tmp_path: Path):
    """Replaces the ``_positive_float`` custom argparse type. ``0`` or
    negative timeout means infinite wait / nonsense — rejected."""
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, dataset_timeout=0.0)
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, dataset_timeout=-1.0)


def test_digest_config_dataset_timeout_upper_bound(tmp_path: Path):
    """Anything > 4 hours is almost certainly a misconfig (a worker
    that legitimately needs 4h should be debugged, not waited for)."""
    with pytest.raises(ValidationError):
        DigestConfig(input=tmp_path, dataset_timeout=60 * 60 * 24)  # 24h


def test_digest_config_datasets_filter(tmp_path: Path):
    """``--datasets ds-001 ds-002`` becomes a list."""
    c = DigestConfig(input=tmp_path, datasets=["ds-001", "ds-002"])
    assert c.datasets == ["ds-001", "ds-002"]


def test_digest_argv_parses_all_flags(tmp_path: Path):
    out_dir = tmp_path / "out"
    c = load_digest_config_from_argv(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--datasets",
            "ds-001",
            "ds-002",
            "--workers",
            "4",
            "--limit",
            "10",
            "--dataset-timeout",
            "300",
        ]
    )
    assert c.input == tmp_path
    assert c.output == out_dir
    assert c.datasets == ["ds-001", "ds-002"]
    assert c.workers == 4
    assert c.limit == 10
    assert c.dataset_timeout == 300.0


def test_digest_argv_env_var_picked_up(tmp_path: Path, monkeypatch):
    """EEGDASH_DIGEST_WORKERS=8 → workers=8 via env."""
    monkeypatch.setenv("EEGDASH_DIGEST_WORKERS", "8")
    c = load_digest_config_from_argv(["--input", str(tmp_path)])
    assert c.workers == 8


def test_digest_argv_validation_error_surfaces(tmp_path: Path):
    """Bad CLI → ValidationError, not a vague argparse failure."""
    with pytest.raises(ValidationError):
        load_digest_config_from_argv(["--input", str(tmp_path), "--workers", "0"])
