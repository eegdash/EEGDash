"""Tests for the Pydantic-settings inject config (C6.5).

Replaces the would-be 460-line argparse subprocess test with direct
construction + validation of :class:`InjectConfig`. Same coverage,
faster, no monkey-patching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _inject_config import (
    DEFAULT_API_URL,
    InjectConfig,
    load_inject_config_from_argv,
)

# ─── Defaults + required fields ───────────────────────────────────────────


def test_config_requires_database():
    """``database`` has no default — must be supplied."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(dry_run=True)
    assert "database" in str(exc.value).lower()


def test_config_rejects_invalid_database():
    """``database`` must match the ValidDatabase Literal set."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(database="not_a_real_db", dry_run=True)
    msg = str(exc.value)
    assert "database" in msg.lower()


def test_config_accepts_each_valid_database():
    """All 5 documented database names accepted."""
    for db in (
        "eegdash",
        "eegdash_dev",
        "eegdash_archive",
        "eegdash_staging",
        "eegdash_v1",
    ):
        c = InjectConfig(database=db, dry_run=True)
        assert c.database == db


def test_config_defaults():
    """Field defaults match the documented behaviour."""
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.api_url == DEFAULT_API_URL
    assert c.input == Path("digestion_output")
    assert c.batch_size == 1000
    assert c.dry_run is True
    assert c.only_datasets is False
    assert c.only_records is False
    assert c.only_montages is False
    assert c.skip_montages is False
    assert c.force is False
    assert c.skip_validation is False
    assert c.data_quality_threshold == 10.0
    assert c.compute_stats is False
    assert c.datasets is None


# ─── Bounds ───────────────────────────────────────────────────────────────


def test_batch_size_lower_bound():
    """``batch_size`` must be ≥ 1."""
    with pytest.raises(ValidationError):
        InjectConfig(database="eegdash_dev", batch_size=0, dry_run=True)


def test_batch_size_upper_bound():
    """``batch_size`` must be ≤ 10_000 (avoids accidentally trying to
    upload a 100k-record batch that the Gateway would reject)."""
    with pytest.raises(ValidationError):
        InjectConfig(database="eegdash_dev", batch_size=20_000, dry_run=True)


def test_data_quality_threshold_bounds():
    """``data_quality_threshold`` is a percentage 0-100."""
    InjectConfig(database="eegdash_dev", data_quality_threshold=0.0, dry_run=True)
    InjectConfig(database="eegdash_dev", data_quality_threshold=100.0, dry_run=True)
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            data_quality_threshold=-1.0,
            dry_run=True,
        )
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            data_quality_threshold=101.0,
            dry_run=True,
        )


# ─── Mutually-exclusive only-* flags ──────────────────────────────────────


def test_two_only_flags_rejected():
    """Any pair of --only-* flags raises."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            dry_run=True,
        )
    assert "mutually exclusive" in str(exc.value).lower()


def test_three_only_flags_rejected():
    """All 3 --only-* set is the worst case → rejected."""
    with pytest.raises(ValidationError):
        InjectConfig(
            database="eegdash_dev",
            only_datasets=True,
            only_records=True,
            only_montages=True,
            dry_run=True,
        )


def test_each_single_only_flag_accepted():
    """Exactly one --only-* flag at a time is fine."""
    for flag in ("only_datasets", "only_records", "only_montages"):
        kw = {"database": "eegdash_dev", "dry_run": True, flag: True}
        c = InjectConfig(**kw)
        assert getattr(c, flag) is True


# ─── only-montages + skip-montages contradict ─────────────────────────────


def test_only_and_skip_montages_contradict():
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            only_montages=True,
            skip_montages=True,
            dry_run=True,
        )
    assert "contradict" in str(exc.value).lower()


# ─── input must exist (unless dry-run) ────────────────────────────────────


def test_missing_input_dir_rejected_when_not_dry_run(tmp_path: Path):
    """A real run needs a real --input dir."""
    with pytest.raises(ValidationError) as exc:
        InjectConfig(
            database="eegdash_dev",
            input=tmp_path / "does_not_exist",
            dry_run=False,
        )
    assert "input" in str(exc.value).lower()


def test_missing_input_dir_allowed_in_dry_run(tmp_path: Path):
    """Dry-run skips the existence check — useful for --help-style runs."""
    InjectConfig(
        database="eegdash_dev",
        input=tmp_path / "does_not_exist",
        dry_run=True,
    )


def test_existing_input_dir_accepted(tmp_path: Path):
    """Normal real-run case."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    c = InjectConfig(database="eegdash_dev", input=real_dir, dry_run=False)
    assert c.input == real_dir


# ─── Env var fallback for token ───────────────────────────────────────────


def test_token_explicit_arg_used_when_provided():
    c = InjectConfig(database="eegdash_dev", token="explicit", dry_run=True)
    assert c.token == "explicit"


def test_token_reads_from_eegdash_admin_token_env(monkeypatch):
    """When --token is missing, read EEGDASH_ADMIN_TOKEN env var
    (matches the legacy ops-script convention)."""
    monkeypatch.setenv("EEGDASH_ADMIN_TOKEN", "from_env")
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token == "from_env"


def test_token_none_when_neither_arg_nor_env(monkeypatch):
    monkeypatch.delenv("EEGDASH_ADMIN_TOKEN", raising=False)
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.token is None


# ─── Convenience accessors (want_datasets / want_records / want_montages) ─


def test_want_accessors_default_true():
    """Default config injects everything."""
    c = InjectConfig(database="eegdash_dev", dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is True
    assert c.want_montages is True


def test_only_datasets_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_datasets=True, dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is False
    assert c.want_montages is False


def test_only_records_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_records=True, dry_run=True)
    assert c.want_datasets is False
    assert c.want_records is True
    assert c.want_montages is False


def test_only_montages_filters_other_paths():
    c = InjectConfig(database="eegdash_dev", only_montages=True, dry_run=True)
    assert c.want_datasets is False
    assert c.want_records is False
    assert c.want_montages is True


def test_skip_montages_disables_montage_only():
    """``--skip-montages`` only affects the montage leg; datasets +
    records still inject."""
    c = InjectConfig(database="eegdash_dev", skip_montages=True, dry_run=True)
    assert c.want_datasets is True
    assert c.want_records is True
    assert c.want_montages is False


# ─── CLI parsing ──────────────────────────────────────────────────────────


def test_argv_parses_minimum_required():
    c = load_inject_config_from_argv(["--database", "eegdash_dev", "--dry-run"])
    assert c.database == "eegdash_dev"
    assert c.dry_run is True


def test_argv_parses_all_string_flags():
    c = load_inject_config_from_argv(
        [
            "--database",
            "eegdash_dev",
            "--api-url",
            "https://test.example.com",
            "--token",
            "my-token",
            "--batch-size",
            "500",
            "--dry-run",
        ]
    )
    assert c.api_url == "https://test.example.com"
    assert c.token == "my-token"
    assert c.batch_size == 500


def test_argv_parses_only_datasets_flag():
    c = load_inject_config_from_argv(
        ["--database", "eegdash_dev", "--only-datasets", "--dry-run"]
    )
    assert c.only_datasets is True
    assert c.want_records is False


def test_argv_parses_dataset_filter_list():
    c = load_inject_config_from_argv(
        [
            "--database",
            "eegdash_dev",
            "--datasets",
            "ds-001",
            "ds-002",
            "ds-003",
            "--dry-run",
        ]
    )
    assert c.datasets == ["ds-001", "ds-002", "ds-003"]


def test_argv_validation_surfaces_via_validation_error():
    """An argparse-parsed config that fails validation raises
    ``pydantic.ValidationError`` (cleaner than argparse's stderr noise)."""
    with pytest.raises(ValidationError):
        load_inject_config_from_argv(
            [
                "--database",
                "eegdash_dev",
                "--only-datasets",
                "--only-records",
                "--dry-run",
            ]
        )


def test_argv_unknown_flag_rejected_by_argparse():
    """Unknown flags still produce argparse's SystemExit (preserves the
    CLI ergonomics)."""
    with pytest.raises(SystemExit):
        load_inject_config_from_argv(
            ["--database", "eegdash_dev", "--bogus-flag", "--dry-run"]
        )
