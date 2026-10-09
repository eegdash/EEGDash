import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import eegdash.paths as paths_module
from eegdash.paths import get_default_cache_dir


@pytest.fixture(autouse=True)
def _clean_cache_env(monkeypatch):
    """Clear every environment variable consulted by get_default_cache_dir."""
    for var in ("EEGDASH_CACHE_DIR", "SCRATCH", "TMPDIR"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("env_value", ["~/custom_eegdash_cache", "/opt/eegdash-cache"])
def test_get_default_cache_dir_from_env(monkeypatch, env_value):
    monkeypatch.setenv("EEGDASH_CACHE_DIR", env_value)

    assert get_default_cache_dir() == Path(env_value).expanduser().resolve()


def test_get_default_cache_dir_prefers_scratch_over_tmpdir(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    tmpdir = tmp_path / "tmpdir"
    scratch.mkdir()
    tmpdir.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setenv("TMPDIR", str(tmpdir))

    assert get_default_cache_dir() == scratch / "eegdash"


def test_get_default_cache_dir_uses_tmpdir(monkeypatch, tmp_path):
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))

    assert get_default_cache_dir() == tmpdir / "eegdash"


@pytest.mark.parametrize("var", ["SCRATCH", "TMPDIR"])
def test_get_default_cache_dir_skips_unwritable_env_dirs(monkeypatch, tmp_path, var):
    not_writable = tmp_path / var.lower()
    not_writable.mkdir()
    mne_dir = tmp_path / "mne_data"
    monkeypatch.setenv(var, str(not_writable))

    real_access = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: (
            False if Path(path) == not_writable else real_access(path, mode)
        ),
    )

    with patch("eegdash.paths.mne_get_config", return_value=str(mne_dir)):
        assert get_default_cache_dir() == mne_dir


def test_get_default_cache_dir_skips_missing_scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH", str(tmp_path / "does-not-exist"))
    mne_dir = tmp_path / "mne_data"

    with patch("eegdash.paths.mne_get_config", return_value=str(mne_dir)):
        assert get_default_cache_dir() == mne_dir


def test_get_default_cache_dir_uses_mne_data(monkeypatch, tmp_path):
    mne_dir = tmp_path / "mne_data"

    with patch("eegdash.paths.mne_get_config", return_value=str(mne_dir)):
        assert get_default_cache_dir() == mne_dir


def test_get_default_cache_dir_uses_platform_user_cache(monkeypatch, tmp_path):
    platform_cache = tmp_path / "user-cache" / "eegdash"

    with patch("eegdash.paths.mne_get_config", return_value=None):
        with patch("eegdash.paths.user_cache_dir", return_value=str(platform_cache)):
            resolved = get_default_cache_dir()

    assert resolved == platform_cache
    assert resolved.exists()


def test_get_default_cache_dir_falls_back_to_local_hidden_folder(monkeypatch, tmp_path):
    with patch("eegdash.paths.mne_get_config", return_value=None):
        with patch(
            "eegdash.paths.user_cache_dir",
            return_value=str(tmp_path / "a-file" / "eegdash"),
        ):
            # The platform cache cannot be created below an existing file.
            (tmp_path / "a-file").touch()
            with patch.object(Path, "cwd", return_value=tmp_path):
                resolved = get_default_cache_dir()

    assert resolved == tmp_path / ".eegdash_cache"
    assert resolved.exists()


@pytest.mark.parametrize(
    "mne_value,expected_name",
    [
        ("~/mne-data", "mne-data"),
        (None, ".eegdash_cache"),
    ],
)
def test_get_default_cache_dir_local_last_resort_when_mkdir_fails(
    monkeypatch, tmp_path, mne_value, expected_name
):
    with patch.object(Path, "cwd", return_value=tmp_path):
        with patch.object(Path, "mkdir", side_effect=PermissionError("readonly")):
            with patch("eegdash.paths.mne_get_config", return_value=mne_value):
                with patch(
                    "eegdash.paths.user_cache_dir",
                    return_value=str(tmp_path / "user-cache"),
                ):
                    resolved = get_default_cache_dir()

    assert resolved.name == expected_name
    if mne_value is None:
        assert resolved == tmp_path / ".eegdash_cache"
        assert not resolved.exists()
    else:
        assert resolved == Path(mne_value).expanduser().resolve()


def test_get_default_cache_dir_resolution_order(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    tmpdir = tmp_path / "tmpdir"
    scratch.mkdir()
    tmpdir.mkdir()
    env_dir = tmp_path / "from-env"
    mne_dir = tmp_path / "mne-data"
    platform_cache = tmp_path / "user-cache" / "eegdash"

    monkeypatch.setenv("EEGDASH_CACHE_DIR", str(env_dir))
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setenv("TMPDIR", str(tmpdir))

    with patch("eegdash.paths.mne_get_config", return_value=str(mne_dir)):
        with patch("eegdash.paths.user_cache_dir", return_value=str(platform_cache)):
            with patch.object(Path, "cwd", return_value=tmp_path):
                assert get_default_cache_dir() == env_dir

                monkeypatch.delenv("EEGDASH_CACHE_DIR")
                assert get_default_cache_dir() == scratch / "eegdash"

                monkeypatch.delenv("SCRATCH")
                assert get_default_cache_dir() == tmpdir / "eegdash"

                monkeypatch.delenv("TMPDIR")
                assert get_default_cache_dir() == mne_dir

                with patch("eegdash.paths.mne_get_config", return_value=None):
                    assert get_default_cache_dir() == platform_cache


def test_get_default_cache_dir_logs_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(paths_module, "_cache_dir_logged", False)
    monkeypatch.setenv("SCRATCH", str(tmp_path))

    with caplog.at_level(logging.INFO, logger="eegdash.paths"):
        get_default_cache_dir()
        get_default_cache_dir()

    messages = [
        record.getMessage()
        for record in caplog.records
        if "cache directory" in record.getMessage().lower()
    ]
    assert len(messages) == 1
    assert str(tmp_path / "eegdash") in messages[0]
