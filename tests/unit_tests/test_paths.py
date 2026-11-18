import logging
import os
from pathlib import Path

import pytest

import eegdash.paths as paths_module
from eegdash.paths import get_default_cache_dir


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Empty working directory, no cache env vars, no MNE_DATA, a temp user cache."""
    for var in ("EEGDASH_CACHE_DIR", "SCRATCH", "TMPDIR"):
        monkeypatch.delenv(var, raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(paths_module, "mne_get_config", lambda key: None)
    monkeypatch.setattr(
        paths_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "user-cache")
    )
    return tmp_path


@pytest.mark.parametrize("env_value", ["~/custom_eegdash_cache", "/opt/eegdash-cache"])
def test_env_var_wins(monkeypatch, env_value):
    monkeypatch.setenv("EEGDASH_CACHE_DIR", env_value)
    assert get_default_cache_dir() == Path(env_value).expanduser().resolve()


def test_resolution_order(monkeypatch, isolated):
    scratch, mne_dir = isolated / "scratch", isolated / "mne"
    scratch.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setattr(paths_module, "mne_get_config", lambda key: str(mne_dir))
    local = Path.cwd() / ".eegdash_cache"

    local.mkdir()  # an existing working-directory cache is kept
    assert get_default_cache_dir() == local
    local.rmdir()
    assert get_default_cache_dir() == scratch / "eegdash"
    monkeypatch.delenv("SCRATCH")
    assert get_default_cache_dir() == mne_dir
    monkeypatch.setattr(paths_module, "mne_get_config", lambda key: None)
    assert get_default_cache_dir() == isolated / "user-cache"
    assert not local.exists()


def test_tmpdir_is_not_used(monkeypatch, isolated):
    # TMPDIR is purged by macOS and wiped after a Slurm job: never a default.
    monkeypatch.setenv("TMPDIR", str(isolated))
    assert get_default_cache_dir() == isolated / "user-cache"


@pytest.mark.parametrize("writable", [False, None])  # unwritable, missing
def test_unusable_scratch_is_skipped(monkeypatch, isolated, writable):
    scratch = isolated / "scratch"
    if writable is False:
        scratch.mkdir()
        real = os.access
        monkeypatch.setattr(
            os, "access", lambda p, m: False if Path(p) == scratch else real(p, m)
        )
    monkeypatch.setenv("SCRATCH", str(scratch))
    assert get_default_cache_dir() == isolated / "user-cache"


def test_working_directory_is_the_last_resort(monkeypatch, isolated):
    (isolated / "a-file").touch()  # the user cache cannot be created below a file
    monkeypatch.setattr(
        paths_module, "user_cache_dir", lambda *a, **k: str(isolated / "a-file" / "x")
    )
    resolved = get_default_cache_dir()
    assert resolved == Path.cwd() / ".eegdash_cache"
    assert resolved.is_dir()


def test_logs_once(monkeypatch, caplog):
    monkeypatch.setattr(paths_module, "_cache_dir_logged", False)
    with caplog.at_level(logging.INFO, logger="eegdash.paths"):
        get_default_cache_dir()
        get_default_cache_dir()
    assert sum("cache directory" in r.getMessage() for r in caplog.records) == 1
