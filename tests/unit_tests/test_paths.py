from pathlib import Path
from unittest.mock import patch

import pytest

from eegdash.paths import get_default_cache_dir


@pytest.mark.parametrize("env_value", ["~/custom_eegdash_cache", "/opt/eegdash-cache"])
def test_get_default_cache_dir_from_env(monkeypatch, env_value):
    monkeypatch.setenv("EEGDASH_CACHE_DIR", env_value)

    assert get_default_cache_dir() == Path(env_value).expanduser().resolve()


def test_get_default_cache_dir_prefers_local_hidden_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)

    with patch("eegdash.paths.mne_get_config", return_value=str(tmp_path / "mne_data")):
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
def test_get_default_cache_dir_fallbacks_when_local_mkdir_fails(
    monkeypatch, tmp_path, mne_value, expected_name
):
    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)

    with patch.object(Path, "cwd", return_value=tmp_path):
        with patch.object(Path, "mkdir", side_effect=PermissionError("readonly")):
            with patch("eegdash.paths.mne_get_config", return_value=mne_value):
                resolved = get_default_cache_dir()

    assert resolved.name == expected_name
    if mne_value is None:
        assert resolved == tmp_path / ".eegdash_cache"
    else:
        assert resolved == Path(mne_value).expanduser().resolve()
