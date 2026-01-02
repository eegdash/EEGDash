from pathlib import Path
from unittest.mock import patch

from eegdash.paths import get_default_cache_dir


def test_get_default_cache_dir_env(monkeypatch):
    # Case 1: EEGDASH_CACHE_DIR environment variable is set
    mock_path = "/tmp/mock_eegdash_cache"
    monkeypatch.setenv("EEGDASH_CACHE_DIR", mock_path)

    path = get_default_cache_dir()
    assert path == Path(mock_path).resolve()


def test_get_default_cache_dir_local(monkeypatch, tmp_path):
    # Case 2: No env var, no MNE config -> defaults to local .eegdash_cache
    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)

    with patch("eegdash.paths.mne_get_config", return_value=None):
        with patch.object(Path, "cwd", return_value=tmp_path):
            path = get_default_cache_dir()
            expected = tmp_path / ".eegdash_cache"
            assert path == expected
            assert path.exists()


def test_get_default_cache_dir_mne(monkeypatch, tmp_path):
    # Case 3: No env var, MNE_DATA config is set -> fallback to MNE_DATA
    monkeypatch.delenv("EEGDASH_CACHE_DIR", raising=False)

    mne_data_path = tmp_path / "mne_data"

    # We simulate that local .eegdash_cache creation fails or we prefer checking priority order?
    # Actually, the code checks local *before* MNE data if local creation succeeds.
    # Wait, reading the code:
    # 1. Env var
    # 2. Local .eegdash_cache (if writable)
    # 3. MNE_DATA

    # So to test MNE_DATA, we need local creation to fail OR we need to verify priority.
    # The docstring says: "2. A hidden directory ... 3. ... MNE_DATA (fallback)"
    # But the code says:
    # 44: local = Path.cwd() / ".eegdash_cache"
    # 45: try: local.mkdir(...) return local
    # So if local creation works, it returns local!

    # Thus, MNE_DATA is only reached if local.mkdir raises Exception (e.g. read-only fs)

    with patch("eegdash.paths.mne_get_config", return_value=str(mne_data_path)):
        # Mock mkdir to raise PermissionError
        with patch.object(Path, "mkdir", side_effect=PermissionError):
            path = get_default_cache_dir()
            assert path == mne_data_path.resolve()
