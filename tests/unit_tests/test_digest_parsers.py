"""Tests for format-specific fallback parsers in the digestion script."""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The digestion script uses relative imports (from _constants import ...)
# that only work when run from scripts/ingestions/. Add that dir to sys.path.
_INGESTIONS_DIR = str(Path(__file__).resolve().parents[2] / "scripts" / "ingestions")


@pytest.fixture()
def fif_parser():
    """Import _parse_fif_with_mne from the digestion script."""
    import importlib.util

    old_path = sys.path.copy()
    sys.path.insert(0, _INGESTIONS_DIR)
    try:
        spec = importlib.util.spec_from_file_location(
            "_digest_mod",
            Path(_INGESTIONS_DIR) / "3_digest.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._parse_fif_with_mne
    finally:
        sys.path = old_path


def test_parse_fif_nonexistent_file(fif_parser, tmp_path):
    """Returns (None, False) for a file that does not exist."""
    result, is_split = fif_parser(tmp_path / "nonexistent.fif")
    assert result is None
    assert is_split is False


def test_parse_fif_broken_symlink(fif_parser, tmp_path):
    """Returns (None, False) for a broken symlink (git-annex stub)."""
    link = tmp_path / "broken.fif"
    link.symlink_to(tmp_path / "target_does_not_exist.fif")
    result, is_split = fif_parser(link)
    assert result is None
    assert is_split is False


def test_parse_fif_extracts_metadata(fif_parser, tmp_path):
    """Extracts sfreq and channel info from a real FIF file via MNE."""
    import mne

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3"], sfreq=256.0, ch_types="eeg"
    )
    raw = mne.io.RawArray([[0] * 100, [0] * 100, [0] * 100], info)

    fif_path = tmp_path / "test_raw.fif"
    raw.save(str(fif_path), overwrite=True, verbose=False)

    result, is_split = fif_parser(fif_path)
    assert result is not None
    assert result["sampling_frequency"] == 256.0
    assert result["nchans"] == 3
    assert result["ch_names"] == ["EEG1", "EEG2", "EEG3"]
    assert is_split is False


def test_parse_fif_passes_on_split_missing_warn(fif_parser, tmp_path):
    """Calls read_raw_fif with on_split_missing='warn' and detects split files."""
    fif_path = tmp_path / "test.fif"
    fif_path.touch()

    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 1000.0, "ch_names": ["MEG1"]}

    with patch("mne.io.read_raw_fif", return_value=mock_raw) as mock_read:
        result, is_split = fif_parser(fif_path)

    mock_read.assert_called_once_with(
        str(fif_path), preload=False, on_split_missing="warn", verbose=False
    )
    assert result["sampling_frequency"] == 1000.0
    assert result["nchans"] == 1
    assert is_split is False


def test_parse_fif_detects_split_warning(fif_parser, tmp_path):
    """Sets is_split=True when MNE emits a split raw file warning."""
    fif_path = tmp_path / "test.fif"
    fif_path.touch()

    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 1000.0, "ch_names": ["MEG1"]}

    def fake_read(*args, **kwargs):
        warnings.warn("Split raw file detected. Missing continuation not found.")
        return mock_raw

    with patch("mne.io.read_raw_fif", side_effect=fake_read):
        result, is_split = fif_parser(fif_path)

    assert result["sampling_frequency"] == 1000.0
    assert is_split is True


def test_parse_fif_returns_none_on_read_error(fif_parser, tmp_path):
    """Returns (None, False) when MNE cannot read the file."""
    fif_path = tmp_path / "bad.fif"
    fif_path.write_bytes(b"not a fif file")

    result, is_split = fif_parser(fif_path)
    assert result is None
    assert is_split is False
