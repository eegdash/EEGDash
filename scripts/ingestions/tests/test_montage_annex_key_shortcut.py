"""Tests for the annex-key size shortcut (Task 2 — Perf sprint).

OpenNeuro / NEMAR FIF files are git-annex-managed; the file size is
encoded in the symlink target (MD5E-s{size}--{hash}.fif). Reading
the symlink first lets us skip a network HEAD round-trip for every
MEG record. Non-annex datasets (Zenodo, Figshare with raw S3 URLs)
fall back to head_content_length.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _montage import _resolve_fif_total_size


def test_returns_annex_size_when_symlink_present(tmp_path: Path) -> None:
    """Broken git-annex symlink → size parsed from symlink target."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    # Annex key format: MD5E-s{size}--{hash}.{ext}
    target = "../.git/annex/objects/aa/bb/MD5E-s4194304--abc123def.fif/MD5E-s4194304--abc123def.fif"
    fif.symlink_to(target)
    assert not fif.exists()  # broken — annex content not fetched

    with patch("_montage.head_content_length") as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 4_194_304
    mock_head.assert_not_called()  # No HEAD round-trip


def test_falls_back_to_head_when_no_annex_symlink(tmp_path: Path) -> None:
    """Plain file (no annex) → HEAD round-trip."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"\x00" * 1024)  # 1 KB regular file

    with patch("_montage.head_content_length", return_value=8_192) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    # Annex size returns 0 for non-annex, falls through to HEAD.
    assert size == 8_192
    mock_head.assert_called_once()


def test_falls_back_to_head_on_malformed_annex_key(tmp_path: Path) -> None:
    """Symlink target doesn't match MD5E-s{size}-- pattern → HEAD fallback."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("not-an-annex-key.fif")

    with patch("_montage.head_content_length", return_value=2_048) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 2_048
    mock_head.assert_called_once()


def test_returns_none_when_neither_annex_nor_head_succeeds(
    tmp_path: Path,
) -> None:
    """All paths exhausted → None (caller treats as transient failure)."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.write_bytes(b"")

    with patch("_montage.head_content_length", return_value=None):
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size is None


def test_zero_byte_annex_key_falls_back_to_head(tmp_path: Path) -> None:
    """An annex key reporting size=0 is nonsensical for FIF — fall back."""
    fif = tmp_path / "sub-01_run-01_meg.fif"
    fif.symlink_to("../.git/annex/objects/aa/bb/MD5E-s0--abc.fif/MD5E-s0--abc.fif")

    with patch("_montage.head_content_length", return_value=5_000) as mock_head:
        size = _resolve_fif_total_size(fif, "https://s3.example.com/sub-01.fif")

    assert size == 5_000
    mock_head.assert_called_once()
