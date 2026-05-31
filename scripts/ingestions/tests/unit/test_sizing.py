"""File-size resolution without reading signal data (Phase 2)."""

from __future__ import annotations

import os
from pathlib import Path

from _sizing import data_file_size


def test_size_of_real_file(tmp_path: Path):
    p = tmp_path / "x.eeg"
    p.write_bytes(b"\x00" * 4096)
    assert data_file_size(p) == 4096


def test_size_from_broken_annex_symlink(tmp_path: Path):
    # A git-annex *broken symlink* whose target encodes the real size.
    target = (
        "../../.git/annex/objects/aa/bb/"
        "MD5E-s123456--abcdef0123456789.eeg/MD5E-s123456--abcdef0123456789.eeg"
    )
    link = tmp_path / "sub-01_eeg.eeg"
    os.symlink(target, link)  # broken on purpose
    assert data_file_size(link) == 123456


def test_size_from_annex_pointer_text(tmp_path: Path):
    # A git-annex pointer *file* (small text file whose content is the key).
    p = tmp_path / "sub-02_eeg.eeg"
    p.write_text("/annex/objects/MD5E-s987654--0123456789abcdef.eeg\n")
    assert data_file_size(p) == 987654


def test_size_missing_returns_none(tmp_path: Path):
    assert data_file_size(tmp_path / "nope.eeg") is None
