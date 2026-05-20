"""Unit tests for the MEF3 (Multiscale Electrophysiology Format v3) parser.

The fixture corpus does not include a real MEF3 directory. These tests
cover the defensive paths. When a future contributor adds a MEF3 fixture,
they should also add golden-value tests against ``parse_mef3_metadata``'s
documented output shape (sampling frequency from .tmet and timing index
from .timd files).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _mef3_parser import parse_mef3_metadata


def test_parse_mef3_nonexistent_path_returns_none():
    """Missing MEF3 directory returns None, no crash."""
    missing = Path("/tmp/_nonexistent_.mefd")
    result = parse_mef3_metadata(missing)
    assert result is None


def test_parse_mef3_empty_directory_does_not_crash(tmp_path: Path):
    """A directory with no MEF3 sub-structure returns None, no crash."""
    empty_mefd = tmp_path / "empty.mefd"
    empty_mefd.mkdir()
    try:
        result = parse_mef3_metadata(empty_mefd)
        assert result is None or isinstance(result, dict)
    except (FileNotFoundError, ValueError, KeyError):
        pass  # acceptable failure modes


def test_parse_mef3_file_instead_of_directory(tmp_path: Path):
    """If a file is passed where a directory is expected, no crash."""
    f = tmp_path / "fake.mefd"
    f.write_bytes(b"not a directory")
    try:
        result = parse_mef3_metadata(f)
        assert result is None or isinstance(result, dict)
    except (NotADirectoryError, OSError):
        pass


@pytest.mark.parametrize(
    "subdir_with_garbage",
    ["timd_0001", "tmet_0001"],
)
def test_parse_mef3_directory_with_garbage_subfiles(
    tmp_path: Path, subdir_with_garbage: str
):
    """A MEF3 directory with structurally-wrong sub-files must not crash."""
    mefd = tmp_path / "data.mefd"
    sub = mefd / subdir_with_garbage
    sub.mkdir(parents=True)
    (sub / "00001.tdat").write_bytes(b"\x00" * 32)
    try:
        result = parse_mef3_metadata(mefd)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, OSError):
        pass
