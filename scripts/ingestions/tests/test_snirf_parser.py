"""Unit tests for the fNIRS SNIRF parser.

The fixture corpus does not currently include a real .snirf file — these
tests cover the defensive paths (missing input, garbage, wrong file type).
A future contributor adding a SNIRF fixture should ALSO add golden-value
tests against ``parse_snirf_metadata``'s documented output shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _snirf_parser import parse_snirf_metadata


def test_parse_snirf_nonexistent_path_returns_none():
    """Missing files return None, do not crash."""
    missing = Path("/tmp/_nonexistent_.snirf")
    result = parse_snirf_metadata(missing)
    assert result is None


@pytest.mark.parametrize(
    "garbage",
    [b"", b"\x00" * 16, b"NOT HDF5", b"\xff\xfe\xfd\xfc" * 64],
)
def test_parse_snirf_garbage_input_does_not_crash(tmp_path: Path, garbage: bytes):
    """Various malformed inputs must not crash the process.

    SNIRF is an HDF5 container; non-HDF5 input must be rejected gracefully.
    """
    f = tmp_path / "garbage.snirf"
    f.write_bytes(garbage)
    try:
        result = parse_snirf_metadata(f)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, OSError, RuntimeError):
        pass  # documented failure modes


def test_parse_snirf_directory_path_does_not_crash(tmp_path: Path):
    """Passing a directory must not crash."""
    try:
        result = parse_snirf_metadata(tmp_path)
        assert result is None or isinstance(result, dict)
    except (IsADirectoryError, OSError, PermissionError):
        pass
