"""Tests for the MEF3 parser: defensive paths and golden values on real fixture."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from _helpers.builders import build_mefd_around_real_tmet

from _mef3_parser import _parse_tmet_sampling_frequency, parse_mef3_metadata
from eegdash.testing import data_file

_TMET_FIXTURE = data_file("ieeg/EKG-000000.tmet")


# ─── 1. Defensive paths ────────────────────────────────────────────────────


def test_parse_mef3_nonexistent_path_returns_none():
    missing = Path("/tmp/_nonexistent_.mefd")
    result = parse_mef3_metadata(missing)
    assert result is None


def test_parse_mef3_empty_directory_does_not_crash(tmp_path: Path):
    empty_mefd = tmp_path / "empty.mefd"
    empty_mefd.mkdir()
    try:
        result = parse_mef3_metadata(empty_mefd)
        assert result is None or isinstance(result, dict)
    except (FileNotFoundError, ValueError, KeyError):
        pass  # acceptable failure modes


def test_parse_mef3_file_instead_of_directory(tmp_path: Path):
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
    mefd = tmp_path / "data.mefd"
    sub = mefd / subdir_with_garbage
    sub.mkdir(parents=True)
    (sub / "00001.tdat").write_bytes(b"\x00" * 32)
    try:
        result = parse_mef3_metadata(mefd)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, OSError):
        pass


# ─── 2. Real fixture — golden values + tolerance ───────────────────────────


def test_real_tmet_extracts_sampling_frequency():
    """ds003708 .tmet: parser walks offsets 1272/1280/1288 and must surface a sfreq in (0.1, 1_000_000)."""
    sfreq = _parse_tmet_sampling_frequency(_TMET_FIXTURE)
    assert sfreq is not None
    assert 0.1 < sfreq < 1_000_000
    assert sfreq >= 100, f"sfreq = {sfreq} is suspiciously low for clinical iEEG"


def test_real_tmet_bytes_are_long_enough_for_parser():
    """Parser requires >= 1300 bytes; pin fixture size as regression check."""
    size = _TMET_FIXTURE.stat().st_size
    assert size >= 1300


@pytest.mark.parametrize(
    ("channels", "expected_sorted"),
    [
        pytest.param(
            ["EKG", "LAD1", "LAD2", "LAD3"],
            ["EKG", "LAD1", "LAD2", "LAD3"],
            id="alphabetic_in_alphabetic_out",
        ),
        pytest.param(
            ["ZZZ", "AAA", "MMM"],
            ["AAA", "MMM", "ZZZ"],
            id="non_alphabetic_input_sorted_output",
        ),
        pytest.param(["solo"], ["solo"], id="single_channel"),
    ],
)
def test_parse_mef3_metadata_channel_layout(
    tmp_path: Path, channels: list[str], expected_sorted: list[str]
):
    """Build a .mefd dir and assert canonical (nchans, sorted ch_names, sampling_frequency) shape."""
    mefd = build_mefd_around_real_tmet(tmp_path, _TMET_FIXTURE, channels=channels)
    out = parse_mef3_metadata(mefd)
    assert out is not None
    assert out["nchans"] == len(expected_sorted)
    assert out["ch_names"] == expected_sorted
    assert "sampling_frequency" in out
    assert 0.1 < out["sampling_frequency"] < 1_000_000


def test_parse_tmet_truncated_file_returns_none(tmp_path: Path):
    """A .tmet truncated below 1300 bytes returns None (minimum-size check)."""
    truncated = tmp_path / "EKG-000000.tmet"
    truncated.write_bytes(_TMET_FIXTURE.read_bytes()[:1200])  # < 1300
    assert _parse_tmet_sampling_frequency(truncated) is None


def test_parse_tmet_with_zeroed_offset_falls_back_to_none(tmp_path: Path):
    """Sampling-frequency offset holding 0.0 is rejected by the sanity check."""
    zeroed = tmp_path / "zero.tmet"
    zeroed.write_bytes(b"\x00" * 16384)
    assert _parse_tmet_sampling_frequency(zeroed) is None


def test_parse_tmet_with_pathological_double_returns_none(tmp_path: Path):
    """sfreq offset holding an out-of-range value (e.g. 1e10) is rejected."""
    bad = tmp_path / "bad.tmet"
    buf = bytearray(b"\x00" * 16384)
    buf[1272:1280] = struct.pack("<d", 1e10)
    bad.write_bytes(bytes(buf))
    assert _parse_tmet_sampling_frequency(bad) is None
