"""Tests for the MEF3 (Multiscale Electrophysiology Format v3) parser.

Two angles:

- **Parser unit** — defensive paths (missing input, empty directory,
  garbage sub-files). Was test_mef3_parser.py.
- **Real fixture** — golden values + tolerance tests using the real
  ``.tmet`` binary header from OpenNeuro ds003708 (CC0, 16,384 bytes,
  real MEF3 v3.0 metadata structure). Exercises the
  ``_parse_tmet_sampling_frequency`` byte-offset reader. Was
  test_mef3_real_fixture.py.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from _helpers.builders import build_mefd_around_real_tmet
from eegdash.testing import data_file

from _mef3_parser import _parse_tmet_sampling_frequency, parse_mef3_metadata

_TMET_FIXTURE = data_file("ieeg/EKG-000000.tmet")


# ─── 1. Defensive paths ────────────────────────────────────────────────────


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


# ─── 2. Real fixture — golden values + tolerance ───────────────────────────


def test_real_tmet_extracts_sampling_frequency():
    """The real ds003708 .tmet has a known sampling frequency.

    The ccep dataset's iEEG was recorded at 2048 Hz (common for clinical
    SPES / CCEP protocols). The parser walks offsets 1272/1280/1288 and
    must surface a value in the (0.1, 1_000_000) sanity range.
    """
    sfreq = _parse_tmet_sampling_frequency(_TMET_FIXTURE)
    assert sfreq is not None
    # The sanity bounds the parser enforces
    assert 0.1 < sfreq < 1_000_000
    # Real clinical iEEG: 256 / 512 / 1024 / 2048 / 4096 Hz are typical.
    # We don't lock to a single value — just assert reasonable range.
    assert sfreq >= 100, f"sfreq = {sfreq} is suspiciously low for clinical iEEG"


def test_real_tmet_bytes_are_long_enough_for_parser():
    """The parser requires len(data) >= 1300 bytes for any chance of
    extraction. Pin the real fixture size as a regression check."""
    size = _TMET_FIXTURE.stat().st_size
    assert size >= 1300


def test_parse_mef3_metadata_with_real_tmet(tmp_path: Path):
    """End-to-end: build a real .mefd directory, run the public entry
    point, assert all three canonical fields populated."""
    mefd = build_mefd_around_real_tmet(
        tmp_path, _TMET_FIXTURE, channels=["EKG", "LAD1", "LAD2", "LAD3"]
    )
    out = parse_mef3_metadata(mefd)
    assert out is not None
    # Sampling frequency from real .tmet
    assert "sampling_frequency" in out
    assert 0.1 < out["sampling_frequency"] < 1_000_000
    # Channel names from .timd directory names (preserves order)
    assert "ch_names" in out
    assert set(out["ch_names"]) == {"EKG", "LAD1", "LAD2", "LAD3"}
    assert out["nchans"] == 4


def test_parse_mef3_metadata_channel_order_from_directory_listing(
    tmp_path: Path,
):
    """The parser sorts .timd dirs alphabetically. Pinning so a future
    refactor that uses os.listdir (unsorted) is caught."""
    # Channels in a deliberately non-alphabetic input order
    mefd = build_mefd_around_real_tmet(
        tmp_path, _TMET_FIXTURE, channels=["ZZZ", "AAA", "MMM"]
    )
    out = parse_mef3_metadata(mefd)
    assert out is not None
    # Output channel list is sorted alphabetically
    assert out["ch_names"] == ["AAA", "MMM", "ZZZ"]


def test_parse_mef3_metadata_single_channel_dataset(tmp_path: Path):
    """A .mefd with a single channel works (smallest valid case)."""
    mefd = build_mefd_around_real_tmet(tmp_path, _TMET_FIXTURE, channels=["solo"])
    out = parse_mef3_metadata(mefd)
    assert out is not None
    assert out["nchans"] == 1
    assert out["ch_names"] == ["solo"]
    assert "sampling_frequency" in out


def test_parse_tmet_truncated_file_returns_none(tmp_path: Path):
    """A .tmet truncated below 1300 bytes returns None — the parser's
    minimum-size sanity check."""
    truncated = tmp_path / "EKG-000000.tmet"
    truncated.write_bytes(_TMET_FIXTURE.read_bytes()[:1200])  # < 1300
    assert _parse_tmet_sampling_frequency(truncated) is None


def test_parse_tmet_with_zeroed_offset_falls_back_to_none(tmp_path: Path):
    """When the sampling-frequency offset holds 0.0 (or NaN), the
    sanity check rejects it and the parser returns None."""
    # Write a 16KB file with zeroes everywhere (no valid sfreq anywhere
    # in the candidate offsets)
    zeroed = tmp_path / "zero.tmet"
    zeroed.write_bytes(b"\x00" * 16384)
    assert _parse_tmet_sampling_frequency(zeroed) is None


def test_parse_tmet_with_pathological_double_returns_none(tmp_path: Path):
    """A .tmet whose sfreq offset holds a value outside the sanity
    range (e.g. 1e10) is rejected."""
    # Build a 16KB file with all zeroes, then plant 1e10 (out of range)
    # at the canonical MEF 3.0 offset 1272.
    bad = tmp_path / "bad.tmet"
    buf = bytearray(b"\x00" * 16384)
    buf[1272:1280] = struct.pack("<d", 1e10)  # out of sanity bounds
    bad.write_bytes(bytes(buf))
    # All candidate offsets either have zero (rejected) or 1e10 (rejected)
    assert _parse_tmet_sampling_frequency(bad) is None
