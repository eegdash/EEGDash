"""Happy-path tests for _mef3_parser using a real MEF3 .tmet fixture.

C3.2 covered the fail paths via synthetic
.mefd directories. This file uses a real ``.tmet`` binary header
extracted from OpenNeuro ds003708 (CC0) to exercise the
``_parse_tmet_sampling_frequency`` byte-offset reader.

The fixture comes from:
  ``s3://openneuro.org/ds003708/sub-01/ses-ieeg01/ieeg/
    sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd/
    EKG.timd/EKG-000000.segd/EKG-000000.tmet``

License: CC0 (OpenNeuro standard). Size: 16,384 bytes. Real MEF3 v3.0
metadata structure; the parser walks it at offsets 1272/1280/1288 for
sampling frequency stored as little-endian double.

If the fixture file is missing (e.g., on a fresh checkout), the
whole module skips via ``pytestmark = pytest.mark.skipif(...)``.
This is the same pattern test_snirf_happy_path.py uses for h5py
absence, but here it's the fixture rather than the library.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

# Real .tmet from OpenNeuro ds003708 (CC0), shipped in the
# ``eegdash-testing-data`` corpus. Skip the module if the corpus is
# unavailable (offline + no cache hit).
try:
    from eegdash.testing import data_file

    _TMET_FIXTURE = data_file("ieeg/EKG-000000.tmet")
    _SKIP_REASON: str | None = None
except Exception as exc:  # fetch failure = skip
    _TMET_FIXTURE = Path(__file__).parent / "_unreachable_.tmet"
    _SKIP_REASON = f"eegdash-testing-data unavailable: {exc}"

pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=_SKIP_REASON or "",
)

from _mef3_parser import (
    _parse_tmet_sampling_frequency,
    parse_mef3_metadata,
)


def _build_mefd_around_real_tmet(
    tmp_path: Path, *, channels: list[str] | None = None
) -> Path:
    """Construct a .mefd directory using the real .tmet fixture.

    Per the MEF 3.0 directory layout, each channel needs its own
    ``<channel>.timd/<channel>-000000.segd/<channel>-000000.tmet``
    path. We copy the same real .tmet into each channel's location
    so every channel "loads" the canonical fixture.

    Returns the .mefd path so callers can pass it to
    ``parse_mef3_metadata``.
    """
    if channels is None:
        channels = ["EKG", "LAD1", "LAD2"]

    mefd = tmp_path / "sub-test_ieeg.mefd"
    mefd.mkdir()
    for ch in channels:
        segd = mefd / f"{ch}.timd" / f"{ch}-000000.segd"
        segd.mkdir(parents=True)
        # Copy the real fixture's bytes into each channel's .tmet
        shutil.copy(_TMET_FIXTURE, segd / f"{ch}-000000.tmet")

    return mefd


# ─── _parse_tmet_sampling_frequency on the real fixture ───────────────────


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


# ─── parse_mef3_metadata against a real .mefd structure ───────────────────


def test_parse_mef3_metadata_with_real_tmet(tmp_path: Path):
    """End-to-end: build a real .mefd directory, run the public entry
    point, assert all three canonical fields populated."""
    mefd = _build_mefd_around_real_tmet(
        tmp_path, channels=["EKG", "LAD1", "LAD2", "LAD3"]
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
    mefd = _build_mefd_around_real_tmet(tmp_path, channels=["ZZZ", "AAA", "MMM"])
    out = parse_mef3_metadata(mefd)
    assert out is not None
    # Output channel list is sorted alphabetically
    assert out["ch_names"] == ["AAA", "MMM", "ZZZ"]


def test_parse_mef3_metadata_single_channel_dataset(tmp_path: Path):
    """A .mefd with a single channel works (smallest valid case)."""
    mefd = _build_mefd_around_real_tmet(tmp_path, channels=["solo"])
    out = parse_mef3_metadata(mefd)
    assert out is not None
    assert out["nchans"] == 1
    assert out["ch_names"] == ["solo"]
    assert "sampling_frequency" in out


# ─── Truncation / corruption tolerance ───────────────────────────────────


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
    import struct

    # Build a 16KB file with all zeroes, then plant 1e10 (out of range)
    # at the canonical MEF 3.0 offset 1272.
    bad = tmp_path / "bad.tmet"
    buf = bytearray(b"\x00" * 16384)
    buf[1272:1280] = struct.pack("<d", 1e10)  # out of sanity bounds
    bad.write_bytes(bytes(buf))
    # All candidate offsets either have zero (rejected) or 1e10 (rejected)
    assert _parse_tmet_sampling_frequency(bad) is None
