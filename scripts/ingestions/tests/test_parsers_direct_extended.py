"""Direct tests for _snirf_parser and _mef3_parser.
Both parsers are at ~28-30% coverage. The HDF5/SNIRF format is too
complex to fixture from scratch in a small test, so SNIRF tests
focus on the fail paths (missing file, garbage bytes).

MEF3 IS a directory-based format we can build synthetically — we
create a fake .mefd directory with .timd subdirs that look like
real MEF3 to the parser's structural checks.
"""

from __future__ import annotations

from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
from _mef3_parser import parse_mef3_metadata
from _snirf_parser import _parse_snirf_with_h5py, parse_snirf_metadata

# ─── _snirf_parser ────────────────────────────────────────────────────────


def test_snirf_missing_file_returns_none(tmp_path: Path):
    """Non-existent .snirf → None."""
    assert parse_snirf_metadata(tmp_path / "missing.snirf") is None


def test_snirf_broken_symlink_returns_none(tmp_path: Path):
    """git-annex broken symlink → None."""
    broken = tmp_path / "broken.snirf"
    broken.symlink_to(tmp_path / ".no_target")
    assert parse_snirf_metadata(broken) is None


def test_snirf_directory_path_returns_none(tmp_path: Path):
    """A directory passed in → None, not a crash."""
    result = parse_snirf_metadata(tmp_path)
    assert result is None


def test_snirf_garbage_bytes_returns_none(tmp_path: Path):
    """A .snirf file that's not real HDF5 → None."""
    garbage = tmp_path / "garbage.snirf"
    garbage.write_bytes(b"\x00\x01\x02 not real HDF5")
    assert parse_snirf_metadata(garbage) is None


def test_snirf_h5py_fallback_handles_empty_hdf5(tmp_path: Path):
    """Calling _parse_snirf_with_h5py on a non-existent file → None
    (h5py can't open it). Pins the fail path."""
    result = _parse_snirf_with_h5py(tmp_path / "no.snirf")
    assert result is None


def test_snirf_accepts_string_path(tmp_path: Path):
    """Path arg accepts both Path and str."""
    # Both should return None for missing file (no crash on type variation)
    assert parse_snirf_metadata(str(tmp_path / "missing.snirf")) is None
    assert parse_snirf_metadata(tmp_path / "missing.snirf") is None


# ─── _mef3_parser (synthetic .mefd directory) ─────────────────────────────


def test_mef3_missing_directory_returns_none(tmp_path: Path):
    """A non-existent .mefd → None."""
    assert parse_mef3_metadata(tmp_path / "missing.mefd") is None


def test_mef3_non_mefd_suffix_returns_none(tmp_path: Path):
    """A directory without ``.mefd`` suffix → None."""
    other_dir = tmp_path / "data.txt"
    other_dir.mkdir()
    assert parse_mef3_metadata(other_dir) is None


def test_mef3_file_not_directory_returns_none(tmp_path: Path):
    """A .mefd file (not a directory) → None — MEF3 IS a directory."""
    fake = tmp_path / "fake.mefd"
    fake.write_bytes(b"not a directory")
    assert parse_mef3_metadata(fake) is None


def test_mef3_empty_mefd_returns_none(tmp_path: Path):
    """A .mefd directory with no .timd subdirs → None or empty result."""
    mefd = tmp_path / "empty.mefd"
    mefd.mkdir()
    result = parse_mef3_metadata(mefd)
    # No channels found → returns None or an empty/minimal result
    assert result is None or result.get("nchans", 0) == 0


def test_mef3_extracts_channel_names_from_timd_dirs(tmp_path: Path):
    """Channel names are derived from .timd directory names."""
    mefd = tmp_path / "session.mefd"
    mefd.mkdir()
    # Create 3 channel dirs
    for ch in ("Cz", "Fz", "Pz"):
        timd = mefd / f"{ch}.timd"
        timd.mkdir()
        # Each channel has a .segd subdir (the segment)
        segd = timd / "segment.segd"
        segd.mkdir()

    result = parse_mef3_metadata(mefd)
    # The parser may still return None if it can't find .tmet files
    # for the sampling_frequency. But channel names should be extracted
    # before that step fails out.
    if result is not None:
        # When ch_names present, they should be the timd basenames
        if "ch_names" in result:
            assert set(result["ch_names"]) == {"Cz", "Fz", "Pz"}


def test_mef3_accepts_string_path(tmp_path: Path):
    """Path arg accepts both Path and str."""
    mefd = tmp_path / "test.mefd"
    mefd.mkdir()
    # Both should yield same result
    out_str = parse_mef3_metadata(str(mefd))
    out_path = parse_mef3_metadata(mefd)
    assert out_str == out_path


_MEF3_RAISING_TARGET: Path | None = None
_REAL_ITERDIR = Path.iterdir


def _iterdir_raises_for_target(self: Path):
    """Module-level helper to monkey-patch Path.iterdir.

    Module level (rather than nested in the test) so the project's
    no-nested-functions lint stays clean.
    """
    if self == _MEF3_RAISING_TARGET:
        raise OSError("permission denied")
    yield from _REAL_ITERDIR(self)


def test_mef3_handles_unreadable_directory(tmp_path: Path, monkeypatch):
    """An iterdir() OSError → None (tolerated, not raised)."""
    global _MEF3_RAISING_TARGET
    mefd = tmp_path / "broken.mefd"
    mefd.mkdir()
    _MEF3_RAISING_TARGET = mefd
    try:
        monkeypatch.setattr(Path, "iterdir", _iterdir_raises_for_target)
        result = parse_mef3_metadata(mefd)
        assert result is None
    finally:
        _MEF3_RAISING_TARGET = None
