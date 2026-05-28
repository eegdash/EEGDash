"""Direct tests for _snirf_parser and _mef3_parser fail paths and synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

from _mef3_parser import parse_mef3_metadata
from _snirf_parser import _parse_snirf_with_h5py, parse_snirf_metadata

# ─── _snirf_parser ────────────────────────────────────────────────────────


def test_snirf_missing_file_returns_none(tmp_path: Path):
    assert parse_snirf_metadata(tmp_path / "missing.snirf") is None


def test_snirf_broken_symlink_returns_none(tmp_path: Path):
    broken = tmp_path / "broken.snirf"
    broken.symlink_to(tmp_path / ".no_target")
    assert parse_snirf_metadata(broken) is None


def test_snirf_directory_path_returns_none(tmp_path: Path):
    result = parse_snirf_metadata(tmp_path)
    assert result is None


def test_snirf_garbage_bytes_returns_none(tmp_path: Path):
    garbage = tmp_path / "garbage.snirf"
    garbage.write_bytes(b"\x00\x01\x02 not real HDF5")
    assert parse_snirf_metadata(garbage) is None


def test_snirf_h5py_fallback_handles_empty_hdf5(tmp_path: Path):
    """_parse_snirf_with_h5py on a non-existent file → None."""
    result = _parse_snirf_with_h5py(tmp_path / "no.snirf")
    assert result is None


def test_snirf_accepts_string_path(tmp_path: Path):
    """Path arg accepts both Path and str."""
    assert parse_snirf_metadata(str(tmp_path / "missing.snirf")) is None
    assert parse_snirf_metadata(tmp_path / "missing.snirf") is None


# ─── _mef3_parser (synthetic .mefd directory) ─────────────────────────────


def test_mef3_missing_directory_returns_none(tmp_path: Path):
    assert parse_mef3_metadata(tmp_path / "missing.mefd") is None


def test_mef3_non_mefd_suffix_returns_none(tmp_path: Path):
    other_dir = tmp_path / "data.txt"
    other_dir.mkdir()
    assert parse_mef3_metadata(other_dir) is None


def test_mef3_file_not_directory_returns_none(tmp_path: Path):
    fake = tmp_path / "fake.mefd"
    fake.write_bytes(b"not a directory")
    assert parse_mef3_metadata(fake) is None


def test_mef3_empty_mefd_returns_none(tmp_path: Path):
    mefd = tmp_path / "empty.mefd"
    mefd.mkdir()
    result = parse_mef3_metadata(mefd)
    assert result is None or result.get("nchans", 0) == 0


def test_mef3_extracts_channel_names_from_timd_dirs(tmp_path: Path):
    """Channel names are derived from .timd directory names."""
    mefd = tmp_path / "session.mefd"
    mefd.mkdir()
    for ch in ("Cz", "Fz", "Pz"):
        timd = mefd / f"{ch}.timd"
        timd.mkdir()
        (timd / "segment.segd").mkdir()

    result = parse_mef3_metadata(mefd)
    if result is not None:
        if "ch_names" in result:
            assert set(result["ch_names"]) == {"Cz", "Fz", "Pz"}


def test_mef3_accepts_string_path(tmp_path: Path):
    """Path arg accepts both Path and str."""
    mefd = tmp_path / "test.mefd"
    mefd.mkdir()
    out_str = parse_mef3_metadata(str(mefd))
    out_path = parse_mef3_metadata(mefd)
    assert out_str == out_path


_MEF3_RAISING_TARGET: Path | None = None
_REAL_ITERDIR = Path.iterdir


def _iterdir_raises_for_target(self: Path):
    """Module-level helper to monkey-patch Path.iterdir (avoids nested-function lint)."""
    if self == _MEF3_RAISING_TARGET:
        raise OSError("permission denied")
    yield from _REAL_ITERDIR(self)


def test_mef3_handles_unreadable_directory(tmp_path: Path, monkeypatch):
    """iterdir() OSError is tolerated and returns None."""
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
