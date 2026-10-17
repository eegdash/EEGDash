"""Unit tests for the format-parser registry."""

from __future__ import annotations

from pathlib import Path

from _format_parser_registry import (
    FormatParserResult,
    _parse_edf_with_mne,
    get_parser_for_extension,
    registered_extensions,
)

# ─── Registry coverage ────────────────────────────────────────────────────


def test_registry_covers_six_known_extensions():
    expected = {".edf", ".bdf", ".set", ".vhdr", ".snirf", ".mefd"}
    assert set(registered_extensions()) == expected


def test_get_parser_returns_none_for_unknown_extension():
    assert get_parser_for_extension(".nii") is None
    assert get_parser_for_extension("") is None
    assert get_parser_for_extension(".unknown") is None


def test_get_parser_returns_callable_for_known_extension():
    for ext in registered_extensions():
        parser = get_parser_for_extension(ext)
        assert parser is not None, f"no parser for {ext}"
        assert callable(parser)


def test_edf_and_bdf_share_the_same_parser():
    """Both EDF and BDF share the same MNE wrapper — changing one must affect the other."""
    assert get_parser_for_extension(".edf") is get_parser_for_extension(".bdf")
    assert get_parser_for_extension(".edf") is _parse_edf_with_mne


# ─── Contract: parsers return None for nonexistent paths ──────────────────


def test_edf_parser_returns_none_for_missing_file(tmp_path: Path):
    result = _parse_edf_with_mne(tmp_path / "does_not_exist.edf")
    assert result is None


def test_edf_parser_returns_none_for_broken_symlink(tmp_path: Path):
    broken = tmp_path / "broken.edf"
    broken.symlink_to(tmp_path / ".no_target")
    result = _parse_edf_with_mne(broken)
    assert result is None


# ─── FormatParserResult shape ─────────────────────────────────────────────


def test_format_parser_result_has_expected_optional_fields():
    """TypedDict accepts the 5 documented keys (all optional, total=False)."""
    r1: FormatParserResult = {}
    r2: FormatParserResult = {"sampling_frequency": 250.0}
    r3: FormatParserResult = {"nchans": 64, "ch_names": ["Fp1", "Fp2"]}
    r4: FormatParserResult = {"n_times": 5000}
    r5: FormatParserResult = {"n_samples": 5000}
    assert all(isinstance(r, dict) for r in (r1, r2, r3, r4, r5))


def _field(value, width):
    return str(value).ljust(width).encode("ascii")


def _write_bdf(path, n_records=2, spr=4):
    """Minimal valid BDF: one channel, ``n_records`` 1-s records of ``spr`` samples."""
    f = _field
    head = (
        b"\xffBIOSEMI"
        + f("x", 80)
        + f("x", 80)
        + f("01.01.20", 8)
        + f("00.00.00", 8)
        + f(512, 8)
        + f("24BIT", 44)
        + f(n_records, 8)
        + f(1, 8)
        + f(1, 4)
    )
    sig = (
        f("Cz", 16)
        + f("", 80)
        + f("uV", 8)
        + f(-100, 8)
        + f(100, 8)
        + f(-8388608, 8)
        + f(8388607, 8)
        + f("", 80)
        + f(spr, 8)
        + f("", 32)
    )
    path.write_bytes(head + sig + b"\x00\x00\x00" * n_records * spr)
    return path


def test_parse_bdf_header_gives_ntimes(tmp_path):
    """read_raw_edf refuses .bdf since MNE 1.2; the parser must use read_raw_bdf."""
    result = _parse_edf_with_mne(_write_bdf(tmp_path / "sub-01_task-x_eeg.bdf"))
    assert result is not None
    assert result["sampling_frequency"] == 4.0
    assert result["n_times"] == 8
