"""Unit tests for the format-parser registry (ROADMAP P2.2)."""

from __future__ import annotations

import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from _format_parser_registry import (
    FormatParserResult,
    _parse_edf_with_mne,
    get_parser_for_extension,
    registered_extensions,
)

# ─── Registry coverage ────────────────────────────────────────────────────


def test_registry_covers_six_known_extensions():
    """The 6 formats in the cascade's dispatch are all registered."""
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
    """Both EDF and BDF go through the same MNE wrapper —
    pinned because changing the EDF parser must also affect BDF."""
    assert get_parser_for_extension(".edf") is get_parser_for_extension(".bdf")
    assert get_parser_for_extension(".edf") is _parse_edf_with_mne


# ─── Contract: parsers return None for nonexistent paths ──────────────────


def test_edf_parser_returns_none_for_missing_file(tmp_path: Path):
    """The MNE EDF parser must return None (not raise) on a missing file."""
    result = _parse_edf_with_mne(tmp_path / "does_not_exist.edf")
    assert result is None


def test_edf_parser_returns_none_for_broken_symlink(tmp_path: Path):
    """git-annex broken symlinks should resolve to None, not crash."""
    broken = tmp_path / "broken.edf"
    broken.symlink_to(tmp_path / ".no_target")
    result = _parse_edf_with_mne(broken)
    assert result is None


# ─── FormatParserResult shape ─────────────────────────────────────────────


def test_format_parser_result_has_expected_optional_fields():
    """The TypedDict accepts the 5 documented keys (all optional)."""
    # All optional, total=False — these constructions must type-check
    # in mypy and run without error.
    r1: FormatParserResult = {}
    r2: FormatParserResult = {"sampling_frequency": 250.0}
    r3: FormatParserResult = {"nchans": 64, "ch_names": ["Fp1", "Fp2"]}
    r4: FormatParserResult = {"n_times": 5000}
    r5: FormatParserResult = {"n_samples": 5000}
    assert all(isinstance(r, dict) for r in (r1, r2, r3, r4, r5))
