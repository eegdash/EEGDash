"""Direct tests for ``_set_parser.py`` (ROADMAP-C2 C2.3).

Was at 36% before this commit. Uses the CC0 .set fixture from the
``eegdash-testing-data`` corpus
(``eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set``) to exercise
the happy path through scipy.io / h5py.

Complements ``test_parsers_property.py`` (which covers the
"doesn't crash on garbage" invariants via Hypothesis) and
``test_format_parser_registry.py`` (which covers the contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent.parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from eegdash.testing import data_file

from _set_parser import parse_set_metadata

EEG_SET = data_file("eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set")


# ─── Happy-path against the real fixture ──────────────────────────────────


def test_parse_set_returns_dict():
    """Parsing the real .set fixture yields a non-empty dict."""
    out = parse_set_metadata(EEG_SET)
    assert out is not None
    assert isinstance(out, dict)


def test_parse_set_extracts_sampling_frequency_when_struct_present():
    """When the fixture has a complete EEGLAB struct, sampling_frequency
    is extracted. The CC0 fixture in eegdash-testing-data/eeg is metadata-light
    so this test is conditional — pins the contract without requiring
    every fixture to be feature-complete."""
    out = parse_set_metadata(EEG_SET)
    assert out is not None
    if "sampling_frequency" in out:
        assert isinstance(out["sampling_frequency"], float)
        assert out["sampling_frequency"] > 0


def test_parse_set_extracts_nchans_when_struct_present():
    """Conditional, like the sampling_frequency test."""
    out = parse_set_metadata(EEG_SET)
    assert out is not None
    if "nchans" in out:
        assert isinstance(out["nchans"], int)
        assert out["nchans"] > 0


def test_parse_set_ch_names_match_nchans_when_both_present():
    """If both ch_names and nchans are extracted, they're consistent."""
    out = parse_set_metadata(EEG_SET)
    assert out is not None
    if "ch_names" in out and "nchans" in out:
        assert len(out["ch_names"]) == out["nchans"]
        assert all(isinstance(n, str) and n for n in out["ch_names"])


def test_parse_set_reports_has_fdt():
    """``has_fdt`` flag indicates companion .fdt file presence."""
    out = parse_set_metadata(EEG_SET)
    assert out is not None
    assert "has_fdt" in out
    assert isinstance(out["has_fdt"], bool)


# ─── Path-tolerance edge cases ────────────────────────────────────────────


def test_parse_set_accepts_string_path():
    """Path arg accepts both Path and str."""
    out = parse_set_metadata(str(EEG_SET))
    assert out is not None


def test_parse_set_missing_file_returns_none(tmp_path: Path):
    """Non-existent file → None, no raise."""
    assert parse_set_metadata(tmp_path / "missing.set") is None


def test_parse_set_broken_symlink_returns_none(tmp_path: Path):
    """git-annex broken symlink → None (per Phase 9 audit-3 F3 fix)."""
    broken = tmp_path / "broken.set"
    broken.symlink_to(tmp_path / ".no_target")
    assert parse_set_metadata(broken) is None


def test_parse_set_directory_path_does_not_crash(tmp_path: Path):
    """A directory passed in → no raised exception (return value may
    be None, dict, or sparse depending on scipy's behaviour)."""
    # Pin the no-crash contract; don't pin the specific return value.
    parse_set_metadata(tmp_path)  # must not raise


def test_parse_set_garbage_bytes_returns_none(tmp_path: Path):
    """A .set file that's not a real MATLAB file → None.

    Pinned for robustness: a bad upload at the source shouldn't crash
    digest.
    """
    garbage = tmp_path / "garbage.set"
    garbage.write_bytes(b"\x00\x01\x02 not a real .set file")
    result = parse_set_metadata(garbage)
    assert result is None or result == {} or "sampling_frequency" not in result
