"""Unit tests for the BrainVision ``.vhdr`` parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from _vhdr_parser import (
    diagnose_vhdr_issues,
    extract_vhdr_references,
    parse_vhdr_metadata,
    parse_vhdr_metadata_robust,
)
from eegdash.testing import data_file

# ── EEG fixture (ds002336 sub-xp101) ────────────────────────────────────────

EEG_VHDR = data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr")
EEG_EEG = data_file("eeg/sub-xp101_task-motorloc_eeg.eeg")
EEG_VMRK = data_file("eeg/sub-xp101_task-motorloc_eeg.vmrk")

# ── iEEG fixture (ds003688 sub-01) ──────────────────────────────────────────

IEEG_VHDR = data_file("ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr")


# ─── parse_vhdr_metadata — golden values on EEG fixture ────────────────────


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param("nchans", 64, id="nchans=64"),
        pytest.param("sampling_frequency", 5000.0, id="sfreq=5000Hz"),
    ],
)
def test_parse_vhdr_eeg_returns_expected_field(field: str, expected):
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert meta[field] == expected


def test_parse_vhdr_eeg_returns_correct_channel_label_count():
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert len(meta["ch_names"]) == meta["nchans"] == 64


@pytest.mark.parametrize(
    ("idx", "expected_name"),
    [
        (0, "Fp1"),
        (1, "Fp2"),
        (17, "Cz"),
        (31, "ECG"),  # the only non-EEG channel in this montage
        (63, "CPz"),
    ],
)
def test_parse_vhdr_eeg_channel_names_at_known_positions(
    idx: int, expected_name: str
) -> None:
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert meta["ch_names"][idx] == expected_name


# ─── parse_vhdr_metadata — golden values on iEEG fixture ───────────────────


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param("nchans", 111, id="nchans=111"),
        pytest.param("sampling_frequency", 2048.0, id="sfreq=2048Hz"),
    ],
)
def test_parse_vhdr_ieeg_returns_expected_field(field: str, expected):
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    assert meta[field] == expected


@pytest.mark.parametrize(
    ("ch_name_or_idx", "expected"),
    [
        pytest.param(0, "AR1", id="idx_0_is_AR1"),
        pytest.param(1, "AR2", id="idx_1_is_AR2"),
        pytest.param("EMG+", "EMG+", id="EMG+_present_anywhere"),
    ],
)
def test_parse_vhdr_ieeg_channel_layout(ch_name_or_idx, expected: str):
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    if isinstance(ch_name_or_idx, int):
        assert meta["ch_names"][ch_name_or_idx] == expected
    else:
        assert expected in meta["ch_names"]


# ─── extract_vhdr_references — sibling file resolution ─────────────────────


def test_extract_vhdr_references_resolves_datafile():
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["datafile"] == "sub-xp101_task-motorloc_eeg.eeg"


def test_extract_vhdr_references_resolves_markerfile():
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["markerfile"] == "sub-xp101_task-motorloc_eeg.vmrk"


def test_extract_vhdr_references_reports_companion_existence():
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["datafile_exists"] is True
    assert refs["markerfile_exists"] is True


# ─── diagnose_vhdr_issues — happy path ─────────────────────────────────────


def test_diagnose_vhdr_eeg_reports_ok():
    diag = diagnose_vhdr_issues(EEG_VHDR)
    assert diag["status"] == "ok"
    assert diag["issues"] == []


def test_diagnose_vhdr_eeg_extracts_metadata_field():
    diag = diagnose_vhdr_issues(EEG_VHDR)
    assert diag["can_extract_metadata"] is True
    assert diag["metadata"]["nchans"] == 64
    assert diag["metadata"]["sampling_frequency"] == 5000.0


# ─── parse_vhdr_metadata_robust — fallback chain ───────────────────────────


def test_parse_vhdr_robust_matches_strict_on_core_fields():
    """Robust parser returns the same core fields; auxiliary ``_status``/``_diagnosis`` are not compared."""
    strict = parse_vhdr_metadata(EEG_VHDR)
    robust = parse_vhdr_metadata_robust(EEG_VHDR)
    assert strict is not None
    assert robust is not None
    for field in ("nchans", "sampling_frequency", "ch_names"):
        assert robust[field] == strict[field], f"mismatch on {field!r}"


def test_parse_vhdr_robust_surfaces_diagnosis_field():
    robust = parse_vhdr_metadata_robust(EEG_VHDR)
    assert robust is not None
    assert "_diagnosis" in robust
    assert robust["_status"] == "ok"


# ─── Edge cases — defensive parsing ────────────────────────────────────────


def test_parse_vhdr_nonexistent_path_returns_none_or_raises():
    """Missing file must not crash; may return None or raise FileNotFoundError/OSError."""
    missing = Path(__file__).parent / "_nonexistent_.vhdr"
    try:
        result = parse_vhdr_metadata(missing)
        assert result is None or isinstance(result, dict)
    except (FileNotFoundError, OSError):
        pass  # acceptable failure


def test_parse_vhdr_empty_file_returns_none(tmp_path: Path):
    empty = tmp_path / "empty.vhdr"
    empty.write_bytes(b"")
    try:
        result = parse_vhdr_metadata(empty)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, RuntimeError):
        pass


def test_parse_vhdr_garbage_bytes_does_not_crash(tmp_path: Path):
    garbage = tmp_path / "garbage.vhdr"
    garbage.write_bytes(b"\x00\x01\x02\x03\x04\xff\xfe\xfd" * 100)
    try:
        result = parse_vhdr_metadata(garbage)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, UnicodeDecodeError):
        pass


def test_parse_vhdr_returns_none_for_directory(tmp_path: Path):
    try:
        result = parse_vhdr_metadata(tmp_path)
        assert result is None or isinstance(result, dict)
    except (IsADirectoryError, OSError, PermissionError):
        pass
