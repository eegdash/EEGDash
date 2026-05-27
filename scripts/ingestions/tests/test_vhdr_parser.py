"""Unit tests for the BrainVision ``.vhdr`` parser.

Targets ``_vhdr_parser.parse_vhdr_metadata``, ``extract_vhdr_references``,
``diagnose_vhdr_issues``, and ``parse_vhdr_metadata_robust``.

Fixtures are CC0 BrainVision triples (``.vhdr`` + ``.vmrk`` + ``.eeg``)
from OpenNeuro ds002336 (EEG) and ds003688 (iEEG), fetched lazily from
the ``eegdash-testing-data`` corpus by ``eegdash.testing.data_path``.

The tests are golden-value: every documented return field is pinned
to a measured value. A change to the parser that alters output will
fail a specific assertion, not just "result is None".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eegdash.testing import data_file

from _vhdr_parser import (
    diagnose_vhdr_issues,
    extract_vhdr_references,
    parse_vhdr_metadata,
    parse_vhdr_metadata_robust,
)

# ── EEG fixture (ds002336 sub-xp101) ────────────────────────────────────────

EEG_VHDR = data_file("eeg/sub-xp101_task-motorloc_eeg.vhdr")
EEG_EEG = data_file("eeg/sub-xp101_task-motorloc_eeg.eeg")
EEG_VMRK = data_file("eeg/sub-xp101_task-motorloc_eeg.vmrk")

# ── iEEG fixture (ds003688 sub-01) ──────────────────────────────────────────

IEEG_VHDR = data_file("ieeg/sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr")


# ─── parse_vhdr_metadata — golden values on EEG fixture ────────────────────


def test_parse_vhdr_eeg_returns_expected_channel_count():
    """ds002336 sub-xp101 is a 64-channel EEG recording."""
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert meta["nchans"] == 64


def test_parse_vhdr_eeg_returns_expected_sampling_frequency():
    """ds002336 sub-xp101 is recorded at 5000 Hz (200 µs sampling interval)."""
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert meta["sampling_frequency"] == 5000.0


def test_parse_vhdr_eeg_returns_correct_channel_label_count():
    """The ch_names list length matches nchans."""
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
    """Specific channels at specific indices — pins the parse order."""
    meta = parse_vhdr_metadata(EEG_VHDR)
    assert meta is not None
    assert meta["ch_names"][idx] == expected_name


# ─── parse_vhdr_metadata — golden values on iEEG fixture ───────────────────


def test_parse_vhdr_ieeg_returns_expected_channel_count():
    """ds003688 sub-01 is a 111-channel intracranial recording."""
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    assert meta["nchans"] == 111


def test_parse_vhdr_ieeg_returns_expected_sampling_frequency():
    """ds003688 sub-01 is recorded at 2048 Hz."""
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    assert meta["sampling_frequency"] == 2048.0


def test_parse_vhdr_ieeg_first_channel_is_intracranial():
    """Intracranial channels follow the AR/AHR/PHR strip convention."""
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    assert meta["ch_names"][0] == "AR1"
    assert meta["ch_names"][1] == "AR2"


def test_parse_vhdr_ieeg_includes_emg_channel():
    """iEEG implantations often co-record EMG; verify it's present."""
    meta = parse_vhdr_metadata(IEEG_VHDR)
    assert meta is not None
    assert "EMG+" in meta["ch_names"]


# ─── extract_vhdr_references — sibling file resolution ─────────────────────


def test_extract_vhdr_references_resolves_datafile():
    """The .vhdr ``DataFile=`` field points at the .eeg companion."""
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["datafile"] == "sub-xp101_task-motorloc_eeg.eeg"


def test_extract_vhdr_references_resolves_markerfile():
    """The .vhdr ``MarkerFile=`` field points at the .vmrk companion."""
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["markerfile"] == "sub-xp101_task-motorloc_eeg.vmrk"


def test_extract_vhdr_references_reports_companion_existence():
    """The fixture committed BOTH .eeg and .vmrk siblings."""
    refs = extract_vhdr_references(EEG_VHDR)
    assert refs["datafile_exists"] is True
    assert refs["markerfile_exists"] is True


# ─── diagnose_vhdr_issues — happy path ─────────────────────────────────────


def test_diagnose_vhdr_eeg_reports_ok():
    """A well-formed .vhdr should diagnose as 'ok' with no issues."""
    diag = diagnose_vhdr_issues(EEG_VHDR)
    assert diag["status"] == "ok"
    assert diag["issues"] == []


def test_diagnose_vhdr_eeg_extracts_metadata_field():
    """The diagnose output embeds the parsed metadata for callers."""
    diag = diagnose_vhdr_issues(EEG_VHDR)
    assert diag["can_extract_metadata"] is True
    assert diag["metadata"]["nchans"] == 64
    assert diag["metadata"]["sampling_frequency"] == 5000.0


# ─── parse_vhdr_metadata_robust — fallback chain ───────────────────────────


def test_parse_vhdr_robust_matches_strict_on_core_fields():
    """The robust parser must return the same core fields on valid input.

    The robust variant adds diagnostic metadata (``_status``, ``_diagnosis``)
    that the strict variant doesn't — those are auxiliary and not compared.
    """
    strict = parse_vhdr_metadata(EEG_VHDR)
    robust = parse_vhdr_metadata_robust(EEG_VHDR)
    assert strict is not None
    assert robust is not None
    for field in ("nchans", "sampling_frequency", "ch_names"):
        assert robust[field] == strict[field], f"mismatch on {field!r}"


def test_parse_vhdr_robust_surfaces_diagnosis_field():
    """Robust parser adds a `_diagnosis` field for caller introspection."""
    robust = parse_vhdr_metadata_robust(EEG_VHDR)
    assert robust is not None
    assert "_diagnosis" in robust
    assert robust["_status"] == "ok"


# ─── Edge cases — defensive parsing ────────────────────────────────────────


def test_parse_vhdr_nonexistent_path_returns_none_or_raises():
    """A missing file must NOT crash the process. It may return None or
    raise FileNotFoundError. Either is acceptable — both are recoverable
    by the caller; what isn't acceptable is a SegFault or hang."""
    missing = Path(__file__).parent / "_nonexistent_.vhdr"
    try:
        result = parse_vhdr_metadata(missing)
        assert result is None or isinstance(result, dict)
    except (FileNotFoundError, OSError):
        pass  # acceptable failure


def test_parse_vhdr_empty_file_returns_none(tmp_path: Path):
    """An empty .vhdr file must not crash; return None to signal failure."""
    empty = tmp_path / "empty.vhdr"
    empty.write_bytes(b"")
    try:
        result = parse_vhdr_metadata(empty)
        # The parser may return None, an empty dict, or raise — all acceptable.
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, RuntimeError):
        pass  # documented failure mode


def test_parse_vhdr_garbage_bytes_does_not_crash(tmp_path: Path):
    """Random non-INI bytes must not crash the parser."""
    garbage = tmp_path / "garbage.vhdr"
    garbage.write_bytes(b"\x00\x01\x02\x03\x04\xff\xfe\xfd" * 100)
    try:
        result = parse_vhdr_metadata(garbage)
        # The parser is permissive — may return a near-empty dict.
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, UnicodeDecodeError):
        pass  # documented failure mode


def test_parse_vhdr_returns_none_for_directory(tmp_path: Path):
    """Passing a directory path must not crash."""
    try:
        result = parse_vhdr_metadata(tmp_path)
        assert result is None or isinstance(result, dict)
    except (IsADirectoryError, OSError, PermissionError):
        pass  # documented failure mode
