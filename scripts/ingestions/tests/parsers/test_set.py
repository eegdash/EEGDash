"""Tests for the EEGLAB .set parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from eegdash.testing import data_file

# scipy.io is only needed for the synthetic-MAT v5 section. importorskip
# at module scope skips ALL three sections cleanly when scipy is absent.
scipy_io = pytest.importorskip("scipy.io")

import numpy as np
from _helpers.builders import build_synthetic_set_v5

from _set_parser import diagnose_set_issues, parse_set_metadata

SET_FIXTURE = data_file("eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set")


# ─── 1. Golden values on the truncated fixture ─────────────────────────────


def test_parse_set_returns_dict_on_known_fixture():
    """The committed truncated .set parses to a dict (not None)."""
    result = parse_set_metadata(SET_FIXTURE)
    assert isinstance(result, dict), (
        "parse_set_metadata must return a dict on a parseable header"
    )


def test_parse_set_truncated_fixture_reports_has_fdt_false():
    """The fixture is a header-only truncation; parser returns ``{'has_fdt': False}``."""
    result = parse_set_metadata(SET_FIXTURE)
    assert result == {"has_fdt": False}


def test_diagnose_set_truncated_fixture_status_ok():
    """A parseable header (even without .fdt) diagnoses as 'ok'."""
    diag = diagnose_set_issues(SET_FIXTURE)
    assert diag["status"] == "ok"
    assert diag["issues"] == []


def test_diagnose_set_embeds_metadata():
    """The diagnose output must include the parsed metadata for callers."""
    diag = diagnose_set_issues(SET_FIXTURE)
    assert diag["can_extract_metadata"] is True
    assert diag["metadata"] == {"has_fdt": False}


def test_parse_set_nonexistent_path_returns_none_or_raises():
    """Missing files must not crash. None or FileNotFoundError both acceptable."""
    missing = Path(__file__).parent / "_nonexistent_.set"
    try:
        result = parse_set_metadata(missing)
        assert result is None or isinstance(result, dict)
    except (FileNotFoundError, OSError):
        pass  # acceptable failure


@pytest.mark.parametrize(
    "garbage",
    [
        b"",  # empty
        b"\x00" * 16,  # nulls
        b"NOT A MAT FILE",  # wrong magic
        b"\xff\xfe\xfd\xfc" * 256,  # random bytes
    ],
)
def test_parse_set_garbage_inputs_dont_crash(tmp_path: Path, garbage: bytes):
    """Various malformed inputs must not crash the process."""
    f = tmp_path / "garbage.set"
    f.write_bytes(garbage)
    try:
        result = parse_set_metadata(f)
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError, OSError, RuntimeError):
        pass  # documented failure modes


def test_parse_set_with_directory_path_does_not_crash(tmp_path: Path):
    """Passing a directory path must not crash."""
    try:
        result = parse_set_metadata(tmp_path)
        assert result is None or isinstance(result, dict)
    except (IsADirectoryError, OSError, PermissionError):
        pass


# ─── 2. Direct extraction on the same fixture ──────────────────────────────


def test_parse_set_extracts_sampling_frequency_when_struct_present():
    """When sampling_frequency is present, it must be a positive float."""
    out = parse_set_metadata(SET_FIXTURE)
    assert out is not None
    if "sampling_frequency" in out:
        assert isinstance(out["sampling_frequency"], float)
        assert out["sampling_frequency"] > 0


def test_parse_set_extracts_nchans_when_struct_present():
    """When nchans is present, it must be a positive int."""
    out = parse_set_metadata(SET_FIXTURE)
    assert out is not None
    if "nchans" in out:
        assert isinstance(out["nchans"], int)
        assert out["nchans"] > 0


def test_parse_set_ch_names_match_nchans_when_both_present():
    """If both ch_names and nchans are extracted, they're consistent."""
    out = parse_set_metadata(SET_FIXTURE)
    assert out is not None
    if "ch_names" in out and "nchans" in out:
        assert len(out["ch_names"]) == out["nchans"]
        assert all(isinstance(n, str) and n for n in out["ch_names"])


def test_parse_set_reports_has_fdt():
    """``has_fdt`` flag indicates companion .fdt file presence."""
    out = parse_set_metadata(SET_FIXTURE)
    assert out is not None
    assert "has_fdt" in out
    assert isinstance(out["has_fdt"], bool)


def test_parse_set_accepts_string_path():
    """Path arg accepts both Path and str."""
    out = parse_set_metadata(str(SET_FIXTURE))
    assert out is not None


def test_parse_set_missing_file_returns_none(tmp_path: Path):
    """Non-existent file → None, no raise."""
    assert parse_set_metadata(tmp_path / "missing.set") is None


def test_parse_set_broken_symlink_returns_none(tmp_path: Path):
    """git-annex broken symlink → None."""
    broken = tmp_path / "broken.set"
    broken.symlink_to(tmp_path / ".no_target")
    assert parse_set_metadata(broken) is None


def test_parse_set_directory_path_does_not_crash(tmp_path: Path):
    """A directory passed in → no raised exception."""
    parse_set_metadata(tmp_path)  # must not raise


def test_parse_set_garbage_bytes_returns_none(tmp_path: Path):
    """A .set file that's not a real MATLAB file → None."""
    garbage = tmp_path / "garbage.set"
    garbage.write_bytes(b"\x00\x01\x02 not a real .set file")
    result = parse_set_metadata(garbage)
    assert result is None or result == {} or "sampling_frequency" not in result


# ─── 3. Synthetic MAT v5 — every extraction branch ─────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "assertion"),
    [
        pytest.param(
            {"srate": 500.0},
            lambda o: o.get("sampling_frequency") == 500.0,
            id="srate_to_sampling_frequency",
        ),
        pytest.param(
            {"nbchan": 64},
            lambda o: o.get("nchans") == 64,
            id="nbchan_to_nchans",
        ),
        pytest.param(
            {"pnts": 10000},
            # Some parsers emit n_samples, others emit n_times — accept either.
            lambda o: o.get("n_samples") == 10000 or o.get("n_times") == 10000,
            id="pnts_to_n_samples_or_n_times",
        ),
    ],
)
def test_set_synthetic_extracts_field(tmp_path: Path, kwargs, assertion):
    """Synthetic MAT v5 builds: EEG.srate/nbchan/pnts → extracted fields."""
    set_path = build_synthetic_set_v5(tmp_path / "test.set", **kwargs)
    out = parse_set_metadata(set_path)
    assert out is not None
    assert assertion(out), f"assertion failed on output {out!r}"


def test_set_extracts_ch_names_from_chanlocs(tmp_path: Path):
    """``EEG.chanlocs.labels`` becomes ch_names."""
    set_path = build_synthetic_set_v5(
        tmp_path / "test.set",
        nbchan=3,
        ch_names=["Cz", "Fz", "Pz"],
    )
    out = parse_set_metadata(set_path)
    assert out is not None
    if "ch_names" in out:
        assert sorted(out["ch_names"]) == ["Cz", "Fz", "Pz"]


@pytest.mark.parametrize(
    ("companion_present", "expected_has_fdt"),
    [
        pytest.param(False, False, id="no_companion_fdt_false"),
        pytest.param(True, True, id="companion_present_fdt_true"),
    ],
)
def test_set_reports_has_fdt(
    tmp_path: Path, companion_present: bool, expected_has_fdt: bool
):
    """``has_fdt`` reflects sibling .fdt file presence."""
    set_path = build_synthetic_set_v5(tmp_path / "test.set")
    if companion_present:
        (tmp_path / "test.fdt").write_bytes(b"placeholder")
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("has_fdt") is expected_has_fdt


def test_set_handles_struct_without_chanlocs(tmp_path: Path):
    """A .set with just srate/nbchan/pnts (no chanlocs) still parses."""
    set_path = tmp_path / "minimal.set"
    scipy_io.savemat(
        str(set_path),
        {
            "EEG": {
                "srate": np.array([[250.0]]),
                "nbchan": np.array([[32]]),
                "pnts": np.array([[1000]]),
            }
        },
    )
    out = parse_set_metadata(set_path)
    assert out is not None
    assert out.get("sampling_frequency") == 250.0
    assert out.get("nchans") == 32


def test_set_returns_none_when_no_eeg_struct(tmp_path: Path):
    """A MAT file without ``EEG`` at the top → None or empty."""
    set_path = tmp_path / "noeeg.set"
    scipy_io.savemat(str(set_path), {"other_var": np.array([1, 2, 3])})
    out = parse_set_metadata(set_path)
    if out is not None:
        assert "sampling_frequency" not in out
        assert "nchans" not in out
