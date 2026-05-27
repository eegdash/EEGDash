"""Unit tests for the EEGLAB ``.set`` header parser.

Targets ``_set_parser.parse_set_metadata`` and ``diagnose_set_issues``.

The fixture is a 64 KB **truncated** prefix of the original 67 MB ds002893
.set file, fetched from ``eegdash-testing-data``
(see ``eeg/LICENSE-ATTRIBUTION.md`` in that corpus).
The truncation captures the MAT-v5 header but not the full structure,
so the parser's robust path returns the minimal ``{'has_fdt': False}``
report. This is intentional — golden-test the behaviour, not what we
wish the behaviour was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eegdash.testing import data_file

from _set_parser import diagnose_set_issues, parse_set_metadata

SET_FIXTURE = data_file("eeg/sub-001_task-AuditoryVisualShift_run-01_eeg.set")


# ─── parse_set_metadata — golden return on the truncated fixture ───────────


def test_parse_set_returns_dict_on_known_fixture():
    """The committed truncated .set parses to a dict (not None)."""
    result = parse_set_metadata(SET_FIXTURE)
    assert isinstance(result, dict), (
        "parse_set_metadata must return a dict on a parseable header"
    )


def test_parse_set_truncated_fixture_reports_has_fdt_false():
    """The fixture is a header-only truncation; no companion .fdt is present.

    The parser must surface that as ``has_fdt: False`` so callers know they
    cannot read the data buffer.
    """
    result = parse_set_metadata(SET_FIXTURE)
    assert result == {"has_fdt": False}


# ─── diagnose_set_issues — composition ─────────────────────────────────────


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


# ─── Edge cases — defensive parsing ────────────────────────────────────────


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
    """Various malformed inputs must not crash the process.

    The parser may return None, return a near-empty dict, or raise a
    documented exception class. What it must NOT do is segfault, hang,
    or raise BaseException (KeyboardInterrupt, SystemExit).
    """
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
