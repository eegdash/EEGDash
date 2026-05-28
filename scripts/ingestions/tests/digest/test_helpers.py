"""Characterisation tests for the pure helpers inside ``3_digest.py``."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest
from _helpers import INGEST_DIR as _INGEST_DIR

import digest_telemetry
from record_enumerator import EnumerationResult

_DIGEST_PATH = _INGEST_DIR / "3_digest.py"


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load ``3_digest.py`` as an importable module despite its numeric name."""
    spec = importlib.util.spec_from_file_location("digest_under_test", _DIGEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── parse_bids_entities_from_path ──────────────────────────────────────────


def test_parse_entities_full_path_with_ses_and_run(digest: ModuleType) -> None:
    """Fully-entity-typed BIDS path returns every field populated."""
    path = "ds002893/sub-001/ses-01/eeg/sub-001_ses-01_task-rest_run-01_eeg.edf"
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == "001"
    assert result["session"] == "01"
    assert result["task"] == "rest"
    assert result["run"] == "01"
    assert result["modality"] == "eeg"
    assert result["datatype"] == "eeg"
    assert result["suffix"] == "eeg"


def test_parse_entities_minimal_path_no_session_no_run(digest: ModuleType) -> None:
    """Minimal BIDS path omits session/run; missing keys must be absent, not None."""
    path = "ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.vhdr"
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == "xp101"
    assert result["task"] == "motorloc"
    assert result["modality"] == "eeg"
    assert "session" not in result
    assert "run" not in result


@pytest.mark.parametrize(
    ("path", "expected_subject"),
    [
        ("ds00/sub-01/eeg/sub-01_eeg.edf", "01"),
        ("ds00/sub-001/eeg/sub-001_eeg.edf", "001"),
        ("ds00/sub-A1B2/eeg/sub-A1B2_eeg.edf", "A1B2"),
        ("ds00/sub-xp101/eeg/sub-xp101_eeg.edf", "xp101"),
    ],
)
def test_parse_entities_subject_parses_alphanumeric(
    digest: ModuleType, path: str, expected_subject: str
) -> None:
    """Subject IDs can be alphanumeric (``sub-xp101`` is legal BIDS)."""
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == expected_subject


@pytest.mark.parametrize(
    ("ext", "modality"),
    [
        (".edf", "eeg"),
        (".bdf", "eeg"),
        (".vhdr", "eeg"),
        (".set", "eeg"),
    ],
)
def test_parse_entities_recovers_modality_from_path(
    digest: ModuleType, ext: str, modality: str
) -> None:
    """Modality is derived from the BIDS datatype directory, not the extension."""
    path = f"ds00/sub-01/{modality}/sub-01_task-rest_{modality}{ext}"
    result = digest.parse_bids_entities_from_path(path)
    assert result["modality"] == modality
    assert result["datatype"] == modality


# ─── is_neuro_data_file ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "is_neuro"),
    [
        # Binary recording formats ARE neuro data
        ("sub-001_eeg.edf", True),
        ("sub-001_eeg.bdf", True),
        ("sub-001_eeg.vhdr", True),
        ("sub-001_eeg.set", True),
        # Sidecars are NOT
        ("sub-001_eeg.json", False),
        ("sub-001_channels.tsv", False),
        ("participants.tsv", False),
        ("dataset_description.json", False),
        # BIDS folder-structure files are NOT
        ("README.md", False),
        ("CHANGES", False),
    ],
)
def test_is_neuro_data_file_classifies_correctly(
    digest: ModuleType, filename: str, is_neuro: bool
) -> None:
    """Recording binaries → True; sidecars/metadata → False."""
    assert digest.is_neuro_data_file(filename) is is_neuro


def test_is_neuro_data_file_is_case_sensitive_on_extension(
    digest: ModuleType,
) -> None:
    """Extension is normalised to lowercase internally; ``.EDF`` and ``.Edf`` both pass."""
    assert digest.is_neuro_data_file("sub-001_eeg.EDF") is True
    assert digest.is_neuro_data_file("sub-001_eeg.Edf") is True


# ─── Mega-function canary ─────────────────────────────────────────────────────


def test_megafunction_line_counts_are_known_baseline(digest: ModuleType) -> None:
    """LOC for big helpers must stay within budget (actual + 20) and above floor."""
    big_functions = {
        "_enumerate_via_manifest": 240,
        "_enumerate_via_bids": 130,
        "extract_record": 240,
        "extract_dataset_metadata": 235,
    }
    for name, baseline_loc in big_functions.items():
        fn = getattr(digest, name, None)
        if fn is None:
            pytest.skip(f"{name} not found — function may have been split already")
        source_lines = inspect.getsourcelines(fn)[0]
        actual_loc = len(source_lines)
        assert actual_loc <= baseline_loc, (
            f"{name} grew to {actual_loc} LOC (budget {baseline_loc}). "
            "Mega-functions should only shrink. If a legitimate growth "
            "happened, bump the baseline in the same commit."
        )

    wrapper_loc = len(inspect.getsourcelines(digest.digest_dataset)[0])
    assert wrapper_loc <= 120, (
        f"digest_dataset grew to {wrapper_loc} LOC — it should stay "
        "a thin orchestrator; the algorithm lives in the helpers."
    )

    assert not hasattr(digest, "digest_from_manifest"), (
        "digest_from_manifest reappeared in 3_digest.py. "
        "The orchestrator routes via ManifestEnumerator from record_enumerator.py."
    )

    # Lower-bound canary: surprise shrinkage can hide a deleted branch.
    big_function_floors = {
        "_enumerate_via_manifest": 120,
        "_enumerate_via_bids": 60,
        "extract_record": 120,
        "extract_dataset_metadata": 120,
    }
    for name, floor_loc in big_function_floors.items():
        fn = getattr(digest, name, None)
        if fn is None:
            continue
        actual_loc = len(inspect.getsourcelines(fn)[0])
        assert actual_loc >= floor_loc, (
            f"{name} shrank to {actual_loc} LOC (floor {floor_loc}). "
            "A whole branch may have been deleted by mistake; if the "
            "shrinkage is intentional, lower the floor in this commit."
        )


# ─── Stage-3D helper unit tests ───────────────────────────────────────────────


def _make_dummy_result(records=(), errors=(), total_files=None):
    """Build an EnumerationResult with controlled fields for helper tests."""
    return EnumerationResult(
        dataset_meta={"dataset_id": "ds-X", "source": "openneuro"},
        records=list(records),
        errors=list(errors),
        total_files=total_files,
        digest_method="bids_filesystem",
    )


@pytest.mark.parametrize(
    ("ds_id", "make_input", "make_output", "expected"),
    [
        pytest.param(
            "ds-001",
            True,
            True,
            {"status": "skipped", "dataset_id": "ds-001", "reason": "already digested"},
            id="already_digested",
        ),
        pytest.param(
            "ds-002",
            False,
            False,
            {
                "status": "skipped",
                "dataset_id": "ds-002",
                "reason": "directory not found",
            },
            id="missing_input_dir",
        ),
        pytest.param(
            "ds-003",
            True,
            False,
            None,
            id="returns_none_when_both_ok",
        ),
    ],
)
def test_check_skip_conditions(
    digest: ModuleType,
    tmp_path: Path,
    ds_id: str,
    make_input: bool,
    make_output: bool,
    expected,
) -> None:
    """_check_dataset_skip_conditions returns the correct skip dict or None."""
    in_dir = tmp_path / ds_id
    out_dir = tmp_path / f"out-{ds_id}"
    if make_input:
        in_dir.mkdir()
    if make_output:
        out_dir.mkdir()
    result = digest._check_dataset_skip_conditions(ds_id, in_dir, out_dir)
    assert result == expected


@pytest.mark.parametrize(
    ("total_files", "expected_reason"),
    [
        pytest.param(0, "no files in manifest", id="no_files_in_manifest_for_zero"),
        pytest.param(
            12, "no records extracted", id="no_records_extracted_for_positive"
        ),
        pytest.param(
            None, "no neurophysiology files found", id="bids_message_when_total_unknown"
        ),
    ],
)
def test_summarise_empty_or_error_empty_status_reason(
    digest: ModuleType,
    total_files,
    expected_reason: str,
) -> None:
    """Empty result with no errors → status='empty' with the correct reason."""
    res = _make_dummy_result(records=[], errors=[], total_files=total_files)
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "empty"
    assert out["reason"] == expected_reason


def test_summarise_empty_or_error_keeps_status_empty_for_soft_warnings_only(
    digest: ModuleType,
) -> None:
    """Soft per-file warnings ('skipped'/'warning'/no status) must NOT flip result to 'error'."""
    res = _make_dummy_result(
        records=[],
        errors=[
            {"file": "a.fif", "status": "skipped", "reason": "broken symlink"},
            {"file": "b.fif", "status": "warning", "error": "no sidecar"},
            {"file": "c.fif", "error": "incomplete"},  # no status key
        ],
        total_files=5,
    )
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "empty"
    assert out["reason"] == "no records extracted"
    # errors are forwarded so operators can still see them.
    assert len(out.get("errors", [])) == 3


def test_summarise_empty_or_error_flips_to_error_when_structural_error_present(
    digest: ModuleType,
) -> None:
    """Any error entry with status='error' flips the summary to status='error' for CI gates."""
    res = _make_dummy_result(
        records=[],
        errors=[
            {"file": "a.fif", "status": "warning", "error": "soft"},
            {"file": "b.fif", "status": "error", "error": "structural"},
        ],
        total_files=5,
    )
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "error"
    assert out["error"] == "No records extracted"
    assert len(out["errors"]) == 2


class _RecordingEmitter:
    """Test double for the digest_telemetry emitter."""

    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None


def test_emit_dataset_finished_payload_round_trips_summary_fields(
    digest: ModuleType,
) -> None:
    """_emit_dataset_finished forwards the 6 documented summary fields; unknown fields must not leak."""
    recorder = _RecordingEmitter()
    saved = digest_telemetry._EMITTER
    digest_telemetry._EMITTER = recorder
    try:
        digest._emit_dataset_finished(
            "ds-Y",
            {
                "status": "success",
                "record_count": 12,
                "error_count": 1,
                "digest_method": "bids_filesystem",
                "integrity_issues_count": 0,
                "montage_count": 2,
                "ignored_field": "should not leak",
            },
        )
    finally:
        digest_telemetry._EMITTER = saved

    assert len(recorder.events) == 1
    ev = recorder.events[0]
    assert ev.event_kind == "dataset_finished"
    assert ev.dataset_id == "ds-Y"
    payload = ev.payload
    assert payload["status"] == "success"
    assert payload["record_count"] == 12
    assert payload["error_count"] == 1
    assert payload["digest_method"] == "bids_filesystem"
    assert payload["integrity_issues_count"] == 0
    assert payload["montage_count"] == 2
    # Defence: fields the summary didn't request must NOT leak through.
    assert "ignored_field" not in payload


def test_emit_dataset_finished_payload_includes_total_files(
    digest: ModuleType,
) -> None:
    """total_files from the manifest path must propagate to the telemetry event payload."""
    recorder = _RecordingEmitter()
    saved = digest_telemetry._EMITTER
    digest_telemetry._EMITTER = recorder
    try:
        digest._emit_dataset_finished(
            "ds-Z",
            {
                "status": "success",
                "record_count": 5,
                "error_count": 0,
                "digest_method": "manifest_only",
                "integrity_issues_count": 0,
                "montage_count": 0,
                "total_files": 5,
            },
        )
    finally:
        digest_telemetry._EMITTER = saved

    assert len(recorder.events) == 1
    assert recorder.events[0].payload.get("total_files") == 5
