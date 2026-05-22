"""Characterisation tests for the pure helpers inside ``3_digest.py``.

Phase 8 of the robustness programme. These tests pin the CURRENT
behaviour of the small, pure-function helpers inside ``3_digest.py``
so we can refactor the file's mega-functions under a safety net.

The bias here is intentional: we DON'T test the mega-functions
directly — we test the leaves first (entity parsing, neuro-file
detection) and let the snapshot test in ``test_digest_snapshot.py``
cover the orchestrator end-to-end. The LOC canary below tracks the
remaining big helpers and surfaces growth or unexpected shrinkage.

The module is loaded via ``importlib`` because its filename starts
with a digit (``3_digest.py`` is not a legal Python identifier).
This is a known pattern for CLI-style script files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ─── Module loader ──────────────────────────────────────────────────────────

_DIGEST_PATH = Path(__file__).parent.parent / "3_digest.py"


@pytest.fixture(scope="module")
def digest() -> ModuleType:
    """Load ``3_digest.py`` as an importable module despite its numeric name."""
    spec = importlib.util.spec_from_file_location("digest_under_test", _DIGEST_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── parse_bids_entities_from_path — characterisation ──────────────────────


def test_parse_entities_full_path_with_ses_and_run(digest: ModuleType) -> None:
    """The fully-entity-typed BIDS path returns every field populated."""
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
    """The minimal BIDS path (sub + task only) omits session/run gracefully."""
    path = "ds002336/sub-xp101/eeg/sub-xp101_task-motorloc_eeg.vhdr"
    result = digest.parse_bids_entities_from_path(path)
    assert result["subject"] == "xp101"
    assert result["task"] == "motorloc"
    assert result["modality"] == "eeg"
    # Missing keys must be absent (not None / not "")
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


# ─── is_neuro_data_file — characterisation ────────────────────────────────


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
    """An ``.EDF`` uppercase extension follows the same classification."""
    # The function's documented behaviour: it normalises to lowercase
    # internally, so both pass. Pinning that here.
    assert digest.is_neuro_data_file("sub-001_eeg.EDF") is True
    assert digest.is_neuro_data_file("sub-001_eeg.Edf") is True


# ─── A deliberate canary: the mega-functions exist with documented sizes ───
# ─── (informational; not a regression — refactor lands in later commits) ──


def test_megafunction_line_counts_are_known_baseline(digest: ModuleType) -> None:
    """Track LOC drift of the remaining big helpers in ``3_digest.py``.

    The Phase 8 robustness programme keeps a per-function LOC budget
    so growth (or unexpected shrinkage that hides a missing branch)
    gets surfaced in CI. Each baseline is set ~20 LOC above the
    current LOC; an upward drift past that ceiling fails the test.

    The dict is sized to match the post-Stage-3D shape of the file:
    the four remaining big helpers, plus the orchestrator wrapper
    bounded by its own much-tighter budget.
    """
    import inspect

    # Baselines updated 2026-05-22 (Phase 8 Stage 3D — orchestrator
    # collapse). Previous session notes:
    # - Session 5 / Phase 8 Stage 3: digest_from_manifest dropped from
    #   670 → 69 LOC after extracting _enumerate_via_manifest; then
    #   was DELETED in Stage 3D (the orchestrator routes via
    #   ManifestEnumerator from record_enumerator.py).
    # - Stage 3D also dropped digest_dataset from 137 → 90 LOC by
    #   extracting _check_dataset_skip_conditions,
    #   _summarise_empty_or_error, _run_enumerator_with_manifest_fallback,
    #   and _emit_dataset_finished.
    big_functions = {
        # Still big — pending further decomposition. Budget = actual + 20.
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
        # Drift upward = a regression worth investigating.
        assert actual_loc <= baseline_loc, (
            f"{name} grew to {actual_loc} LOC (budget {baseline_loc}). "
            "Mega-functions should only shrink. If a legitimate growth "
            "happened, bump the baseline in the same commit."
        )

    # Sanity-check the orchestrator stayed thin — digest_dataset was
    # 330 LOC before Phase 8 Stage 3 and 90 LOC after Stage 3D. The
    # 120 LOC budget leaves a little headroom for a future log line
    # but fails fast if a chunk of algorithm leaks back in.
    wrapper_loc = len(inspect.getsourcelines(digest.digest_dataset)[0])
    assert wrapper_loc <= 120, (
        f"digest_dataset grew to {wrapper_loc} LOC — it should stay "
        "a thin orchestrator; the algorithm lives in the helpers."
    )

    # digest_from_manifest was removed in Phase 8 Stage 3D; if it
    # ever comes back as a public entry point we want to know.
    assert not hasattr(digest, "digest_from_manifest"), (
        "digest_from_manifest reappeared in 3_digest.py. Stage 3D "
        "removed it; the orchestrator now routes via ManifestEnumerator."
    )

    # Post-review lower-bound canary: surprise shrinkage of the
    # mega-functions can also hide a deleted branch. If any of these
    # drops below the floor, a chunk of algorithm has vanished and
    # the bound below should be revisited.
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


# ─── Direct unit tests for the Stage 3D helpers ───────────────────────────
# Post-review (2026-05-22): the four helpers below were only exercised
# end-to-end via digest_dataset's snapshot fixtures, which only cover
# the happy path. These tests pin each branch directly so a regression
# in (e.g.) `_run_enumerator_with_manifest_fallback`'s try/except is
# caught without needing a malformed-manifest snapshot fixture.


def _make_dummy_result(records=(), errors=(), total_files=None):
    """Build an EnumerationResult with controlled fields for helper tests."""
    from record_enumerator import EnumerationResult

    return EnumerationResult(
        dataset_meta={"dataset_id": "ds-X", "source": "openneuro"},
        records=list(records),
        errors=list(errors),
        total_files=total_files,
        digest_method="bids_filesystem",
    )


def test_check_skip_conditions_already_digested(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Already-digested output dir → skipped/already digested."""
    (tmp_path / "ds-001").mkdir()
    (tmp_path / "out-ds-001").mkdir()
    skip = digest._check_dataset_skip_conditions(
        "ds-001", tmp_path / "ds-001", tmp_path / "out-ds-001"
    )
    assert skip == {
        "status": "skipped",
        "dataset_id": "ds-001",
        "reason": "already digested",
    }


def test_check_skip_conditions_missing_input_dir(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Input dataset dir missing → skipped/directory not found."""
    skip = digest._check_dataset_skip_conditions(
        "ds-002", tmp_path / "absent", tmp_path / "out"
    )
    assert skip == {
        "status": "skipped",
        "dataset_id": "ds-002",
        "reason": "directory not found",
    }


def test_check_skip_conditions_returns_none_when_both_ok(
    digest: ModuleType, tmp_path: Path
) -> None:
    """Input present and output absent → None (proceed with digest)."""
    (tmp_path / "ds-003").mkdir()
    assert (
        digest._check_dataset_skip_conditions(
            "ds-003", tmp_path / "ds-003", tmp_path / "out-ds-003"
        )
        is None
    )


def test_summarise_empty_or_error_picks_no_files_in_manifest_for_zero(
    digest: ModuleType,
) -> None:
    """total_files == 0 → legacy 'no files in manifest' reason."""
    res = _make_dummy_result(records=[], errors=[], total_files=0)
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "empty"
    assert out["reason"] == "no files in manifest"


def test_summarise_empty_or_error_picks_no_records_extracted_for_positive(
    digest: ModuleType,
) -> None:
    """total_files > 0, records empty, no structural errors → 'no records extracted'."""
    res = _make_dummy_result(records=[], errors=[], total_files=12)
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "empty"
    assert out["reason"] == "no records extracted"


def test_summarise_empty_or_error_picks_bids_message_when_total_unknown(
    digest: ModuleType,
) -> None:
    """BIDS path has total_files=None → legacy generic 'no neurophysiology files found'."""
    res = _make_dummy_result(records=[], errors=[], total_files=None)
    out = digest._summarise_empty_or_error("ds-X", res)
    assert out["status"] == "empty"
    assert out["reason"] == "no neurophysiology files found"


def test_summarise_empty_or_error_keeps_status_empty_for_soft_warnings_only(
    digest: ModuleType,
) -> None:
    """Per-file warnings ('skipped' / 'warning' / no status) must NOT flip
    the result to status='error' — old digest_from_manifest always
    returned 'empty' regardless of soft entries.
    """
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
    """Any error entry with status='error' is a structural failure ->
    the summary flips to status='error' so CI gates can distinguish it
    from a benign 'no records extracted' skip.
    """
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
    """_emit_dataset_finished forwards the documented 6 summary fields
    to the telemetry payload. Pins the contract so a future summary
    addition can't silently break dashboards that read the event.
    """
    import digest_telemetry

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
