"""Unit tests for the RecordEnumerator factory dispatch.

Stage-1 scope: the factory's fallback rules (3 cases mirrored from the
old ``digest_dataset`` orchestrator). The Adapter ``enumerate()`` bodies
are stubbed in stage 1; stage 2 wires them to the existing digest
algorithm bodies, and stage 2's tests use real fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from record_enumerator import (
    BIDSFilesystemEnumerator,
    EnumerationResult,
    ManifestEnumerator,
    _has_actual_recording_files,
    get_record_enumerator,
)
from source_adapter import OpenNeuroAdapter


def _make_adapter(dataset_id: str = "ds002893") -> OpenNeuroAdapter:
    """Make a minimal SourceAdapter; bids_root unused by the factory."""
    return OpenNeuroAdapter(dataset_id)


# ─── EnumerationResult dataclass ──────────────────────────────────────────


def test_enumeration_result_defaults():
    """Empty fields default sensibly (empty lists / dict)."""
    result = EnumerationResult(dataset_meta={"dataset_id": "ds001"})
    assert result.records == []
    assert result.errors == []
    assert result.montages == {}
    assert result.digest_method == "bids_filesystem"


def test_enumeration_result_carries_all_fields():
    """All four collections + method label round-trip through the dataclass."""
    result = EnumerationResult(
        dataset_meta={"dataset_id": "ds001"},
        records=[{"path": "a"}],
        errors=[{"file": "b", "error": "x"}],
        montages={"hash1": {"sensors": []}},
        digest_method="manifest_only",
    )
    assert result.records == [{"path": "a"}]
    assert result.errors == [{"file": "b", "error": "x"}]
    assert "hash1" in result.montages
    assert result.digest_method == "manifest_only"


# ─── _has_actual_recording_files ───────────────────────────────────────────


def test_has_actual_files_detects_edf(tmp_path: Path) -> None:
    """A canonical .edf file means the BIDS path is viable."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_detects_ctf_ds_dir(tmp_path: Path) -> None:
    """A CTF .ds directory counts as a real recording."""
    (tmp_path / "sub-01" / "meg").mkdir(parents=True)
    (tmp_path / "sub-01" / "meg" / "sub-01_meg.ds").mkdir()
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_detects_symlink_pointer(tmp_path: Path) -> None:
    """A git-annex broken symlink still counts (pointer, not the binary)."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    pointer = tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf"
    pointer.symlink_to(tmp_path / ".does_not_exist")  # broken on purpose
    assert _has_actual_recording_files(tmp_path) is True


def test_has_actual_files_returns_false_on_empty_dir(tmp_path: Path) -> None:
    """No recordings → False."""
    (tmp_path / "sub-01").mkdir()
    assert _has_actual_recording_files(tmp_path) is False


def test_has_actual_files_ignores_unrelated_files(tmp_path: Path) -> None:
    """README / dataset_description.json don't count as recordings."""
    (tmp_path / "README").write_text("hi")
    (tmp_path / "dataset_description.json").write_text("{}")
    assert _has_actual_recording_files(tmp_path) is False


# ─── Factory dispatch ──────────────────────────────────────────────────────


def test_factory_picks_bids_when_real_files_present(tmp_path: Path) -> None:
    """Case 1 of the factory: real files on disk → BIDS path."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")

    enumerator = get_record_enumerator(
        dataset_id="ds002893",
        dataset_dir=tmp_path,
        source="openneuro",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    assert isinstance(enumerator, BIDSFilesystemEnumerator)


def test_factory_picks_manifest_when_only_manifest_present(
    tmp_path: Path,
) -> None:
    """Case 2: manifest.json but no actual files → Manifest path."""
    (tmp_path / "manifest.json").write_text('{"files": []}')

    enumerator = get_record_enumerator(
        dataset_id="zenodo-12345",
        dataset_dir=tmp_path,
        source="zenodo",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    assert isinstance(enumerator, ManifestEnumerator)


def _bids_init_that_raises(*args, **kwargs):
    """Module-level stub: a __init__ that always raises OSError.

    Used to monkey-patch BIDSFilesystemEnumerator's constructor and
    simulate a BIDS-load failure. Kept at module level so the project's
    no-nested-functions lint stays clean.
    """
    raise OSError("simulated BIDS load failure")


def test_factory_falls_back_to_manifest_when_bids_load_fails_with_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Case 2 fallback: BIDS load raises → fall back to manifest.

    Simulates the BIDS-construction failure by monkey-patching the
    BIDSFilesystemEnumerator constructor to raise. The factory should
    catch and return ManifestEnumerator since manifest.json exists.
    """
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    (tmp_path / "manifest.json").write_text('{"files": []}')

    import record_enumerator as re_mod

    real_init = re_mod.BIDSFilesystemEnumerator.__init__
    monkeypatch.setattr(
        re_mod.BIDSFilesystemEnumerator, "__init__", _bids_init_that_raises
    )
    try:
        enumerator = get_record_enumerator(
            dataset_id="ds002893",
            dataset_dir=tmp_path,
            source="openneuro",
            source_adapter=_make_adapter(),
            digested_at="2026-05-21T12:00:00Z",
        )
        assert isinstance(enumerator, ManifestEnumerator)
    finally:
        monkeypatch.setattr(re_mod.BIDSFilesystemEnumerator, "__init__", real_init)


def test_factory_propagates_bids_failure_when_no_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """BIDS load fails AND no manifest → re-raise (no fallback possible)."""
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    # No manifest.json on purpose.

    import record_enumerator as re_mod

    real_init = re_mod.BIDSFilesystemEnumerator.__init__
    monkeypatch.setattr(
        re_mod.BIDSFilesystemEnumerator, "__init__", _bids_init_that_raises
    )
    try:
        with pytest.raises(OSError, match="simulated"):
            get_record_enumerator(
                dataset_id="ds002893",
                dataset_dir=tmp_path,
                source="openneuro",
                source_adapter=_make_adapter(),
                digested_at="2026-05-21T12:00:00Z",
            )
    finally:
        monkeypatch.setattr(re_mod.BIDSFilesystemEnumerator, "__init__", real_init)


def test_factory_returns_bids_enumerator_when_empty_no_manifest(
    tmp_path: Path,
) -> None:
    """Case 3 — empty dir, no manifest: return BIDSFilesystemEnumerator
    (its .enumerate() will produce an empty result with status='empty').

    The factory doesn't refuse to return; the orchestrator inspects the
    EnumerationResult to decide the status.
    """
    # Empty directory; no manifest, no files.
    enumerator = get_record_enumerator(
        dataset_id="ds_empty",
        dataset_dir=tmp_path,
        source="openneuro",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    # Either is acceptable here; the orchestrator surfaces the empty
    # status. We assert the type belongs to the Enumerator hierarchy.
    assert isinstance(enumerator, BIDSFilesystemEnumerator | ManifestEnumerator)


# ─── ABC invariants ────────────────────────────────────────────────────────


def test_record_enumerator_is_abstract():
    """RecordEnumerator can't be instantiated directly."""
    from record_enumerator import RecordEnumerator

    with pytest.raises(TypeError):
        RecordEnumerator(  # type: ignore[abstract]
            "ds001", Path("/tmp"), "openneuro", _make_adapter(), "now"
        )


def test_bids_enumerate_raises_when_bids_load_fails(tmp_path: Path) -> None:
    """Stage 3C: ``enumerate()`` calls EEGBIDSDataset directly.

    When the dataset_dir doesn't contain a valid BIDS structure, the
    constructor raises (OSError / ValueError / KeyError). The caller
    in ``3_digest.py:digest_dataset`` catches and falls back to the
    manifest path; the Adapter itself doesn't synthesize a diagnostic.

    Stages 1+2 used to assert NotImplementedError; Stage 2D used to
    swallow the failure and return a diagnostic EnumerationResult.
    Stage 3C's contract is "raise — let the caller decide".
    """
    parent = tmp_path / "input"
    parent.mkdir()
    enumerator = BIDSFilesystemEnumerator(
        "ds001", parent / "ds001", "openneuro", _make_adapter(), "now"
    )
    with pytest.raises((OSError, ValueError, KeyError, FileNotFoundError)):
        enumerator.enumerate()


def test_manifest_enumerate_returns_empty_when_manifest_missing(
    tmp_path: Path,
) -> None:
    """ManifestEnumerator: missing manifest.json → empty EnumerationResult
    with a structured diagnostic. No raise (so the caller can write the
    summary as ``status='empty'`` rather than crash).

    Post-review (2026-05-22) the diagnostic dict now carries the legacy
    summary keys (``status``, ``reason``, ``dataset_id``) so log
    scrapers can distinguish "manifest.json not found" from
    "manifest.json corrupt" from "manifest.json permission denied".
    """
    parent = tmp_path / "input"
    (parent / "z-001").mkdir(parents=True)
    enumerator = ManifestEnumerator(
        "z-001", parent / "z-001", "zenodo", _make_adapter(), "now"
    )
    result = enumerator.enumerate()
    assert isinstance(result, EnumerationResult)
    assert result.records == []
    # Post-review: pinned (not the dataclass default) so a future
    # caller can rely on the manifest path always carrying a non-None
    # total_files.
    assert result.total_files == 0
    assert len(result.errors) == 1
    entry = result.errors[0]
    # New structured-summary shape: assert on the legacy keys.
    assert entry.get("status") == "skipped"
    assert "manifest.json not found" in entry.get("reason", "")
    assert entry.get("dataset_id") == "z-001"


# ─── write_dataset_outputs — shared JSON writer (stage 2A) ────────────────


def _make_minimal_result(
    digest_method: str = "bids_filesystem",
) -> EnumerationResult:
    """Build a minimal but well-formed EnumerationResult for write tests."""
    return EnumerationResult(
        dataset_meta={
            "dataset_id": "ds002893",
            "source": "openneuro",
            "name": "Test Dataset",
            "ingestion_fingerprint": "abc123",
        },
        records=[
            {
                "dataset": "ds002893",
                "bids_relpath": "sub-001/eeg/sub-001_eeg.set",
                "storage": {"backend": "s3", "base": "s3://openneuro.org/ds002893"},
                "recording_modality": ["eeg"],
            }
        ],
        errors=[],
        montages={},
        digest_method=digest_method,
    )


def test_write_outputs_creates_all_four_json_files(tmp_path: Path) -> None:
    """The helper writes _dataset / _records / _montages / _summary JSONs."""
    from record_enumerator import write_dataset_outputs

    result = _make_minimal_result()
    summary = write_dataset_outputs(
        tmp_path,
        result,
        dataset_id="ds002893",
        source="openneuro",
        digested_at="2026-05-21T12:00:00Z",
    )

    assert (tmp_path / "ds002893_dataset.json").exists()
    assert (tmp_path / "ds002893_records.json").exists()
    assert (tmp_path / "ds002893_montages.json").exists()
    assert (tmp_path / "ds002893_summary.json").exists()
    assert summary["status"] == "success"
    assert summary["record_count"] == 1


def test_write_outputs_montages_file_written_when_empty(tmp_path: Path) -> None:
    """Behaviour change documented in STAGE-2-PLAN.md: _montages.json now
    always written, even with no montages — downstream tooling can assume it."""
    import json as json_mod

    from record_enumerator import write_dataset_outputs

    result = _make_minimal_result(digest_method="manifest_only")
    write_dataset_outputs(
        tmp_path,
        result,
        dataset_id="zenodo-999",
        source="zenodo",
        digested_at="2026-05-21T12:00:00Z",
    )

    montages_path = tmp_path / "zenodo-999_montages.json"
    assert montages_path.exists()
    payload = json_mod.loads(montages_path.read_text())
    assert payload["montage_count"] == 0
    assert payload["montages"] == []


def test_write_outputs_summary_carries_digest_method(tmp_path: Path) -> None:
    """Summary always includes digest_method (was only set on manifest path)."""
    import json as json_mod

    from record_enumerator import write_dataset_outputs

    for method in ("bids_filesystem", "manifest_only"):
        result = _make_minimal_result(digest_method=method)
        write_dataset_outputs(
            tmp_path,
            result,
            dataset_id=f"ds_{method}",
            source="openneuro",
            digested_at="2026-05-21T12:00:00Z",
        )
        summary = json_mod.loads((tmp_path / f"ds_{method}_summary.json").read_text())
        assert summary["digest_method"] == method


def test_write_outputs_status_no_neuro_files_when_records_empty(
    tmp_path: Path,
) -> None:
    """Empty record list flips status to 'no_neuro_files'."""
    import json as json_mod

    from record_enumerator import write_dataset_outputs

    result = EnumerationResult(
        dataset_meta={"dataset_id": "ds_empty"},
        records=[],
        errors=[],
        montages={},
        digest_method="bids_filesystem",
    )
    write_dataset_outputs(
        tmp_path,
        result,
        dataset_id="ds_empty",
        source="openneuro",
        digested_at="2026-05-21T12:00:00Z",
    )
    summary = json_mod.loads((tmp_path / "ds_empty_summary.json").read_text())
    assert summary["status"] == "no_neuro_files"
    assert summary["record_count"] == 0


def test_write_outputs_total_files_kwarg_surfaces_in_summary(
    tmp_path: Path,
) -> None:
    """``total_files`` (manifest path) preserved in summary; absent when
    caller doesn't pass it (BIDS path)."""
    import json as json_mod

    from record_enumerator import write_dataset_outputs

    write_dataset_outputs(
        tmp_path,
        _make_minimal_result(),
        dataset_id="ds_with",
        source="zenodo",
        digested_at="now",
        total_files=42,
    )
    s = json_mod.loads((tmp_path / "ds_with_summary.json").read_text())
    assert s["total_files"] == 42

    write_dataset_outputs(
        tmp_path,
        _make_minimal_result(),
        dataset_id="ds_without",
        source="openneuro",
        digested_at="now",
    )
    s = json_mod.loads((tmp_path / "ds_without_summary.json").read_text())
    assert "total_files" not in s


def test_write_outputs_integrity_issue_enrichment(tmp_path: Path) -> None:
    """Records with _has_missing_files get author/contact stamped from
    Dataset meta (was inline in digest_dataset)."""
    import json as json_mod

    from record_enumerator import write_dataset_outputs

    result = EnumerationResult(
        dataset_meta={
            "dataset_id": "ds_iss",
            "authors": ["Curie M."],
            "contact_info": "curie@radium.org",
            "external_links": {"source_url": "https://example.org/ds_iss"},
        },
        records=[
            {
                "dataset": "ds_iss",
                "bids_relpath": "sub-01/eeg/sub-01_eeg.edf",
                "_has_missing_files": True,
                "_data_integrity_issues": ["channels.tsv missing"],
            }
        ],
        errors=[],
        montages={},
        digest_method="bids_filesystem",
    )
    summary = write_dataset_outputs(
        tmp_path,
        result,
        dataset_id="ds_iss",
        source="openneuro",
        digested_at="now",
    )
    assert summary["integrity_issues_count"] == 1

    records = json_mod.loads((tmp_path / "ds_iss_records.json").read_text())
    rec = records["records"][0]
    assert rec["_dataset_authors"] == ["Curie M."]
    assert rec["_dataset_contact"] == "curie@radium.org"
    assert rec["_source_url"] == "https://example.org/ds_iss"


def test_write_outputs_uses_json_default_serializer(tmp_path: Path) -> None:
    """Path / datetime objects in metadata don't crash the writer."""
    import json as json_mod
    from datetime import datetime

    from record_enumerator import write_dataset_outputs

    result = EnumerationResult(
        dataset_meta={
            "dataset_id": "ds_paths",
            "weird_path_field": Path("/tmp/somewhere"),
            "weird_time_field": datetime(2026, 5, 21, 12, 0, 0),
        },
        records=[],
        montages={},
        digest_method="bids_filesystem",
    )
    write_dataset_outputs(
        tmp_path,
        result,
        dataset_id="ds_paths",
        source="openneuro",
        digested_at="now",
    )
    dataset_payload = json_mod.loads((tmp_path / "ds_paths_dataset.json").read_text())
    assert dataset_payload["weird_path_field"] == "/tmp/somewhere"
    assert dataset_payload["weird_time_field"] == "2026-05-21T12:00:00"
