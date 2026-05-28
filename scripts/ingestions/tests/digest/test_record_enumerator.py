from __future__ import annotations

import json as json_mod
from datetime import datetime
from pathlib import Path

import pytest

import record_enumerator as re_mod
from record_enumerator import (
    BIDSFilesystemEnumerator,
    EnumerationResult,
    ManifestEnumerator,
    RecordEnumerator,
    _has_actual_recording_files,
    get_record_enumerator,
    write_dataset_outputs,
)
from source_adapter import OpenNeuroAdapter


def _make_adapter(dataset_id: str = "ds002893") -> OpenNeuroAdapter:
    return OpenNeuroAdapter(dataset_id)


# ─── EnumerationResult dataclass ──────────────────────────────────────────


def test_enumeration_result_defaults():
    result = EnumerationResult(dataset_meta={"dataset_id": "ds001"})
    assert result.records == []
    assert result.errors == []
    assert result.montages == {}
    assert result.digest_method == "bids_filesystem"


def test_enumeration_result_carries_all_fields():
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


def _arrange_has_actual_files(tmp_path: Path, case: str) -> None:
    if case == "edf":
        (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
        (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    elif case == "ctf_ds_dir":
        (tmp_path / "sub-01" / "meg").mkdir(parents=True)
        (tmp_path / "sub-01" / "meg" / "sub-01_meg.ds").mkdir()
    elif case == "symlink_pointer":
        (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
        pointer = tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf"
        pointer.symlink_to(tmp_path / ".does_not_exist")  # broken on purpose
    elif case == "empty_dir":
        (tmp_path / "sub-01").mkdir()
    elif case == "unrelated_files":
        (tmp_path / "README").write_text("hi")
        (tmp_path / "dataset_description.json").write_text("{}")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        pytest.param("edf", True, id="test_has_actual_files_detects_edf"),
        pytest.param("ctf_ds_dir", True, id="test_has_actual_files_detects_ctf_ds_dir"),
        pytest.param(
            "symlink_pointer",
            True,
            id="test_has_actual_files_detects_symlink_pointer",
        ),
        pytest.param(
            "empty_dir", False, id="test_has_actual_files_returns_false_on_empty_dir"
        ),
        pytest.param(
            "unrelated_files",
            False,
            id="test_has_actual_files_ignores_unrelated_files",
        ),
    ],
)
def test_has_actual_recording_files(tmp_path: Path, case: str, expected: bool) -> None:
    _arrange_has_actual_files(tmp_path, case)
    assert _has_actual_recording_files(tmp_path) is expected


# ─── Factory dispatch ──────────────────────────────────────────────────────


def test_factory_picks_bids_when_real_files_present(tmp_path: Path) -> None:
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
    raise OSError("simulated BIDS load failure")


def test_factory_falls_back_to_manifest_when_bids_load_fails_with_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")
    (tmp_path / "manifest.json").write_text('{"files": []}')

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
    (tmp_path / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "sub-01" / "eeg" / "sub-01_eeg.edf").write_bytes(b"\x00")

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
    enumerator = get_record_enumerator(
        dataset_id="ds_empty",
        dataset_dir=tmp_path,
        source="openneuro",
        source_adapter=_make_adapter(),
        digested_at="2026-05-21T12:00:00Z",
    )
    assert isinstance(enumerator, BIDSFilesystemEnumerator | ManifestEnumerator)


# ─── ABC invariants ────────────────────────────────────────────────────────


def test_record_enumerator_is_abstract():
    with pytest.raises(TypeError):
        RecordEnumerator(  # type: ignore[abstract]
            "ds001", Path("/tmp"), "openneuro", _make_adapter(), "now"
        )


def test_bids_enumerate_raises_when_bids_load_fails(tmp_path: Path) -> None:
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
    parent = tmp_path / "input"
    (parent / "z-001").mkdir(parents=True)
    enumerator = ManifestEnumerator(
        "z-001", parent / "z-001", "zenodo", _make_adapter(), "now"
    )
    result = enumerator.enumerate()
    assert isinstance(result, EnumerationResult)
    assert result.records == []
    assert result.total_files == 0
    assert len(result.errors) == 1
    entry = result.errors[0]
    assert entry.get("status") == "skipped"
    assert "manifest.json not found" in entry.get("reason", "")
    assert entry.get("dataset_id") == "z-001"


# ─── write_dataset_outputs ────────────────────────────────────────────────


def _make_minimal_result(
    digest_method: str = "bids_filesystem",
) -> EnumerationResult:
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
