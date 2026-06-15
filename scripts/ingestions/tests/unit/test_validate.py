"""Tests for ``_validate.py`` — schema gate (validators + helpers)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from _validate import (
    DATA_QUALITY_FIELDS,
    NEURO_EXTENSIONS,
    RECOMMENDED_DATASET_FIELDS,
    VALID_SOURCES,
    VALID_STORAGE_PATTERNS,
    ValidationResult,
    validate_dataset,
    validate_digestion_output,
    validate_record,
    validate_storage_url,
)
from eegdash.testing import data_file


def _minimal_valid_record() -> dict:
    return {
        "dataset": "ds002893",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "digested_at": "2026-05-22T12:00:00+00:00",
        "recording_modality": ["eeg"],
        "storage": {
            "base": "s3://openneuro.org/ds002893",
            "backend": "s3",
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "dep_keys": [],
        },
    }


def _minimal_valid_dataset() -> dict:
    return {
        "dataset_id": "ds002893",
        "source": "openneuro",
        "ingestion_fingerprint": "abc123def456" + "0" * 20,
        "name": "Test dataset",
        "digested_at": "2026-05-22T12:00:00+00:00",
        "recording_modality": ["eeg"],
    }


# ─── validate_record ──────────────────────────────────────────────────────


def test_validate_record_accepts_minimal_valid_record():
    result = ValidationResult()
    rec = _minimal_valid_record()
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.errors == []
    assert result.stats["storage_errors"] == 0


def test_validate_record_flags_storage_url_mismatch():
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"]["base"] = "s3://nemar/ds002893"  # wrong source

    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["storage_errors"] == 1
    assert any(e.get("field") == "storage.base" for e in result.errors), (
        f"no storage.base error in {result.errors}"
    )


def test_validate_record_missing_mandatory_field_raises_pydantic_error():
    result = ValidationResult()
    rec = _minimal_valid_record()
    del rec["bids_relpath"]

    validate_record(rec, "ds002893", "openneuro", result)
    assert len(result.errors) >= 1
    assert any("record" in str(e) for e in result.errors)


@pytest.mark.parametrize(
    ("override", "counter", "expected"),
    [
        pytest.param({}, "missing_nchans", 1, id="nchans_missing"),
        pytest.param({}, "missing_sampling_frequency", 1, id="sfreq_missing"),
        pytest.param(
            {"nchans": 0}, "missing_nchans", 1, id="nchans_zero_treated_missing"
        ),
        pytest.param(
            {"nchans": 64, "sampling_frequency": 250.0},
            "missing_nchans",
            0,
            id="nchans_present_not_counted",
        ),
        pytest.param(
            {"nchans": 64, "sampling_frequency": 250.0},
            "missing_sampling_frequency",
            0,
            id="sfreq_present_not_counted",
        ),
    ],
)
def test_validate_record_missing_field_counters(
    override: dict, counter: str, expected: int
):
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec.update(override)
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats[counter] == expected


def test_validate_record_first_record_only_emits_recommended_warnings():
    """Only record_idx == 0 emits recommended-field warnings."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    validate_record(rec, "ds002893", "openneuro", result, record_idx=0)
    first_warning_count = len(result.warnings)
    assert first_warning_count > 0

    validate_record(rec, "ds002893", "openneuro", result, record_idx=1)
    assert len(result.warnings) == first_warning_count


def test_validate_record_unknown_source_passes_storage_check():
    result = ValidationResult()
    rec = _minimal_valid_record()
    validate_record(rec, "ds-xxx", "brand_new_source", result)
    assert result.stats["storage_errors"] == 0


def test_validate_record_handles_missing_storage_dict():
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"] = {}
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["storage_errors"] == 0


# ─── validate_dataset ─────────────────────────────────────────────────────


def test_validate_dataset_accepts_minimal_valid_dataset():
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    validate_dataset(ds, result)
    assert result.errors == []
    assert result.stats["invalid_source"] == 0


def test_validate_dataset_warns_on_unknown_source():
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    ds["source"] = "made_up_source"
    validate_dataset(ds, result)
    assert result.stats["invalid_source"] == 1
    assert any(w.get("field") == "source" for w in result.warnings), (
        f"no source warning in {result.warnings}"
    )


def test_validate_dataset_accepts_all_known_sources():
    for source in VALID_SOURCES:
        result = ValidationResult()
        ds = _minimal_valid_dataset()
        ds["source"] = source
        validate_dataset(ds, result)
        assert result.stats["invalid_source"] == 0, (
            f"{source} should be a valid source but warned"
        )


def test_validate_dataset_warns_on_missing_recommended():
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    for field in RECOMMENDED_DATASET_FIELDS:
        ds.pop(field, None)
    validate_dataset(ds, result)
    assert len(result.warnings) >= len(RECOMMENDED_DATASET_FIELDS)


def test_validate_dataset_handles_no_dataset_id():
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    del ds["dataset_id"]
    validate_dataset(ds, result)
    assert len(result.errors) >= 1


# ─── validate_digestion_output (entry point) ──────────────────────────────


def test_validate_digestion_output_returns_error_for_missing_dir(tmp_path: Path):
    out = validate_digestion_output(tmp_path / "no_such_dir")
    assert len(out.errors) >= 1
    assert any("does not exist" in str(e) for e in out.errors)


def test_validate_digestion_output_empty_dir(tmp_path: Path):
    out = validate_digestion_output(tmp_path)
    assert out.stats["datasets_checked"] == 0
    assert out.errors == []


def test_validate_digestion_output_walks_dataset_dirs(tmp_path: Path):
    for ds_id in ("ds001", "ds002"):
        ds_dir = tmp_path / ds_id
        ds_dir.mkdir()
        dataset = _minimal_valid_dataset()
        dataset["dataset_id"] = ds_id
        (ds_dir / f"{ds_id}_dataset.json").write_text(json.dumps(dataset))
        records_doc = {
            "dataset_id": ds_id,
            "records": [_minimal_valid_record() | {"dataset": ds_id}],
        }
        (ds_dir / f"{ds_id}_records.json").write_text(json.dumps(records_doc))

    out = validate_digestion_output(tmp_path)
    assert out.stats["datasets_checked"] == 2
    assert out.stats["records_checked"] == 2


def test_validate_digestion_output_surfaces_malformed_json(tmp_path: Path):
    ds_dir = tmp_path / "ds_bad"
    ds_dir.mkdir()
    (ds_dir / "ds_bad_dataset.json").write_text("{ this is not json")

    out = validate_digestion_output(tmp_path)
    assert any("Invalid JSON" in str(e) for e in out.errors), (
        f"expected 'Invalid JSON' error, got {out.errors}"
    )


def test_validate_digestion_output_accepts_snapshot_fixture():
    """End-to-end pin against the committed snapshot fixture."""
    snapshot_root = data_file("digest_snapshots/outputs")
    out = validate_digestion_output(snapshot_root)
    assert out.stats["datasets_checked"] == 3
    assert out.stats["records_checked"] == 5
    assert out.errors == [], f"snapshot has errors: {out.errors}"


def test_validate_digestion_output_full_report_golden():
    """Freeze the COMPLETE validation report against the committed corpus.

    The count-only pin above lets the warnings list, every stats counter, the
    source/modality distributions, and the data-quality issues drift silently.
    This freezes all of them so any change in validation behavior fails here.
    """
    out = validate_digestion_output(data_file("digest_snapshots/outputs"))
    report = {
        "errors": out.errors,
        "warnings": sorted(
            out.warnings, key=lambda w: (w["dataset"], w["field"] or "", w["message"])
        ),
        "stats": out.stats,
        "source_distribution": out.source_distribution,
        "modality_distribution": out.modality_distribution,
        "empty_datasets": sorted(out.empty_datasets),
        "zip_placeholder_datasets": sorted(out.zip_placeholder_datasets),
        "data_quality_issues": sorted(out.data_quality_issues),
    }
    assert report == {
        "errors": [],
        "warnings": [
            {
                "dataset": "ds_snapshot_eeg_montage",
                "field": "source",
                "message": "Unknown source: unknown",
            },
            {
                "dataset": "ds_snapshot_vhdr",
                "field": "source",
                "message": "Unknown source: unknown",
            },
        ],
        "stats": {
            "datasets_checked": 3,
            "records_checked": 5,
            "storage_errors": 0,
            "missing_mandatory": 0,
            "missing_recommended": 0,
            "empty_datasets": 0,
            "invalid_source": 2,
            "zip_placeholders": 0,
            "missing_nchans": 3,
            "missing_sampling_frequency": 3,
        },
        "source_distribution": {"unknown": 2, "zenodo": 1},
        "modality_distribution": {"eeg": 5},
        "empty_datasets": [],
        "zip_placeholder_datasets": [],
        "data_quality_issues": [
            "ds_snapshot_manifest (zenodo): missing nchans, sampling_frequency"
        ],
    }


def test_validate_digestion_output_strict_promotes_warnings(tmp_path: Path):
    """Unknown-source triggers a warning in non-strict mode."""
    ds_dir = tmp_path / "ds_synth"
    ds_dir.mkdir()
    dataset = _minimal_valid_dataset()
    dataset["dataset_id"] = "ds_synth"
    dataset["source"] = "made_up_source"
    (ds_dir / "ds_synth_dataset.json").write_text(json.dumps(dataset))

    non_strict = validate_digestion_output(tmp_path, strict=False)
    assert non_strict.stats["invalid_source"] == 1
    assert any(w.get("field") == "source" for w in non_strict.warnings)


# ─── Integration: storage URL rejection across all known sources ──────────


@pytest.mark.parametrize(
    ("source", "wrong_url"),
    [
        ("openneuro", "s3://nemar/wrong"),
        ("nemar", "s3://openneuro.org/wrong"),
        ("zenodo", "https://figshare.com/wrong"),
        ("figshare", "https://zenodo.org/wrong"),
        ("osf", "s3://openneuro.org/wrong"),
        ("scidb", "https://files.osf.io/wrong"),
        ("datarn", "https://scidb.cn/wrong"),
        ("gin", "s3://nemar/wrong"),
    ],
)
def test_validate_record_rejects_cross_source_storage_urls(source: str, wrong_url: str):
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"]["base"] = wrong_url

    validate_record(rec, "ds-test", source, result)
    assert result.stats["storage_errors"] == 1, (
        f"{source} should have rejected {wrong_url} but didn't"
    )


@pytest.mark.parametrize(
    ("source", "url"),
    [
        ("openneuro", "s3://openneuro.org/ds002893"),
        ("openneuro", "s3://openneuro.org/ds002893/sub-01/eeg/file.edf"),
        ("nemar", "s3://nemar/nm000176"),
        ("nemar", "s3://nmdatasets/nm000176"),
        ("osf", "https://files.osf.io/v1/resources/abc/providers/osfstorage/"),
        ("figshare", "https://figshare.com/ndownloader/files/12345"),
        ("figshare", "https://ndownloader.figshare.com/files/12345"),
        ("figshare", "https://mydomain.figshare.com/articles/12345"),
        ("zenodo", "https://zenodo.org/record/12345"),
        ("zenodo", "https://zenodo.org/records/12345"),
        ("scidb", "https://scidb.cn/dataset/abc"),
        ("scidb", "https://www.scidb.cn/dataset/abc"),
        ("datarn", "https://webdav.data.ru.nl/test/path"),
        ("gin", "https://gin.g-node.org/EEGManyLabs/study"),
    ],
)
def test_storage_url_accepts_canonical_per_source(source: str, url: str):
    ok, msg = validate_storage_url(source, url)
    assert ok is True, f"{source=} {url=} → {msg=}"
    assert msg == ""


@pytest.mark.parametrize(
    ("source", "wrong_url"),
    [
        ("nemar", "s3://openneuro.org/nm000176"),
        ("openneuro", "s3://nemar/ds002893"),
        ("figshare", "https://zenodo.org/record/12345"),
        ("zenodo", "https://files.osf.io/v1/abc"),
        ("openneuro", "https://example.com/ds002893"),
        ("zenodo", "s3://nemar/12345"),
    ],
)
def test_storage_url_rejects_cross_source_mismatch(source: str, wrong_url: str):
    ok, msg = validate_storage_url(source, wrong_url)
    assert ok is False
    assert source in msg
    assert wrong_url in msg


def test_storage_url_unknown_source_passes_through():
    ok, msg = validate_storage_url("brand_new_source", "https://example.com/")
    assert ok is True
    assert msg == ""


def test_storage_url_must_match_at_start():
    ok, _msg = validate_storage_url("openneuro", "garbage_s3://openneuro.org/ds123")
    assert ok is False


# ─── Constant-set invariants ──────────────────────────────────────────────


def test_valid_sources_includes_canonical():
    expected = {
        "openneuro",
        "nemar",
        "gin",
        "figshare",
        "zenodo",
        "osf",
        "scidb",
        "datarn",
    }
    assert expected.issubset(VALID_SOURCES)


def test_valid_storage_patterns_cover_all_valid_sources_except_hbn():
    sources_with_patterns = set(VALID_STORAGE_PATTERNS.keys())
    sources_without_patterns = VALID_SOURCES - sources_with_patterns
    assert sources_without_patterns <= {"hbn"}


def test_neuro_extensions_includes_canonical_formats():
    expected = {".edf", ".bdf", ".vhdr", ".set", ".fif", ".snirf"}
    assert expected.issubset(NEURO_EXTENSIONS)


def test_data_quality_fields_match_record_schema():
    assert "nchans" in DATA_QUALITY_FIELDS
    assert "sampling_frequency" in DATA_QUALITY_FIELDS


def test_storage_patterns_compile():
    for source, pattern in VALID_STORAGE_PATTERNS.items():
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"VALID_STORAGE_PATTERNS[{source!r}] = {pattern!r} fails: {e}")


# ─── ValidationResult ──────────────────────────────────────────────────────


def test_validation_result_initialises_empty():
    r = ValidationResult()
    assert r.errors == []
    assert r.warnings == []
    assert r.stats["datasets_checked"] == 0
    assert r.stats["records_checked"] == 0
    assert r.stats["storage_errors"] == 0


def test_validation_result_stats_keys_are_stable():
    r = ValidationResult()
    expected_keys = {
        "datasets_checked",
        "records_checked",
        "storage_errors",
        "missing_mandatory",
        "missing_recommended",
        "empty_datasets",
        "invalid_source",
        "zip_placeholders",
        "missing_nchans",
        "missing_sampling_frequency",
    }
    assert set(r.stats.keys()) == expected_keys
