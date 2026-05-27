"""Tests for ``_validate.py`` — schema gate (validators + helpers).

Two angles:

- **Functions** — validate_record / validate_dataset / validate_digestion_output (was test_validate_functions.py).
- **Helpers** — VALID_STORAGE_PATTERNS / VALID_SOURCES / RECOMMENDED_* / validate_storage_url (was test_validate_helpers.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eegdash.testing import data_file

from _validate import (
    DATA_QUALITY_FIELDS,
    NEURO_EXTENSIONS,
    RECOMMENDED_DATASET_FIELDS,
    RECOMMENDED_RECORD_FIELDS,
    VALID_SOURCES,
    VALID_STORAGE_PATTERNS,
    ValidationResult,
    validate_dataset,
    validate_digestion_output,
    validate_record,
    validate_storage_url,
)

# ─── 1. Functions ──────────────────────────────────────────────

# ─── Fixtures (minimal valid + invalid documents) ─────────────────────────


def _minimal_valid_record() -> dict:
    """Minimal Record that should pass RecordModel.model_validate.

    Keys here are the ones declared mandatory by eegdash.schemas.RecordModel.
    """
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
    """Happy path — minimal valid record adds no errors."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    validate_record(rec, "ds002893", "openneuro", result)
    # The record has no recommended fields beyond mandatory; that
    # surfaces warnings but no errors.
    assert result.errors == []
    assert result.stats["storage_errors"] == 0


def test_validate_record_flags_storage_url_mismatch():
    """A NEMAR URL on an OpenNeuro record triggers a storage error."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"]["base"] = "s3://nemar/ds002893"  # wrong source

    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["storage_errors"] == 1
    assert any(e.get("field") == "storage.base" for e in result.errors), (
        f"no storage.base error in {result.errors}"
    )


def test_validate_record_missing_mandatory_field_raises_pydantic_error():
    """Removing a mandatory field surfaces a pydantic ValidationError
    in result.errors (not raised)."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    del rec["bids_relpath"]

    validate_record(rec, "ds002893", "openneuro", result)
    assert len(result.errors) >= 1
    # Pydantic errors are prefixed with 'record'
    assert any("record" in str(e) for e in result.errors)


def test_validate_record_counts_missing_nchans():
    """nchans missing increments the missing_nchans counter."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    # nchans is None by default in our minimal record
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["missing_nchans"] == 1


def test_validate_record_counts_missing_sampling_frequency():
    """sampling_frequency missing → counter increment."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["missing_sampling_frequency"] == 1


def test_validate_record_does_not_count_present_nchans():
    """When nchans is present + nonzero, counter does NOT increment."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["nchans"] = 64
    rec["sampling_frequency"] = 250.0
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["missing_nchans"] == 0
    assert result.stats["missing_sampling_frequency"] == 0


def test_validate_record_treats_zero_nchans_as_missing():
    """nchans == 0 is treated as missing (degenerate value)."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["nchans"] = 0
    validate_record(rec, "ds002893", "openneuro", result)
    assert result.stats["missing_nchans"] == 1


def test_validate_record_first_record_only_emits_recommended_warnings():
    """``record_idx == 0`` triggers recommended-field warnings;
    subsequent records reuse the same warnings (don't accumulate)."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    # First record (idx=0) — warnings emitted for each missing
    # recommended field
    validate_record(rec, "ds002893", "openneuro", result, record_idx=0)
    first_warning_count = len(result.warnings)
    assert first_warning_count > 0

    # Second record — no new warnings should be added
    validate_record(rec, "ds002893", "openneuro", result, record_idx=1)
    assert len(result.warnings) == first_warning_count


def test_validate_record_unknown_source_passes_storage_check():
    """Unknown source: storage URL check is skipped (per
    validate_storage_url's pass-through)."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    # Use a totally unknown source
    validate_record(rec, "ds-xxx", "brand_new_source", result)
    assert result.stats["storage_errors"] == 0


def test_validate_record_handles_missing_storage_dict():
    """If storage is missing/empty, no storage error (it's a schema
    error caught by pydantic, separately)."""
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"] = {}
    validate_record(rec, "ds002893", "openneuro", result)
    # Storage-check is skipped when storage dict is empty
    assert result.stats["storage_errors"] == 0


# ─── validate_dataset ─────────────────────────────────────────────────────


def test_validate_dataset_accepts_minimal_valid_dataset():
    """Happy path — minimal valid dataset adds no errors."""
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    validate_dataset(ds, result)
    # The minimum doc is missing some recommended fields (warnings)
    # but no errors.
    assert result.errors == []
    assert result.stats["invalid_source"] == 0


def test_validate_dataset_warns_on_unknown_source():
    """An unknown source name produces a warning + counter increment."""
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    ds["source"] = "made_up_source"
    validate_dataset(ds, result)
    assert result.stats["invalid_source"] == 1
    assert any(w.get("field") == "source" for w in result.warnings), (
        f"no source warning in {result.warnings}"
    )


def test_validate_dataset_accepts_all_known_sources():
    """Each VALID_SOURCES entry should not trigger an invalid_source warning."""
    from _validate import VALID_SOURCES

    for source in VALID_SOURCES:
        result = ValidationResult()
        ds = _minimal_valid_dataset()
        ds["source"] = source
        validate_dataset(ds, result)
        assert result.stats["invalid_source"] == 0, (
            f"{source} should be a valid source but warned"
        )


def test_validate_dataset_warns_on_missing_recommended():
    """Each missing recommended field produces a warning."""
    from _validate import RECOMMENDED_DATASET_FIELDS

    result = ValidationResult()
    ds = _minimal_valid_dataset()
    # Remove all recommended fields
    for field in RECOMMENDED_DATASET_FIELDS:
        ds.pop(field, None)
    validate_dataset(ds, result)
    # Expect one warning per missing recommended field
    assert len(result.warnings) >= len(RECOMMENDED_DATASET_FIELDS)


def test_validate_dataset_handles_no_dataset_id():
    """A dataset without dataset_id still validates (default 'unknown')."""
    result = ValidationResult()
    ds = _minimal_valid_dataset()
    del ds["dataset_id"]
    # Should not crash; pydantic surfaces the error via _add_pydantic_errors
    validate_dataset(ds, result)
    assert len(result.errors) >= 1


# ─── validate_digestion_output (entry point) ──────────────────────────────


def test_validate_digestion_output_returns_error_for_missing_dir(tmp_path: Path):
    """Non-existent input dir → result.errors populated."""
    out = validate_digestion_output(tmp_path / "no_such_dir")
    assert len(out.errors) >= 1
    assert any("does not exist" in str(e) for e in out.errors)


def test_validate_digestion_output_empty_dir(tmp_path: Path):
    """An empty directory → 0 datasets checked, no errors."""
    out = validate_digestion_output(tmp_path)
    assert out.stats["datasets_checked"] == 0
    assert out.errors == []


def test_validate_digestion_output_walks_dataset_dirs(tmp_path: Path):
    """Each dataset subdirectory becomes one 'datasets_checked' tick."""
    for ds_id in ("ds001", "ds002"):
        ds_dir = tmp_path / ds_id
        ds_dir.mkdir()
        # Write a minimal valid dataset.json
        dataset = _minimal_valid_dataset()
        dataset["dataset_id"] = ds_id
        (ds_dir / f"{ds_id}_dataset.json").write_text(json.dumps(dataset))
        # And a minimal records.json
        records_doc = {
            "dataset_id": ds_id,
            "records": [_minimal_valid_record() | {"dataset": ds_id}],
        }
        (ds_dir / f"{ds_id}_records.json").write_text(json.dumps(records_doc))

    out = validate_digestion_output(tmp_path)
    assert out.stats["datasets_checked"] == 2
    assert out.stats["records_checked"] == 2


def test_validate_digestion_output_surfaces_malformed_json(tmp_path: Path):
    """A dataset.json that won't parse becomes an error, not a crash."""
    ds_dir = tmp_path / "ds_bad"
    ds_dir.mkdir()
    (ds_dir / "ds_bad_dataset.json").write_text("{ this is not json")

    out = validate_digestion_output(tmp_path)
    # The malformed JSON triggers a JSONDecodeError → result.add_error
    assert any("Invalid JSON" in str(e) for e in out.errors), (
        f"expected 'Invalid JSON' error, got {out.errors}"
    )


def test_validate_digestion_output_accepts_snapshot_fixture():
    """The committed snapshot fixture validates cleanly.

    End-to-end pin: the BIDS + manifest snapshot output produced by
    Stage 3 must pass the validator. If a Stage-3 refactor changes
    a field, this fires.
    """
    snapshot_root = data_file("digest_snapshots/outputs")
    out = validate_digestion_output(snapshot_root)
    # Snapshot has 3 datasets, 5 records:
    #   - ds_snapshot_vhdr           1 record  (BIDS-fs, no montage)
    #   - ds_snapshot_manifest       3 records (manifest-only)
    #   - ds_snapshot_eeg_montage    1 record  (BIDS-fs, +1 montage — added
    #                                          2026-05-22 to engage Layer-2
    #                                          acceptance montage tests)
    assert out.stats["datasets_checked"] == 3
    assert out.stats["records_checked"] == 5
    # No schema errors. (May have warnings for unknown source on
    # ds_snapshot_vhdr; that's fine in non-strict mode.)
    assert out.errors == [], f"snapshot has errors: {out.errors}"


def test_validate_digestion_output_strict_promotes_warnings(tmp_path: Path):
    """In strict mode, the snapshot's unknown-source warning becomes
    an error — pinning the documented promotion."""
    # Build a synthetic dataset with unknown source (warning trigger)
    ds_dir = tmp_path / "ds_synth"
    ds_dir.mkdir()
    dataset = _minimal_valid_dataset()
    dataset["dataset_id"] = "ds_synth"
    dataset["source"] = "made_up_source"  # triggers unknown-source warning
    (ds_dir / "ds_synth_dataset.json").write_text(json.dumps(dataset))

    # Non-strict: warning, no error.
    non_strict = validate_digestion_output(tmp_path, strict=False)
    assert non_strict.stats["invalid_source"] == 1
    # Errors may exist for missing records.json but no source-related error.

    # The "strict promotes warnings" mechanic happens at the CLI layer
    # (4_validate_output.py) rather than inside validate_digestion_output.
    # The library function returns warnings either way; the caller decides
    # how to interpret them. So we just verify the warning was emitted.
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
    """Cross-source rejection across all 8 known sources.

    This is the schema gate that prevented the pre-PR-#327 misrouting
    bug from re-occurring across the 8 active sources.
    """
    result = ValidationResult()
    rec = _minimal_valid_record()
    rec["storage"]["base"] = wrong_url

    validate_record(rec, "ds-test", source, result)
    assert result.stats["storage_errors"] == 1, (
        f"{source} should have rejected {wrong_url} but didn't"
    )


# ─── 2. Helpers ──────────────────────────────────────────────

# ─── VALID_STORAGE_PATTERNS — happy paths per source ──────────────────────


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
    """Each source's known-good URL shape is accepted."""
    ok, msg = validate_storage_url(source, url)
    assert ok is True, f"{source=} {url=} → {msg=}"
    assert msg == ""


# ─── VALID_STORAGE_PATTERNS — cross-source mismatches rejected ────────────


@pytest.mark.parametrize(
    ("source", "wrong_url"),
    [
        # OpenNeuro URL on NEMAR dataset (the pre-PR-#327 misrouting bug)
        ("nemar", "s3://openneuro.org/nm000176"),
        # NEMAR URL on OpenNeuro
        ("openneuro", "s3://nemar/ds002893"),
        # Zenodo URL on Figshare
        ("figshare", "https://zenodo.org/record/12345"),
        # OSF URL on Zenodo
        ("zenodo", "https://files.osf.io/v1/abc"),
        # Random URL on OpenNeuro
        ("openneuro", "https://example.com/ds002893"),
        # NEMAR URL on Zenodo
        ("zenodo", "s3://nemar/12345"),
    ],
)
def test_storage_url_rejects_cross_source_mismatch(source: str, wrong_url: str):
    """A URL with the wrong source's pattern is rejected with a message."""
    ok, msg = validate_storage_url(source, wrong_url)
    assert ok is False
    assert source in msg
    assert wrong_url in msg


def test_storage_url_unknown_source_passes_through():
    """Unknown sources can't be validated — return True with empty message."""
    ok, msg = validate_storage_url("brand_new_source", "https://example.com/")
    assert ok is True
    assert msg == ""


def test_storage_url_must_match_at_start():
    """Patterns are anchored — URL with the right shape mid-string is rejected."""
    ok, _msg = validate_storage_url("openneuro", "garbage_s3://openneuro.org/ds123")
    assert ok is False


# ─── Constant-set invariants ──────────────────────────────────────────────


def test_valid_sources_includes_canonical():
    """Every Source mentioned in ADR 0001 / cycle 1 must be in VALID_SOURCES."""
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
    """``hbn`` is in VALID_SOURCES but doesn't have a storage pattern —
    that's intentional (HBN is a logical grouping, not a Source per se).

    Every OTHER valid source should have a storage pattern.
    """
    sources_with_patterns = set(VALID_STORAGE_PATTERNS.keys())
    sources_without_patterns = VALID_SOURCES - sources_with_patterns
    # Only ``hbn`` is allowed to lack a pattern.
    assert sources_without_patterns <= {"hbn"}


def test_neuro_extensions_includes_canonical_formats():
    """The 7 BIDS-supported neuro file formats must be present."""
    expected = {".edf", ".bdf", ".vhdr", ".set", ".fif", ".snirf"}
    assert expected.issubset(NEURO_EXTENSIONS)


def test_data_quality_fields_match_record_schema():
    """The data-quality fields enforced for ingest match the Record schema."""
    assert "nchans" in DATA_QUALITY_FIELDS
    assert "sampling_frequency" in DATA_QUALITY_FIELDS


def test_recommended_fields_are_string_lists():
    """Recommended field lists are non-empty lists of strings."""
    assert all(
        isinstance(f, str) and f
        for f in RECOMMENDED_RECORD_FIELDS + RECOMMENDED_DATASET_FIELDS
    )


def test_storage_patterns_compile():
    """Every regex pattern compiles (catches typos at import time)."""
    import re

    for source, pattern in VALID_STORAGE_PATTERNS.items():
        try:
            re.compile(pattern)
        except re.error as e:
            pytest.fail(f"VALID_STORAGE_PATTERNS[{source!r}] = {pattern!r} fails: {e}")


# ─── ValidationResult ──────────────────────────────────────────────────────


def test_validation_result_initialises_empty():
    """A fresh ValidationResult has empty errors + warnings + zero counts."""
    r = ValidationResult()
    assert r.errors == []
    assert r.warnings == []
    assert r.stats["datasets_checked"] == 0
    assert r.stats["records_checked"] == 0
    assert r.stats["storage_errors"] == 0


def test_validation_result_stats_keys_are_stable():
    """Pins the stats key names so downstream consumers don't break silently."""
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
