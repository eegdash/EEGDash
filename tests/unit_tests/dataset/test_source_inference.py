"""Tests for the runtime source / storage inference helper."""

from __future__ import annotations

import pytest

from eegdash.dataset._source_inference import (
    correct_storage_inplace,
    expected_storage_base,
    infer_source_from_dataset_id,
)


@pytest.mark.parametrize(
    "dataset_id, expected",
    [
        ("nm000237", "nemar"),
        ("nm000001", "nemar"),
        ("ds005505", "openneuro"),
        ("ds002718", "openneuro"),
        ("EEG2025r1mini", "nemar"),
        ("EEGManyLabs_MMN", "gin"),
        ("totally-custom-id", None),
        ("", None),
    ],
)
def test_infer_source_from_dataset_id(dataset_id, expected):
    assert infer_source_from_dataset_id(dataset_id) == expected


def test_expected_storage_base_only_resolves_pattern_sources():
    assert expected_storage_base("nm000237") == "s3://nemar/nm000237"
    assert expected_storage_base("ds005505") == "s3://openneuro.org/ds005505"
    # GIN and unknown patterns require extra metadata, so we don't fabricate one.
    assert expected_storage_base("EEGManyLabs_MMN") is None
    assert expected_storage_base("totally-custom-id") is None


def test_correct_storage_reroutes_misrouted_nemar_record():
    rec = {
        "dataset": "nm000237",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org/nm000237",
            "raw_key": "sub-13/ses-3/eeg/sub-13_ses-3_task-imagery_run-5_eeg.bdf",
        },
    }
    corrected, old = correct_storage_inplace(rec)
    assert corrected is True
    assert old == "s3://openneuro.org/nm000237"
    assert rec["storage"]["base"] == "s3://nemar/nm000237"
    # NEMAR records use the dedicated ``"nemar"`` backend tag because
    # direct S3 fetches against ``s3://nemar/<id>/<bidspath>`` don't work
    # (filenames are SHA-resolved); the runtime resolver converts BIDS
    # paths to SHA-keyed objects via the dataset's git-annex pointers.
    assert rec["storage"]["backend"] == "nemar"
    # Idempotent: a second call leaves the record untouched.
    assert correct_storage_inplace(rec) == (False, None)


def test_correct_storage_fixes_stale_backend_on_canonical_nemar_base():
    # Records ingested between PR #327 and the nemar-backend split sit in
    # the DB with the right base but the wrong backend tag. Self-heal
    # should still fire so the runtime takes the NEMAR resolution path.
    rec = {
        "dataset": "nm000237",
        "storage": {
            "backend": "s3",
            "base": "s3://nemar/nm000237",
            "raw_key": "sub-13/ses-3/eeg/sub-13_ses-3_task-imagery_run-5_eeg.bdf",
        },
    }
    corrected, old = correct_storage_inplace(rec)
    assert corrected is True
    assert old is None  # base wasn't rewritten, only backend
    assert rec["storage"]["base"] == "s3://nemar/nm000237"
    assert rec["storage"]["backend"] == "nemar"
    assert correct_storage_inplace(rec) == (False, None)


def test_correct_storage_leaves_correct_records_alone():
    rec = {
        "dataset": "ds005505",
        "storage": {"backend": "s3", "base": "s3://openneuro.org/ds005505"},
    }
    assert correct_storage_inplace(rec) == (False, None)


def test_correct_storage_does_not_touch_unknown_patterns():
    # We only auto-correct when the dataset_id pattern unambiguously
    # implies a source — never touch arbitrary IDs.
    rec = {
        "dataset": "totally-custom-id",
        "storage": {"base": "s3://openneuro.org/totally-custom-id"},
    }
    assert correct_storage_inplace(rec) == (False, None)


def test_correct_storage_skips_user_supplied_buckets():
    # A user-supplied mirror (not in STORAGE_CONFIGS) must not be silently
    # rewritten — only known foreign buckets are auto-rerouted.
    rec = {
        "dataset": "nm000237",
        "storage": {"base": "s3://my-private-mirror/nm000237", "backend": "s3"},
    }
    assert correct_storage_inplace(rec) == (False, None)


def test_correct_storage_handles_missing_storage():
    # No storage block at all -> no-op (no base to compare against).
    rec = {"dataset": "nm000237"}
    assert correct_storage_inplace(rec) == (False, None)
