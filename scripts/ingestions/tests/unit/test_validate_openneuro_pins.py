"""NEMAR records may carry an approved OpenNeuro object pin (listings a242cf7).

Nightly 2026-10-24: 4_validate_output.py and the inject gate rejected 8
records of on003574, on003768, on004346 and on005815 whose storage is a
version-pinned OpenNeuro object (backend s3, base s3://openneuro.org/dsN,
version_id). Inject wrote nothing. Such a record must validate; an unpinned
OpenNeuro base under a NEMAR id still must not.
"""

from __future__ import annotations

from _validate import ValidationResult, validate_record


def _record(storage):
    return {
        "dataset": "on003574",
        "bids_relpath": "sub-17/eeg/sub-17_task-rest_eeg.edf",
        "storage": storage,
    }


def _errors(storage):
    result = ValidationResult()
    validate_record(_record(storage), "on003574", "nemar", result)
    return [e for e in result.errors if "storage.base" in str(e)]


def test_pinned_openneuro_object_is_valid_for_nemar():
    pin = {
        "backend": "s3",
        "base": "s3://openneuro.org/ds003574",
        "raw_key": "sub-17/eeg/sub-17_task-rest_eeg.edf",
        "version_id": "nvQ123",
    }
    assert _errors(pin) == []


def test_unpinned_openneuro_base_still_invalid_for_nemar():
    assert _errors({"backend": "s3", "base": "s3://openneuro.org/ds003574"}) != []


def test_pin_must_be_the_same_accession():
    pin = {"backend": "s3", "base": "s3://openneuro.org/ds999999", "version_id": "x"}
    assert _errors(pin) != []


def test_nemar_base_unchanged():
    assert _errors({"backend": "nemar", "base": "s3://nemar/on003574"}) == []


def test_per_file_version_ids_pin_is_valid():
    # on003768 / on005815 store the pin per file.
    pin = {
        "backend": "s3",
        "base": "s3://openneuro.org/ds003574",
        "version_ids": {"sub-17/eeg/sub-17_task-rest_eeg.edf": "abc"},
    }
    assert _errors(pin) == []
