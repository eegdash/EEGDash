"""NEMAR records may carry an approved OpenNeuro object pin (listings a242cf7).

Nightly 2026-10-24: 4_validate_output.py and the inject gate rejected 8
records of on003574, on003768, on004346 and on005815 whose storage is a
version-pinned OpenNeuro object (backend s3, base s3://openneuro.org/dsN,
version_id). Inject wrote nothing. Such a record must validate; an unpinned
OpenNeuro base under a NEMAR id still must not.
"""

from __future__ import annotations

import pytest

from _validate import ValidationResult, validate_record
from _validate import _is_approved_openneuro_pin as pin_ok


def _record(storage):
    return {
        "dataset": "on003574",
        "bids_relpath": "sub-17/eeg/sub-17_task-rest_eeg.edf",
        "storage": storage,
    }


def _errors(storage, dataset_id="on003574", source="nemar"):
    result = ValidationResult()
    validate_record(_record(storage), dataset_id, source, result)
    return [e for e in result.errors if e.get("field") == "storage.base"]


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


_PIN = {"backend": "s3", "base": "s3://openneuro.org/ds003574", "version_id": "v1"}


@pytest.mark.parametrize("source", ["gin", "osf", "openneuro"])
def test_other_sources_do_not_get_the_exception(source):
    # Only NEMAR records use the pin exception; for others the normal rule applies.
    assert pin_ok("on003574", source, _PIN) is False


@pytest.mark.parametrize("backend", ["https", "nemar", None])
def test_backend_must_be_s3(backend):
    assert _errors({**_PIN, "backend": backend}) != []


@pytest.mark.parametrize(
    "base",
    [
        "s3://openneuro.org/ds003574/sub-17",
        "s3://openneuro.org/ds003574x",
        "s3://openneuro.org/ds0035740",
        "s3://openneuro.org/ds00357",
    ],
)
def test_base_must_be_exactly_the_same_accession(base):
    assert _errors({**_PIN, "base": base}) != []


@pytest.mark.parametrize(
    "dataset_id", ["on003574x", "ds003574", "nm003574", "xon003574"]
)
def test_dataset_id_must_be_exactly_on_digits(dataset_id):
    assert _errors(_PIN, dataset_id=dataset_id) != []
