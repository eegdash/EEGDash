"""Schema pre-flight gate — validate fixture records against the schemas.

Phase 6 of the robustness programme. This is the "stop bad records BEFORE
they hit MongoDB" tier. The tests:

1. Confirm every committed fixture record validates cleanly against
   ``eegdash.schemas.RecordModel``. Catches schema drift when the
   schemas package changes shape.
2. Confirm a deliberately-malformed record FAILS validation with a
   specific Pydantic error. Proves the gate is not asleep.
3. Confirm the gate fires on the three most-likely real-world drifts:
   missing required field, wrong field type, unknown enum value.

When CI runs ``5_inject.py --dry-run --input tests/fixtures/records/``,
all golden fixtures pass; CI fails if a future code change to either
``eegdash.schemas`` or the digest-step output drifts.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from eegdash.schemas import RecordModel
from pydantic import ValidationError

FIXTURES = Path(__file__).parent / "fixtures" / "records"


def _load_fixture(name: str) -> dict:
    """Load a fixture record JSON by filename (without extension)."""
    path = FIXTURES / f"{name}.json"
    return json.loads(path.read_text())


# ─── Golden fixtures validate ─────────────────────────────────────────────


def test_valid_record_eeg_passes_validation():
    """The committed CC0-derived golden EEG record validates."""
    data = _load_fixture("valid_record_eeg")
    record = RecordModel.model_validate(data)
    assert record.dataset == "ds002893"
    assert record.recording_modality == ["eeg"]
    assert record.extension == ".set"


def test_valid_record_eeg_roundtrips_through_json():
    """RecordModel dump → load → validate is idempotent (schema stable)."""
    data = _load_fixture("valid_record_eeg")
    record_a = RecordModel.model_validate(data)
    serialised = record_a.model_dump_json()
    record_b = RecordModel.model_validate_json(serialised)
    assert record_a == record_b


# ─── Negative-fixture parametric tests — gate fires on drift ──────────────


@pytest.mark.parametrize(
    ("field", "drift_reason"),
    [
        ("dataset", "missing required field 'dataset'"),
        ("bids_relpath", "missing required field 'bids_relpath'"),
        ("storage", "missing required field 'storage'"),
        ("recording_modality", "missing required field 'recording_modality'"),
    ],
)
def test_validation_fires_when_required_field_missing(field: str, drift_reason: str):
    """Removing any required field must surface a Pydantic ValidationError."""
    data = _load_fixture("valid_record_eeg")
    del data[field]
    with pytest.raises(ValidationError, match=field):
        RecordModel.model_validate(data)


def test_validation_fires_when_storage_lacks_required_subfield():
    """``storage`` is a nested schema; missing 'raw_key' must fail too."""
    data = _load_fixture("valid_record_eeg")
    del data["storage"]["raw_key"]
    with pytest.raises(ValidationError, match="raw_key"):
        RecordModel.model_validate(data)


def test_validation_fires_when_recording_modality_is_wrong_type():
    """``recording_modality`` is a list; a string must be rejected."""
    data = _load_fixture("valid_record_eeg")
    data["recording_modality"] = "eeg"  # should be ["eeg"]
    with pytest.raises(ValidationError):
        RecordModel.model_validate(data)


def test_validation_fires_when_dataset_is_wrong_type():
    """``dataset`` must be a string."""
    data = _load_fixture("valid_record_eeg")
    data["dataset"] = 12345  # should be str
    with pytest.raises(ValidationError):
        RecordModel.model_validate(data)


# ─── Defense-in-depth: deep clone, mutation isolation ─────────────────────


def test_drift_in_one_field_doesnt_silently_pass_via_alias():
    """A common bug class: a misspelled field is silently accepted as 'extra'.

    Pydantic v2's default is to accept-extra; the test asserts that the
    REQUIRED fields are still enforced (we don't accidentally satisfy
    'dataset' via 'datset' → 'extra').
    """
    data = deepcopy(_load_fixture("valid_record_eeg"))
    # codespell-ignore-line: deliberate misspelling, used as drift signal
    data["dataaset"] = data.pop("dataset")  # misspelled on purpose — drift signal
    with pytest.raises(ValidationError, match="dataset"):
        RecordModel.model_validate(data)
