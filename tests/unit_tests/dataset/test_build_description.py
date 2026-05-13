"""Tests for the unified _build_description helper and the three EEGDashDataset
initialization paths (records=, offline, query).

Covers:
- participant_tsv precedence conflict logging
- None-padding for missing description fields
- Case/hyphen-insensitive key matching
- Description parity across all three construction paths
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash.dataset.dataset import EEGDashDataset


# ---------------------------------------------------------------------------
# Lightweight EEGDashRaw stub
# ---------------------------------------------------------------------------


class _FakeRaw:
    """Minimal stand-in for EEGDashRaw.

    Satisfies the two things BaseConcatDataset needs from each element:
      - __len__ returning an integer
      - description as a pd.Series

    Stores the description kwarg so tests can inspect it later.
    """

    def __init__(self, record, cache_dir=None, description=None, **kwargs):
        _ = cache_dir, kwargs  # accepted to match EEGDashRaw's signature; not needed in stub
        self.record = record
        self.description = pd.Series(description or {}, dtype=object)

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_ds(tmp_path):
    """Lightweight EEGDashDataset used as a host for direct _build_description calls."""
    record = {
        "dataset": "ds_bd_test",
        "bidspath": "ds_bd_test/a.set",
        "bids_relpath": "a.set",
        "extension": ".set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    with patch("eegdash.dataset.dataset.EEGDashRaw", _FakeRaw):
        ds = EEGDashDataset(cache_dir=tmp_path, records=[record], download=True)
    return ds


@pytest.fixture
def parity_record(tmp_path):
    """A v2-format record suitable for all three construction paths."""
    return {
        "dataset": "ds_parity",
        "bidspath": "ds_parity/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "extension": ".set",
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
        "storage": {"backend": "local", "base": str(tmp_path / "ds_parity")},
    }


# ---------------------------------------------------------------------------
# 1. Precedence conflict: top-level record value wins over participant_tsv
# ---------------------------------------------------------------------------


def test_build_description_precedence_conflict(minimal_ds, caplog):
    """Top-level record value is kept when participant_tsv carries a different value.

    A debug-level log must be emitted to make the silent priority explicit.
    """
    record = {
        "age": 30,  # top-level value that should win
        "participant_tsv": {"age": 99, "sex": "M"},
    }

    with caplog.at_level(logging.DEBUG, logger="eegdash"):
        desc = minimal_ds._build_description(record, description_fields=["age", "sex"])

    assert desc["age"] == 30, "Top-level record value must take precedence"
    assert desc["sex"] == "M", "Uncontested participant_tsv field must be merged"

    conflict_logs = [m for m in caplog.messages if "age" in m and "30" in m]
    assert conflict_logs, (
        "Expected a debug log mentioning the 'age' conflict, got: "
        + str(caplog.messages)
    )
    assert any("kept" in m for m in caplog.messages), (
        "Default 'record' precedence must log 'kept', not 'overwrote'"
    )


# ---------------------------------------------------------------------------
# 2. Missing fields are padded with None, not omitted
# ---------------------------------------------------------------------------


def test_build_description_missing_fields_padding(minimal_ds):
    """Fields absent from the record appear in the description as None.

    Accessing description["missing_field_xyz"] must not raise KeyError.
    """
    record = {"entities": {"subject": "01"}}
    description_fields = ["subject", "missing_field_xyz"]

    desc = minimal_ds._build_description(record, description_fields=description_fields)

    assert desc["subject"] == "01"
    assert "missing_field_xyz" in desc, (
        "Missing field must be present in description (padded with None)"
    )
    assert desc["missing_field_xyz"] is None, "Missing field value must be None"


# ---------------------------------------------------------------------------
# 3. Key lookup is case- and separator-insensitive
# ---------------------------------------------------------------------------


def test_build_description_key_insensitivity(minimal_ds):
    """_find_key_in_nested_dict maps 'Subject-ID' in the record to 'subject_id' in fields."""
    record = {"Subject-ID": "sub-007"}

    desc = minimal_ds._build_description(
        record, description_fields=["subject_id", "task"]
    )

    assert desc["subject_id"] == "sub-007", (
        "Normalised key lookup must resolve 'Subject-ID' → 'subject_id'"
    )
    assert desc["task"] is None, "Absent field must be None"


# ---------------------------------------------------------------------------
# 4. Path parity: records=, offline, and query paths produce identical descriptions
# ---------------------------------------------------------------------------


def test_dataset_initialization_path_parity(tmp_path, parity_record):
    """All three EEGDashDataset construction paths must build identical descriptions.

    _FakeRaw is used instead of MagicMock so BaseConcatDataset can safely
    call len() and access .description on each element.  The description
    passed to each _FakeRaw constructor is extracted and compared via
    pandas.testing.assert_frame_equal.
    """
    description_fields = ["subject", "task"]
    (tmp_path / "ds_parity").mkdir(parents=True, exist_ok=True)

    with patch("eegdash.dataset.dataset.EEGDashRaw", _FakeRaw):

        # -- Path 1: records= ------------------------------------------------
        ds_records = EEGDashDataset(
            cache_dir=tmp_path,
            records=[parity_record],
            download=True,
            description_fields=description_fields,
        )

        # -- Path 2: offline (download=False) ---------------------------------
        # discover_local_bids_records returns the same record; EEGBIDSDataset
        # is made to fail so no participant enrichment happens, keeping the
        # result identical to what the records= path produces.
        with (
            patch(
                "eegdash.dataset.dataset.discover_local_bids_records",
                return_value=[parity_record],
            ),
            patch(
                "eegdash.dataset.dataset.EEGBIDSDataset",
                side_effect=Exception("no bids"),
            ),
        ):
            ds_offline = EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds_parity",
                download=False,
                description_fields=description_fields,
            )

        # -- Path 3: query (mocked API) ---------------------------------------
        mock_api = MagicMock()
        mock_api.find.return_value = [parity_record]
        with patch("eegdash.dataset.dataset.validate_record", return_value=[]):
            ds_query = EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds_parity",
                eeg_dash_instance=mock_api,
                download=True,
                description_fields=description_fields,
            )

    # BaseConcatDataset.description builds pd.DataFrame([ds.description for ds in datasets])
    # _FakeRaw.description is a real pd.Series, so this works correctly.
    pd.testing.assert_frame_equal(
        ds_records.description,
        ds_offline.description,
        check_like=True,
        obj="records= vs offline",
    )
    pd.testing.assert_frame_equal(
        ds_records.description,
        ds_query.description,
        check_like=True,
        obj="records= vs query",
    )


# ---------------------------------------------------------------------------
# 5. description_precedence="participant_tsv" — participant_tsv values win
# ---------------------------------------------------------------------------


def test_build_description_participant_tsv_precedence(tmp_path):
    """participant_tsv values overwrite conflicting record values when precedence='participant_tsv'.

    Also verifies that a None value in participant_tsv overwrites a non-None
    record value — this is intentional when the caller trusts that source fully.
    """
    _stub_record = {
        "dataset": "ds_prec",
        "bidspath": "ds_prec/a.set",
        "bids_relpath": "a.set",
        "extension": ".set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    with patch("eegdash.dataset.dataset.EEGDashRaw", _FakeRaw):
        ds = EEGDashDataset(
            cache_dir=tmp_path,
            records=[_stub_record],
            download=True,
            description_precedence="participant_tsv",
        )

    record = {
        "age": 30,
        "participant_tsv": {"age": 99, "sex": "M"},
    }
    desc = ds._build_description(record, description_fields=["age", "sex"])

    assert desc["age"] == 99, "participant_tsv value must win when precedence='participant_tsv'"
    assert desc["sex"] == "M"

    # None in participant_tsv overwrites a real record value (documented behaviour).
    record_none = {
        "age": 30,
        "participant_tsv": {"age": None},
    }
    desc_none = ds._build_description(record_none, description_fields=["age"])
    assert desc_none["age"] is None, (
        "None in participant_tsv must overwrite record value when precedence='participant_tsv'"
    )


# ---------------------------------------------------------------------------
# 6. Invalid description_precedence raises ValueError at construction
# ---------------------------------------------------------------------------


def test_dataset_invalid_description_precedence(tmp_path):
    """An unsupported description_precedence value raises ValueError at construction."""
    _stub_record = {
        "dataset": "ds_inv",
        "bidspath": "ds_inv/a.set",
        "bids_relpath": "a.set",
        "extension": ".set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="description_precedence must be one of"):
        EEGDashDataset(
            cache_dir=tmp_path,
            records=[_stub_record],
            download=True,
            description_precedence="invalid_mode",
        )
