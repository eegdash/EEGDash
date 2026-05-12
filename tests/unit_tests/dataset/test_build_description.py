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
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_ds(tmp_path):
    """Lightweight EEGDashDataset used as a host for direct _build_description calls.

    EEGDashRaw is mocked so no filesystem or network access is required.
    """
    record = {
        "dataset": "ds_bd_test",
        "bidspath": "ds_bd_test/a.set",
        "bids_relpath": "a.set",
        "extension": ".set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    with patch("eegdash.dataset.dataset.EEGDashRaw"):
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
        desc = minimal_ds._build_description(
            record, description_fields=["age", "sex"]
        )

    assert desc["age"] == 30, "Top-level record value must take precedence"
    assert desc["sex"] == "M", "Uncontested participant_tsv field must be merged"

    conflict_logs = [m for m in caplog.messages if "age" in m and "30" in m]
    assert conflict_logs, (
        "Expected a debug log mentioning the 'age' conflict, got: "
        + str(caplog.messages)
    )


# ---------------------------------------------------------------------------
# 2. Missing fields are padded with None, not omitted
# ---------------------------------------------------------------------------


def test_build_description_missing_fields_padding(minimal_ds):
    """Fields absent from the record appear in the description as None.

    Accessing description["missing_field_xyz"] must not raise KeyError.
    """
    record = {
        "entities": {"subject": "01"},
    }
    description_fields = ["subject", "missing_field_xyz"]

    desc = minimal_ds._build_description(record, description_fields=description_fields)

    assert "subject" in desc
    assert desc["subject"] == "01"

    assert "missing_field_xyz" in desc, (
        "Missing field must be present in description (padded with None)"
    )
    assert desc["missing_field_xyz"] is None, (
        "Missing field value must be None, not absent"
    )


# ---------------------------------------------------------------------------
# 3. Key lookup is case- and separator-insensitive
# ---------------------------------------------------------------------------


def test_build_description_key_insensitivity(minimal_ds):
    """_find_key_in_nested_dict maps 'Subject-ID' in the record to 'subject_id' in fields."""
    record = {
        "Subject-ID": "sub-007",
    }

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

    EEGDashRaw is mocked throughout; the description dicts passed to each
    constructor call are captured and compared via pandas.testing.assert_frame_equal.
    """
    description_fields = ["subject", "task"]
    (tmp_path / "ds_parity").mkdir(parents=True, exist_ok=True)

    with patch("eegdash.dataset.dataset.EEGDashRaw") as mock_raw_cls:

        # -- Path 1: records= ------------------------------------------------
        EEGDashDataset(
            cache_dir=tmp_path,
            records=[parity_record],
            download=True,
            description_fields=description_fields,
        )
        desc_records = mock_raw_cls.call_args_list[-1].kwargs["description"]
        mock_raw_cls.reset_mock()

        # -- Path 2: offline (download=False) ---------------------------------
        # discover_local_bids_records is mocked to return the same record;
        # EEGBIDSDataset is made to fail so no participant enrichment happens,
        # keeping the result identical to what records= produces.
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
            EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds_parity",
                download=False,
                description_fields=description_fields,
            )
        desc_offline = mock_raw_cls.call_args_list[-1].kwargs["description"]
        mock_raw_cls.reset_mock()

        # -- Path 3: query (mocked API) ---------------------------------------
        mock_api = MagicMock()
        mock_api.find.return_value = [parity_record]
        with patch("eegdash.dataset.dataset.validate_record", return_value=[]):
            EEGDashDataset(
                cache_dir=tmp_path,
                dataset="ds_parity",
                eeg_dash_instance=mock_api,
                download=True,
                description_fields=description_fields,
            )
        desc_query = mock_raw_cls.call_args_list[-1].kwargs["description"]

    # Wrap each description dict in a single-row DataFrame and compare strictly.
    df_records = pd.DataFrame([desc_records])
    df_offline = pd.DataFrame([desc_offline])
    df_query = pd.DataFrame([desc_query])

    pd.testing.assert_frame_equal(
        df_records, df_offline, check_like=True, obj="records= vs offline"
    )
    pd.testing.assert_frame_equal(
        df_records, df_query, check_like=True, obj="records= vs query"
    )
