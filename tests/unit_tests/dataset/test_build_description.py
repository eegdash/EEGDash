"""Tests for build_description and the three EEGDashDataset initialization paths."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eegdash.bids_metadata import build_description
from eegdash.dataset.dataset import EEGDashDataset

# ---------------------------------------------------------------------------
# Lightweight EEGDashRaw stub
# ---------------------------------------------------------------------------


class _FakeRaw:
    """Minimal stand-in for EEGDashRaw.

    This stub provides a real pd.Series and __len__.
    """

    def __init__(self, record, cache_dir=None, description=None, **kwargs):
        self.record = record
        self.description = pd.Series(description or {}, dtype=object)

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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
# 1. Path parity: records=, offline, and query paths produce identical descriptions
# ---------------------------------------------------------------------------


def test_dataset_initialization_path_parity(tmp_path, parity_record):
    """All three EEGDashDataset construction paths must build identical descriptions."""
    description_fields = ["subject", "task"]
    (tmp_path / "ds_parity").mkdir(parents=True, exist_ok=True)

    with patch("eegdash.dataset.dataset.EEGDashRaw", _FakeRaw):
        # Path 1: records=
        ds_records = EEGDashDataset(
            cache_dir=tmp_path,
            records=[parity_record],
            download=True,
            description_fields=description_fields,
        )

        # Path 2: offline
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

        # Path 3: query (mocked API)
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
# 2. description_precedence — parametrized over all conflict scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precedence,record_val,tsv_val,expected",
    [
        ("record", 30, 99, 30),
        ("participant_tsv", 30, 99, 99),
        ("participant_tsv", 30, None, None),
    ],
)
def test_build_description_precedence(precedence, record_val, tsv_val, expected):
    record = {
        "age": record_val,
        "participant_tsv": {"age": tsv_val, "sex": "M"},
    }
    desc = build_description(record, ["age", "sex"], description_precedence=precedence)
    assert desc["age"] == expected


# ---------------------------------------------------------------------------
# 3. Invalid description_precedence raises ValueError at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid", ["invalid_mode", "RECORD", "none", "both"])
def test_dataset_invalid_description_precedence(tmp_path, invalid):
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
            description_precedence=invalid,
        )
