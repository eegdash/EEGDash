from unittest.mock import MagicMock, patch

import pytest

from eegdash.dataset.dataset import EEGChallengeDataset, EEGDashDataset


@pytest.fixture
def mock_dash_client():
    mock = MagicMock()
    # Mock find to return one record
    mock.find.return_value = [
        {
            "dataset": "ds1",
            "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "storage": {"backend": "s3", "base": "s3://bucket/ds1"},
            "entities_mne": {"subject": "01", "task": "rest"},
        }
    ]
    return mock


def test_eegdashdataset_records_init(tmp_path):
    # Init directly with records (no DB call)
    records = [
        {
            "dataset": "ds1",
            "bids_relpath": "sub-01/eeg/file.edf",
            "storage": {"backend": "local", "base": str(tmp_path)},
            "entities_mne": {"subject": "01"},
        }
    ]
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset="ds1", records=records, download=False
    )
    assert len(ds) == 1
    assert ds.datasets[0].bidspath.subject == "01"


def test_eegdashdataset_infer_dataset_id(tmp_path):
    # Infer dataset ID from records
    records = [
        {
            "dataset": "ds_inferred",
            "bids_relpath": "file.edf",
            "storage": {"backend": "local", "base": str(tmp_path)},
        }
    ]
    ds = EEGDashDataset(cache_dir=tmp_path, records=records, download=False)
    assert ds.query["dataset"] == "ds_inferred"


def test_eegdashdataset_deduplication(tmp_path):
    # Duplicate records
    rec = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg/file.edf",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    records = [rec, rec]

    # Without dedupe
    ds = EEGDashDataset(
        cache_dir=tmp_path, dataset="ds1", records=records, download=False
    )
    assert len(ds) == 2

    # With dedupe
    ds_dedupe = EEGDashDataset(
        cache_dir=tmp_path,
        dataset="ds1",
        records=records,
        download=False,
        _dedupe_records=True,
    )
    assert len(ds_dedupe) == 1


def test_eegdashdataset_offline_missing_dir(tmp_path):
    # Download=False but dir missing
    with pytest.raises(ValueError, match="Offline mode is enabled, but local data_dir"):
        EEGDashDataset(cache_dir=tmp_path, dataset="ds_missing", download=False)


def test_eegchallenge_validation(tmp_path):
    # Unknown release
    with pytest.raises(ValueError, match="Unknown release: RX"):
        EEGChallengeDataset(release="RX", cache_dir=tmp_path)

    # Query with dataset key
    with pytest.raises(ValueError, match="Query using the parameters `dataset`"):
        EEGChallengeDataset(release="R1", cache_dir=tmp_path, query={"dataset": "foo"})


def test_eegchallenge_mini_subject_validation(tmp_path):
    # Mock map
    with patch(
        "eegdash.dataset.dataset.SUBJECT_MINI_RELEASE_MAP", {"R-Test": ["001", "002"]}
    ):
        with patch(
            "eegdash.dataset.dataset.RELEASE_TO_OPENNEURO_DATASET_MAP",
            {"R-Test": "dsX"},
        ):
            # Valid subject
            ds = EEGChallengeDataset(
                release="R-Test",
                cache_dir=tmp_path,
                subject="001",
                download=False,
                records=[],
            )
            assert ds.query["subject"] == [
                "001"
            ]  # or similar logic depending on implementation detail

            # Invalid subject for mini
            with pytest.raises(ValueError, match="Some requested subject"):
                EEGChallengeDataset(
                    release="R-Test", cache_dir=tmp_path, subject="999", mini=True
                )
