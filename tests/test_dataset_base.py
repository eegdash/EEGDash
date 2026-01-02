from unittest.mock import patch

import pytest

from eegdash.dataset.base import EEGDashRaw
from eegdash.schemas import create_record


@pytest.fixture
def sample_record():
    return create_record(
        dataset="ds1",
        storage_base="s3://bucket/ds1",
        bids_relpath="sub-01/eeg/sub-01_eeg.edf",
        subject="01",
    )


def test_eegdashraw_init_validation(tmp_path):
    # Invalid record (missing required fields)
    with pytest.raises(ValueError, match="Invalid record"):
        EEGDashRaw(record={}, cache_dir=str(tmp_path))


def test_eegdashraw_paths(tmp_path, sample_record):
    # S3 backend
    ds = EEGDashRaw(sample_record, cache_dir=str(tmp_path))
    assert ds.s3file == "s3://bucket/ds1/sub-01/eeg/sub-01_eeg.edf"
    assert ds.bids_root == tmp_path / "ds1"

    # Local backend
    sample_record["storage"]["backend"] = "local"
    sample_record["storage"]["base"] = str(tmp_path / "local")
    ds_local = EEGDashRaw(sample_record, cache_dir=str(tmp_path))
    assert ds_local.s3file is None
    assert ds_local.bids_root == tmp_path / "local"


def test_eegdashraw_len_error_handling(tmp_path, sample_record):
    ds = EEGDashRaw(sample_record, cache_dir=str(tmp_path))

    # Mock _ensure_raw to fail
    def mock_ensure(*args):
        raise RuntimeError("Load failed")

    ds._ensure_raw = mock_ensure

    # Should return 0 and log warning (handled inside __len__)
    assert len(ds) == 0


def test_eegdashraw_len_metadata_fallback(tmp_path, sample_record):
    # If ntimes/sfreq present, used for length
    sample_record["ntimes"] = 1000
    sample_record["sampling_frequency"] = 100.0

    ds = EEGDashRaw(sample_record, cache_dir=str(tmp_path))
    # Should NOT trigger download/_ensure_raw if metadata is present
    assert len(ds) == 100000


def test_eegdashraw_raw_property(tmp_path, sample_record):
    ds = EEGDashRaw(sample_record, cache_dir=str(tmp_path))

    with patch.object(ds, "_download_required_files") as mock_dl:
        with patch.object(ds, "_load_raw", return_value="MNE_RAW") as mock_load:
            # Accessing raw triggers download/load
            assert ds.raw == "MNE_RAW"
            mock_dl.assert_called_once()
            mock_load.assert_called_once()

            # Cached
            assert ds.raw == "MNE_RAW"
            assert mock_load.call_count == 1
