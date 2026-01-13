from unittest.mock import MagicMock, patch

import pytest


def test_len_with_ntimes_and_sfreq():
    """Test length calculation when ntimes exists and raw is accessed."""
    from eegdash.dataset.base import EEGDashRaw

    # Create mock with ntimes in record and accessed raw
    mock_raw = MagicMock()
    mock_raw.__len__ = MagicMock(return_value=5000)

    recording = EEGDashRaw.__new__(EEGDashRaw)
    recording._raw = mock_raw
    recording.record = {"ntimes": 10.5, "sampling_frequency": 500}

    # Access length - should use len(self._raw) since _raw exists
    length = len(recording)
    assert length == 5000


def test_len_without_raw_with_ntimes_sfreq():
    """Test length calculation from ntimes*sfreq when raw not loaded."""
    from eegdash.dataset.base import EEGDashRaw

    recording = EEGDashRaw.__new__(EEGDashRaw)
    recording._raw = None
    recording.record = {"ntimes": 10, "sampling_frequency": 100}

    # Line 180: should return int(ntimes * sfreq) = 10 * 100 = 1000
    length = len(recording)
    assert length == 1000


def test_eegdashraw_len_with_sfreq_ntimes(tmp_path):
    """Test __len__ returns ntimes*sfreq when raw not loaded (line 180)."""
    from eegdash.dataset.base import EEGDashRaw

    # Create proper BIDS directory structure
    ds_dir = tmp_path / "ds_test"
    ds_dir.mkdir()
    sub_dir = ds_dir / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    record = {
        "dataset": "ds_test",
        "data_name": "ds_test_sub-01_task-rest_eeg.set",
        "bidspath": "ds_test/sub-01/eeg/sub-01_task-rest_eeg.set",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {
            "backend": "local",
            "base": str(ds_dir),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.set",
        },
        "subject": "01",
        "session": None,
        "run": None,
        "task": "rest",
        "modality": "eeg",
        "suffix": "eeg",
        "datatype": "eeg",
        "extension": ".set",
        "ntimes": 10.5,  # seconds
        "sfreq": 256,  # Hz
        "sampling_frequency": 256,
        "entities_mne": {"subject": "01", "task": "rest"},
        "entities": {"subject": "01", "task": "rest"},
    }

    raw_obj = EEGDashRaw(record=record, cache_dir=tmp_path)

    # Should return int(ntimes * sfreq) = int(10.5 * 256) = 2688
    assert len(raw_obj) == int(10.5 * 256)


def test_base_ensure_raw_failure(tmp_path):
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds",
        "bids_relpath": "f.set",
        "storage": {"base": str(tmp_path), "backend": "local"},
        # Missing ntimes/sfreq to force load
    }
    # Create file so exists check passes
    d = tmp_path / "ds"
    d.mkdir()
    (d / "f.set").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=[]):
        ds = EEGDashRaw(record, str(tmp_path))

        # Mock download success
        with patch.object(ds, "_download_required_files"):
            # Mock load failure
            with patch("mne_bids.read_raw_bids", side_effect=ValueError("Bad file")):
                with pytest.raises(ValueError):
                    ds._ensure_raw()

                # Length check
                # Should return 0 and log warning
                assert len(ds) == 0


def test_base_len_from_metadata(tmp_path):
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds",
        "bids_relpath": "f.set",
        "storage": {"base": str(tmp_path), "backend": "local"},
        "ntimes": 100,
        "sampling_frequency": 10,
    }
    with patch("eegdash.dataset.base.validate_record", return_value=[]):
        ds = EEGDashRaw(record, str(tmp_path))
        assert len(ds) == 1000


@patch("eegdash.dataset.base.validate_record")
def test_len_error_handling(mock_validate, tmp_path):
    from eegdash.dataset.base import EEGDashRaw

    mock_validate.return_value = None  # Simulate valid record
    record = {
        "dataset": "ds002",
        "bids_relpath": "sub-02/eeg.vhdr",
        # Missing ntimes/sfreq to trigger _ensure_raw
    }
    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Mock _ensure_raw to raise exception
    dataset._ensure_raw = MagicMock(side_effect=Exception("Download failed"))

    # Trigger __len__
    length = len(dataset)
    assert length == 0  # Should fallback to 0 on error


@patch("eegdash.dataset.base.validate_record")
@patch("eegdash.dataset.base._repair_vhdr_pointers")
@patch("eegdash.dataset.base._ensure_coordsystem_symlink")
@patch("eegdash.dataset.base.mne_bids")
def test_ensure_raw_integrity(
    mock_mne_bids, mock_ensure_symlink, mock_repair, mock_validate, tmp_path
):
    """Test that _ensure_raw calls the appropriate IO helpers."""
    from eegdash.dataset.base import EEGDashRaw

    mock_validate.return_value = None

    dataset_id = "ds_test"
    record = {
        "dataset": dataset_id,
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path), "raw_key": "dummy"},
    }

    # Setup directory structure so checks pass
    eeg_dir = tmp_path / dataset_id / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    # Init dataset
    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Mock _load_raw to prevent actual loading
    dataset._load_raw = MagicMock()

    # Run _ensure_raw
    dataset._ensure_raw()

    # Check delegation
    mock_ensure_symlink.assert_called_once()
    mock_repair.assert_called_once()

    # Validation: the argument passed should be related to filecache
    args, _ = mock_repair.call_args
    assert args[0] == dataset.filecache


def test_eegdashraw_backend_logic(tmp_path):
    """Test INIT logic for different storage backends."""
    from eegdash.dataset.base import EEGDashRaw

    # Case 1: Local backend logic
    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "local",
            "base": str(tmp_path),
            "raw_key": "raw",
            "dep_keys": ["dep1"],
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
        # Should point to local base
        assert ds.bids_root == tmp_path
        assert ds._raw_uri is None
        assert len(ds._dep_paths) == 1

    # Case 2: S3 backend logic
    record_s3 = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://bucket",
            "raw_key": "raw",
            "dep_keys": ["dep1"],
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record_s3, cache_dir=str(tmp_path))
        assert ds._raw_uri == "s3://bucket/raw"
        assert len(ds._dep_uris) == 1
