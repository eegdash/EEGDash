from unittest.mock import MagicMock
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
    from eegdash.dataset.base import EEGDashRaw
    from unittest.mock import patch
    import pytest
    import mne_bids

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
    from eegdash.dataset.base import EEGDashRaw
    from unittest.mock import patch

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
