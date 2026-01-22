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
    """Test length calculation from ntimes when raw not loaded."""
    from eegdash.dataset.base import EEGDashRaw

    recording = EEGDashRaw.__new__(EEGDashRaw)
    recording._raw = None
    recording.record = {"ntimes": 10, "sampling_frequency": 100}

    # Should return int(ntimes) = 10
    length = len(recording)
    assert length == 10


def test_eegdashraw_len_with_sfreq_ntimes(tmp_path):
    """Test __len__ returns ntimes when raw not loaded."""
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
        "ntimes": 2688,  # samples
        "sfreq": 256,  # Hz
        "sampling_frequency": 256,
        "entities_mne": {"subject": "01", "task": "rest"},
        "entities": {"subject": "01", "task": "rest"},
    }

    raw_obj = EEGDashRaw(record=record, cache_dir=tmp_path)

    # Should return int(ntimes) = 2688
    assert len(raw_obj) == 2688


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
        assert len(ds) == 100


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


def test_eegdashraw_raw_setter(tmp_path):
    """Test the raw property setter (line 210-211)."""
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    ds.raw = mock_raw
    assert ds._raw is mock_raw


def test_eegdashraw_https_backend(tmp_path):
    """Test HTTPS backend storage logic."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "https",
            "base": "https://bucket.s3.amazonaws.com",
            "raw_key": "raw",
            "dep_keys": ["dep1"],
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
        assert ds._raw_uri == "https://bucket.s3.amazonaws.com/raw"
        assert len(ds._dep_uris) == 1


def test_eegdashraw_no_raw_key(tmp_path):
    """Test when storage has backend but no raw_key."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://bucket",
            # No raw_key
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
        # Should not set _raw_uri when raw_key is missing
        assert ds._raw_uri is None


def test_eegdashraw_local_backend_no_raw_key(tmp_path):
    """Test local backend uses filecache when raw_key is empty."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "local",
            "base": str(tmp_path),
            "raw_key": "",  # Empty raw_key
            "dep_keys": [],
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
        # Should use default filecache path
        assert ds._raw_uri is None


def test_eegdashraw_download_required_files_no_uri(tmp_path):
    """Test _download_required_files when _raw_uri is None."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }

    # Create the file so it exists
    ds_dir = tmp_path / "ds1" / "sub-01"
    ds_dir.mkdir(parents=True)
    (ds_dir / "eeg.vhdr").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))
        ds._raw_uri = None
        ds._download_required_files()
        # Should just set filenames
        assert ds.filenames == [ds.filecache]


def test_eegdashraw_bids_root_mkdir(tmp_path):
    """Test bids_root is created when it doesn't exist and _raw_uri is set."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds_new",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://bucket",
            "raw_key": "raw",
        },
    }
    bids_root = tmp_path / "ds_new"
    assert not bids_root.exists()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        EEGDashRaw(record, cache_dir=str(tmp_path))

    # bids_root should be created
    assert bids_root.exists()


def test_eegdashraw_invalid_record():
    """Test EEGDashRaw raises ValueError for invalid record."""
    import pytest

    from eegdash.dataset.base import EEGDashRaw

    # Invalid record missing required fields
    record = {"dataset": "ds1"}
    with pytest.raises(ValueError, match="Invalid record"):
        EEGDashRaw(record, cache_dir="/tmp")


def test_eegdashraw_len_with_raw_loaded(tmp_path):
    """Test __len__ returns len(self._raw) when raw is loaded."""
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "ntimes": 10,
        "sampling_frequency": 100,
    }

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Set a mock _raw object
    mock_raw = MagicMock()
    mock_raw.__len__ = MagicMock(return_value=5000)
    ds._raw = mock_raw

    assert len(ds) == 5000


def test_eegdashraw_ensure_raw_load_error_logging(tmp_path):
    """Test _ensure_raw logs error and re-raises on load failure."""
    from unittest.mock import MagicMock, patch

    import pytest

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }

    # Create directory for filecache
    ds_dir = tmp_path / "ds1" / "sub-01"
    ds_dir.mkdir(parents=True)

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Mock _download_required_files to do nothing
    ds._download_required_files = MagicMock()

    # Mock _load_raw to raise exception
    with patch.object(ds, "_load_raw", side_effect=IOError("Corrupted file")):
        with patch("eegdash.dataset.base.logger") as mock_logger:
            with pytest.raises(IOError):
                ds._ensure_raw()
            # Check error was logged
            mock_logger.error.assert_called_once()


def test_eegdashraw_s3file_attribute(tmp_path):
    """Test s3file attribute matches _raw_uri."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://mybucket",
            "raw_key": "path/to/file",
        },
    }

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    assert ds.s3file == "s3://mybucket/path/to/file"
    assert ds.s3file == ds._raw_uri
