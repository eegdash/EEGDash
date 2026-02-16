from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def base_record(tmp_path):
    """Create a minimal valid record for testing."""
    return {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }


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


@pytest.mark.parametrize(
    "create_vhdr,create_vmrk,expect_repair,expect_generate_vhdr,expect_generate_vmrk",
    [
        (True, True, True, False, False),  # VHDR+VMRK exist: repair only
        (
            True,
            False,
            True,
            False,
            True,
        ),  # VHDR exists, VMRK missing: repair + generate vmrk
        (
            False,
            False,
            False,
            True,
            True,
        ),  # VHDR missing: generate vhdr + vmrk stub (mock doesn't create files)
    ],
    ids=["vhdr_exists", "vmrk_missing", "vhdr_missing"],
)
@patch("eegdash.dataset.base.validate_record")
@patch("eegdash.dataset.base._generate_vmrk_stub")
@patch("eegdash.dataset.base._generate_vhdr_from_metadata")
@patch("eegdash.dataset.base._repair_vhdr_pointers")
@patch("eegdash.dataset.base._ensure_coordsystem_symlink")
@patch("eegdash.dataset.base.mne_bids")
def test_ensure_raw_vhdr_handling(
    mock_mne_bids,
    mock_ensure_symlink,
    mock_repair,
    mock_generate_vhdr,
    mock_generate_vmrk,
    mock_validate,
    tmp_path,
    create_vhdr,
    create_vmrk,
    expect_repair,
    expect_generate_vhdr,
    expect_generate_vmrk,
):
    """Test _ensure_raw handles VHDR/VMRK generation and repair correctly."""
    from eegdash.dataset.base import EEGDashRaw

    mock_validate.return_value = None

    dataset_id = "ds_test"
    bids_relpath = "sub-01/eeg/sub-01_task-rest_eeg.vhdr"
    eeg_dir = tmp_path / dataset_id / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
    }

    if create_vhdr:
        (eeg_dir / "sub-01_task-rest_eeg.vhdr").touch()
    if create_vmrk:
        (eeg_dir / "sub-01_task-rest_eeg.vmrk").touch()

    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))
    dataset._load_raw = MagicMock()
    dataset._ensure_raw()

    assert mock_repair.called == expect_repair
    assert mock_generate_vhdr.called == expect_generate_vhdr
    assert mock_generate_vmrk.called == expect_generate_vmrk


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


def test_eegdashraw_raw_setter(tmp_path, base_record):
    """Test the raw property setter (line 210-211)."""
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.base import EEGDashRaw

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(base_record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    ds.raw = mock_raw
    assert ds._raw is mock_raw


@pytest.mark.parametrize(
    "storage,expected_uri",
    [
        (
            {
                "backend": "https",
                "base": "https://bucket.s3.amazonaws.com",
                "raw_key": "raw",
                "dep_keys": ["dep1"],
            },
            "https://bucket.s3.amazonaws.com/raw",
        ),
        ({"backend": "s3", "base": "s3://bucket"}, None),  # No raw_key
        (
            {"backend": "local", "base": "/tmp", "raw_key": "", "dep_keys": []},
            None,
        ),  # Local empty raw_key
    ],
    ids=["https", "no_raw_key", "local_no_raw_key"],
)
@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_eegdashraw_storage_backends(mock_validate, tmp_path, storage, expected_uri):
    """Test different storage backend configurations."""
    from eegdash.dataset.base import EEGDashRaw

    if "base" in storage and storage["base"] == "/tmp":
        storage["base"] = str(tmp_path)  # Use actual tmp_path
    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg.vhdr",
        "storage": storage,
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    assert ds._raw_uri == expected_uri


def test_eegdashraw_download_required_files_no_uri(tmp_path, base_record):
    """Test _download_required_files when _raw_uri is None."""
    from unittest.mock import patch

    from eegdash.dataset.base import EEGDashRaw

    # Create the file so it exists
    ds_dir = tmp_path / "ds1" / "sub-01"
    ds_dir.mkdir(parents=True)
    (ds_dir / "eeg.vhdr").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(base_record, cache_dir=str(tmp_path))
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


def test_eegdashraw_ensure_raw_load_error_logging(tmp_path, base_record):
    """Test _ensure_raw logs error and re-raises on load failure."""
    from unittest.mock import MagicMock, patch

    import pytest

    from eegdash.dataset.base import EEGDashRaw

    # Create directory for filecache
    ds_dir = tmp_path / "ds1" / "sub-01"
    ds_dir.mkdir(parents=True)

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(base_record, cache_dir=str(tmp_path))

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


# ---- Tests for _load_raw retry logic ----


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_epoched_eeglab_recovery(mock_validate, tmp_path):
    """Test that epoched EEGLAB files trigger _load_epoched_eeglab_as_raw."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds006370",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=TypeError(
            "The number of trials is 1280. It must be 1 for raw files."
        ),
    ):
        with patch(
            "eegdash.dataset.base._load_epoched_eeglab_as_raw",
            return_value=mock_raw,
        ) as mock_epoched:
            result = ds._load_raw()
            mock_epoched.assert_called_once()
            assert result is mock_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_eeglab_extension_mismatch(mock_validate, tmp_path):
    """Test EEGLAB extension mismatch falls back to scipy loader."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds003645",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError(
            "Invalid value for the 'EEGLAB file extension' parameter. "
            "Allowed values are '.set' and '.fdt', but got '' instead."
        ),
    ):
        with patch(
            "eegdash.dataset.base._load_set_via_scipy",
            return_value=mock_raw,
        ) as mock_scipy:
            result = ds._load_raw()
            mock_scipy.assert_called_once()
            assert result is mock_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_channel_type_conflict_recovery(mock_validate, tmp_path):
    """Test channel type conflict retries with on_ch_mismatch='ignore'."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds002712",
        "bids_relpath": "sub-01/meg/sub-01_task-rest_meg.fif",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError(
                "Cannot change channel type for channel MEG0113 in projector"
            )
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        result = ds._load_raw()
        assert result is mock_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_fif_validation_error_direct_fallback(mock_validate, tmp_path):
    """Test FIF validation errors fall back to direct reader."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds000247",
        "bids_relpath": "sub-01/meg/sub-01_task-rest_meg.fif",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("Illegal date: 14-10-1925"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            return_value=mock_raw,
        ) as mock_direct:
            result = ds._load_raw()
            mock_direct.assert_called_once()
            assert result is mock_raw


@pytest.mark.parametrize(
    "error_cls,error_msg,match",
    [
        (
            TypeError,
            "buffer is too small for requested array",
            "corrupted or truncated",
        ),
        (
            ValueError,
            "PosixPath('eeg/sub-01_task-learning_eeg.vhdr') is not in list. "
            "Did you mean one of ['func/sub-01_task-learning_bold.nii.gz']?",
            "BIDS path mismatch",
        ),
        (
            ValueError,
            "setting an array element with a sequence. "
            "The requested array has an inhomogeneous shape",
            "inhomogeneous",
        ),
    ],
    ids=["corrupted", "bids_mismatch", "inhomogeneous"],
)
@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_raises_unsupported(
    mock_validate, tmp_path, error_cls, error_msg, match
):
    """Test various unrecoverable errors raise UnsupportedDataError."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import UnsupportedDataError

    record = {
        "dataset": "ds_test",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with patch("mne_bids.read_raw_bids", side_effect=error_cls(error_msg)):
        with pytest.raises(UnsupportedDataError, match=match):
            ds._load_raw()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_missing_companion_raises_integrity(mock_validate, tmp_path):
    """Test missing companion files raise DataIntegrityError."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds004011",
        "bids_relpath": "sub-01/meg/sub-01_task-rest_meg.fif",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=OSError("Could not find any data, could not find file"),
    ):
        with pytest.raises(DataIntegrityError, match="Missing companion files"):
            ds._load_raw()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_snirf_unsupported_type_code(mock_validate, tmp_path):
    """Test unsupported SNIRF type code raises UnsupportedDataError."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import UnsupportedDataError

    record = {
        "dataset": "ds006545",
        "bids_relpath": "sub-01/fnirs/sub-01_task-rest_nirs.snirf",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError(
            "Expected type code 1 or 99999 but received type code 301"
        ),
    ):
        with patch(
            "eegdash.dataset.base._repair_snirf_bids_metadata", return_value=False
        ):
            with pytest.raises(UnsupportedDataError, match="Cannot load SNIRF file"):
                ds._load_raw()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_unknown_error_reraised(mock_validate, tmp_path):
    """Test that unknown errors are re-raised as-is."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds_unknown",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("Some completely unexpected error"),
    ):
        with pytest.raises(RuntimeError, match="Some completely unexpected error"):
            ds._load_raw()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_non_numeric_run_sanitized_in_init(mock_validate, tmp_path):
    """Test that non-numeric run entities are sanitized to None during init."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds003190",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": {"subject": "01", "task": "rest", "run": "5H"},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Non-numeric run "5H" should be sanitized to None during construction
    assert ds.bidspath.run is None


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_numeric_run_preserved_in_init(mock_validate, tmp_path):
    """Test that valid numeric run entities are preserved."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds_test",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities_mne": {"subject": "01", "task": "rest", "run": "01"},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Valid numeric run should be preserved
    assert ds.bidspath.run == "01"


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_ensure_raw_calls_new_repair_functions(mock_validate, tmp_path):
    """Test that _ensure_raw calls the new repair functions."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds_test",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "local",
            "base": str(tmp_path / "ds_test"),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        },
    }

    eeg_dir = tmp_path / "ds_test" / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_eeg.vhdr").touch()

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    ds._load_raw = MagicMock()

    with (
        patch("eegdash.dataset.base._repair_electrodes_tsv") as mock_elec,
        patch("eegdash.dataset.base._repair_tsv_decimal_separators") as mock_dec,
        patch("eegdash.dataset.base._repair_tsv_na_values") as mock_na,
        patch("eegdash.dataset.base._repair_vhdr_missing_markerfile") as mock_marker,
    ):
        ds._ensure_raw()

        mock_elec.assert_called_once()
        mock_dec.assert_called_once()
        mock_na.assert_called_once()
        mock_marker.assert_called_once()
