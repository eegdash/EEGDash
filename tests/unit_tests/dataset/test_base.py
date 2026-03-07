from unittest.mock import MagicMock, call, patch

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


def test_load_raw_retries_on_ctf_illegal_date(tmp_path):
    """When _read_raw_bids raises 'Illegal date', _load_raw patches CTF parser and retries."""
    from unittest.mock import MagicMock, patch

    from eegdash.dataset.base import EEGDashRaw

    (tmp_path / "sub-01" / "sub-01_meg.ds").mkdir(
        parents=True
    )  # CTF .ds is a directory
    record = {
        "dataset": "ds",
        "bids_relpath": "sub-01/sub-01_meg.ds",
        "storage": {
            "backend": "local",
            "base": str(tmp_path),
            "raw_key": "sub-01/sub-01_meg.ds",
        },
    }
    mock_raw = MagicMock()

    with patch("eegdash.dataset.base.validate_record", return_value=[]):
        ds = EEGDashRaw(record, str(tmp_path))
        with patch.object(ds, "_download_required_files"):
            with patch(
                "eegdash.dataset.base.mne_bids.read_raw_bids",
                side_effect=[
                    RuntimeError("Illegal date: 14-10-1925."),
                    mock_raw,
                ],
            ):
                result = ds._load_raw()

    assert result is mock_raw


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


# Tests for _has_non_numeric_run


@patch("eegdash.dataset.base.validate_record", return_value=None)
@pytest.mark.parametrize(
    "entities,expected",
    [
        ({"run": "5H"}, True),
        ({"run": "1"}, False),
        ({"run": None}, False),
        ({}, False),
        ({"run": "02"}, False),
        ({"run": "abc"}, True),
    ],
    ids=[
        "non_numeric_5H",
        "numeric_1",
        "no_run_none",
        "no_run_absent",
        "numeric_02",
        "non_numeric_abc",
    ],
)
def test_has_non_numeric_run(mock_validate, tmp_path, entities, expected):
    """Test _has_non_numeric_run detects non-numeric BIDS run values."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds1",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {"backend": "local", "base": str(tmp_path)},
        "entities": entities,
    }
    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    assert ds._has_non_numeric_run() is expected


# Tests for VHDR non-numeric run fallback in _load_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_falls_back_for_non_numeric_run_vhdr(mock_validate, tmp_path):
    """Test _load_raw falls back to direct BrainVision loading for non-numeric run."""
    from eegdash.dataset.base import EEGDashRaw

    dataset_id = "ds003190"
    bids_relpath = "sub-019/eeg/sub-019_ses-03_task-ctos_run-5H_eeg.vhdr"
    eeg_dir = tmp_path / dataset_id / "sub-019" / "eeg"
    eeg_dir.mkdir(parents=True)
    vhdr_path = eeg_dir / "sub-019_ses-03_task-ctos_run-5H_eeg.vhdr"
    vhdr_path.touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "019", "session": "03", "task": "ctos", "run": "5H"},
        "entities_mne": {
            "subject": "019",
            "session": "03",
            "task": "ctos",
            "run": None,
        },
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    mock_raw = MagicMock()

    with patch("mne_bids.read_raw_bids", side_effect=ValueError("run is not an index")):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(vhdr_path)
    assert result is mock_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_numeric_run_vhdr_does_not_fallback(mock_validate, tmp_path):
    """Test _load_raw does NOT fall back for VHDR files with numeric runs."""
    from eegdash.dataset.base import EEGDashRaw

    dataset_id = "ds_test"
    bids_relpath = "sub-01/eeg/sub-01_task-rest_run-01_eeg.vhdr"
    eeg_dir = tmp_path / dataset_id / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_run-01_eeg.vhdr").touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "01", "task": "rest", "run": "01"},
        "entities_mne": {"subject": "01", "task": "rest", "run": "01"},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with patch("mne_bids.read_raw_bids", side_effect=ValueError("Some other error")):
        with patch("eegdash.dataset.base._load_raw_direct") as mock_direct:
            with pytest.raises(ValueError, match="Some other error"):
                ds._load_raw()

    mock_direct.assert_not_called()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_vhdr_fallback_failure_chains_exceptions(mock_validate, tmp_path):
    """Test that when the VHDR direct fallback also fails, exceptions are chained."""
    from eegdash.dataset.base import EEGDashRaw

    dataset_id = "ds003190"
    bids_relpath = "sub-019/eeg/sub-019_ses-03_task-ctos_run-5H_eeg.vhdr"
    eeg_dir = tmp_path / dataset_id / "sub-019" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-019_ses-03_task-ctos_run-5H_eeg.vhdr").touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "019", "session": "03", "task": "ctos", "run": "5H"},
        "entities_mne": {
            "subject": "019",
            "session": "03",
            "task": "ctos",
            "run": None,
        },
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    original_error = ValueError("run is not an index")
    fallback_error = IOError("Corrupted VHDR file")

    with patch("mne_bids.read_raw_bids", side_effect=original_error):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            side_effect=fallback_error,
        ):
            with pytest.raises(IOError, match="Corrupted VHDR file") as exc_info:
                ds._load_raw()

    # Verify exception chaining preserves the original MNE-BIDS error
    assert exc_info.value.__cause__ is original_error


# ─── Tests for new error handlers ───────────────────────────────────────


def _make_local_eegdashraw(tmp_path, dataset_id, bids_relpath, ext=".edf", **extra):
    """Helper: create an EEGDashRaw with a local backend and minimal record."""
    from eegdash.dataset.base import EEGDashRaw

    parts = bids_relpath.split("/")
    data_dir = tmp_path / dataset_id / "/".join(parts[:-1])
    data_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / dataset_id / bids_relpath).touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": extra.get("entities", {"subject": "01", "task": "rest"}),
        "entities_mne": extra.get("entities_mne", {"subject": "01", "task": "rest"}),
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(tmp_path))


def _make_remote_eegdashraw(tmp_path, dataset_id, bids_relpath, **extra):
    """Helper: create an EEGDashRaw with an S3 backend and minimal record."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "s3",
            "base": "s3://bucket",
            "raw_key": bids_relpath,
        },
        "entities": extra.get("entities", {"subject": "01", "task": "rest"}),
        "entities_mne": extra.get("entities_mne", {"subject": "01", "task": "rest"}),
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(tmp_path))


# ── Error 2 / 3: Unrecoverable data corruption → DataIntegrityError ──


@pytest.mark.parametrize(
    "error_msg",
    [
        "Bad EDF+ header: invalid data record count",
        "invalid literal for int() with base 10: 'abc'",
        "Could not find any data in the raw file",
        "no valid samples found in data",
    ],
    ids=["bad_edf", "invalid_int", "no_data", "no_samples"],
)
def test_load_raw_unrecoverable_raises_data_integrity_error(tmp_path, error_msg):
    """Unrecoverable ValueError/OSError must become DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_corrupt", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    exc_cls = OSError if "Could not find" in error_msg else ValueError
    with patch("mne_bids.read_raw_bids", side_effect=exc_cls(error_msg)):
        with pytest.raises(DataIntegrityError, match="Cannot read data file"):
            ds._load_raw()


@pytest.mark.parametrize(
    "error_msg",
    [
        "buffer is too small for requested array",
        "iteration over a 0-d array",
        "cannot reshape array of size 0 into shape (64,1000)",
        "setting an array element with a sequence",
    ],
    ids=["buffer_small", "0d_array", "cannot_reshape", "array_element_sequence"],
)
def test_load_raw_type_error_raises_data_integrity_error(tmp_path, error_msg):
    """TypeError from corrupt MAT/EEGLAB files must become DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_corrupt", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    with patch("mne_bids.read_raw_bids", side_effect=TypeError(error_msg)):
        with pytest.raises(DataIntegrityError, match="Cannot read data file"):
            ds._load_raw()


# ── Error: NaN onset/sample in events.tsv → repair + retry / direct fallback ──


def test_load_raw_nan_events_repair_and_retry(tmp_path):
    """NaN in events.tsv sample column should trigger repair then retry."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004841", "sub-01/eeg/sub-01_task-drive_eeg.set"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("cannot convert float NaN to integer")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_events_tsv_nan_samples", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once()
    assert result is mock_raw
    assert call_count == 2


def test_load_raw_nan_events_repair_fails_propagates(tmp_path):
    """NaN events repair fails → error propagates (no direct reader fallback)."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004841", "sub-01/eeg/sub-01_task-drive_eeg.set"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("cannot convert float NaN to integer"),
    ):
        with patch(
            "eegdash.dataset.base._repair_events_tsv_nan_samples",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
                ds._load_raw()


# ── Error 1: Invalid scans.tsv timestamp → repair + retry ──


def test_load_raw_repairs_bad_scans_timestamp_and_retries(tmp_path):
    """'second must be' error should trigger timestamp repair then retry."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003775", "sub-01/ses-t1/eeg/sub-01_ses-t1_task-rest_eeg.edf"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("second must be in 0..59")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_scans_tsv_timestamps", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once()
    assert result is mock_raw
    assert call_count == 2


def test_load_raw_timestamp_repair_fails_still_raises(tmp_path):
    """If timestamp repair returns False, original ValueError re-raises."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003775", "sub-01/ses-t1/eeg/sub-01_ses-t1_task-rest_eeg.edf"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("second must be in 0..59"),
    ):
        with patch(
            "eegdash.dataset.base._repair_scans_tsv_timestamps",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="second must be"):
                ds._load_raw()


# ── Error 5: Unallowed BIDS entity → direct reader fallback ──


def test_load_raw_unallowed_entity_falls_back_to_direct(tmp_path):
    """'Unallowed' ValueError should fall back to _load_raw_direct."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005752", "sub-01/eeg/sub-01_task-my-task_eeg.edf"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("Unallowed -, _, or / in task"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    assert result is mock_raw


def test_load_raw_downloads_split_fif_continuations_until_load_succeeds(tmp_path):
    """Missing FIF continuations should be downloaded and retried on demand."""
    ds = _make_remote_eegdashraw(
        tmp_path, "ds_split", "sub-01/meg/sub-01_task-rest_meg.fif"
    )
    mock_raw = MagicMock()
    annex_next = (
        "/tmp/.git/annex/objects/ab/cd/MD5E-s1055433547--90f455363b16dec91282c634cfb710fd.fif/"
        "MD5E-s1055433547--90f455363b16dec91282c634cfb710fd-1.fif"
    )
    second_next = ds.filecache.with_name("sub-01_task-rest_meg-2.fif")
    mock_fs = object()

    def download_side_effect(s3_path, local_path, filesystem=None):
        if "MD5E-s1055433547" in s3_path:
            raise FileNotFoundError("not found")
        return local_path

    with patch.object(
        ds,
        "_read_raw_bids",
        side_effect=[
            ValueError(
                f"Split raw file detected but next file {annex_next} does not exist"
            ),
            ValueError(
                f"Split raw file detected but next file {second_next} does not exist"
            ),
            mock_raw,
        ],
    ) as mock_read:
        with patch(
            "eegdash.dataset.base.downloader.get_s3_filesystem", return_value=mock_fs
        ):
            with patch(
                "eegdash.dataset.base.downloader.download_s3_file",
                side_effect=download_side_effect,
            ) as mock_download:
                result = ds._load_raw()

    assert result is mock_raw
    assert mock_read.call_count == 3
    assert mock_download.call_args_list == [
        call(
            "s3://bucket/sub-01/meg/MD5E-s1055433547--90f455363b16dec91282c634cfb710fd-1.fif",
            ds.filecache.with_name(
                "MD5E-s1055433547--90f455363b16dec91282c634cfb710fd-1.fif"
            ),
            filesystem=mock_fs,
        ),
        call(
            "s3://bucket/sub-01/meg/sub-01_task-rest_meg-1.fif",
            ds.filecache.with_name("sub-01_task-rest_meg-1.fif"),
            filesystem=mock_fs,
        ),
        call(
            "s3://bucket/sub-01/meg/sub-01_task-rest_meg-2.fif",
            ds.filecache.with_name("sub-01_task-rest_meg-2.fif"),
            filesystem=mock_fs,
        ),
    ]


def test_load_raw_downloads_bids_split_continuation_and_preserves_split_entity(
    tmp_path,
):
    """Split entity records should preserve split in BIDSPath and fetch split-02."""
    ds = _make_remote_eegdashraw(
        tmp_path,
        "ds_split",
        "sub-01/meg/sub-01_task-rest_split-01_meg.fif",
        entities={"subject": "01", "task": "rest", "split": "01"},
        entities_mne={"subject": "01", "task": "rest", "split": "01"},
    )
    mock_raw = MagicMock()
    expected_next = ds.filecache.with_name("sub-01_task-rest_split-02_meg.fif")
    mock_fs = object()

    with patch.object(
        ds,
        "_read_raw_bids",
        side_effect=[
            ValueError(
                f"Split raw file detected but next file {expected_next} does not exist"
            ),
            mock_raw,
        ],
    ):
        with patch(
            "eegdash.dataset.base.downloader.get_s3_filesystem", return_value=mock_fs
        ):
            with patch(
                "eegdash.dataset.base.downloader.download_s3_file",
                return_value=expected_next,
            ) as mock_download:
                result = ds._load_raw()

    assert ds.bidspath.split == "01"
    assert result is mock_raw
    mock_download.assert_called_once_with(
        "s3://bucket/sub-01/meg/sub-01_task-rest_split-02_meg.fif",
        expected_next,
        filesystem=mock_fs,
    )


# ── Error 4: CTF .ds directory completeness ──


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_ensure_ctf_directory_complete_passes_when_files_exist(mock_validate, tmp_path):
    """CTF check should pass when .meg4 and .res4 exist."""
    from eegdash.dataset.base import EEGDashRaw

    dataset_id = "ds005279"
    bids_relpath = "sub-01/meg/sub-01_task-rest_meg.ds"
    ds_dir = tmp_path / dataset_id / "sub-01" / "meg" / "sub-01_task-rest_meg.ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "sub-01_task-rest_meg.meg4").touch()
    (ds_dir / "sub-01_task-rest_meg.res4").touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    # Should not raise — files exist
    ds._ensure_ctf_directory_complete()


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_ensure_ctf_directory_incomplete_raises(mock_validate, tmp_path):
    """CTF check should raise DataIntegrityError when files are missing."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    dataset_id = "ds005279"
    bids_relpath = "sub-01/meg/sub-01_task-rest_meg.ds"
    ds_dir = tmp_path / dataset_id / "sub-01" / "meg" / "sub-01_task-rest_meg.ds"
    ds_dir.mkdir(parents=True)
    # Only create .res4, missing .meg4
    (ds_dir / "sub-01_task-rest_meg.res4").touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with pytest.raises(DataIntegrityError, match="CTF .ds directory incomplete"):
        ds._ensure_ctf_directory_complete()


# ── _repair_scans_tsv_timestamps standalone tests ──


def test_repair_scans_tsv_timestamps_fixes_bad_seconds(tmp_path):
    """Timestamps with seconds >= 60 should be replaced with n/a."""
    from eegdash.dataset.io import _repair_scans_tsv_timestamps

    scans = tmp_path / "sub-01_scans.tsv"
    scans.write_text(
        "filename\tacq_time\n"
        "eeg/file1.edf\t2020-01-01T12:30:60\n"
        "eeg/file2.edf\t2020-01-01T12:30:45\n"
    )

    assert _repair_scans_tsv_timestamps(tmp_path) is True

    lines = scans.read_text().strip().split("\n")
    assert lines[1].split("\t")[1] == "n/a"
    assert lines[2].split("\t")[1] == "2020-01-01T12:30:45"


def test_repair_scans_tsv_timestamps_noop_when_valid(tmp_path):
    """No changes when all timestamps are valid."""
    from eegdash.dataset.io import _repair_scans_tsv_timestamps

    scans = tmp_path / "sub-01_scans.tsv"
    scans.write_text("filename\tacq_time\neeg/file1.edf\t2020-01-01T12:30:45\n")

    assert _repair_scans_tsv_timestamps(tmp_path) is False


def test_repair_scans_tsv_timestamps_handles_na(tmp_path):
    """Existing n/a values should not be modified."""
    from eegdash.dataset.io import _repair_scans_tsv_timestamps

    scans = tmp_path / "sub-01_scans.tsv"
    scans.write_text("filename\tacq_time\neeg/file1.edf\tn/a\n")

    assert _repair_scans_tsv_timestamps(tmp_path) is False


# ── _load_raw_direct standalone tests ──


def test_load_raw_direct_unsupported_extension():
    """Unsupported extension should raise ValueError."""
    from pathlib import Path

    from eegdash.dataset.io import _load_raw_direct

    with pytest.raises(ValueError, match="No direct reader"):
        _load_raw_direct(Path("/tmp/test.xyz"))


def test_load_raw_direct_calls_correct_reader(tmp_path):
    """_load_raw_direct should map extensions to correct MNE readers."""
    from eegdash.dataset.io import _load_raw_direct

    edf_path = tmp_path / "test.edf"
    edf_path.touch()

    mock_raw = MagicMock()
    with patch("mne.io.read_raw_edf", return_value=mock_raw) as mock_reader:
        result = _load_raw_direct(edf_path)

    mock_reader.assert_called_once_with(str(edf_path), preload=False, verbose="ERROR")
    assert result is mock_raw
