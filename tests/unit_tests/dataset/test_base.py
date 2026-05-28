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
    recording._skipped = False
    recording.record = {"ntimes": 10.5, "sampling_frequency": 500}

    # Access length - should use len(self._raw) since _raw exists
    length = len(recording)
    assert length == 5000


def test_len_without_raw_with_ntimes_sfreq():
    """Test length calculation from ntimes when raw not loaded."""
    from eegdash.dataset.base import EEGDashRaw

    recording = EEGDashRaw.__new__(EEGDashRaw)
    recording._raw = None
    recording._skipped = False
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


def test_nemar_download_required_files_uses_annex_object_uri(tmp_path):
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.schemas import create_record

    annex_key = "MD5E-s117384448--3f1620cf0484cce641d500a406630b89.edf"
    bids_relpath = "sub-70495563/emg/sub-70495563_task-typing_emg.edf"
    record = create_record(
        dataset="nm000104",
        storage_base="s3://nemar/nm000104",
        storage_backend="nemar",
        bids_relpath=bids_relpath,
        subject="70495563",
        task="typing",
        datatype="emg",
        suffix="emg",
        sampling_frequency=1000.0,
        ntimes=1000,
    )
    record["storage"]["annex_keys"] = {bids_relpath: annex_key}

    ds = EEGDashRaw(record=record, cache_dir=tmp_path)

    with (
        patch("eegdash.dataset.base.downloader.get_s3_filesystem"),
        patch("eegdash.dataset.base.downloader.download_files") as mock_deps,
        patch("eegdash.dataset.base.downloader.download_s3_file") as mock_download,
        patch.object(ds, "_fetch_nemar_root_metadata"),
    ):
        ds._download_required_files()

    mock_deps.assert_called_once()
    mock_download.assert_called_once_with(
        f"s3://nemar/nm000104/objects/{annex_key}",
        tmp_path / "nm000104" / bids_relpath,
        filesystem=mock_deps.call_args.kwargs["filesystem"],
    )


def test_nemar_download_required_files_falls_back_to_data_portal(tmp_path):
    """After an S3 403 on the original URI, eegdash must delegate to
    ``nemar-py`` (NEMARClient + download_one) for the recovery path.
    """
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.schemas import create_record

    annex_key = "MD5E-s117384448--3f1620cf0484cce641d500a406630b89.edf"
    bids_relpath = "sub-70495563/emg/sub-70495563_task-typing_emg.edf"
    record = create_record(
        dataset="nm000104",
        storage_base="s3://nemar/nm000104",
        storage_backend="nemar",
        bids_relpath=bids_relpath,
        subject="70495563",
        task="typing",
        datatype="emg",
        suffix="emg",
        sampling_frequency=1000.0,
        ntimes=1000,
    )
    record["storage"]["annex_keys"] = {bids_relpath: annex_key}

    ds = EEGDashRaw(record=record, cache_dir=tmp_path)

    fake_file = MagicMock(path=bids_relpath, size=117384448)
    fake_manifest = MagicMock()
    fake_manifest.__contains__ = lambda self, key: key == bids_relpath
    fake_manifest.file = MagicMock(return_value=fake_file)

    with (
        patch("eegdash.dataset.base.downloader.get_s3_filesystem"),
        patch("eegdash.dataset.base.downloader.download_files"),
        patch(
            "eegdash.dataset.base.downloader.download_s3_file",
            side_effect=PermissionError("Forbidden"),
        ),
        patch(
            "eegdash.dataset.base._fetch_nemar_manifest",
            return_value=fake_manifest,
        ) as mock_fetch,
        patch("nemar.download_one") as mock_download_one,
        patch.object(ds, "_fetch_nemar_root_metadata"),
    ):
        ds._download_required_files()

    mock_fetch.assert_called_once_with("nm000104")
    fake_manifest.file.assert_called_once_with(bids_relpath)
    mock_download_one.assert_called_once_with(
        fake_file, tmp_path / "nm000104" / bids_relpath
    )


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
        (eeg_dir / "sub-01_task-rest_eeg.vhdr").write_text(
            "[Common Infos]\nDataFile=sub-01_task-rest_eeg.eeg\n"
        )
    if create_vmrk:
        (eeg_dir / "sub-01_task-rest_eeg.vmrk").touch()

    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))
    mock_raw = MagicMock()
    mock_raw.n_times = 100
    dataset._load_raw = MagicMock(return_value=mock_raw)
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

    # Use .edf (not .set) so buffer_small doesn't trigger the latin-1 retry
    ds = _make_local_eegdashraw(
        tmp_path, "ds_corrupt", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    with patch("mne_bids.read_raw_bids", side_effect=TypeError(error_msg)):
        with pytest.raises(DataIntegrityError, match="Cannot read data file"):
            ds._load_raw()


# ── EEGLAB .set char-encoding retry with uint16_codec='latin-1' ──


def test_load_raw_set_buffer_error_retries_with_latin1(tmp_path):
    """EEGLAB .set 'buffer is too small' retries with uint16_codec='latin-1'."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004166", "sub-01/eeg/sub-01_ses-01_task-WM_run-1_eeg.set"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TypeError("buffer is too small for requested array")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        result = ds._load_raw()

    assert result is mock_raw
    assert call_count == 2


def test_load_raw_set_buffer_error_retry_fails_propagates(tmp_path):
    """EEGLAB .set latin-1 retry also fails -> error propagates."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004166", "sub-01/eeg/sub-01_ses-01_task-WM_run-1_eeg.set"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=TypeError("buffer is too small for requested array"),
    ):
        with pytest.raises(TypeError, match="buffer is too small"):
            ds._load_raw()


@pytest.mark.parametrize(
    "exc_type, error_msg",
    [
        (AttributeError, "'numpy.ndarray' object has no attribute 'get'"),
        (AttributeError, "'Bunch' object has no attribute 'chanlocs'"),
        (TypeError, "Expecting matrix here"),
        (TypeError, "argument of type 'NoneType' is not iterable"),
        (
            ValueError,
            "Invalid value for the 'EEGLAB file extension' parameter",
        ),
    ],
    ids=[
        "chaninfo_ndarray",
        "missing_chanlocs",
        "malformed_mat",
        "nonetype_iterable",
        "extension_mismatch",
    ],
)
def test_load_raw_eeglab_errors_try_fallback_then_raise(tmp_path, exc_type, error_msg):
    """EEGLAB reader errors try fallback; if fallback fails, raise DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_corrupt", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    with patch("mne_bids.read_raw_bids", side_effect=exc_type(error_msg)):
        with patch(
            "eegdash.dataset.base._load_raw_eeglab_fallback",
            side_effect=ValueError("fallback failed"),
        ):
            with pytest.raises(DataIntegrityError, match="Cannot read data file"):
                ds._load_raw()


def test_load_raw_eeglab_fallback_success(tmp_path):
    """When MNE reader fails but EEGLAB fallback succeeds, return fallback result."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds_ok", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=TypeError("argument of type 'NoneType' is not iterable"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_eeglab_fallback",
            return_value=mock_raw,
        ) as mock_fallback:
            result = ds._load_raw()

    assert result is mock_raw
    mock_fallback.assert_called_once()


def test_load_raw_eeglab_fallback_failure_raises_data_integrity(tmp_path):
    """When both MNE reader and EEGLAB fallback fail, raise DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_bad", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=AttributeError("'Bunch' object has no attribute 'chanlocs'"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_eeglab_fallback",
            side_effect=ValueError("Cannot parse .set file"),
        ):
            with pytest.raises(DataIntegrityError, match="Cannot read data file"):
                ds._load_raw()


# ── Error: NaN onset/sample in events.tsv → repair + retry / direct fallback ──


def test_load_raw_na_whitespace_repair_and_retry(tmp_path):
    """Whitespace-padded n/a in TSV triggers repair then retry."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004860", "sub-01/eeg/sub-01_task-rest_eeg.bdf"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("could not convert string to float: 'n/a      '")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_tsv_na_whitespace", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once()
    assert result is mock_raw
    assert call_count == 2


def test_load_raw_na_whitespace_repair_no_change_falls_through(tmp_path):
    """n/a whitespace repair returns False → error propagates."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds004860", "sub-01/eeg/sub-01_task-rest_eeg.bdf"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("could not convert string to float: 'n/a      '"),
    ):
        with patch(
            "eegdash.dataset.base._repair_tsv_na_whitespace",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="could not convert string to float"):
                ds._load_raw()


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
    assert lines[2].split("\t")[1] == "2020-01-01T12:30:45.000000"


def test_repair_scans_tsv_timestamps_noop_when_valid(tmp_path):
    """No changes when all timestamps are valid."""
    from eegdash.dataset.io import _repair_scans_tsv_timestamps

    scans = tmp_path / "sub-01_scans.tsv"
    scans.write_text("filename\tacq_time\neeg/file1.edf\t2020-01-01T12:30:45.000000\n")

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


def test_load_raw_snirf_array_error_raises_data_integrity_error(tmp_path):
    """ds005929: ValueError('setting an array element with a sequence') from SNIRF
    must be caught by _UNRECOVERABLE_PATTERNS as DataIntegrityError, NOT fall
    through to the generic SNIRF `except Exception` handler.
    """
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path,
        "ds005929",
        "sub-01/nirs/sub-01_task-rest_nirs.snirf",
        ext=".snirf",
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("setting an array element with a sequence"),
    ):
        with pytest.raises(DataIntegrityError, match="Cannot read data file"):
            ds._load_raw()


# ── Projector channels not found → direct reader + del_proj ──


def test_load_raw_projector_channels_not_found_retries(tmp_path):
    """ValueError('projector channels not found') falls back to direct reader and del_proj."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds006545", "sub-01/meg/sub-01_task-rest_meg.fif", ext=".fif"
    )

    mock_raw = MagicMock()
    mock_raw.info = {"projs": [MagicMock()]}

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError("projector channels not found in data"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    mock_raw.del_proj.assert_called_once()
    assert result is mock_raw


def test_load_raw_projector_type_conflict_retries(tmp_path):
    """RuntimeError('in projector') falls back to direct reader and del_proj."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds002712", "sub-01/meg/sub-01_task-rest_meg.fif", ext=".fif"
    )

    mock_raw = MagicMock()
    mock_raw.info = {"projs": [MagicMock()]}

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError(
            'Cannot change channel type for channel MEG0113 in projector "PCA-v1"'
        ),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    mock_raw.del_proj.assert_called_once()
    assert result is mock_raw


# ── AssertionError → hide events.tsv and retry ──


def test_load_raw_assertion_error_retries_without_events(tmp_path):
    """AssertionError hides events.tsv, retries, and restores the file."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003690", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    # Create an events.tsv in the data directory
    events_tsv = ds.filecache.parent / "sub-01_task-rest_events.tsv"
    events_tsv.write_text("onset\tduration\n1.0\t0.5\n")

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=[AssertionError("duration mismatch"), mock_raw],
    ):
        result = ds._load_raw()

    assert result is mock_raw
    # events.tsv should be restored
    assert events_tsv.exists()
    assert not events_tsv.with_suffix(".tsv._hidden").exists()


def test_load_raw_assertion_error_no_events_raises(tmp_path):
    """AssertionError with no events.tsv re-raises."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003690", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )
    # No events.tsv created

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=AssertionError("duration mismatch"),
    ):
        with pytest.raises(AssertionError, match="duration mismatch"):
            ds._load_raw()


def test_load_raw_assertion_error_restores_events_on_failure(tmp_path):
    """Both attempts fail → events.tsv is still restored."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003690", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    events_tsv = ds.filecache.parent / "sub-01_task-rest_events.tsv"
    events_tsv.write_text("onset\tduration\n1.0\t0.5\n")

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=AssertionError("duration mismatch"),
    ):
        with pytest.raises(AssertionError):
            ds._load_raw()

    # events.tsv must be restored even when retry fails
    assert events_tsv.exists()
    assert not events_tsv.with_suffix(".tsv._hidden").exists()


# ── RuntimeError unrecoverable patterns → DataIntegrityError ──


@pytest.mark.parametrize(
    "error_msg",
    [
        "incorrect number of samples in the data",
        "MNE only supports reading continuous wave amplitude and processed "
        "haemoglobin SNIRF files. Expected type code 1 or 99999 but received "
        "type code 301",
    ],
    ids=["fif-samples", "snirf-td-nirs"],
)
def test_load_raw_runtime_error_unrecoverable_raises_data_integrity(
    tmp_path, error_msg
):
    """RuntimeError with unrecoverable pattern becomes DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_corrupt", "sub-01/meg/sub-01_task-rest_meg.fif", ext=".fif"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError(error_msg),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            side_effect=Exception("direct reader failed too"),
        ):
            with pytest.raises(DataIntegrityError, match="Cannot read data file"):
                ds._load_raw()


# ── CTF mandatory HPI fix → retry with patched kind dict ──


def test_load_raw_mandatory_hpi_retries_with_fix(tmp_path):
    """RuntimeError('mandatory HPI') retries with extended _kind_dict."""
    ds = _make_local_eegdashraw(
        tmp_path,
        "ds006502",
        "sub-01/ses-meg1/meg/sub-01_ses-meg1_task-rest_run-1_meg.ds",
        ext=".ds",
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=[
            RuntimeError(
                "Some of the mandatory HPI device-coordinate info was not there."
            ),
            mock_raw,
        ],
    ):
        result = ds._load_raw()

    assert result is mock_raw


# ── KeyError FIFFV_COIL_NONE → direct reader fallback ──


def test_load_raw_fiffv_coil_none_keyerror_falls_back(tmp_path):
    """KeyError('FIFFV_COIL_NONE') from montage falls back to direct reader."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds003690", "sub-01/eeg/sub-01_task-rest_eeg.set", ext=".set"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=KeyError("0 (FIFFV_COIL_NONE)"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    assert result is mock_raw


def test_load_raw_direct_fif_passes_on_split_missing(tmp_path):
    """_load_raw_direct should pass on_split_missing='warn' for .fif files."""
    from eegdash.dataset.io import _load_raw_direct

    fif_path = tmp_path / "test.fif"
    fif_path.touch()

    mock_raw = MagicMock()
    with patch("mne.io.read_raw_fif", return_value=mock_raw) as mock_reader:
        result = _load_raw_direct(fif_path)

    mock_reader.assert_called_once_with(
        str(fif_path), preload=False, verbose="ERROR", on_split_missing="warn"
    )
    assert result is mock_raw


# ── _download_companion_files tests ──


def _make_s3_raw(tmp_path, ext, record_overrides=None):
    """Helper to create an EEGDashRaw with S3 backend for companion tests."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds_test",
        "bids_relpath": f"sub-01/eeg/sub-01_task-rest_eeg{ext}",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org/ds_test",
            "raw_key": f"sub-01/eeg/sub-01_task-rest_eeg{ext}",
        },
        "subject": "01",
        "session": None,
        "run": None,
        "task": "rest",
        "modality": "eeg",
        "suffix": "eeg",
        "datatype": "eeg",
        "extension": ext,
        "entities_mne": {"subject": "01", "task": "rest"},
        "entities": {"subject": "01", "task": "rest"},
    }
    if record_overrides:
        record.update(record_overrides)
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        return EEGDashRaw(record, cache_dir=str(tmp_path))


def test_download_companion_files_set_fdt(tmp_path):
    """Companion .fdt should be downloaded for .set files."""
    ds = _make_s3_raw(tmp_path, ".set")
    mock_fs = MagicMock()

    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_companion_files(mock_fs)

    expected_uri = "s3://openneuro.org/ds_test/sub-01/eeg/sub-01_task-rest_eeg.fdt"
    expected_local = ds.filecache.with_suffix(".fdt")
    mock_dl.assert_called_once_with(expected_uri, expected_local, filesystem=mock_fs)


def test_download_companion_files_vhdr_eeg_vmrk_dat(tmp_path):
    """All .eeg, .vmrk, and .dat companions should be downloaded for .vhdr files."""
    ds = _make_s3_raw(tmp_path, ".vhdr")
    mock_fs = MagicMock()

    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_companion_files(mock_fs)

    assert mock_dl.call_count == 3
    uris = [call.args[0] for call in mock_dl.call_args_list]
    assert "s3://openneuro.org/ds_test/sub-01/eeg/sub-01_task-rest_eeg.eeg" in uris
    assert "s3://openneuro.org/ds_test/sub-01/eeg/sub-01_task-rest_eeg.vmrk" in uris
    assert "s3://openneuro.org/ds_test/sub-01/eeg/sub-01_task-rest_eeg.dat" in uris


def test_download_companion_files_skips_existing(tmp_path):
    """Already-downloaded companion files should be skipped."""
    ds = _make_s3_raw(tmp_path, ".set")
    # Create the .fdt locally so it's already present
    fdt_path = ds.filecache.with_suffix(".fdt")
    fdt_path.parent.mkdir(parents=True, exist_ok=True)
    fdt_path.touch()

    mock_fs = MagicMock()

    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_companion_files(mock_fs)

    mock_dl.assert_not_called()


def test_download_companion_files_tries_embedded_fdt_on_missing(tmp_path):
    """When BIDS-named .fdt is missing, _download_embedded_fdt is tried."""
    ds = _make_s3_raw(tmp_path, ".set")
    mock_fs = MagicMock()

    with (
        patch(
            "eegdash.dataset.base.downloader.download_s3_file",
            side_effect=FileNotFoundError,
        ),
        patch.object(ds, "_download_embedded_fdt") as mock_embedded,
    ):
        ds._download_companion_files(mock_fs)

    mock_embedded.assert_called_once_with(mock_fs)


def test_download_companion_files_noop_for_unrelated_format(tmp_path):
    """Formats without companions (e.g. .edf) should be a no-op."""
    ds = _make_s3_raw(tmp_path, ".edf")
    mock_fs = MagicMock()

    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_companion_files(mock_fs)

    mock_dl.assert_not_called()


def test_download_embedded_fdt_parses_set_header(tmp_path):
    """_download_embedded_fdt parses the .set header to find .fdt name."""
    ds = _make_s3_raw(tmp_path, ".set")
    # Create a minimal fake .set file with an EEG.datfile field
    ds.filecache.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np

    try:
        import scipy.io

        eeg_struct = {"datfile": np.array(["custom_data.fdt"])}
        scipy.io.savemat(
            str(ds.filecache),
            {"EEG": eeg_struct},
        )
    except ImportError:
        pytest.skip("scipy not available")

    mock_fs = MagicMock()
    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_embedded_fdt(mock_fs)

    expected_uri = "s3://openneuro.org/ds_test/sub-01/eeg/custom_data.fdt"
    expected_local = ds.filecache.parent / "custom_data.fdt"
    mock_dl.assert_called_once_with(expected_uri, expected_local, filesystem=mock_fs)


def test_download_embedded_fdt_skips_when_exists(tmp_path):
    """_download_embedded_fdt does not re-download if file already exists."""
    ds = _make_s3_raw(tmp_path, ".set")
    ds.filecache.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np

    try:
        import scipy.io

        eeg_struct = {"datfile": np.array(["custom_data.fdt"])}
        scipy.io.savemat(str(ds.filecache), {"EEG": eeg_struct})
    except ImportError:
        pytest.skip("scipy not available")

    # Pre-create the embedded .fdt file
    (ds.filecache.parent / "custom_data.fdt").touch()

    mock_fs = MagicMock()
    with patch("eegdash.dataset.base.downloader.download_s3_file") as mock_dl:
        ds._download_embedded_fdt(mock_fs)

    mock_dl.assert_not_called()


def test_load_raw_falls_back_on_file_not_found(tmp_path):
    """FileNotFoundError from read_raw_bids falls back to _load_raw_direct."""
    ds = _make_s3_raw(tmp_path, ".bdf")
    mock_raw = MagicMock()

    with (
        patch.object(ds, "_read_raw_bids", side_effect=FileNotFoundError("missing")),
        patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct,
    ):
        result = ds._load_raw()

    assert result is mock_raw
    mock_direct.assert_called_once_with(ds.filecache)


def test_load_raw_bad_task_metadata_raises_data_integrity(tmp_path):
    """RuntimeError about missing 'task' raises DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_s3_raw(tmp_path, ".json")

    with (
        patch.object(
            ds,
            "_read_raw_bids",
            side_effect=RuntimeError(
                '"bids_path" must contain `root`, `subject`, and `task` '
                "attributes but it's missing `task`."
            ),
        ),
        pytest.raises(DataIntegrityError, match="Bad record metadata"),
    ):
        ds._load_raw()


def test_load_raw_retries_on_coordsystem_should_exist(tmp_path):
    """RuntimeError 'coordsystem.json should exist' triggers coordsystem retry."""
    ds = _make_s3_raw(tmp_path, ".edf")
    mock_raw = MagicMock()

    with (
        patch.object(
            ds,
            "_read_raw_bids",
            side_effect=RuntimeError("coordsystem.json should exist for iEEG"),
        ),
        patch.object(
            ds,
            "_retry_with_generated_coordsystem",
            return_value=mock_raw,
        ) as mock_retry,
    ):
        result = ds._load_raw()

    mock_retry.assert_called_once()
    assert result is mock_raw


def test_load_raw_retries_on_coordsystem_key_error(tmp_path):
    """KeyError('iEEGCoordinateSystem') triggers coordsystem retry."""
    ds = _make_s3_raw(tmp_path, ".edf")
    mock_raw = MagicMock()

    with (
        patch.object(
            ds,
            "_read_raw_bids",
            side_effect=KeyError("iEEGCoordinateSystem"),
        ),
        patch.object(
            ds,
            "_retry_with_generated_coordsystem",
            return_value=mock_raw,
        ) as mock_retry,
    ):
        result = ds._load_raw()

    mock_retry.assert_called_once()
    assert result is mock_raw


@patch("eegdash.dataset.base.validate_record", return_value=None)
def test_load_raw_falls_back_for_non_numeric_run_edf(mock_validate, tmp_path):
    """EDF file with non-numeric run falls back to direct MNE reader."""
    from eegdash.dataset.base import EEGDashRaw

    dataset_id = "ds_test"
    bids_relpath = "sub-01/eeg/sub-01_task-rest_run-5H_eeg.edf"
    eeg_dir = tmp_path / dataset_id / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    edf_path = eeg_dir / "sub-01_task-rest_run-5H_eeg.edf"
    edf_path.touch()

    record = {
        "dataset": dataset_id,
        "bids_relpath": bids_relpath,
        "storage": {
            "backend": "local",
            "base": str(tmp_path / dataset_id),
            "raw_key": bids_relpath,
        },
        "entities": {"subject": "01", "task": "rest", "run": "5H"},
        "entities_mne": {"subject": "01", "task": "rest", "run": None},
    }

    ds = EEGDashRaw(record, cache_dir=str(tmp_path))
    mock_raw = MagicMock()

    with patch("mne_bids.read_raw_bids", side_effect=ValueError("run is not an index")):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            ds._load_raw()

    mock_direct.assert_called_once_with(edf_path)


# ── EEGLAB epoch files (trials > 1) → read_epochs_eeglab handler ──


def test_load_raw_eeglab_epochs_trial_error_uses_epoch_loader(tmp_path):
    """TypeError('number of trials is') triggers _load_raw_from_eeglab_epochs."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds_epoch", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=TypeError("number of trials is 40, not 1"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_from_eeglab_epochs",
            return_value=mock_raw,
        ) as mock_epoch:
            result = ds._load_raw()

    mock_epoch.assert_called_once_with(ds.filecache)
    assert result is mock_raw


def test_load_raw_eeglab_epochs_fallback_failure_raises_data_integrity(tmp_path):
    """When epoch loader also fails, DataIntegrityError is raised."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_epoch_bad", "sub-01/eeg/sub-01_task-rest_eeg.set"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=TypeError("number of trials is 10, not 1"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_from_eeglab_epochs",
            side_effect=ValueError("corrupted epoch data"),
        ):
            with pytest.raises(DataIntegrityError, match="Cannot read epoched EEGLAB"):
                ds._load_raw()


# ── CTF trial-size mismatch → direct reader fallback ──


def test_load_raw_ctf_trial_size_mismatch_falls_back(tmp_path):
    """RuntimeError about trial size mismatch retries with tolerant sample info."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds_ctf", "sub-01/meg/sub-01_task-rest_meg.ds", ext=".ds"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("Data size is not an even multiple of the trial size"),
    ):
        with patch("mne.io.read_raw_ctf", return_value=mock_raw) as mock_ctf:
            result = ds._load_raw()

    mock_ctf.assert_called_once()
    assert result is mock_raw


def test_load_raw_ctf_trial_size_mismatch_fallback_fails(tmp_path):
    """CTF trial size mismatch + truncated fix failure → DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(
        tmp_path, "ds_ctf_bad", "sub-01/meg/sub-01_task-rest_meg.ds", ext=".ds"
    )

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("Data size is not an even multiple of the trial size"),
    ):
        with patch(
            "mne.io.read_raw_ctf",
            side_effect=ValueError("cannot read CTF"),
        ):
            with pytest.raises(DataIntegrityError, match="CTF trial size mismatch"):
                ds._load_raw()


# ── configparser.NoOptionError (missing MarkerFile) → repair + retry ──


def test_load_raw_no_option_error_markerfile_repairs_and_retries(tmp_path):
    """configparser.NoOptionError for markerfile triggers VHDR repair then retry."""
    import configparser

    ds = _make_local_eegdashraw(
        tmp_path, "ds003944", "sub-01/eeg/sub-01_task-rest_eeg.vhdr", ext=".vhdr"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise configparser.NoOptionError("markerfile", "Common Infos")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_vhdr_missing_markerfile", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once_with(ds.filecache)
    assert result is mock_raw
    assert call_count == 2


def test_load_raw_no_option_error_repair_fails_reraises(tmp_path):
    """configparser.NoOptionError when repair returns False → re-raises."""
    import configparser

    ds = _make_local_eegdashraw(
        tmp_path, "ds003944", "sub-01/eeg/sub-01_task-rest_eeg.vhdr", ext=".vhdr"
    )

    error = configparser.NoOptionError("markerfile", "Common Infos")
    with patch("mne_bids.read_raw_bids", side_effect=error):
        with patch(
            "eegdash.dataset.base._repair_vhdr_missing_markerfile", return_value=False
        ):
            with pytest.raises(configparser.NoOptionError):
                ds._load_raw()


def test_load_raw_no_option_error_non_markerfile_reraises(tmp_path):
    """configparser.NoOptionError for non-markerfile option → re-raises."""
    import configparser

    ds = _make_local_eegdashraw(
        tmp_path, "ds003944", "sub-01/eeg/sub-01_task-rest_eeg.vhdr", ext=".vhdr"
    )

    error = configparser.NoOptionError("datafile", "Common Infos")
    with patch("mne_bids.read_raw_bids", side_effect=error):
        with pytest.raises(configparser.NoOptionError):
            ds._load_raw()


# ── UnicodeDecodeError → repair TSV encoding + retry ──


def test_load_raw_unicode_decode_error_repairs_encoding(tmp_path):
    """UnicodeDecodeError triggers TSV encoding repair then retry."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005692", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise UnicodeDecodeError("utf-8", b"\xb5", 0, 1, "invalid start byte")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_tsv_encoding", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once_with(ds.filecache.parent)
    assert result is mock_raw
    assert call_count == 2


def test_load_raw_unicode_decode_error_repair_fails_reraises(tmp_path):
    """UnicodeDecodeError when encoding repair returns False → re-raises."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005692", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    error = UnicodeDecodeError("utf-8", b"\xb5", 0, 1, "invalid start byte")
    with patch("mne_bids.read_raw_bids", side_effect=error):
        with patch("eegdash.dataset.base._repair_tsv_encoding", return_value=False):
            with pytest.raises(UnicodeDecodeError):
                ds._load_raw()


# ── scans.tsv path mismatch → direct reader fallback ──


def test_load_raw_scans_tsv_mismatch_falls_back_to_direct(tmp_path):
    """'is not in list. Did you mean' falls back to direct reader."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005795", "sub-01/eeg/sub-01_task-rest_eeg.vhdr", ext=".vhdr"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=ValueError(
            "eeg/sub-01_task-rest_eeg.vhdr is not in list. "
            "Did you mean func/sub-01_task-rest_bold.nii.gz?"
        ),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct", return_value=mock_raw
        ) as mock_direct:
            result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    assert result is mock_raw


def test_load_raw_scans_tsv_mismatch_fallback_fails_chains(tmp_path):
    """scans.tsv mismatch direct reader failure chains exceptions."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005795", "sub-01/eeg/sub-01_task-rest_eeg.vhdr", ext=".vhdr"
    )

    original = ValueError(
        "eeg/sub-01_task-rest_eeg.vhdr is not in list. "
        "Did you mean func/sub-01_task-rest_bold.nii.gz?"
    )
    fallback = IOError("cannot read file")

    with patch("mne_bids.read_raw_bids", side_effect=original):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            side_effect=fallback,
        ):
            with pytest.raises(IOError, match="cannot read file") as exc_info:
                ds._load_raw()

    assert exc_info.value.__cause__ is original


def test_load_raw_is_not_in_list_without_did_you_mean_tries_participants(tmp_path):
    """'is not in list' without 'Did you mean' triggers participants.tsv repair."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds005795", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    mock_raw = MagicMock()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("sub-01 is not in list")
        return mock_raw

    with patch("mne_bids.read_raw_bids", side_effect=side_effect):
        with patch(
            "eegdash.dataset.base._repair_participants_tsv_ids", return_value=True
        ) as mock_repair:
            result = ds._load_raw()

    mock_repair.assert_called_once()
    assert result is mock_raw


# ── on_error parameter tests ──


def test_raw_property_on_error_warn_skips_integrity_error(tmp_path):
    """on_error='warn' catches DataIntegrityError, sets .raw to None."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "storage": {
            "backend": "local",
            "base": str(tmp_path / "ds"),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    (tmp_path / "ds" / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "ds" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path), on_error="warn")

    err = DataIntegrityError(message="corrupt", record=record, issues=["bad"])
    with patch.object(ds, "_ensure_raw", side_effect=err):
        result = ds.raw

    assert result is None
    assert ds._skipped is True
    assert ds._integrity_error is err


def test_raw_property_on_error_raise_still_raises(tmp_path):
    """on_error='raise' (default) propagates DataIntegrityError."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "storage": {
            "backend": "local",
            "base": str(tmp_path / "ds"),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    (tmp_path / "ds" / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "ds" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    err = DataIntegrityError(message="corrupt", record=record, issues=["bad"])
    with patch.object(ds, "_ensure_raw", side_effect=err):
        with pytest.raises(DataIntegrityError):
            _ = ds.raw


def test_raw_property_on_error_skip_silent(tmp_path):
    """on_error='skip' silently sets .raw to None, no warning logged."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "storage": {
            "backend": "local",
            "base": str(tmp_path / "ds"),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    (tmp_path / "ds" / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "ds" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path), on_error="skip")

    err = DataIntegrityError(message="corrupt", record=record, issues=["bad"])
    with patch.object(ds, "_ensure_raw", side_effect=err):
        with patch.object(err, "log_warning") as mock_warn:
            result = ds.raw

    assert result is None
    assert ds._skipped is True
    mock_warn.assert_not_called()


def test_len_returns_zero_when_skipped():
    """Skipped record returns 0 from __len__."""
    from eegdash.dataset.base import EEGDashRaw

    ds = EEGDashRaw.__new__(EEGDashRaw)
    ds._raw = None
    ds._skipped = True
    ds.record = {"ntimes": 1000}

    assert len(ds) == 0


def test_raw_property_skipped_does_not_retry(tmp_path):
    """Accessing .raw twice on a skipped record doesn't re-attempt loading."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        "storage": {
            "backend": "local",
            "base": str(tmp_path / "ds"),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.edf",
        },
        "entities": {"subject": "01", "task": "rest"},
        "entities_mne": {"subject": "01", "task": "rest"},
    }
    (tmp_path / "ds" / "sub-01" / "eeg").mkdir(parents=True)
    (tmp_path / "ds" / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf").touch()

    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path), on_error="warn")

    err = DataIntegrityError(message="corrupt", record=record, issues=["bad"])
    with patch.object(ds, "_ensure_raw", side_effect=err) as mock_ensure:
        ds.raw  # first access — triggers _ensure_raw
        ds.raw  # second access — should NOT trigger _ensure_raw again

    mock_ensure.assert_called_once()


# ── Fallback tests ──


def test_load_raw_missing_task_falls_back_to_direct_reader(tmp_path):
    """RuntimeError about missing 'task' entity falls back to direct reader."""
    ds = _make_local_eegdashraw(tmp_path, "ds_task", "sub-01/eeg/sub-01_eeg.edf")

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("BIDSPath must contain 'task' entity"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            return_value=mock_raw,
        ):
            result = ds._load_raw()

    assert result is mock_raw


def test_load_raw_missing_task_direct_reader_also_fails(tmp_path):
    """RuntimeError about missing 'task' + direct reader failure → DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_local_eegdashraw(tmp_path, "ds_task", "sub-01/eeg/sub-01_eeg.edf")

    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("BIDSPath must contain 'task' entity"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            side_effect=Exception("reader failed"),
        ):
            with pytest.raises(
                DataIntegrityError, match="Bad record metadata and direct reader failed"
            ):
                ds._load_raw()


def test_load_raw_runtime_unrecoverable_set_tries_eeglab_fallback(tmp_path):
    """Unrecoverable RuntimeError on .set file tries EEGLAB fallback first."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds_set", "sub-01/eeg/sub-01_task-rest_eeg.set", ext=".set"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("Allowed values for the Extension"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_eeglab_fallback",
            return_value=mock_raw,
        ):
            result = ds._load_raw()

    assert result is mock_raw


def test_load_raw_runtime_unrecoverable_tries_direct_reader(tmp_path):
    """Unrecoverable RuntimeError on .edf tries direct reader before raising."""
    ds = _make_local_eegdashraw(
        tmp_path, "ds_edf", "sub-01/eeg/sub-01_task-rest_eeg.edf"
    )

    mock_raw = MagicMock()
    with patch(
        "mne_bids.read_raw_bids",
        side_effect=RuntimeError("incorrect number of samples in the data"),
    ):
        with patch(
            "eegdash.dataset.base._load_raw_direct",
            return_value=mock_raw,
        ):
            result = ds._load_raw()

    assert result is mock_raw


# ---------------------------------------------------------------------------
# Git-annex key resolution in _download_required_files
# ---------------------------------------------------------------------------


def test_resolve_annex_key_uri_md5e(tmp_path):
    """_resolve_annex_key_uri detects MD5E annex keys and returns BIDS URI."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds002158",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds002158/sub-01/eeg/MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vhdr",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    resolved = ds._resolve_annex_key_uri(ds._raw_uri)
    assert resolved == (
        "s3://openneuro.org/ds002158/sub-01/eeg/sub-01_task-rest_eeg.vhdr"
    )


def test_resolve_annex_key_uri_sha256e(tmp_path):
    """_resolve_annex_key_uri detects SHA256E annex keys."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds003848",
        "bids_relpath": "sub-02/eeg/sub-02_task-rest_eeg.eeg",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds003848/sub-02/eeg/SHA256E-s2313--abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789.eeg",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    resolved = ds._resolve_annex_key_uri(ds._raw_uri)
    assert resolved == (
        "s3://openneuro.org/ds003848/sub-02/eeg/sub-02_task-rest_eeg.eeg"
    )


def test_resolve_annex_key_uri_normal_path_returns_none(tmp_path):
    """_resolve_annex_key_uri returns None for normal BIDS paths."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds001",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds001/sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    assert ds._resolve_annex_key_uri(ds._raw_uri) is None


def test_download_required_files_annex_key_fallback(tmp_path):
    """_download_required_files tries BIDS name when annex key URI fails."""
    from eegdash.dataset.base import EEGDashRaw

    record = {
        "dataset": "ds002158",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds002158/sub-01/eeg/MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vhdr",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    bids_uri = "s3://openneuro.org/ds002158/sub-01/eeg/sub-01_task-rest_eeg.vhdr"

    def download_side_effect(uri, path, filesystem=None):
        if "MD5E-s11657--" in uri:
            raise FileNotFoundError(uri)
        # BIDS-named URI succeeds
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with (
        patch("eegdash.dataset.base.downloader.get_s3_filesystem"),
        patch(
            "eegdash.dataset.base.downloader.download_s3_file",
            side_effect=download_side_effect,
        ) as mock_dl,
        patch("eegdash.dataset.base.downloader.download_files"),
    ):
        ds._download_required_files()

    # First two calls: annex URI (fails), then BIDS URI (succeeds).
    # Additional calls are companion file downloads (.eeg, .vmrk, .dat).
    assert mock_dl.call_count >= 2
    assert "MD5E-s11657--" in mock_dl.call_args_list[0][0][0]
    assert mock_dl.call_args_list[1][0][0] == bids_uri


def test_download_required_files_annex_key_both_fail(tmp_path):
    """DataIntegrityError when both annex key and BIDS name fail."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds002158",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds002158/sub-01/eeg/MD5E-s11657--7a519e74754041a678931b7b7d72f0ab.vhdr",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with (
        patch("eegdash.dataset.base.downloader.get_s3_filesystem"),
        patch(
            "eegdash.dataset.base.downloader.download_s3_file",
            side_effect=FileNotFoundError("not found"),
        ),
        patch("eegdash.dataset.base.downloader.download_files"),
        pytest.raises(DataIntegrityError, match="annex key and BIDS name"),
    ):
        ds._download_required_files()


def test_download_required_files_normal_uri_no_annex_fallback(tmp_path):
    """Normal (non-annex) URI that fails raises DataIntegrityError directly."""
    from eegdash.dataset.base import EEGDashRaw
    from eegdash.dataset.exceptions import DataIntegrityError

    record = {
        "dataset": "ds005170",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.set",
        "storage": {
            "backend": "s3",
            "base": "s3://openneuro.org",
            "raw_key": "ds005170/sub-01/eeg/sub-01_task-rest_eeg.set",
        },
    }
    with patch("eegdash.dataset.base.validate_record", return_value=None):
        ds = EEGDashRaw(record, cache_dir=str(tmp_path))

    with (
        patch("eegdash.dataset.base.downloader.get_s3_filesystem"),
        patch(
            "eegdash.dataset.base.downloader.download_s3_file",
            side_effect=FileNotFoundError("not found"),
        ),
        patch("eegdash.dataset.base.downloader.download_files"),
        pytest.raises(DataIntegrityError, match="not found on S3"),
    ):
        ds._download_required_files()


# ── Split FIF annex path resolution ──


def test_split_fif_continuation_copies_to_expected_annex_path(tmp_path):
    """Downloaded continuation should also be placed at the expected (annex) path."""
    ds = _make_remote_eegdashraw(
        tmp_path, "ds_annex", "sub-01/meg/sub-01_task-rest_meg.fif"
    )
    annex_dir = tmp_path / "annex_objects" / "ab" / "cd"
    annex_next = annex_dir / "MD5E-s123--abc-1.fif"
    mock_raw = MagicMock()
    mock_fs = object()

    def download_side_effect(s3_path, local_path, filesystem=None):
        # Simulate successful download by creating the file
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake fif data")
        return local_path

    with patch.object(
        ds,
        "_read_raw_bids",
        side_effect=[
            ValueError(
                f"Split raw file detected but next file {annex_next} does not exist"
            ),
            mock_raw,
        ],
    ):
        with patch(
            "eegdash.dataset.base.downloader.get_s3_filesystem", return_value=mock_fs
        ):
            with patch(
                "eegdash.dataset.base.downloader.download_s3_file",
                side_effect=download_side_effect,
            ):
                result = ds._load_raw()

    assert result is mock_raw
    # The file should also exist at the annex path
    assert annex_next.exists()
    assert annex_next.read_bytes() == b"fake fif data"


def test_split_fif_download_fails_falls_back_to_direct_reader(tmp_path):
    """When split FIF download fails, should fall back to _load_raw_direct."""
    ds = _make_remote_eegdashraw(
        tmp_path, "ds_split_fail", "sub-01/meg/sub-01_task-rest_meg.fif"
    )
    mock_raw = MagicMock()
    mock_fs = object()

    with patch.object(
        ds,
        "_read_raw_bids",
        side_effect=ValueError(
            "Split raw file detected but next file /tmp/missing-1.fif does not exist"
        ),
    ):
        with patch(
            "eegdash.dataset.base.downloader.get_s3_filesystem", return_value=mock_fs
        ):
            with patch(
                "eegdash.dataset.base.downloader.download_s3_file",
                side_effect=FileNotFoundError("not on S3"),
            ):
                with patch(
                    "eegdash.dataset.base._load_raw_direct",
                    return_value=mock_raw,
                ) as mock_direct:
                    result = ds._load_raw()

    mock_direct.assert_called_once_with(ds.filecache)
    assert result is mock_raw


def test_split_fif_download_and_direct_both_fail_raises(tmp_path):
    """When both split download and direct reader fail → DataIntegrityError."""
    from eegdash.dataset.exceptions import DataIntegrityError

    ds = _make_remote_eegdashraw(
        tmp_path, "ds_split_bad", "sub-01/meg/sub-01_task-rest_meg.fif"
    )
    mock_fs = object()

    with patch.object(
        ds,
        "_read_raw_bids",
        side_effect=ValueError(
            "Split raw file detected but next file /tmp/missing-1.fif does not exist"
        ),
    ):
        with patch(
            "eegdash.dataset.base.downloader.get_s3_filesystem", return_value=mock_fs
        ):
            with patch(
                "eegdash.dataset.base.downloader.download_s3_file",
                side_effect=FileNotFoundError("not on S3"),
            ):
                with patch(
                    "eegdash.dataset.base._load_raw_direct",
                    side_effect=ValueError("cannot read"),
                ):
                    with pytest.raises(
                        DataIntegrityError,
                        match="Split FIF continuation missing",
                    ):
                        ds._load_raw()


# ── CTF .ds direct reader support ──


def test_load_raw_direct_ctf_calls_read_raw_ctf(tmp_path):
    """_load_raw_direct should support .ds extension via read_raw_ctf."""
    from eegdash.dataset.io import _load_raw_direct

    ds_path = tmp_path / "sub-01_task-rest_meg.ds"
    ds_path.mkdir()

    mock_raw = MagicMock()
    with patch("mne.io.read_raw_ctf", return_value=mock_raw) as mock_reader:
        result = _load_raw_direct(ds_path)

    mock_reader.assert_called_once_with(str(ds_path), preload=False, verbose="ERROR")
    assert result is mock_raw
