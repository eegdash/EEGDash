from unittest.mock import MagicMock, patch

from eegdash.dataset.base import EEGDashRaw


@patch("eegdash.dataset.base.validate_record")
@patch("eegdash.dataset.base.mne_bids")
def test_ensure_raw_symlink_logic(mock_mne_bids, mock_validate, tmp_path):
    mock_validate.return_value = None  # Simulate valid record
    # Setup mock record
    record = {
        "dataset": "ds001",
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "local",
            "base": str(tmp_path),
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        },
    }

    # Create directory structure
    dataset_root = tmp_path
    eeg_dir = dataset_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    # Create dummy files
    (eeg_dir / "sub-01_task-rest_eeg.vhdr").touch()
    (eeg_dir / "sub-01_task-rest_electrodes.tsv").touch()

    # Create coordsystem in subject root (parent of eeg)
    subject_root = dataset_root / "sub-01"
    (subject_root / "sub-01_coordsystem.json").touch()

    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Run _ensure_raw which triggers symlink logic
    # We mock _load_raw to avoid actual MNE reading
    dataset._load_raw = MagicMock()
    dataset._ensure_raw()

    # Verify symlink created
    expected_link = eeg_dir / "sub-01_coordsystem.json"
    assert expected_link.exists()
    assert expected_link.is_symlink()
    assert expected_link.read_text() == ""  # It's empty but exists


@patch("eegdash.dataset.base.validate_record")
def test_len_error_handling(mock_validate, tmp_path):
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
def test_vhdr_auto_repair(mock_validate, tmp_path):
    mock_validate.return_value = None

    # Setup dataset structure
    dataset_id = "ds_bad_vhdr"
    record = {
        "dataset": dataset_id,
        "bids_relpath": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "storage": {
            "backend": "s3",
            "base": "s3://bucket",
            "raw_key": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        },
    }

    eeg_dir = tmp_path / dataset_id / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)

    # Create the BIDS files (what we want to point to)
    (eeg_dir / "sub-01_task-rest_eeg.eeg").touch()
    (eeg_dir / "sub-01_task-rest_eeg.vmrk").touch()

    # Create the VHDR with BAD pointers
    vhdr_path = eeg_dir / "sub-01_task-rest_eeg.vhdr"
    vhdr_content = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=INTERNAL_NAME.eeg
MarkerFile=INTERNAL_NAME.vmrk
"""
    vhdr_path.write_text(vhdr_content)

    # Init dataset
    dataset = EEGDashRaw(record, cache_dir=str(tmp_path))

    # Trigger repair manually or via _ensure_raw
    # We'll call _patch_vhdr_pointers directly to test logic
    dataset._patch_vhdr_pointers()

    # Verify content updated
    new_content = vhdr_path.read_text()
    assert "DataFile=sub-01_task-rest_eeg.eeg" in new_content
    assert "MarkerFile=sub-01_task-rest_eeg.vmrk" in new_content
    assert "INTERNAL_NAME" not in new_content
