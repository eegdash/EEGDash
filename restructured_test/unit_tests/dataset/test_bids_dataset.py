from pathlib import Path

import mne
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset.bids_dataset import EEGBIDSDataset


@pytest.fixture
def mock_bids_dir(tmp_path):
    """Create a valid BIDS dataset using MNE-BIDS."""
    bids_root = tmp_path / "dsX"
    bids_root.mkdir()

    # Create dummy raw data
    sfreq = 100
    info = mne.create_info(ch_names=["O1", "O2", "Cz"], sfreq=sfreq, ch_types="eeg")
    data = np.random.randn(3, sfreq * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)

    # Write to BIDS (enforce .set for EEGLAB)
    bids_path = BIDSPath(
        subject="01", task="rest", datatype="eeg", root=bids_root, extension=".set"
    )
    write_raw_bids(
        raw,
        bids_path,
        verbose=False,
        overwrite=True,
        allow_preload=True,
        format="EEGLAB",
    )

    # Force clean participants.tsv to avoid merging issues or formatting errors
    part_tsv_file = bids_root / "participants.tsv"
    # Ensure standard format
    part_tsv_file.write_text("participant_id\tage\nsub-01\t25\n")

    return bids_root


def test_eegbidsdataset_init(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    assert len(ds.files) > 0
    assert ds.detected_modality == "eeg"
    assert ds.check_eeg_dataset() is True


def test_eegbidsdataset_init_mismatch(tmp_path):
    root = tmp_path / "dsY"
    root.mkdir()
    with pytest.raises(AssertionError, match="does not correspond to dataset"):
        EEGBIDSDataset(data_dir=root, dataset="dsX")


def test_file_attributes(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    # Find the generated file (EEGLAB uses .set)
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])

    # Direct attributes
    assert ds.get_bids_file_attribute("subject", fpath) == "01"
    assert ds.get_bids_file_attribute("task", fpath) == "rest"

    # JSON attributes
    assert ds.get_bids_file_attribute("sfreq", fpath) == 100
    # Allow some tolerance due to float/duration encoding
    assert ds.get_bids_file_attribute("duration", fpath) == pytest.approx(10.0, abs=0.1)


def test_num_times(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    # 10s * 100Hz = 1000 samples
    # Using pytest.approx for robustness against float duration
    samples = ds.num_times(fpath)
    assert samples == pytest.approx(1000, abs=1)


def test_channel_labels(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    labels = ds.channel_labels(fpath)
    assert "O1" in labels
    assert "Cz" in labels


def test_subject_participant_tsv(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])

    info = ds.subject_participant_tsv(fpath)
    assert info["age"] == "25"


def test_get_relative_bidspath(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(list((mock_bids_dir / "sub-01" / "eeg").glob("*.set"))[0])
    filename = Path(fpath).name

    # Should return "dsX/sub-01/eeg/..."
    rel = ds.get_relative_bidspath(fpath)
    assert rel == f"dsX/sub-01/eeg/{filename}"


def test_bids_dataset_gaps(tmp_path):
    # Trigger bids_dataset.py 552, 616, 621
    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    # We need a directory that looks like a BIDS dataset
    # Recursion happens in get_bids_metadata_files -> _get_bids_file_inheritance
    # which goes up until it finds dataset_description.json or hits root.
    # We should make it hit dataset_description.json.
    ds_path = Path(tmp_path).resolve() / "ds000123"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "dataset_description.json").touch()
    file_path = ds_path / "some_file.set"
    file_path.touch()

    # Mock file discovery to avoid AssertionError
    with patch(
        "eegdash.dataset.bids_dataset._find_bids_files", return_value=[str(file_path)]
    ):
        ds = EEGBIDSDataset(data_dir=str(ds_path), dataset="ds000123")

    # 552: No channels.tsv
    with pytest.raises(FileNotFoundError, match="No channels.tsv"):
        ds.channel_types(str(file_path))

    # 616: subj_val is None
    with patch.object(ds, "get_bids_file_attribute", return_value=None):
        assert ds.subject_participant_tsv(str(file_path)) == {}

    # 621: subj_val starts with sub-
    with patch.object(ds, "get_bids_file_attribute", return_value="sub-001"):
        # We also need to avoid failing later on participants_tsv
        with patch.object(ds, "get_bids_metadata_files", return_value=[]):
            assert ds.subject_participant_tsv(str(file_path)) == {}


def test_bids_subject_participant_tsv_gap(tmp_path):
    # EEGBIDSDataset.subject_participant_tsv
    # Missing 2 lines: probably file not found or empty

    # We need to mock get_bids_file_attribute to return a subject
    # And mock read_csv to return/fail

    from eegdash.dataset.bids_dataset import EEGBIDSDataset

    with patch(
        "eegdash.dataset.bids_dataset._find_bids_files", return_value=["some_file.set"]
    ):
        ds_dir = tmp_path / "ds001"
        ds_dir.mkdir()
        p = ds_dir / "dataset_description.json"
        p.write_text('{"Name": "Test Dataset", "BIDSVersion": "1.8.0"}')
        # Also patch validation to be safe
        # with patch("eegdash.dataset.bids_dataset._validate_bids_dataset"): # Removed invalid patch
        ds = EEGBIDSDataset(data_dir=str(ds_dir), dataset="ds001")

    with patch.object(ds, "get_bids_file_attribute", return_value="sub-01"):
        # Mock _find_bids_files recursion for participants.tsv?
        # The method calls self.find_file("participants.tsv")? No, it looks up inheritance.

        # Let's try to mock the internal call that finds the csv
        # It likely uses _get_bids_file_inheritance
        # Create a dummy participants.tsv
        participants_tsv = ds_dir / "participants.tsv"
        participants_tsv.write_text("participant_id\tage\tsex\nsub-01\t25\tM\n")

        # We need to mock get_bids_metadata_files to return this file
        with patch.object(
            ds, "get_bids_metadata_files", return_value=[participants_tsv]
        ):
            # Now call the method
            # The method checks if subject ("sub-01") is in the participants tsv
            # Our mock get_bids_file_attribute returns "sub-01"

            # Also need to ensure the filepath argument is handled correctly.
            # The method calls get_bids_metadata_files(filepath, "participants.tsv")

            info = ds.subject_participant_tsv("some_file.set")
            assert info["age"] == "25"
            assert info["sex"] == "M"
