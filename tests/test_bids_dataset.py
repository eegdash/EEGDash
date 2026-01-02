import json

import pytest

from eegdash.dataset.bids_dataset import EEGBIDSDataset


@pytest.fixture
def mock_bids_dir(tmp_path):
    """Create a minimal BIDS structure."""
    ds_root = tmp_path / "dsX"
    ds_root.mkdir()

    # Dataset description (required for inheritance stop)
    (ds_root / "dataset_description.json").write_text(json.dumps({"Name": "Test"}))

    # Root metadata
    (ds_root / "task-rest_eeg.json").write_text(json.dumps({"SamplingFrequency": 100}))

    # Subject dir
    sub_dir = ds_root / "sub-01" / "eeg"
    sub_dir.mkdir(parents=True)

    # Data file
    (sub_dir / "sub-01_task-rest_eeg.vhdr").touch()

    # Specific JSON (inheritance)
    (sub_dir / "sub-01_task-rest_eeg.json").write_text(
        json.dumps({"RecordingDuration": 10})
    )

    # Channels TSV
    (sub_dir / "sub-01_task-rest_channels.tsv").write_text("name\ttype\nFp1\tEEG\n")

    # Participants TSV
    (ds_root / "participants.tsv").write_text("participant_id\tage\nsub-01\t25\n")

    return ds_root


def test_eegbidsdataset_init(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    assert ds.files  # Should find the file
    assert ds.detected_modality == "eeg"
    assert ds.check_eeg_dataset() is True


def test_eegbidsdataset_init_mismatch(tmp_path):
    root = tmp_path / "dsY"
    root.mkdir()
    with pytest.raises(AssertionError, match="does not correspond to dataset"):
        EEGBIDSDataset(data_dir=root, dataset="dsX")


def test_file_attributes(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(mock_bids_dir / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr")

    # Direct attributes
    assert ds.get_bids_file_attribute("subject", fpath) == "01"
    assert ds.get_bids_file_attribute("task", fpath) == "rest"

    # JSON attributes (inherited) - 100Hz from root, duration from specific
    assert ds.get_bids_file_attribute("sfreq", fpath) == 100
    assert ds.get_bids_file_attribute("ntimes", fpath) == 10


def test_num_times(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(mock_bids_dir / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr")
    # 100 Hz * 10s = 1000 samples
    assert ds.num_times(fpath) == 1000


def test_channel_labels(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(mock_bids_dir / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr")
    labels = ds.channel_labels(fpath)
    assert labels == ["Fp1"]


def test_subject_participant_tsv(mock_bids_dir):
    ds = EEGBIDSDataset(data_dir=mock_bids_dir, dataset="dsX")
    fpath = str(mock_bids_dir / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr")

    info = ds.subject_participant_tsv(fpath)
    assert info["participant_id"] == "sub-01"
    assert info["age"] == "25"
