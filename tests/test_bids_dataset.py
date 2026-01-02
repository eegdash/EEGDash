from pathlib import Path

import mne
import numpy as np
import pytest
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
