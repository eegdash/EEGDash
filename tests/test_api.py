import mne
import numpy as np
import pytest
import xarray as xr
from mne_bids import BIDSPath, write_raw_bids

from eegdash.api import EEGDash


# Fixture to create a dummy BIDS dataset for testing
@pytest.fixture(scope="module")
def dummy_bids_dataset(tmpdir_factory):
    bids_root = tmpdir_factory.mktemp("bids")
    # Create a simple MNE Raw object
    ch_names = ["EEG 001", "EEG 002", "EEG 003"]
    ch_types = ["eeg"] * 3
    sfreq = 100
    n_times = 100
    data = np.random.randn(len(ch_names), n_times)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Define BIDS path
    subject_id = "01"
    session_id = "01"
    task_name = "test"
    run_id = "01"
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_id,
        task=task_name,
        run=run_id,
        root=bids_root,
        datatype="eeg",
    )

    # Write BIDS data
    write_raw_bids(raw, bids_path, overwrite=True, format="EEGLAB", allow_preload=True)

    return str(bids_path.fpath)


def test_load_eeg_data_from_bids_file(dummy_bids_dataset):
    eegdash = EEGDash()
    data = eegdash.load_eeg_data_from_bids_file(dummy_bids_dataset)
    assert isinstance(data, xr.DataArray)


def test_load_eeg_data_from_bids_file_content(dummy_bids_dataset):
    eegdash = EEGDash()
    data = eegdash.load_eeg_data_from_bids_file(dummy_bids_dataset)

    # Check dimensions
    assert data.dims == ("channel", "time")

    # Check shape
    assert data.shape == (3, 100)

    # Check channel names
    assert list(data.channel.values) == ["EEG 001", "EEG 002", "EEG 003"]

    # Check time values
    assert len(data.time.values) == 100
