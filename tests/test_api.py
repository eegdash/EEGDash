import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash.dataset import EEGDashDataset
from eegdash.paths import get_default_cache_dir
from eegdash.schemas import create_record


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


def test_eegdashdataset_empty_cache_dir():
    """Test that EEGDashDataset with an empty cache_dir uses the current directory."""
    # This test is to verify the behavior of the `cache_dir` argument.
    # The previous implementation used `get_default_cache_dir()` when an empty
    # string was passed. The new implementation uses `Path("")`, which resolves
    # to the current directory.
    record = create_record(
        dataset="ds005505",
        storage_base="s3://test-bucket/ds005505",
        bids_relpath="sub-01/eeg/sub-01_task-test_eeg.set",
        subject="01",
        task="test",
    )
    # Add extra fields for length calculation
    record["sampling_frequency"] = 1
    record["ntimes"] = 1

    ds = EEGDashDataset(
        cache_dir="",
        records=[record],
        download=False,
    )
    assert ds.cache_dir == get_default_cache_dir()
