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


def test_eegdash_api_actions():
    """Test EEGDash convenience methods with mocked client."""
    from unittest.mock import MagicMock

    from eegdash.api import EEGDash

    # Mock client
    with pytest.MonkeyPatch.context() as mp:
        mock_client = MagicMock()
        mock_get_client = MagicMock(return_value=mock_client)
        mp.setattr("eegdash.api.get_client", mock_get_client)

        eeg = EEGDash()

        # 1. find
        mock_client.find.return_value = [{"id": 1}]
        assert eeg.find({"dataset": "ds1"}) == [{"id": 1}]
        # Kwargs handled
        eeg.find(dataset="ds1", subject="01")
        # Check call arguments (merged query)
        args, kwargs = mock_client.find.call_args
        assert args[0] == {"dataset": "ds1", "subject": "01"}

        # 2. exists (uses find_one)
        mock_client.find_one.return_value = {"id": 1}
        assert eeg.exists(dataset="ds1") is True

        mock_client.find_one.return_value = None
        assert eeg.exists(dataset="missing") is False

        # 3. count
        mock_client.count_documents.return_value = 42
        assert eeg.count(dataset="ds1") == 42

        # 4. insert
        mock_client.insert_one.return_value = "id123"
        eeg.insert({"dataset": "ds1"})
        assert mock_client.insert_one.call_count == 1

        mock_client.insert_many.return_value = 2
        eeg.insert([{"a": 1}, {"b": 2}])
        assert mock_client.insert_many.call_count == 1

        # 5. update_field
        mock_client.update_many.return_value = (5, 3)  # matched, modified
        matched, modified = eeg.update_field(
            {"dataset": "ds1"}, update={"field": "val"}
        )
        assert matched == 5
        assert modified == 3
