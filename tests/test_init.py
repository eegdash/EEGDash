from pathlib import Path

from mne import get_config
from torch.utils.data import Dataset

from eegdash import EEGDash, EEGDashDataset

cache_folder = Path(get_config("MNE_DATA"))


def test_set_import_instanciate_eegdash():
    eeg_dash_instance = EEGDash()
    assert isinstance(eeg_dash_instance, EEGDash)

    eeg_pytorch_dataset_instance = EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_folder,
    )
    assert isinstance(eeg_pytorch_dataset_instance, Dataset)


def test_dataset_api():
    eegdash = EEGDash()
    record = eegdash.find({"dataset": "ds005511", "subject": "NDARUF236HM7"})
    print(record)
    assert isinstance(record, list)


def test_number_recordings():
    eeg_dash_instance = EEGDash()

    records = eeg_dash_instance.find({})

    assert isinstance(records, list)
    assert len(records) >= 55088
    # As of the last known count in 9 of jun of 2025, there are 55088 recordings in the dataset
