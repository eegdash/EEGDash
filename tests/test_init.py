from torch.utils.data import Dataset

from eegdash import EEGDash, EEGDashDataset


def test_set_import_instanciate_eegdash():
    eeg_dash_instance = EEGDash()
    assert isinstance(eeg_dash_instance, EEGDash)

    eeg_pytorch_dataset_instance = EEGDashDataset(
        {"dataset": "ds005514", "task": "RestingState", "subject": "NDARDB033FW5"}
    )
    assert isinstance(eeg_pytorch_dataset_instance, Dataset)


def test_number_recordings():
    eeg_dash_instance = EEGDash()

    records = eeg_dash_instance.find({})

    assert isinstance(records, list)
    assert len(records) >= 55088
    # As of the last known count in 9 of jun of 2025, there are 55088 recordings in the dataset
