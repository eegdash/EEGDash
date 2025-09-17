from pathlib import Path

import pytest
from torch.utils.data import Dataset

from eegdash import EEGDash, EEGDashDataset


@pytest.fixture(scope="module")
def cache_dir():
    """Provide a shared cache directory for tests that need to cache datasets."""
    from pathlib import Path

    from eegdash.paths import get_default_cache_dir

    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_set_import_instanciate_eegdash(cache_dir: Path):
    eeg_dash_instance = EEGDash()
    assert isinstance(eeg_dash_instance, EEGDash)

    eeg_pytorch_dataset_instance = EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_dir,
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
