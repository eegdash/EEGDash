from pathlib import Path

from torch.utils.data import Dataset

from eegdash import EEGDash, EEGDashDataset


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

    count = eeg_dash_instance.count({})

    assert isinstance(count, int)
    assert count >= 55088
    # As of the last known count in 9 of jun of 2025, there are 55088 recordings in the dataset


def test_lazy_load_eegdash_dataset():
    """Test lazy loading of EEGDashDataset."""
    import eegdash

    # Access EEGDashDataset through __getattr__
    cls = getattr(eegdash, "EEGDashDataset")
    assert cls is not None


def test_lazy_load_eegchallenge_dataset():
    """Test lazy loading of EEGChallengeDataset."""
    import eegdash

    # Access EEGChallengeDataset through __getattr__
    cls = getattr(eegdash, "EEGChallengeDataset")
    assert cls is not None


def test_lazy_load_preprocessing():
    """Test lazy loading of preprocessing module."""
    import eegdash

    # Access preprocessing through __getattr__
    module = getattr(eegdash, "preprocessing")
    assert module is not None


def test_dir_function_in_init():
    """Test __dir__ function in __init__.py."""
    import eegdash

    dir_list = dir(eegdash)
    assert "EEGDash" in dir_list
    assert "EEGDashDataset" in dir_list


def test_invalid_attribute_in_init():
    """Test __getattr__ raises AttributeError for invalid names."""
    import pytest

    import eegdash

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = eegdash.nonexistent_attribute


def test_lazy_import_eegdash():
    """Test lazy import of EEGDash (line 31)."""
    import eegdash

    cls = eegdash.EEGDash
    assert cls is not None


def test_lazy_import_datasets():
    """Test lazy import of datasets (line 33)."""
    import eegdash

    cls1 = eegdash.EEGDashDataset
    cls2 = eegdash.EEGChallengeDataset
    assert cls1 is not None
    assert cls2 is not None


def test_lazy_import_preprocessing():
    """Test lazy import of preprocessing module (line 38)."""
    import eegdash

    mod = eegdash.preprocessing
    assert mod is not None


def test_lazy_import_invalid():
    """Test AttributeError for invalid attribute."""
    import pytest

    import eegdash

    with pytest.raises(AttributeError):
        _ = eegdash.nonexistent_attribute


def test_dir_function():
    """Test __dir__ returns expected attributes."""
    import eegdash

    attrs = dir(eegdash)
    assert "EEGDash" in attrs
    assert "EEGDashDataset" in attrs
