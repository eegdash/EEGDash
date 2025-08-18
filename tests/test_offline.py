from pathlib import Path

import pytest

from eegdash import EEGDash, EEGDashDataset

CACHE_DIR = (Path.home() / "mne_data" / "eeg_challenge_cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def test_dataset_loads_without_eegdash(monkeypatch):
    """Dataset should load from records without contacting network resources."""
    eeg_dash = EEGDash()

    records = eeg_dash.find(subject="NDARAC350XUM", task="RestingState")

    # test with internet
    dataset_internet = EEGDashDataset(
        query=dict(task="RestingState", subject="NDARAC350XUM", dataset="ds005509"),
        cache_dir=CACHE_DIR,
        eeg_dash_instance=eeg_dash,
    )

    # Monkeypatch any network calls inside EEGDashDataset to raise if called
    monkeypatch.setattr(
        EEGDashDataset,
        "find_datasets",
        lambda *args, **kwargs: pytest.skip(
            "Skipping network download in offline test"
        ),
    )
    monkeypatch.setattr(
        EEGDashDataset,
        "find_datasets",
        lambda *args, **kwargs: pytest.skip(
            "Skipping network download in offline test"
        ),
    )
    # TO-DO: discover way to do this pytest

    dataset_without_internet = EEGDashDataset(
        records=records, cache_dir=CACHE_DIR, eeg_dash_instance=None
    )

    assert dataset_internet.datasets[0].raw == dataset_without_internet.datasets[0].raw
    assert (
        dataset_internet.datasets[0].record
        == dataset_without_internet.datasets[0].record
    )
