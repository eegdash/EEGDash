"""Fixtures for features module test."""

import shutil
from pathlib import Path

import pytest

from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash import EEGDashDataset
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation
from eegdash.logging import logger


@pytest.fixture(scope="session")
def eeg_dash_dataset(cache_dir: Path):
    """Fixture to create an instance of EEGDashDataset."""
    return EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_dir,
    )


@pytest.fixture(scope="session")
def preprocess_instance(eeg_dash_dataset, cache_dir: Path):
    """Fixture to create an instance of EEGDashDataset with preprocessing."""
    selected_channels = [
        "E22",
        "E9",
        "E33",
        "E24",
        "E11",
        "E124",
        "E122",
        "E29",
        "E6",
        "E111",
        "E45",
        "E36",
        "E104",
        "E108",
        "E42",
        "E55",
        "E93",
        "E58",
        "E52",
        "E62",
        "E92",
        "E96",
        "E70",
        "Cz",
    ]
    pre_processed_dir = cache_dir / "preprocessed"
    pre_processed_dir.mkdir(parents=True, exist_ok=True)
    loaded_ds = None
    if pre_processed_dir.exists() and any(pre_processed_dir.iterdir()):
        try:
            loaded_ds = load_concat_dataset(
                pre_processed_dir,
                preload=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load dataset, recreating: {e}")

    if loaded_ds is not None:
        return loaded_ds

    if pre_processed_dir.exists():
        # folder with issue or empty, erasing and creating again
        shutil.rmtree(pre_processed_dir)
    pre_processed_dir.mkdir(parents=True, exist_ok=True)

    preprocessors = [
        hbn_ec_ec_reannotation(),
        Preprocessor(
            "pick_channels",
            ch_names=selected_channels,
        ),
        Preprocessor("resample", sfreq=128),
        Preprocessor("filter", l_freq=1, h_freq=55),
    ]

    eeg_dash_dataset = preprocess(
        eeg_dash_dataset, preprocessors, n_jobs=-1, save_dir=pre_processed_dir
    )

    return eeg_dash_dataset


@pytest.fixture(scope="session")
def windows_ds(preprocess_instance):
    """Fixture to create windows from the preprocessed EEG dataset."""
    windows = create_windows_from_events(
        preprocess_instance,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=256,
        preload=True,
    )
    return windows
