from pathlib import Path

import pytest

from eegdash.paths import get_default_cache_dir


def is_bids_dataset_available() -> tuple[bool, str]:
    """Check if the BIDS test dataset is available and valid.

    Returns a tuple of (is_available, reason).
    """
    cache_dir = Path(get_default_cache_dir())
    path = cache_dir / "ds005509-bdf-mini"

    if not path.exists():
        return False, f"BIDS dataset not found at {path}"

    # Check for basic BIDS structure (dataset_description.json)
    if not (path / "dataset_description.json").exists():
        return False, "Not a valid BIDS dataset (missing dataset_description.json)"

    # Check for at least one data file
    bdf_files = list(path.rglob("*.bdf"))
    edf_files = list(path.rglob("*.edf"))
    if not bdf_files and not edf_files:
        return False, "No BDF/EDF data files found in dataset"

    return True, ""


@pytest.fixture(scope="session")
def cache_dir():
    """Provide a shared cache directory for tests that need to cache datasets.

    This fixture ensures all tests use the same cache directory (via
    get_default_cache_dir()) to avoid redundant downloads across test runs.
    The directory is created if it doesn't exist.

    Returns
    -------
    Path
        The shared cache directory path.

    """
    cache_dir = Path(get_default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    return cache_dir


@pytest.fixture(scope="session")
def bids_mini_dataset_path(cache_dir: Path):
    """Get the path to the mini BIDS dataset for benchmark tests.

    This fixture provides a consistent path to the ds005509-bdf-mini dataset
    used in performance/benchmark tests. If the dataset doesn't exist,
    the test will be skipped.

    Parameters
    ----------
    cache_dir : Path
        The shared cache directory from the cache_dir fixture.

    Returns
    -------
    Path
        Path to the mini BIDS dataset.

    """
    path = cache_dir / "ds005509-bdf-mini"
    if not path.exists():
        pytest.skip(f"BIDS dataset not found at {path}")
    return path


import numpy as np
import pandas as pd

from eegdash.features.datasets import FeaturesConcatDataset, FeaturesDataset


@pytest.fixture
def features_dataset():
    df = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": [0, 1],
            "i_start_in_trial": [0, 100],
            "i_stop_in_trial": [100, 200],
            "target": [0, 1],
        }
    )
    description = {"subject": "sub-01", "task": "rest"}
    return FeaturesDataset(df, metadata, description)


@pytest.fixture
def concat_dataset(features_dataset):
    ds2 = FeaturesDataset(
        pd.DataFrame({"feat1": [5.0, 6.0], "feat2": [7.0, 8.0]}),
        pd.DataFrame(
            {
                "i_window_in_trial": [0, 1],
                "i_start_in_trial": [0, 100],
                "i_stop_in_trial": [100, 200],
                "target": [0, 1],
            }
        ),
        {"subject": "sub-02", "task": "task"},
    )
    return FeaturesConcatDataset([features_dataset, ds2])


@pytest.fixture
def signal_2d():
    return np.random.randn(2, 100)


@pytest.fixture
def signal_1d():
    return np.random.randn(100)


import shutil

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
