import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from mne import get_config
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from skorch.helper import SliceDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from braindecode.datautil import load_concat_dataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.util import set_random_seeds
from eegdash import EEGDashDataset
from eegdash.preprocessing import hbn_ec_ec_reannotation

cache_folder = Path(get_config("MNE_DATA"))
logger = logging.getLogger("eegdash")

seed = 42
set_random_seeds(seed, cuda=False)
random_state = check_random_state(seed)


@pytest.fixture(scope="module")
def eeg_dash_dataset():
    """Fixture to create an instance of EEGDashDataset."""
    return EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_folder,
    )


@pytest.fixture(scope="module")
def preprocess_instance(eeg_dash_dataset):
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
    pre_processed_dir = cache_folder / "preprocessed"
    pre_processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        eeg_dash_dataset = load_concat_dataset(
            pre_processed_dir,
            preload=True,
        )
        return eeg_dash_dataset

    except ValueError as e:
        warnings.warn(f"Failed to load dataset creating a new instance: {e}. ")

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


@pytest.fixture(scope="module")
def windows_ds(preprocess_instance):
    """Fixture to create windows from the preprocessed EEG dataset."""
    windows = create_windows_from_events(
        preprocess_instance,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=256,
        preload=True,
    )
    return windows


def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
    x = (x - mean) / std
    x = x.to(dtype=torch.float32)
    return x


def test_complete_train(windows_ds):
    """Test the complete training process with the EEG dataset."""
    label_eye_open_closed = np.array(
        SliceDataset(windows_ds, idx=1)
    ).T  # Extract labels (eyes open/closed)

    # train-test split
    train_idx, valid_idx = train_test_split(
        range(len(windows_ds)),
        test_size=0.2,
        stratify=label_eye_open_closed,
        random_state=random_state,
    )

    # Convert the data to tensors
    X_train = SliceDataset(windows_ds, idx=0, indices=train_idx)
    # Convert list of arrays to torch
    X_train = torch.FloatTensor(np.array(X_train))

    X_valid = SliceDataset(windows_ds, idx=0, indices=valid_idx)
    # Convert list of arrays to torch
    X_valid = torch.FloatTensor(np.array(X_valid))
    # Convert targets to tensor
    y_train = torch.LongTensor(label_eye_open_closed[train_idx])
    y_valid = torch.LongTensor(label_eye_open_closed[valid_idx])

    dataset_train = TensorDataset(X_train, y_train)
    dataset_valid = TensorDataset(X_valid, y_valid)

    # Create data loaders for training and valid (batch size 10)
    train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset_valid, batch_size=10, shuffle=False)

    logger.info(
        f"Shape of data {X_train.shape} number of samples - Train: {len(train_loader)}, Test: {len(val_loader)}"
    )
    logger.info(
        f"Eyes-Open/Eyes-Closed balance, train: {np.mean(label_eye_open_closed[train_idx]):.2f}, test: {np.mean(label_eye_open_closed[valid_idx]):.2f}"
    )

    model = ShallowFBCSPNet(n_chans=24, n_outputs=2, n_times=256)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    device = torch.device("cpu")
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    epochs = 3

    for e in range(epochs):
        # training
        correct_train = 0
        for _, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            scores = model(normalize_data(x))
            _, preds = scores.max(1)
            correct_train += (preds == y).sum() / len(dataset_train)

            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        correct_val = 0

        for t, (x, y) in enumerate(val_loader):
            model.eval()  # put model to testing mode
            scores = model(normalize_data(x))
            y = y.to(device=device, dtype=torch.long)
            _, preds = scores.max(1)
            correct_val += (preds == y).sum() / len(dataset_valid)

        logger.info(
            f"Epoch {e}, Train accuracy: {correct_train:.2f}, \
            Valid accuracy: {correct_val:.2f}"
        )
