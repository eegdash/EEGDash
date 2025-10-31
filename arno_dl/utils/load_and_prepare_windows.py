# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset for a collection of subject. The dataset ds005505 contains 136 subjects with both male and female participants.
# %%
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.models import EEGConformer, EEGNeX
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset


def load_and_prepare_windows(
    releases,
    tasks,
    target_name,
    folds=10,
    it_fold=0,
    split_val=0,
    output_dir="data",
    batch_size=128,
    random_add=42,
):
    if not isinstance(releases, list):
        releases = [releases]
    if not isinstance(tasks, list):
        tasks = [tasks]

    cached_data_folder_names = []
    for release in releases:
        for task in tasks:
            cached_data_folder_name = os.path.join(output_dir, release + "_" + task)
            if os.path.exists(cached_data_folder_name):
                cached_data_folder_names.append(cached_data_folder_name)
            else:
                raise FileNotFoundError(
                    f"Missing DataError({cached_data_folder_name}): You first run process_data to run the task for each release"
                )

    print("Loading data from disk")
    windows_ds = []
    for cached_data_folder_name in cached_data_folder_names:
        windows_ds_tmp = load_concat_dataset(
            path=cached_data_folder_name, preload=False
        )
        windows_ds.extend([ds for ds in windows_ds_tmp.datasets])
        print(
            f"Number of datasets in {cached_data_folder_name}: {len(windows_ds_tmp.datasets)}"
        )

    windows_ds = BaseConcatDataset(windows_ds)
    print(f"Number of datasets in all releases: {len(windows_ds.datasets)}")
    print(f"number of samples in windows_ds: {len(windows_ds)}")

    # check target
    if windows_ds.target_name != target_name:
        raise ValueError(
            f"Target name {windows_ds.target_name} does not match {target_name}"
        )

    # ## Creating a Training and Test Set
    unique_subjects, unique_indices = np.unique(
        windows_ds.description["subject"], return_index=True
    )

    if folds > 1:
        # Use KFold; StratifiedKFold requires labels which we don't have here
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_add)
        splits = list(splitter.split(unique_subjects))
        train_idx, val_idx = splits[it_fold]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(unique_subjects)), train_size=0.8, random_state=random_add
        )
        splits = [(train_idx, val_idx)]

    train_ds = BaseConcatDataset(
        [
            ds
            for ds in windows_ds.datasets
            if ds.description.subject in unique_subjects[train_idx]
            and ds.description.get(target_name, None) is not None
            and not isinstance(ds.description.get(target_name, None), str)
            and np.isfinite(float(ds.description.get(target_name, np.nan)))
            and (
                (abs(float(ds.description.get(target_name))) >= split_val)
                if split_val > 0
                else True
            )
        ]
    )
    val_ds = BaseConcatDataset(
        [
            ds
            for ds in windows_ds.datasets
            if ds.description.subject in unique_subjects[val_idx]
            and ds.description.get(target_name, None) is not None
            and not isinstance(ds.description.get(target_name, None), str)
            and np.isfinite(float(ds.description.get(target_name, np.nan)))
            and (
                (abs(float(ds.description.get(target_name))) >= split_val)
                if split_val > 0
                else True
            )
        ]
    )

    # Create dataloaders with smaller batch size to save memory
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4
    )

    # REPLACE HERE WITH NEW CODE AND ALSO TRY OVERWRITING METADATA AND RECREATE WINDOWS_DS
    # PROBABLY MODIFYING WINDOWS_DS IS NEEDED (maybe simply set targetname?)
    raw_targets = []
    for ds in train_ds.datasets:
        v = ds.description.get(target_name, None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            raw_targets.append(v)
    targets_np = np.array(raw_targets, dtype=np.float32)
    if targets_np.size == 0:
        target_std = 1.0
        train_baseline_mae = 1.0
    else:
        dd = 1 if targets_np.size > 1 else 0
        target_std = float(np.std(targets_np, ddof=dd))
        med = float(np.median(targets_np))
        train_baseline_mae = float(np.mean(np.abs(targets_np - med)))
        if not np.isfinite(target_std) or target_std <= 0:
            target_std = 1.0
        if not np.isfinite(train_baseline_mae) or train_baseline_mae <= 0:
            train_baseline_mae = 1.0

    return train_loader, val_loader, train_ds, val_ds, target_std, train_baseline_mae


if __name__ == "__main__":
    load_and_prepare_windows(
        releases=["ds003061"],
        tasks=["P300"],
        target_name="P300",
        folds=1,
        it_fold=0,
        split_val=0,
    )
