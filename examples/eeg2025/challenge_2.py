""".. _challenge_2:

Challenge 2 - Predicting p-factor from EEG
============================================

This tutorial shows you how to start to play with the challenge 2, the prediction of p-factor.
The design of this task was inspired by the need to identify and extract relevant biomarkers 
from EEG signals that can predict mental health outcomes.

The design is aimed to force the model of EEG signal processing to focus on the relevant 
features that are indicative of mental health states. The emerging behavior of the large or small
model should ideally highlight these features and improve the interpretability of the results.

We need to go beyond traditional classification and focus on challenges that highlight the 
extrapolation and the generalization of the model.

For this challenge, we assume that you already have the dataset downloaded and prepared for use.

Several ways to download the dataset can be check in the [documentation](https://eeg2025.github.io/data/#downloading-the-data),
if you haven't done so already.

"""
######################################################################
# Fill...
# -----------------------------------------
# Short answer: ....
#
# Fill here
#
# - Fill here?
#
# Some figure here!



######################################################################
# .. warning::
#    Some warning about the data leakage, the pre-processed steps
#

######################################################################
# Loading and preprocessing of data, defining a model, etc.
# ----------------------------------------------------------
#

import random
from pathlib import Path
from eegdash import EEGChallengeDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset



######################################################################
# Loading data
# ~~~~~~~~~~~~~
#
# In this challenge 2 example, we load the EEG 2025 release using EEG Dash and Braindecode,
# we load all the public datasets available in the EEG 2025 release.
#
# The first step is define the cache folder!
cache_dir = (Path.home() / "mne_data" / "eeg_challenge_cache").resolve()
cache_dir = cache_dir.expanduser()
# Creating the path if it does not exist
cache_dir.mkdir(parents=True, exist_ok=True)
# We load all the releases
# We are loading the releasing between 1 to 11.

release_list = ["R{}".format(i) for i in range(1, 11+1)]

print(release_list)

# For this challenge, we gonna sample across all the possibles task, but for start,
# we recommend to focus on the Resting State task.

all_datasets = [
    EEGChallengeDataset(
        release=release,
        query=dict(
            task="RestingState",
        ),
        description_fields=[
            "subject",
            "session",
            "run",
            "task",
            "age",
            "gender",
            "sex",
            "p_factor",
        ],
    )
    for release in release_list
]

# Issue with eegchallenge object again.
print(all_datasets[0].datasets[0].raw)

######################################################################
# Combine the PyTorch datasets into single dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combine all datasets into a single BaseConcatDataset
print("Combining all datasets into a single PyTorch dataset object with braindecode.")
all_datasets = BaseConcatDataset(all_datasets)

description = all_datasets.description

print(description)

######################################################################
# How to inspect your data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can check what is inside the dataset consuming the
# MNE-object inside the Braindecode dataset

# raw = all_datasets.datasets[0].raw
# print(raw.info)

# raw.plot(duration=10, scalings="auto", show=True)

# The visualization shows the raw EEG signal for the first 10 seconds.
# We can also inspect the data further by looking at the events and annotations.
# We strong recommend you to take a look into the details and check how the events are structured.

# print(raw.annotations)

SFREQ = 100

# Extract 2-second windows, uniformly sampled over the whole signal
class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        p_factor = self.dataset.raw.info["subject_info"]["p_factor"]
        p_factor = float(p_factor)

        # Addtional information:
        infos = {
            "subject": self.dataset.raw.info["subject_info"]["his_id"],
            "sex": self.dataset.raw.info["subject_info"]["sex"],
            "age": float(self.dataset.raw.info["subject_info"]["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, p_factor, (i_window_in_trial, i_start, i_stop), infos


windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

windows_ds = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
)

# Now we have our pytorch dataset necessary for the training!

######################################################################
# %%
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from braindecode.models import EEGNetv4

# Create PyTorch Dataloader
dataloader = DataLoader(windows_ds, batch_size=10, shuffle=True)

# Initialize model
model = EEGNetv4(n_chans=129, n_outputs=1, n_times=2 * SFREQ)

# All the braindecode models expect the input to be of shape (batch_size, n_channels, n_times)
# and have a test coverage about the behavior of the model.
print(model)


# Specify optimizer
optimizer = optim.Adamax(params=model.parameters(), lr=0.002)

# Train model for 1 epoch
for epoch in range(1):

    for idx, batch in enumerate(dataloader):
        X, y, crop_inds, infos = batch
        X = X.to(torch.float)
        y = y.to(torch.float32)
        y_pred = torch.squeeze(model(X))

        loss = l1_loss(y_pred, y)

        print(f"Epoch {0} - step {idx}, loss: {loss.item()}")

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


