"""Challenge 2: Predicting the p-factor from EEG
=============================================

.. _challenge-2:
.. meta::
   :html_theme.sidebar_secondary.remove: true
.. contents:: This example covers:
   :local:
   :depth: 2
"""
######################################################################
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_2.ipynb
#    :alt: Open In Colab
######################################################################
# Preliminary notes
# -----------------
# Before we begin, I just want to make a deal with you, ok?
# This is a community competition with a strong open-source foundation.
# When I say open-source, I mean volunteer work.
#
# So, if you see something that does not work or could be improved, first, **please be kind**, and
# we will fix it together on GitHub, okay?
#
# The entire decoding community will only go further when we stop
# solving the same problems over and over again, and it starts working together.

######################################################################
# Overview
# --------
# The psychopathology factor (P-factor) is a widely recognized construct in mental health research, representing a common underlying dimension of psychopathology across various disorders.
# Currently, the P-factor is often assessed using self-report questionnaires or clinician ratings, which can be subjective, prone to bias, and time-consuming.
# **The Challenge 2** consists of developing a model to predict the P-factor from EEG recordings.
#
# The challenge encourages learning physiologically meaningful signal representations and discovery of reproducible biomarkers.
# Models of any size should emphasize robust, interpretable features that generalize across subjects,
# sessions, and acquisition sites.
#
# Unlike a standard in-distribution classification task, this regression problem stresses out-of-distribution robustness
# and extrapolation. The goal is not only to minimize error on seen subjects, but also to transfer effectively to unseen data.
# Ensure the dataset is available locally. If not, see the
# `dataset download guide <https://eeg2025.github.io/data/#downloading-the-data>`__.
######################################################################
# Contents of this start kit
# --------------------------
# .. note:: If you need additional explanations on the
#    :doc:`EEGChallengeDataset
#    </api/dataset/eegdash.dataset.EEGChallengeDataset>` class, dataloading,
#    `braindecode <https://braindecode.org/stable/models/models_table.html>`__'s
#    deep learning models, or brain decoding in general, please refer to the
#    start-kit of challenge 1 which delves deeper into these topics.
#
# More contents will be released during the competition inside the
# :mod:`eegdash` `examples webpage <https://eeglab.org/EEGDash/generated/auto_examples/index.html>`__.
#
# .. admonition:: Prerequisites
#    :class: important
#
#    The tutorial assumes prior knowledge of:
#
#    - Standard neural network architectures (e.g., CNNs)
#    - Optimization by batch gradient descent and backpropagation
#    - Overfitting, early stopping, and regularization
#    - Some knowledge of PyTorch
#    - Basic familiarity with EEG and preprocessing
#    - An appreciation for open-source work :)
######################################################################
# Install dependencies on Colab
# -----------------------------
#
# .. note:: These installs are optional; skip on local environments
#    where you already have the dependencies installed.
#
# .. code-block:: bash
#
#    pip install eegdash
######################################################################
# Imports
# -------
from pathlib import Path
import math
import os
import random
from joblib import Parallel, delayed

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset

######################################################################
# .. warning::
#    In case of Colab, before starting, make sure you're on a GPU instance
#    for faster training! If running on Google Colab, please request a GPU runtime
#    by clicking `Runtime/Change runtime type` in the top bar menu, then selecting
#    'T4 GPU' under 'Hardware accelerator'.
######################################################################
# Identify whether a CUDA-enabled GPU is available
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    msg = "CUDA-enabled GPU found. Training should be faster."
else:
    msg = (
        "No GPU found. Training will be carried out on CPU, which might be "
        "slower.\n\nIf running on Google Colab, you can request a GPU runtime by"
        " clicking\n`Runtime/Change runtime type` in the top bar menu, then "
        "selecting 'T4 GPU'\nunder 'Hardware accelerator'."
    )
print(msg)
######################################################################
# Understanding the P-factor regression task.
# -------------------------------------------
#
# The psychopathology factor (P-factor) is a widely recognized construct in mental health research, representing a common underlying dimension of psychopathology across various disorders.
# The P-factor is thought to reflect the shared variance among different psychiatric conditions, suggesting that individuals with higher P-factor scores may be more vulnerable to a range of mental health issues.
# Currently, the P-factor is often assessed using self-report questionnaires or clinician ratings, which can be subjective, prone to bias, and time-consuming.
# In the dataset of this challenge, the P-factor was assessed using the Child
# Behavior Checklist (CBCL) `McElroy et al., (2017) <https://doi.org/10.1111/jcpp.12849>`__.
#
# The goal of Challenge 2 is to develop a model to predict the P-factor from EEG recordings.
# **The feasibility of using EEG data for this purpose is still an open question**.
# The solution may involve finding meaningful representations of the EEG data that correlate with the P-factor scores.
# The challenge encourages learning physiologically meaningful signal representations and discovery of reproducible biomarkers.
# If contestants are successful in this task, it could pave the way for more objective and efficient assessments of the P-factor in clinical settings.
######################################################################
# Define local path and (down)load the data
# -----------------------------------------
# In this challenge 2 example, we load the EEG 2025 release using
# :doc:`EEGChallengeDataset </api/dataset/eegdash.dataset.EEGChallengeDataset>`.
# **Note:** in this example notebook, we load the contrast change detection task from one mini release only as an example. Naturally, you are encouraged to train your models on all complete releases, using data from all the tasks you deem relevant.
######################################################################
# The first step is to define the cache folder!
# Match tests' cache layout under ~/eegdash_cache/eeg_challenge_cache
DATA_DIR = Path(os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")).resolve()
# Creating the path if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
# We define the list of releases to load.
# Here, only release 5 is loaded.
release_list = ["R5"]
all_datasets_list = [
    EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",
        mini=True,
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
        cache_dir=DATA_DIR,
    )
    for release in release_list
]
print("Datasets loaded")
sub_rm = ["NDARWV769JM7"]
######################################################################
# Combine the datasets into a single one
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here, we combine the datasets from the different releases into a single
# ``BaseConcatDataset`` object.
# %%
all_datasets = BaseConcatDataset(all_datasets_list)
print(all_datasets.description)
raws = Parallel(n_jobs=os.cpu_count())(
    delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
)
######################################################################
# Inspect your data
# -----------------
# We can check what is inside the dataset consuming the
# MNE-object inside the Braindecode dataset.
#
# The following snippet, if uncommented, will show the first 10 seconds of the raw EEG signal.
# We can also inspect the data further by looking at the events and annotations.
# We strongly recommend you to take a look into the details and check how the events are structured.
######################################################################
raw = all_datasets.datasets[0].raw  # mne.io.Raw object
print(raw.info)
raw.plot(duration=10, scalings="auto", show=True)
print(raw.annotations)
SFREQ = 100
######################################################################
# Wrap the data into a PyTorch-compatible dataset
# --------------------------------------------------
# The class below defines a dataset wrapper that will extract 2-second windows,
# uniformly sampled over the whole signal. In addition, it will add useful information
# about the extracted windows, such as the p-factor, the subject or the task.


# %%
class DatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]
        # P-factor label:
        p_factor = self.dataset.description["p_factor"]
        p_factor = float(p_factor)
        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
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


# We filter out certain recordings, create fixed length windows and finally make use of our `DatasetWrapper`.
# %%
# Filter out recordings that are too short or missing p_factor
all_datasets = BaseConcatDataset(
    [
        ds
        for ds in all_datasets.datasets
        if ds.description.subject not in sub_rm
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and "p_factor" in ds.description
        and ds.description["p_factor"] is not None
        and not math.isnan(ds.description["p_factor"])
    ]
)
# Create 4-seconds windows with 2-seconds stride
windows_ds = create_fixed_length_windows(
    all_datasets,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)
# Wrap each sub-dataset in the windows_ds
windows_ds = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
)

######################################################################
# Inspect the label distribution
# -------------------------------
#
import numpy as np
from skorch.helper import SliceDataset

y_label = np.array(list(SliceDataset(windows_ds, 1)))
# Plot histogram of the response times with matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_label)
ax.set_title("Response Time Distribution")
ax.set_xlabel("Response Time (s)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
######################################################################
# Define, train and save a model
# ---------------------------------
# Now we have our pytorch dataset necessary for the training!
#
# Below, we define a simple EEGNeX model from Braindecode.
# All the braindecode models expect the input to be of shape (batch_size, n_channels, n_times)
# and have a test coverage about the behavior of the model.
# However, you can use any pytorch model you want.
#
######################################################################
# Initialize model
# -----------------
model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * SFREQ).to(device)
# Specify optimizer
optimizer = optim.Adamax(params=model.parameters(), lr=0.002)
print(model)

# Finally, we can train our model. Here we define a simple training loop using pure PyTorch.
# In this example, we only train for a single epoch. Feel free to increase the number of epochs.
# Create PyTorch Dataloader
num_workers = (
    0  # Set num_workers to 0 to avoid multiprocessing issues in notebooks/tutorials.
)
dataloader = DataLoader(
    windows_ds, batch_size=128, shuffle=True, num_workers=num_workers
)
n_epochs = 1
# Train model for 1 epoch
for epoch in range(n_epochs):
    for idx, batch in enumerate(dataloader):
        # Reset gradients
        optimizer.zero_grad()
        # Unpack the batch
        X, y, crop_inds, infos = batch
        X = X.to(dtype=torch.float32, device=device)
        y = y.to(dtype=torch.float32, device=device).unsqueeze(1)
        # Forward pass
        y_pred = model(X)
        # Compute loss
        loss = l1_loss(y_pred, y)
        print(f"Epoch {0} - step {idx}, loss: {loss.item()}")
        # Gradient backpropagation
        loss.backward()
        optimizer.step()
# Finally, we can save the model for later use
torch.save(model.state_dict(), "weights_challenge_2.pt")
print("Model saved as 'weights_challenge_2.pt'")
