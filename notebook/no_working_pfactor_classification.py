# %% [markdown]
""".. _pfactor-classification:

P-factor Regression Example
===========================

A minimal EEGDash + Braindecode pipeline for predicting the p-factor.
"""

from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from eegdash import EEGDash, EEGDashDataset
import matplotlib.pyplot as plt

CACHE_DIR = Path(os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ID = os.getenv("EEGDASH_DATASET_ID", "EEG2025r5")
TASK = os.getenv("EEGDASH_TASK", "").strip() or None
RECORD_LIMIT = int(os.getenv("EEGDASH_RECORD_LIMIT", "80"))
WINDOW_DIR = CACHE_DIR / "pfactor_windows"

eegdash = EEGDash()
query = {"dataset": DATASET_ID, "p_factor": {"$ne": None}}
if TASK:
    query["task"] = TASK
records = eegdash.find(query, limit=RECORD_LIMIT)
if not records:
    records = eegdash.find({"p_factor": {"$ne": None}}, limit=RECORD_LIMIT)
if records:
    dataset_id = records[0].get("dataset")
    if dataset_id:
        records = [rec for rec in records if rec.get("dataset") == dataset_id]
if not records:
    raise RuntimeError("No records with p_factor metadata found from the API.")

raw_ds = EEGDashDataset(
    cache_dir=CACHE_DIR,
    records=records,
    description_fields=[
        "subject",
        "session",
        "run",
        "task",
        "age",
        "sex",
        "gender",
        "p_factor",
    ],
)

filtered = []
for ds in raw_ds.datasets:
    val = ds.description.get("p_factor")
    try:
        if pd.isna(val):
            continue
        float(val)
    except (TypeError, ValueError):
        continue
    filtered.append(ds)
raw_ds = BaseConcatDataset(filtered)

preprocessors = [
    Preprocessor("resample", sfreq=256),
    Preprocessor("filter", l_freq=1, h_freq=55),
]
for ds in raw_ds.datasets:
    ds.target_name = "p_factor"

preprocess(raw_ds, preprocessors, n_jobs=-1)

windows_ds = create_fixed_length_windows(
    raw_ds,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=256,
    window_stride_samples=256,
    drop_last_window=True,
    preload=False,
)
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
windows_ds.save(str(WINDOW_DIR), overwrite=True)

print(
    f"Created {len(windows_ds)} windows across {len(windows_ds.datasets)} recordings."
)

plt.figure()
plt.plot(windows_ds[0][0][0, :].transpose())
plt.show()

# %% [markdown]
# ## Load pre-saved data
#
# If you have run the previous steps before, the data should be saved and may be reloaded here. If you are simply running this notebook for the first time, there is no need to reload the data, and this step may be skipped. However, it is quick, so you might as well execute the cell; it will have no consequences and will allow you to check that the data was saved properly.

# %%
from braindecode.datautil import load_concat_dataset

print("Loading data from disk")
windows_ds = load_concat_dataset(
    path="data/hbn_preprocessed_restingstate_256", preload=False
)

# %%
windows_ds[1000][0].shape

# %% [markdown]
# ## Creating a Training and Test Set
#
# The code below creates a training and test set. We first split the data using the **train_test_split** function and then create a **TensorDataset** for both sets.
#
# 1. **Set Random Seed** – The random seed is fixed using `torch.manual_seed(random_state)` to ensure reproducibility in dataset splitting and model training.
# 2. **Get Balanced Indices for Male and Female Subjects** – We ensure a 50/50 split of male and female subjects in both the training and test sets. Additionally, we prevent subject leakage, meaning the same subjects do not appear in both sets. The dataset is split into training (90%) and testing (10%) subsets using `train_test_split()`, ensuring balanced stratification based on gender.
# 3. **Convert Data to PyTorch Tensors** – The selected training and testing samples are converted into `FloatTensor` for input features and `LongTensor` for labels, making them compatible with PyTorch models.
# 4. **Create DataLoaders** – The datasets are wrapped in PyTorch `DataLoader` objects with a batch size of 100, allowing efficient mini-batch training and shuffling. Although there are only 136 subjects, the dataset contains more than 10,000 2-second samples.
#

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# %%
from braindecode.datasets import BaseConcatDataset

# random seed for reproducibility
random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)

# Get balanced indices for male and female subjects and create a balanced dataset
male_subjects = windows_ds.description["subject"][windows_ds.description["sex"] == "M"]
female_subjects = windows_ds.description["subject"][
    windows_ds.description["sex"] == "F"
]
n_samples = min(len(male_subjects), len(female_subjects))
balanced_subjects = np.concatenate(
    [male_subjects[:n_samples], female_subjects[:n_samples]]
)
balanced_gender = ["M"] * n_samples + ["F"] * n_samples
train_subj, val_subj, train_gender, val_gender = train_test_split(
    balanced_subjects,
    balanced_gender,
    train_size=0.9,
    stratify=balanced_gender,
    random_state=random_state,
)

# Create datasets
train_ds = BaseConcatDataset(
    [ds for ds in windows_ds.datasets if ds.description.subject in train_subj]
)
val_ds = BaseConcatDataset(
    [ds for ds in windows_ds.datasets if ds.description.subject in val_subj]
)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=250, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=250, shuffle=True)

# Check the balance of the dataset
assert len(balanced_subjects) == len(balanced_gender)
print(f"Number of subjects in balanced dataset: {len(balanced_subjects)}")
print(
    f"Gender distribution in balanced dataset: {np.unique(balanced_gender, return_counts=True)}"
)

# %% [markdown]
# # Check labels
#
# It is good practice to verify the labels and ensure the random seed is functioning correctly. If all labels are 'M' (male) or 'F' (female), it could indicate an issue with data loading or stratification, requiring further investigation.

# %%
# get the first batch to check the labels
dataiter = iter(train_loader)
first_item, label, sz = dataiter.__next__()
np.array(label).T

# %% [markdown]
# # Create model
#
# The model is a custom convolutional neural network with 24 input channels (EEG channels), 2 output classes (male vs. female), and an input window size of 256 samples (2 seconds of EEG data). See the reference below for more information.
#
# [1] Truong, D., Milham, M., Makeig, S., & Delorme, A. (2021). Deep Convolutional Neural Network Applied to Electroencephalography: Raw Data vs Spectral Features. IEEE Engineering in Medicine and Biology Society. Annual International Conference, 2021, 1039–1042. https://doi.org/10.1109/EMBC46164.2021.9630708
#
#

from torch import nn

# %%
# create model
from torchinfo import summary

model = nn.Sequential(
    # First VGG block
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # Second VGG block
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # Third VGG block
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # Flatten and FC layers
    nn.Flatten(),
    nn.Linear(64 * 16 * 32, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 1),
)

print(summary(model, input_size=(1, 1, 129, 256)))

# %% [markdown]
# # Model Training and Evaluation Process
#
# This section trains the neural network using the Adamax optimizer, normalizes input data, computes cross-entropy loss, updates model parameters, and tracks accuracy across six epochs.
#
# 1. **Set Up Optimizer and Learning Rate Scheduler** – The `Adamax` optimizer initializes with a learning rate of 0.002 and weight decay of 0.001 for regularization.
#
# 2. **Allocate Model to Device** – The model moves to the specified device (CPU, GPU, or MPS for Mac silicon) to optimize computation efficiency.
#
# 3. **Normalize Input Data** – The `normalize_data` function standardizes input data by subtracting the mean and dividing by the standard deviation along the time dimension before transferring it to the appropriate device.
#
# 4. **Train the Model for Two Epochs** – The training loop iterates through data batches with the model in training mode. It normalizes inputs, computes predictions, calculates cross-entropy loss, performs backpropagation, updates model parameters, and steps the learning rate scheduler. It tracks correct predictions to compute accuracy.
#
# 5. **Evaluate on Test Data** – After each epoch, the model runs in evaluation mode on the test set. It computes predictions on normalized data and calculates test accuracy by comparing outputs with actual labels.
#

# %%
from torch.nn import functional as F

optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device=device)


def normalize_data(x):
    x = x.reshape(x.shape[0], 1, 129, 256)
    mean = x.mean(dim=3, keepdim=True)
    std = x.std(dim=3, keepdim=True) + 1e-7  # add small epsilon for numerical stability
    x = (x - mean) / std
    x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
    return x


# dictionary of genders for converting sample labels to numerical values
gender_dict = {"M": 0, "F": 1}

epochs = 10
for e in range(epochs):
    # training
    correct_train = 0
    for t, (x, y, sz) in enumerate(train_loader):
        model.train()  # put model to training mode
        scores = model(normalize_data(x))
        _, preds = scores.max(1)
        # y = torch.tensor([gender_dict[gender] for gender in y], device=device, dtype=torch.long)
        y = torch.tensor(y, device=device, dtype=torch.float)
        # correct_train += (preds == y).sum()/len(train_ds)
        mse_loss_train = F.mse_loss(scores.flatten(), y.float())

        # Calculates the cross-entropy loss and performs backpropagation
        loss = F.mse_loss(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 50 == 0:
            print("Epoch %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))

    # validation
    correct_test = 0
    for t, (x, y, sz) in enumerate(val_loader):
        model.eval()  # put model to testing mode
        scores = model(normalize_data(x))
        _, preds = scores.max(1)
        y = torch.tensor(y, device=device, dtype=torch.float)
        # y = torch.tensor([gender_dict[gender] for gender in y], device=device, dtype=torch.long)
        # correct_test += (preds == y).sum()/len(val_ds)
        mse_loss_test = F.mse_loss(scores.flatten(), y.float())

    # print(f'Epoch {e}, Train accuracy: {correct_train:.2f}, Test accuracy: {correct_test:.2f}\n')
    print(f"Train MSE Loss: {mse_loss_train:.4f}, Test MSE Loss: {mse_loss_test:.4f}")

# %%
