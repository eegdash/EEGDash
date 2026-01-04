# %% [markdown]
""".. _auditory-oddball-tutorial:

Oddball Classification
=======================

This tutorial demonstrates using the *EEGDash* library with PyTorch to classify EEG responses in an oddball paradigm.

1. **Data Description**: Dataset contains EEG recordings during an oddball task with two stimulus types:
   - Standard (non-target)
   - Oddball (target)

2. **Data Preprocessing**:
   - Applies bandpass filtering (1-55 Hz)
   - Selects all 64 EEG channels
   - Creates event-based windows
   - Processes data in batches for memory efficiency

3. **Dataset Preparation**:
   - Remaps events into two classes: oddball, standard
   - Splits into training (80%) and test (20%) sets
   - Creates PyTorch DataLoaders

4. **Model**:
   - ShallowFBCSPNet architecture
   - 64 input channels, 2 output classes
   - 256-sample input windows

5. **Training**:
   - Adamax optimizer with learning rate decay
   - A few training epochs (configurable)
   - Reports accuracy on train and test sets
"""

# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# Data retrieved via the EEGDash API. Use EEGDASH_DATASET_ID/EEGDASH_TASK
# to override the defaults.

# %%
from pathlib import Path
import os


from eegdash import EEGDash, EEGDashDataset
from eegdash.paths import get_default_cache_dir

CACHE_DIR = Path(get_default_cache_dir()).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ID = "ds005863"
TASK = "visualoddball"
RECORD_LIMIT = 20

eegdash = EEGDash()
records = eegdash.find({"dataset": DATASET_ID, "task": TASK}, limit=RECORD_LIMIT)
if not records:
    records = eegdash.find(
        {"task": {"$regex": "oddball", "$options": "i"}}, limit=RECORD_LIMIT
    )
if records:
    dataset_id = records[0].get("dataset")
    if dataset_id:
        records = [rec for rec in records if rec.get("dataset") == dataset_id]
if not records:
    raise RuntimeError("No oddball task records found from the API.")

dataset_concat = EEGDashDataset(cache_dir=CACHE_DIR, records=records)

# %% [markdown]
# ## Data Preprocessing Using Braindecode
#
# [Braindecode](https://braindecode.org/) provides a powerful framework for EEG data preprocessing and analysis.
#
# We apply three preprocessing steps in Braindecode:
#
# 1.**Event Remapping** using event markers to convert:
#   - 3,4 → oddball (0)
#   - 6,7 → standard (1)
#
# 2.**Channel Selection & Filtering**:
#   - Selecting first 64 EEG channels
#   - Bandpass filtering between 1 Hz and 55 Hz
#
# When calling the **preprocess** function, the data is retrieved from the files.
#
# Finally, we use **create_windows_from_events** to extract windows centered on events (-128 to +128 samples around each event).

import logging
import warnings

import numpy as np
import mne

# %%
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

mne.set_log_level("ERROR")
logging.getLogger("joblib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# BrainDecode preprocessors
preprocessors = [
    Preprocessor(
        "pick_channels", ch_names=dataset_concat.datasets[0].raw.ch_names[:64]
    ),
    Preprocessor("resample", sfreq=128),
    Preprocessor("filter", l_freq=1, h_freq=55),
]
preprocess(dataset_concat, preprocessors)

# Extract windows
event_mapping = {
    "Target": 1,
    "NonTarget": 0,
    "target": 1,
    "standard": 0,
    "oddball": 1,
    "3": 1,
    "4": 1,
    "6": 0,
    "7": 0,
}

windows_ds = create_windows_from_events(
    dataset_concat,
    trial_start_offset_samples=-128,
    trial_stop_offset_samples=128,
    preload=False,
    drop_bad_windows=True,
    mapping=event_mapping,
)

print(f"\nAll files processed, total number of windows: {len(windows_ds)}")
print(f"Window shape: {windows_ds[0][0].shape}")

# %% [markdown]
# ## Creating training and test sets
#
# The data preparation pipeline consists of these key steps:
#
# 1. **Dataset Creation** - The processed windows are automatically labeled (0=oddball, 1=standard) by the OddballPreprocessor using efficient array operations.
#
# 2. **Train-Test Split** - Using sklearn's train_test_split with 80-20 split and stratified sampling.
#
# 3. **PyTorch Data Preparation** - Converting to tensors and creating DataLoader objects for mini-batch training.

# %%
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

# Extract data and labels using array operations
data = np.stack([windows_ds[i][0] for i in range(len(windows_ds))])
labels = np.array([windows_ds[i][1] for i in range(len(windows_ds))])

# Print dataset information
print(f"Dataset size: {len(data)}")
print(f"Data shape: {data.shape}")
print("Distribution of labels:", np.unique(labels, return_counts=True))
print("Label meanings: 0=oddball, 1=standard")

# Split into train and test sets
train_indices, test_indices = train_test_split(
    range(len(data)), test_size=0.2, stratify=labels, random_state=random_state
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(data[train_indices])
X_test = torch.FloatTensor(data[test_indices])
y_train = torch.LongTensor(labels[train_indices])
y_test = torch.LongTensor(labels[test_indices])

# Create data loaders
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)

# Print dataset information
print("\nDataset size:")
print(f"Training set: {X_train.shape}, labels: {y_train.shape}")
print(f"Test set: {X_test.shape}, labels: {y_test.shape}")
print("\nProportion of samples of each class in training set:")
for label in np.unique(labels):
    ratio = np.mean(y_train.numpy() == label)
    print(f"Category {label}: {ratio:.3f}")

# %% [markdown]
# # Create model
#
# The model is a shallow convolutional neural network (ShallowFBCSPNet) with 64 input channels (EEG channels), 2 output classes (oddball, standard), and an input window size of 256 samples (1 seconds of EEG data).

from torchinfo import summary

# %%
from braindecode.models import ShallowFBCSPNet

model = ShallowFBCSPNet(
    in_chans=64, n_classes=2, input_window_samples=256, final_conv_length="auto"
)

summary(model, input_size=(1, 64, 256))

# %% [markdown]
# ## Model Training and Evaluation Process
#
# The training and evaluation pipeline runs for 5 epochs using Adamax optimization. Key components include:
#
# 1. **Hardware Setup** - Model allocation to CPU/GPU for optimal computation.
#
# 2. **Data Processing** - Channel-wise normalization of input data using mean and standard deviation.
#
# 3. **Training Process** - Each epoch performs forward passes, computes cross-entropy loss, updates parameters, and tracks accuracy.
#
# 4. **Evaluation** - Model performance is assessed on the test set after each training epoch.
#
# The process monitors both training and test accuracy to track model learning progress.

# %%
# Set up device, optimizer, and learning rate scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)


def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7
    x = (x - mean) / std
    x = x.to(device=device, dtype=torch.float32)
    return x


print("\nStart training...")
epochs = int(os.getenv("EEGDASH_EPOCHS", "2"))

for e in range(epochs):
    model.train()
    correct_train = 0
    for t, (x, y) in enumerate(train_loader):
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)
        _, preds = scores.max(1)
        correct_train += (preds == y).sum() / len(dataset_train)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    correct_test = 0
    with torch.no_grad():
        for t, (x, y) in enumerate(test_loader):
            scores = model(normalize_data(x))
            y = y.to(device=device, dtype=torch.long)
            _, preds = scores.max(1)
            correct_test += (preds == y).sum() / len(dataset_test)

    print(
        f"epoch {e + 1}, training accuracy: {correct_train:.3f}, test accuracy: {correct_test:.3f}"
    )
