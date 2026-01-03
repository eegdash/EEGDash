# %% [markdown]
""".. _p3-visual-oddball:

P3 Visual Oddball Classification (API-driven)
=============================================

This tutorial demonstrates using the *EEGDash* library with PyTorch to classify EEG responses from a visual P3 oddball paradigm.

1. **Data Description**: Dataset contains EEG recordings during a visual oddball task where:

   - Letters A, B, C, D, and E were presented randomly (p = .2 for each)
   - One letter was designated as target (oddball) for each block
   - Other letters served as non-targets (standard)
   - Participants responded whether each letter was target or non-target

2. **Data Preprocessing**:

   - Applies bandpass filtering (1-55 Hz)
   - Selects first 30 EEG channels
   - Downsamples to 256Hz
   - Creates event-based windows (0.1s to 0.6s post-stimulus)

3. **Dataset Preparation**:

   - Maps events into two classes (target vs. standard) using annotation names
   - Splits into training (80%) and test (20%) sets
   - Creates PyTorch DataLoaders

4. **Model**:

   - ShallowFBCSPNet architecture
   - 30 input channels, 2 output classes
   - 128-sample input windows (0.5s at 256Hz)

5. **Training**:

   - Adamax optimizer with learning rate decay
   - A few training epochs (configurable)
   - Reports accuracy on train and test sets
"""
# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# The P3 oddball dataset is fetched from the EEGDash API.

# %%
from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from eegdash import EEGDash, EEGDashDataset

CACHE_DIR = Path(os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ID = os.getenv("EEGDASH_DATASET_ID", "ds005863")
TASK = os.getenv("EEGDASH_TASK", "visualoddball")
RECORD_LIMIT = int(os.getenv("EEGDASH_RECORD_LIMIT", "20"))

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
# [Braindecode](https://braindecode.org/) provides powerful tools for EEG data preprocessing and analysis. Our implementation processes EEG data with these key steps:
#
# 1. **Channel Selection & Signal Processing**:
#    - Selecting first 30 EEG channels
#    - Bandpass filtering between 1-55 Hz
#    - Downsampling from 1024Hz to 256Hz
#
# 2. **Event Processing**:
#    - Map target vs. standard events based on annotation labels (e.g., Target/NonTarget).
#    - Response-only events are ignored by the mapping.
#
# 3. **Window Creation**:
#
#    - Window duration: 1s
#    - Efficient memory usage with on-demand loading

import logging
import warnings

import mne
import numpy as np

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
        "pick_channels", ch_names=dataset_concat.datasets[0].raw.ch_names[:30]
    ),
    Preprocessor("resample", sfreq=256),
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
    trial_start_offset_samples=26,
    trial_stop_offset_samples=154,
    preload=False,
    window_size_samples=None,
    window_stride_samples=None,
    drop_bad_windows=True,
    mapping=event_mapping,
)

print(f"\nAll files processed, total number of windows: {len(windows_ds)}")
print(f"Window shape: {windows_ds[0][0].shape}")

# %% [markdown]
# ## Creating Training and Test Sets
#
# The data preparation pipeline consists of these key steps:
#
# 1. **Data Extraction** - Windows are automatically labeled (0=standard, 1=oddball) by the P3OddballPreprocessor.
#
# 2. **Train-Test Split** - Using sklearn's train_test_split with:
#    - 80-20 split ratio
#    - Stratified sampling to maintain class proportions
#    - Fixed random seed for reproducibility
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
print("Label meanings: 0=standard, 1=oddball")

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
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=True)

# Print dataset information
print("\nDataset size:")
print(f"Training set: {X_train.shape}, labels: {y_train.shape}")
print(f"Test set: {X_test.shape}, labels: {y_test.shape}")
print("\nProportion of samples of each class in training set:")
for label in np.unique(labels):
    ratio = np.mean(y_train.numpy() == label)
    print(f"Category {label}: {ratio:.3f}")

# %% [markdown]
# ## Create Model
#
# The model is a shallow convolutional neural network (ShallowFBCSPNet) with:
# - 30 input channels (EEG channels)
# - 2 output classes (oddball, standard)
# - 128-sample input windows (0.5s at 256Hz)
#
# This architecture is particularly effective for EEG classification tasks, incorporating frequency-band specific spatial patterns.

from torchinfo import summary

# %%
from braindecode.models import ShallowFBCSPNet

model = ShallowFBCSPNet(
    in_chans=30,
    n_classes=2,
    input_window_samples=128,  # 0.5s at 256Hz
    final_conv_length="auto",
)

summary(model, input_size=(1, 30, 128))

# %% [markdown]
# ## Model Training and Evaluation
#
# The training pipeline consists of:
#
# 1. **Optimization Setup**:
#    - Adamax optimizer with learning rate 0.002
#    - Weight decay for regularization
#    - Learning rate scheduler
#
# 2. **Training Process**:
#    - 5 epochs of training
#    - Mini-batch processing
#    - Cross-entropy loss function
#
# 3. **Evaluation**:
#    - Accuracy tracking for both training and test sets
#    - Batch normalization applied to input data

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.005)
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
