"""# EEG Classification Tutorial: P3 Visual Oddball Task

This tutorial demonstrates using the *EEGDash* library with PyTorch to classify EEG responses from a visual P3 oddball paradigm.

1. **Data Description**: Dataset contains EEG recordings during a visual oddball task where:

   - Letters A, B, C, D, and E were presented randomly (p = .2 for each)
   - One letter was designated as target (oddball) for each block
   - Other letters served as non-targets (standard)
   - Participants responded whether each letter was target or non-target

2. **Data Preprocessing**:

   - Applies bandpass filtering (1-55 Hz)
   - Selects first 30 EEG channels
   - Downsamples from 1024Hz to 256Hz
   - Creates event-based windows (0.1s to 0.6s post-stimulus)

3. **Dataset Preparation**:

   - Maps events into two classes:
     * oddball: events where block target matches trial stimulus (11,22,33,44,55)
     * standard: events where block target differs from trial stimulus
     * Note: Response events (201, 202) are excluded
   - Splits into training (80%) and test (20%) sets
   - Creates PyTorch DataLoaders

4. **Model**:

   - ShallowFBCSPNet architecture
   - 30 input channels, 2 output classes
   - 128-sample input windows (0.5s at 256Hz)

5. **Training**:

   - Adamax optimizer with learning rate decay
   - 5 training epochs
   - Reports accuracy on train and test sets
"""
# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# The P3 Visual Oddball dataset is stored in BIDS format. We use EEGDash to load and manage the data efficiently.

# %%
from eegdash.data_utils import EEGBIDSDataset

dataset = EEGBIDSDataset(
    data_dir="d:/Users/vivian/Desktop/UCSD/EEG/P3 Raw Data BIDS-Compatible",
    dataset="P3 Raw Data BIDS-Compatible",
)

all_files = dataset.get_files()

# Select files from subject-001
subject_files = [f for f in all_files if "sub-001" in f]
print("\nSelected files from subject-001:")
for i, file in enumerate(subject_files):
    print(f"{i + 1}. {file}")

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
#    - Reading events from events.tsv file:
#      * Block A: 11=oddball, 12-15=standard
#      * Block B: 22=oddball, 21,23-25=standard
#      * Block C: 33=oddball, 31-32,34-35=standard
#      * Block D: 44=oddball, 41-43,45=standard
#      * Block E: 55=oddball, 51-54=standard
#      * Response events (201, 202) are excluded
#
# 3. **Window Creation**:
#
#    - Window duration: 1s
#    - Efficient memory usage with on-demand loading

import logging
import warnings

import mne
import numpy as np
from mne.io import read_raw_eeglab

from braindecode.datasets import BaseConcatDataset, BaseDataset

# %%
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

mne.set_log_level("ERROR")
logging.getLogger("joblib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class P3OddballPreprocessor(Preprocessor):
    """A preprocessor that combines channel selection, filtering, and event mapping
    for P3 oddball paradigm EEG data.

    Maps events based on block target and trial stimulus:
        Block A: 11=oddball, 12-15=standard
        Block B: 22=oddball, 21,23-25=standard
        Block C: 33=oddball, 31-32,34-35=standard
        Block D: 44=oddball, 41-43,45=standard
        Block E: 55=oddball, 51-54=standard
        Response events (201, 202) are excluded
    """

    def __init__(self):
        super().__init__(fn=self.transform, apply_on_array=False)

    def transform(self, raw):
        """Transform the raw data by selecting channels and mapping events."""
        # Filter for stimulus events only
        events, _ = mne.events_from_annotations(raw)

        # Define oddball events (11,22,33,44,55)
        oddball_codes = np.array([11, 22, 33, 44, 55])

        # Map events: oddball=1, standard=0
        oddball_mask = np.isin(events[:, 2], oddball_codes)
        events[oddball_mask, 2] = 1
        events[~oddball_mask, 2] = 0

        # Print event counts for verification
        oddball_count = np.sum(oddball_mask)
        standard_count = len(events) - oddball_count
        print("\nEvent distribution:")
        print(f"Oddball events (11,22,33,44,55): {oddball_count}")
        print(f"Standard events: {standard_count}")

        # Create annotations from events
        annot_from_events = mne.annotations_from_events(
            events=events,
            event_desc={0: "standard", 1: "oddball"},
            sfreq=raw.info["sfreq"],
        )
        raw.set_annotations(annot_from_events)

        return raw


# Create dataset from all files
all_datasets = [
    BaseDataset(read_raw_eeglab(f, preload=False), target_name=None)
    for f in subject_files
]
dataset_concat = BaseConcatDataset(all_datasets)

# BrainDecode preprocessors
preprocessors = [
    P3OddballPreprocessor(),
    Preprocessor(
        "pick_channels",
        ch_names=read_raw_eeglab(subject_files[0], preload=False).ch_names[:30],
    ),
    Preprocessor("resample", sfreq=256),
    Preprocessor("filter", l_freq=1, h_freq=55),
]

preprocess(dataset_concat, preprocessors)

# Extract windows
windows_ds = create_windows_from_events(
    dataset_concat,
    trial_start_offset_samples=26,
    trial_stop_offset_samples=154,
    preload=False,
    window_size_samples=None,
    window_stride_samples=None,
    drop_bad_windows=True,
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
epochs = 5

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
