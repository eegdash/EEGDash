""".. _tutorial-eoec:

Using the EEGPrep preprocessor
==============================

EEGDash example for using the EEGPrep preprocessor.

This example builds on the eyes-open/eyes-closed tutorial (tutorial_eoec.py) and
introduces the EEGPrep processor. It is assumed that readers have first familiarized
themselves with and are able to run the basic eo/ec tutorial.

The conventional approach in deep learning is to perform only minimal preprocessing
and leave most processing to the deep model itself. Given the right architecture, this
can work; however, in many cases, the preprocessing stage can perform adaptive
computations that are not easy to replicate in the deep-learning stage. Such advanced
preprocessing can go hand in hand with simpler DL pipelines and result in an overall
more robust, better generalizing EEG decoding approach. EEGPrep
(https://github.com/sccn/eegprep) is a sister project of EEGDash, and is exposed as a
braindecode preprocessing step.

The following tutorial code shows how to insert EEGPrep into your preprocessing chain.

"""

# NOTE: this tutorial depends on an update to `braindecode` that is not yet in the master
# branch as of this writing, so you won't be able to run this just yet.


# %%
# Data Retrieval Using EEGDash
# ----------------------------
#
# Here we instantiate an :class:`eegdash.api.EEGDashDataset` to fetch
# the metadata for the experiment before requesting any recordings; this is the
# same as in the base tutorial.

from time import time as now
from pathlib import Path

cache_folder = Path.home() / "eegdash"
# %%
from eegdash import EEGDashDataset

ds_eoec = EEGDashDataset(
    query={"dataset": "ds005514", "task": "RestingState", "subject": "NDARDB033FW5"},
    cache_dir=cache_folder,
)

# %%
# Data Preprocessing Using Braindecode and EEGPrep
# ------------------------------------------------
#
# Here we adapt the basic tutorial's preprocessing chain by inserting EEGPrep.
# To use EEGPrep, import it from braindecode.preprocessing. You will have to have
# a recent version of both braindecode and eegprep installed; if any of the features
# appear to be missing, consider installing from the respective git repositories
# using the syntax `[uv] pip install git+https://github.com/user/repo.git@branch-or-tag`
# where the uv part only applies if you created your environment using uv.
#
# The preprocessing chain is structured as follows:
# 1. Event marker curation. This step is identical to the basic EO/EC tutorial.
# 2. EEGPrep usage. Using EEGPrep in its default configuration is as simple as
#    listing the EEGPrep() object in your chain. Here we show how to perform a few
#    minimal customizations, but these are all optional.
# 3. Channel selection. This is also identical to the base tutorial.
# 4. Bandpass filtering. Also unchanged from the base tutorial.

# If you run this, you should see that EO/EC classification accuracy improved from
# ca. 0.56 to ca. 0.86, a considerable improvement. However, note that this tutorial
# trains and tests on different portions of the same session, which is not the ideal setup
# for using preprocessing such as EEGPrep due to a mild risk of train/test information
# leakage, and this might explain some of the improvment. Instead the main use case of
# EEGPrep is in between-session transfer.
#
# For additional details, review the class documentation of the EEGPrep object. A few
# additional tips apply when using EEGPrep:
# - EEGPrep by default accepts and outputs continuous (i.e., non-segmented) EEG data;
#   therefore this should be one of the earliest, and often the first step in your
#   preprocessing chain
# - EEGprep works best when operating on all EEG channels; therefore it is recommended
#   to perform channel selection only after EEGPrep, as done in this example
# - likewise, it is recommended to perform bandpass filtering only after EEGPrep, since
#   the method works best when operating on all-frequency content, which helps identify
#   artifacts.
# - EEGPrep is highly configurable and in particular you can disable one or more of the
#   processing stages. One of the stages is removal of time segments that could not be
#   cleaned of artifacts using other means. Removal of these time windows will logically
#   imply your training and/or test datasets may be reduced. This can have a range of
#   repercussions that are important to be aware of:
#   * your training set may be smaller than otherwise
#   * your test set may be missing some "hard" data that is full of artifacts, and your
#     test-set performance may as a result look better than it otherwise would
#   * the balance of your classes may be different, especially when one of the classes
#     is more frequently associated with artifacts; this can cause your model to be
#     biased and can in extreme cases raise you chance-level (random guessing)
#     performance
#   For these reasons, removal of bad time windows should be carefully considered,
#   and can be disabled by passing in a bad_window_max_bad_channels=None argument;
#   this is done in the example below.

# %%
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    EEGPrep,
    create_windows_from_events,
)
import numpy as np
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# this toggle allows you to test two different ways of using EEGPrep
use_end2end_pipeline = False  # True or False

if use_end2end_pipeline:
    # the default use of EEGPrep is just instantiating the class and using it in your
    # processing chain, but we may tailor the operation to our needs; below are a
    # few common adjustments; all are optional
    eegprep_preprocessors = [
        EEGPrep(
            # ensure we have a consistent and sensible sampling rate for downstream
            # processing and modeling
            resample_to=128,
            # since we're not analyzing low-frequency activity we can use a less tight
            # highpass filter here
            highpass_frequencies=(0.5, 1.0),
            # disable removal of bad windows by default since this is usually what's
            # desired in a single-trial decoding scenario
            bad_window_max_bad_channels=None,

            # for completeness, we show here a few other options that are frequently
            # tuned, but which we are here leaving at their defaults:
            # burst_removal_cutoff=15.0,       # 15.0 -> less aggressive burst removal
            # bad_channel_corr_threshold=0.75, # 0.75 -> less aggressive channel removal
        ),
    ]
else:
    # alternatively, we can customize the EEGPrep pipeline by constructing the pipeline
    # stages individually, as below; this gives you more control over detailed parameters
    # that may not all be exposed in the EEGPrep class. Please refer to the EEGPrep
    # class documentation as well as that of the individual preprocessors for details.
    from braindecode.preprocessing.eegprep_preprocess import *

    # the below is the canonical order of these pipeline stages; we highly recommend
    # to *not* reorder these stages, although you may drop/add stages as needed.
    # If you want bit-identical results to EEGPrep (up to float32/64 discrepancies),
    # make sure you use functions from the eegprep_preprocess module rather than
    # MNE analogs; the following is configured to be a near-identical analog of the
    # above end-to-end pipeline usage.
    eegprep_preprocessors = [
        RemoveDCOffset(),
        # note: to preserve non-EEG channels, you would need to use the MNE variant atm:
        Resampling(128),  # OR Preprocessor("resample", sfreq=128), for the MNE variant
        RemoveFlatChannels(),
        RemoveDrifts((0.5, 1.0)),
        RemoveBadChannels(),  # OR RemoveBadChannelsNoLocs() if chn locations missing
        RemoveBursts(),
        # you may enable this if you understand the implications
        # RemoveBadWindows(),
        ReinterpolateRemovedChannels(),
        RemoveCommonAverageReference(),  # OR Preprocessor("set_eeg_reference", ref_channels='average')
    ]

# BrainDecode preprocessors
preprocessors = [
    hbn_ec_ec_reannotation(),
    *eegprep_preprocessors,
    Preprocessor(
        "pick_channels",
        ch_names=[
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
        ],
    ),
    Preprocessor("filter", l_freq=1, h_freq=55),
]
preprocess(ds_eoec, preprocessors)

# Extract 2-second segments
windows_ds = create_windows_from_events(
    ds_eoec,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=256,
    preload=True,
)


# %%
# Creating training and test sets
# -------------------------------
#
# The code below creates a training and test set. We first split the data into
# training and test sets using the **train_test_split** function from the
# **sklearn** library. We then create a **TensorDataset** for the training and
# test sets.
#
# 1. **Set Random Seed** – The random seed is fixed using
#    `torch.manual_seed(random_state)` to ensure reproducibility in dataset
#    splitting and model training.
# 2. **Extract Labels from the Dataset** – Labels (eye-open or eye-closed
#    events) are extracted from `windows_ds`, stored as a NumPy array, and
#    printed for verification.
# 3. **Split Dataset into Train and Test Sets** – The dataset is split into
#    training (80%) and testing (20%) subsets using `train_test_split()`,
#    ensuring balanced stratification based on the extracted labels.
# 4. **Convert Data to PyTorch Tensors** – The selected training and testing
#    samples are converted into `FloatTensor` for input features and
#    `LongTensor` for labels, making them compatible with PyTorch models.
# 5. **Create DataLoaders** – The datasets are wrapped in PyTorch DataLoader
#    objects with a batch size of 10, enabling efficient mini-batch training and
#    shuffling.
#

# %%
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

# Extract labels from the dataset
eo_ec = np.array([ds[1] for ds in windows_ds]).transpose()  # check labels
print("labels: ", eo_ec)

# Get balanced indices for male and female subjects
train_indices, test_indices = train_test_split(
    range(len(windows_ds)), test_size=0.2, stratify=eo_ec, random_state=random_state
)

# Convert the data to tensors
X_train = torch.FloatTensor(
    np.array([windows_ds[i][0] for i in train_indices])
)  # Convert list of arrays to single tensor
X_test = torch.FloatTensor(
    np.array([windows_ds[i][0] for i in test_indices])
)  # Convert list of arrays to single tensor
y_train = torch.LongTensor(eo_ec[train_indices])  # Convert targets to tensor
y_test = torch.LongTensor(eo_ec[test_indices])  # Convert targets to tensor
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

# Create data loaders for training and testing (batch size 10)
train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)

# Print shapes and sizes to verify split
print(
    f"Shape of data {X_train.shape} number of samples - Train: {len(train_loader)}, Test: {len(test_loader)}"
)
print(
    f"Eyes-Open/Eyes-Closed balance, train: {np.mean(eo_ec[train_indices]):.2f}, test: {np.mean(eo_ec[test_indices]):.2f}"
)


# %%
# Create model
# ------------
#
# The model is a shallow convolutional neural network (ShallowFBCSPNet) with 24
# input channels (EEG channels), 2 output classes (eyes-open and eyes-closed),
# and an input window size of 256 samples (2 seconds of EEG data).

# %%
import torch
import numpy as np
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

torch.manual_seed(random_state)
model = ShallowFBCSPNet(24, 2, n_times=256, final_conv_length="auto")
summary(model, input_size=(1, 24, 256))

# %%
# Model Training and Evaluation Process
# -------------------------------------
#
# This section trains the neural network using the Adamax optimizer, normalizes
# input data, computes cross-entropy loss, updates model parameters, and tracks
# accuracy across six epochs.
#
# 1. **Set Up Optimizer and Learning Rate Scheduler** – The `Adamax` optimizer
#    initializes with a learning rate of 0.002 and weight decay of 0.001 for
#    regularization. An `ExponentialLR` scheduler with a decay factor of 1 keeps
#    the learning rate constant.
# 2. **Allocate Model to Device** – The model moves to the specified device
#    (CPU, GPU, or MPS for Mac silicon) to optimize computation efficiency.
# 3. **Normalize Input Data** – The `normalize_data` function standardizes input
#    data by subtracting the mean and dividing by the standard deviation along
#    the time dimension before transferring it to the appropriate device.
# 4. **Evaluates Classification Accuracy Over Six Epochs** – The training loop
#    iterates through data batches with the model in training mode. It
#    normalizes inputs, computes predictions, calculates cross-entropy loss,
#    performs backpropagation, updates model parameters, and steps the learning
#    rate scheduler. It tracks correct predictions to compute accuracy.
# 5. **Evaluate on Test Data** – After each epoch, the model runs in evaluation
#    mode on the test set. It computes predictions on normalized data and
#    calculates test accuracy by comparing outputs with actual labels.

# %%
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device=device)  # move the model parameters to CPU/GPU
epochs = 6


def normalize_data(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
    x = (x - mean) / std
    x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
    return x


for e in range(epochs):
    # training
    correct_train = 0
    for t, (x, y) in enumerate(train_loader):
        model.train()  # put model to training mode
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)
        _, preds = scores.max(1)
        correct_train += (preds == y).sum() / len(dataset_train)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    correct_test = 0
    for t, (x, y) in enumerate(test_loader):
        model.eval()  # put model to testing mode
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)
        _, preds = scores.max(1)
        correct_test += (preds == y).sum() / len(dataset_test)

    # Reporting
    print(
        f"Epoch {e}, Train accuracy: {correct_train:.2f}, Test accuracy: {correct_test:.2f}"
    )
