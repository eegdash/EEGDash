""".. _challenge-1:

Challenge 1: Cross-Task Transfer Learning!
==========================================

.. contents:: This example covers:
   :local:
   :depth: 2
"""

######################################################################
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_1.ipynb
#    :alt: Open In Colab

######################################################################
# How can we use the knowledge from one EEG Decoding task into another?
# ---------------------------------------------------------------------
# Transfer learning is a widespread technique used in deep learning. It
# uses knowledge learned from one source task/domain in another target
# task/domain. It has been studied in depth in computer vision, natural
# language processing, and speech, but what about EEG brain decoding?
#
# The cross-task transfer learning scenario in EEG decoding is remarkably
# underexplored compared to the development of new models,
# `Aristimunha et al. (2023) <https://arxiv.org/abs/2308.02408>`__, even
# though it can be much more useful for real applications, see
# `Wimpff et al. (2025) <https://arxiv.org/abs/2502.06828>`__,
# `Wu et al. (2025) <https://arxiv.org/abs/2507.09882>`__.
#
# Our Challenge 1 addresses a key goal in neurotechnology: decoding
# cognitive function from EEG using the pre-trained knowledge from another.
# In other words, developing models that can effectively
# transfer/adapt/adjust/fine-tune knowledge from passive EEG tasks to
# active tasks.
#
# The ability to generalize and transfer is something critical that we
# believe should be focused on. To go beyond just comparing metrics numbers
# that are often not comparable, given the specificities of EEG, such as
# pre-processing, inter-subject variability, and many other unique
# components of this type of data.
#
# This means your submitted model might be trained on a subset of tasks
# and fine-tuned on data from another condition, evaluating its capacity to
# generalize with task-specific fine-tuning.

######################################################################
# __________
#
# Note: For simplicity purposes, we will only show how to do the decoding directly in our target task, and it is up to the teams to think about how to use the passive task to perform the pre-training.

######################################################################
# Summary table for this start kit
# --------------------------------
# In this tutorial, we are going to show in more detail what we want
# from Challenge 1:
#
# **Contents**:
#
# 0. Understand the Contrast Change Detection - CCD task.
# 1. Understand the `EEGChallengeDataset <https://eeglab.org/EEGDash/api/eegdash.html#eegdash.EEGChallengeDataset>`__ object.
# 2. Preparing the dataloaders.
# 3. Building the deep learning model with `braindecode <https://braindecode.org/stable/models/models_table.html>`__.
# 4. Designing the training loop.
# 5. Training the model.
# 6. Evaluating test performance.
# 7. Going further, *benchmark go brrr!*
# **NOTE: You will still be able to run the whole notebook at your own pace and learn about these concepts along the way**
# **NOTE: If you just want run the code and start to play, please go to the challenge version 1 python, clean in the folder**

###### ----
# For the challenge, we will need two significant dependencies:
# `braindecode` and `eegdash`. The libraries will install PyTorch, Pytorch Audio, Scikit-learn, MNE, MNE-BIDS, and many other packages necessary for the many functions.
# %% Install dependencies on colab or your local machine
# !pip install braindecode
# !pip install eegdash

######################################################################
# Imports and setup
# -----------------
from pathlib import Path
import torch
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from braindecode.models import EEGNeX
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from typing import Optional
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import copy
from joblib import Parallel, delayed

######################################################################
# 0. Check GPU availability
# -------------------------
#
# Identify whether a CUDA-enabled GPU is available
# and set the device accordingly.
# If using Google Colab, ensure that the runtime is set to use a GPU.
# This can be done by navigating to `Runtime` > `Change runtime type` and selecting
# `GPU` as the hardware accelerator.
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
# 1. What are we decoding?
# ------------------------
#
# The Contrast Change Detection (CCD) task relates to
# `Steady-State Visual Evoked Potentials (SSVEP) <https://en.wikipedia.org/wiki/Steady-state_visually_evoked_potential>`__
# and `Event-Related Potentials (ERP) <https://en.wikipedia.org/wiki/Event-related_potential>`__.
#
# Algorithmically, what the subject sees during recording is:
#
# * Two flickering striped discs: one tilted left, one tilted right.
# * After a variable delay, **one disc's contrast gradually increases** **while the other decreases**.
# * They **press left or right** to indicate which disc got stronger.
# * They receive **feedback** (🙂 correct / 🙁 incorrect).
#
# **The task parallels SSVEP and ERP:**
#
# * The continuous flicker **tags the EEG at fixed frequencies (and harmonics)** → SSVEP-like signals.
# * The **ramp onset**, the **button press**, and the **feedback** are **time-locked events** that yield ERP-like components.
#
# Your task (**label**) is to predict the response time for the subject during this windows.

######################################################################
# Stimulus demonstration
# ----------------------
# .. raw:: html
#
#    <div class="video-wrapper">
#      <iframe src="https://www.youtube.com/embed/tOW2Vu2zHoU?start=1630"
#              title="Contrast Change Detection (CCD) task demo"
#              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
#              allowfullscreen></iframe>
#    </div>

######################################################################
# 2) Load the competition dataset
# -------------------------------
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    keep_only_recordings_with,
    add_extras_columns,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

dataset_ccd = EEGChallengeDataset(
    task="contrastChangeDetection", release="R5", cache_dir=DATA_DIR, mini=True
)

# For visualization purposes, we will see just one object.
raw = dataset_ccd.datasets[0].raw  # get the Raw object of the first recording

# To download all data directly, you can do:
raws = Parallel(n_jobs=-1)(delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets)

######################################################################
# 3) Create windows of interest
# -----------------------------
EPOCH_LEN_S = 2.0
SFREQ = 100  # by definition here

transformation_offline = [
    Preprocessor(
        annotate_trials_with_target,
        target_field="rt_from_stimulus",
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
        apply_on_array=False,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
]
preprocess(dataset_ccd, transformation_offline, n_jobs=1)

ANCHOR = "stimulus_anchor"
SHIFT_AFTER_STIM = 0.5
WINDOW_LEN = 2.0

# Keep only recordings that actually contain stimulus anchors
dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

# Create single-interval windows (stim-locked, long enough to include the response)
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),  # +0.5 s
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),  # +2.5 s
    window_size_samples=int(EPOCH_LEN_S * SFREQ),
    window_stride_samples=SFREQ,
    preload=True,
)

# Injecting metadata into the extra mne annotation.
single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=(
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
)

######################################################################
# 4) Split the data
# -----------------
# Extract meta information
meta_information = single_windows.get_metadata()

valid_frac = 0.1
test_frac = 0.1
seed = 2025

subjects = meta_information["subject"].unique()

train_subj, valid_test_subject = train_test_split(
    subjects,
    test_size=(valid_frac + test_frac),
    random_state=check_random_state(seed),
    shuffle=True,
)

valid_subj, test_subj = train_test_split(
    valid_test_subject,
    test_size=test_frac,
    random_state=check_random_state(seed + 1),
    shuffle=True,
)

# Sanity check
assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

# Create train/valid/test splits for the windows
subject_split = single_windows.split("subject")
train_set = []
valid_set = []
test_set = []

for s in subject_split:
    if s in train_subj:
        train_set.append(subject_split[s])
    elif s in valid_subj:
        valid_set.append(subject_split[s])
    elif s in test_subj:
        test_set.append(subject_split[s])

train_set = BaseConcatDataset(train_set)
valid_set = BaseConcatDataset(valid_set)
test_set = BaseConcatDataset(test_set)

print("Number of examples in each split in the minirelease")
print(f"Train:\t{len(train_set)}")
print(f"Valid:\t{len(valid_set)}")
print(f"Test:\t{len(test_set)}")

######################################################################
# 5) Create dataloaders
# ---------------------
batch_size = 128
# Set num_workers to 0 to avoid multiprocessing issues in notebooks/tutorials
num_workers = 0

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
valid_loader = DataLoader(
    valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

######################################################################
# 6) Build the model
# ------------------
# For any braindecode model, you can initialize only inputing the signal related parameters
# You can use any Pytorch module that you want here.
model = EEGNeX(
    n_chans=129,  # 129 channels
    n_outputs=1,  # 1 output for regression
    n_times=200,  # 2 seconds
    sfreq=100,  # sample frequency 100 Hz
)

print(model)
model.to(device)


######################################################################
# 7) Define training and validation functions
# -------------------------------------------
def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    device,
    print_batch_stats: bool = True,
):
    model.train()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), disable=not print_batch_stats
    )

    for batch_idx, batch in progress_bar:
        # Support datasets that may return (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Flatten to 1D for regression metrics and accumulate squared error
        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_description(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse


@torch.no_grad()
def valid_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn,
    device,
    print_batch_stats: bool = True,
):
    model.eval()

    total_loss = 0.0
    sum_sq_err = 0.0
    n_batches = len(dataloader)
    n_samples = 0

    iterator = tqdm(
        enumerate(dataloader), total=n_batches, disable=not print_batch_stats
    )

    for batch_idx, batch in iterator:
        # Supports (X, y) or (X, y, ...)
        X, y = batch[0], batch[1]
        X, y = X.to(device).float(), y.to(device).float()

        preds = model(X)
        batch_loss = loss_fn(preds, y).item()
        total_loss += batch_loss

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
        n_samples += y_flat.numel()

        if print_batch_stats:
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            iterator.set_description(
                f"Val Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
            )

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

    print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
    return avg_loss, rmse


######################################################################
# 8) Train the model
# ------------------
lr = 1e-3
weight_decay = 1e-5
n_epochs = (
    5  # For demonstration purposes, we use just 5 epochs here. You can increase this.
)
early_stopping_patience = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - 1)
loss_fn = torch.nn.MSELoss()

patience = 5
min_delta = 1e-4
best_rmse = float("inf")
epochs_no_improve = 0
best_state, best_epoch = None, None

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_rmse = train_one_epoch(
        train_loader, model, loss_fn, optimizer, scheduler, epoch, device
    )
    val_loss, val_rmse = valid_model(test_loader, model, loss_fn, device)

    print(
        f"Train RMSE: {train_rmse:.6f}, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Val RMSE: {val_rmse:.6f}, "
        f"Average Val Loss: {val_loss:.6f}"
    )

    if val_rmse < best_rmse - min_delta:
        best_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best Val RMSE: {best_rmse:.6f} (epoch {best_epoch})"
            )
            break

if best_state is not None:
    model.load_state_dict(best_state)

######################################################################
# 9) Save the model
# -----------------
torch.save(model.state_dict(), "weights_challenge_1.pt")
print("Model saved as 'weights_challenge_1.pt'")
