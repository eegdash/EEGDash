"""How do I adapt a pretrained EEG encoder with a linear probe?
============================================================

A shorter companion to the full fine-tuning example
(``plot_73_finetune_pretrained_model.py``). It adapts the published CBraMod
checkpoint to real HBN eyes-open/closed cues with the simplest transfer
strategy: freeze the pretrained encoder and train only a linear head on top.
The checkpoint is https://huggingface.co/braindecode/cbramod-pretrained
(documented by Braindecode's CBraMod model). It was pretrained on TUH EEG;
this example uses the distinct HBN R5 mini cohort. It downloads about 20 MB
of weights and eighteen challenge recordings (roughly 320 MB total).

Six subject-grouped folds each hold three participants out for testing
while three others select the epoch and twelve train, so every participant
is scored exactly once. The training loop is written out in one block, with
no helper functions, so each step can be read top to bottom. The small fixed
budget demonstrates adaptation, not a foundation-model benchmark.
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, Braindecode, PyTorch, NumPy, scikit-learn and Matplotlib.
# The public checkpoint requires network access on first use;
# ``from_pretrained`` caches its weights separately from ``EEGDASH_CACHE_DIR``,
# which caches the eighteen EEG recordings. The revision below pins the actual
# checkpoint rather than relying on a changing default branch.
#
# The question here is narrow: does a frozen published representation, read
# out by a single linear layer, separate the two eye states in participants it
# has never seen? For the comparison with training from random weights and with
# full fine-tuning, see the full example. For the architecture and checkpoint
# contract, consult `Braindecode's pretrained-model example <https://braindecode.org/dev/auto_examples/model_building/plot_load_pretrained_models.html>`_.

import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.models import CBraMod
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from sklearn.metrics import balanced_accuracy_score

from eegdash import EEGChallengeDataset
from eegdash.const import SUBJECT_MINI_RELEASE_MAP

CACHE_DIR = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
CHANNELS = ["E70", "E75", "E83"]  # three posterior (occipital) channels
SFREQ = 200  # CBraMod expects 200 Hz: one-second patches of 200 samples
# Steady-state window (start, stop) in seconds after each cue, as in the eyes-open/closed tutorial
CUE_WINDOW = {"instructed_toOpenEyes": (5, 19), "instructed_toCloseEyes": (15, 29)}

# %%
# Match the real signal to the encoder input
# ------------------------------------------
#
# E70, E75 and E83 are an explicit small posterior-channel subset, not a
# claim that the complete HBN montage is interchangeable with the pretraining
# montage. The source challenge derivative has already been filtered and sampled
# at 100 Hz; resampling to 200 Hz satisfies CBraMod's one-second, 200-sample
# patch contract but cannot recover information above the source Nyquist
# frequency.
#
# Two of the twenty mini-release recordings have flat or saturated posterior
# electrodes (found by inspecting per-channel amplitudes) and are excluded by
# ID. Posterior amplitudes still differ by orders of magnitude across the
# remaining recordings, so each recording is standardized per channel with its
# own mean and standard deviation. That step uses no labels and is applied
# identically to every recording.
#
# Windows sample the steady state of each eye condition rather than the
# instruction transient, following the eyes-open/closed tutorial: seven
# two-second windows from 5 to 19 s after an open-eyes cue (the open block
# lasts 20 s) and from 15 to 29 s after a close-eyes cue (the closed block
# lasts 40 s). The final open-eyes cue falls a few seconds before the recording
# ends and cannot supply a full window; Braindecode refuses such a cue, so it
# is removed first. Labels
# represent the observed open/close instructions, not eye tracking.

# Load the resting-state recordings of the HBN R5 mini release, except two with
# flat or saturated posterior channels
BAD_RECORDINGS = ["NDARAP785CTE", "NDARCA740UC8"]
subjects = [
    s for s in sorted(SUBJECT_MINI_RELEASE_MAP["R5"]) if s not in BAD_RECORDINGS
]
dataset = EEGChallengeDataset(
    release="R5", mini=True, task="RestingState", subject=subjects, cache_dir=CACHE_DIR
)


def drop_late_cues(raw):
    """The last eyes-open cue comes ~5 s before the end and cannot supply a full window."""
    fits = (
        raw.annotations.onset + max(stop for _, stop in CUE_WINDOW.values())
        < raw.times[-1]
    )
    return raw.set_annotations(raw.annotations[fits])


preprocess(
    dataset,
    [
        Preprocessor("pick", picks=CHANNELS),
        Preprocessor("resample", sfreq=SFREQ),
        # standardize each channel of each recording with its own mean and std (uses no labels)
        Preprocessor(
            lambda x: (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        ),
        Preprocessor(drop_late_cues, apply_on_array=False),
    ],
)

# Cut 2 s windows: label 0 = eyes open, 1 = eyes closed
windows = create_windows_from_events(
    dataset,
    mapping={"instructed_toOpenEyes": 0, "instructed_toCloseEyes": 1},
    trial_start_offset_samples={
        cue: start * SFREQ for cue, (start, _) in CUE_WINDOW.items()
    },
    trial_stop_offset_samples={
        cue: stop * SFREQ for cue, (_, stop) in CUE_WINDOW.items()
    },
    window_size_samples=2 * SFREQ,
    window_stride_samples=2 * SFREQ,
)

# %%
# Reserve participants, not windows
# ---------------------------------
#
# ``X`` has shape ``(windows, 3, 400)`` and ``y`` contains the two integer
# instruction classes. Participants, rather than windows, determine the folds:
# six subject-grouped folds each hold out three participants for testing, three
# others select the epoch, and the remaining twelve train. Every participant is
# tested exactly once.
#
# Epoch selection uses validation balanced accuracy only. Test labels are read
# once per fold, for the final score.

# Arrays for PyTorch: X = (windows, channels, samples), y = label, subject = participant of each window
X = np.stack([x for x, _, _ in windows]).astype("float32")
metadata = windows.get_metadata()
y = metadata["target"].to_numpy()
subject = metadata["subject"].to_numpy()
print(f"X {X.shape} | classes {np.bincount(y)} | participants {len(subjects)}")

# Six folds of three participants each; every participant is tested exactly once
folds = np.array_split(np.array(subjects), 6)

# %%
# Load the checkpoint and understand the head dimensions
# ------------------------------------------------------
#
# ``return_encoder_output=True`` exposes CBraMod's patch representations.
# For three channels and two one-second patches, each with 200 representation
# coordinates, flattening gives ``3 × 2 × 200 = 1200`` inputs to the two-class
# linear head. The head returns logits; cross-entropy consumes logits directly,
# so no softmax is inserted in the training loop.
#
# If you change channels or duration, derive the new head width from a real
# encoder forward pass. Merely changing ``n_outputs`` cannot fix a mismatched
# feature shape. The same principle is illustrated in Braindecode's
# `foundation-model fine-tuning walkthrough <https://braindecode.org/dev/auto_examples/advanced_training/plot_finetune_foundation_model.html>`_.

# Download the published checkpoint (about 20 MB); the revision pins the exact weights
pretrained = CBraMod.from_pretrained(
    "braindecode/cbramod-pretrained",
    revision="584cdc415913739a05d84bf0c1cb3db397764507",
    return_encoder_output=True,  # return the patch features instead of class scores
    n_chans=len(CHANNELS),
    n_times=2 * SFREQ,
    sfreq=SFREQ,
)

# %%
# Train a linear head on the frozen encoder
# -----------------------------------------
#
# For each fold, copy the pretrained encoder, freeze it with
# ``requires_grad_(False)``, and append ``Flatten`` and a ``Linear(1200, 2)``
# head. Only the head's parameters are passed to AdamW (learning rate
# ``1e-3``). During training the encoder is also put in evaluation mode:
# freezing gradients alone would not switch off its dropout.
#
# Each epoch reshuffles the mini-batches of 16 windows (the windows are
# otherwise ordered by participant and cue, which would make consecutive batches
# nearly single-class), then scores the validation participants. The weights of
# the best validation epoch are restored before the test participants are
# scored. Six epochs per fold keep CPU execution to a few minutes; they are not
# a recommended training budget. Each held-out participant receives one score,
# the balanced accuracy over that participant's windows.

torch.manual_seed(0)
scores = []

for test_subjects in folds:
    others = [s for s in subjects if s not in test_subjects]

    train = np.isin(subject, others[:-3])
    valid = np.isin(subject, others[-3:])
    test = np.isin(subject, test_subjects)

    # Frozen CBraMod + linear classifier
    encoder = copy.deepcopy(pretrained)
    encoder.requires_grad_(False)

    model = torch.nn.Sequential(
        encoder,
        torch.nn.Flatten(),
        torch.nn.Linear(1200, 2),
    )

    optimizer = torch.optim.AdamW(model[-1].parameters(), lr=1e-3)

    best_score = -1

    for epoch in range(6):
        model.train()
        model[0].eval()

        idx_train = np.flatnonzero(train)

        for batch in torch.randperm(len(idx_train)).split(16):
            idx = idx_train[batch.numpy()]

            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(X[idx])),
                torch.from_numpy(y[idx]),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[valid])).argmax(1).numpy()

        score = balanced_accuracy_score(y[valid], pred)

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    with torch.no_grad():
        pred = model(torch.from_numpy(X[test])).argmax(1).numpy()

    for s in test_subjects:
        m = subject[test] == s
        scores.append(balanced_accuracy_score(y[test][m], pred[m]))

    print(test_subjects, np.mean(scores[-len(test_subjects) :]))

# %%
# Plot one score per participant
# ------------------------------
#
# Each dot is one held-out participant; the bar is the mean over the eighteen
# participants and the dashed line is the two-class chance level.

plt.bar(0, np.mean(scores), color="lightgray")
plt.scatter(np.linspace(-0.2, 0.2, len(scores)), scores, color="k")
plt.axhline(0.5, ls="--", color="gray")
plt.ylim(0, 1)
plt.ylabel("Balanced accuracy")
plt.xticks([0], ["Linear probe"])
plt.show()

# %%
# Read the result and design the next experiment
# ----------------------------------------------
#
# The frozen published representation, read out by a linear head trained on
# three posterior channels, separates the two eye states above chance for most
# held-out participants. A few participants sit near 0.5; those recordings show
# little alpha reactivity in the selected channels, which is a data property to
# inspect, not a modeling failure to tune away. This page does not show that
# the pretrained weights are responsible for the result: that requires the
# same pipeline trained from random weights, which the full example
# (``plot_73_finetune_pretrained_model.py``) runs alongside full fine-tuning,
# with paired tests over participants.
#
# To extend this page, unfreeze the encoder with a smaller learning rate
# (fine-tuning), add repeated seeds, and prespecify longer budgets. Save the
# selected weights together with the checkpoint revision, channel order, rate,
# window offsets, standardization rule and subject lists if you reuse the
# trained model; weights alone omit the input contract.
