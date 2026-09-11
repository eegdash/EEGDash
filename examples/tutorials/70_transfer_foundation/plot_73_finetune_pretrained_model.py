"""How do I fine-tune a published pretrained EEG encoder?
======================================================

Adapt the published CBraMod checkpoint to real HBN eyes-open/closed cues.
The checkpoint is https://huggingface.co/braindecode/cbramod-pretrained
(documented by Braindecode's CBraMod model). It was pretrained on TUH EEG;
this example uses the distinct HBN R5 mini cohort. It downloads about 20 MB
of weights and eighteen challenge recordings (roughly 320 MB total).

Six subject-grouped folds each hold three participants out for testing
while three others select the epoch and twelve train, so every participant
is scored exactly once. Compare scratch, a frozen encoder with a learned linear
head, and fine-tuning. The small fixed budget demonstrates adaptation, not a
foundation-model benchmark.
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, Braindecode, PyTorch, NumPy, SciPy, scikit-learn and Matplotlib.
# The public checkpoint requires network access on first use;
# ``from_pretrained`` caches its weights separately from ``EEGDASH_CACHE_DIR``,
# which caches the eighteen EEG recordings. The revision below pins the actual
# checkpoint rather than relying on a changing default branch.
#
# The executable comparison answers a narrow question: what happens when the
# same downstream task is learned from random weights, a frozen published
# representation, or an adaptable published representation? It does not repeat
# the checkpoint's large-scale pretraining. For the architecture and checkpoint
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
from scipy.stats import wilcoxon
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
# All checkpoint selection uses validation balanced accuracy. Test labels are
# read only for the final score of each prespecified regime. Reporting several
# regimes is not permission to choose a winning configuration on test performance
# and call that a new independent result.

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
# Compare optimization regimes with validation-only selection
# -----------------------------------------------------------
#
# Three small functions keep the loop readable. ``make_model`` builds the
# encoder plus a linear head: the scratch model has the same encoder
# architecture but no downloaded weights, the linear probe freezes the
# pretrained encoder so that only the head learns, and fine-tuning lets both
# change. ``fit`` trains with AdamW, reshuffles the mini-batches every epoch
# (the windows are otherwise ordered by participant and cue, which would make
# consecutive batches nearly single-class), and keeps the weights of the epoch
# with the best validation score. For the probe it also puts the frozen encoder
# in evaluation mode, because freezing gradients alone would not switch off its
# dropout. ``predict`` returns the predicted class of each window.
#
# The learning rate is ``1e-4`` for the scratch and fine-tune regimes and
# ``1e-3`` for the linear head alone. Six epochs per fold keep CPU execution to
# a few minutes; they are not a recommended training budget. Each regime is
# scored once per held-out participant, as the balanced accuracy over that
# participant's windows.

REGIMES = ["scratch", "linear probe", "fine-tune"]


def make_model(regime):
    """CBraMod encoder + linear head. 3 channels x 2 patches x 200 features = 1200 inputs."""
    if regime == "scratch":
        encoder = CBraMod(
            n_chans=len(CHANNELS),
            n_times=2 * SFREQ,
            sfreq=SFREQ,
            return_encoder_output=True,
        )
    else:
        encoder = copy.deepcopy(pretrained)
    if regime == "linear probe":
        encoder.requires_grad_(False)  # freeze the encoder: only the head learns
    return torch.nn.Sequential(encoder, torch.nn.Flatten(), torch.nn.Linear(1200, 2))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).argmax(1).numpy()


def fit(model, train, valid, lr, frozen, n_epochs=6, batch_size=16):
    """Train with AdamW; return the model with the weights of the best validation epoch."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    best_score, best_state = -1, None
    for epoch in range(n_epochs):
        model.train()
        if frozen:
            model[0].eval()  # frozen encoder: also switch off its dropout
        for batch in torch.randperm(int(train.sum())).split(
            batch_size
        ):  # shuffled mini-batches
            idx = np.flatnonzero(train)[batch.numpy()]
            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(X[idx])), torch.from_numpy(y[idx])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = balanced_accuracy_score(y[valid], predict(model, X[valid]))
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


torch.manual_seed(0)
scores = {
    regime: [] for regime in REGIMES
}  # one balanced accuracy per held-out participant
for test_subjects in folds:
    others = [s for s in subjects if s not in test_subjects]
    train = np.isin(subject, others[:-3])  # 12 participants train
    valid = np.isin(subject, others[-3:])  # 3 participants select the epoch
    test = np.isin(subject, test_subjects)  # 3 participants are scored once
    for regime in REGIMES:
        frozen = regime == "linear probe"
        model = fit(
            make_model(regime), train, valid, lr=1e-3 if frozen else 1e-4, frozen=frozen
        )
        pred = predict(model, X[test])
        for s in test_subjects:
            m = subject[test] == s
            scores[regime].append(balanced_accuracy_score(y[test][m], pred[m]))
    n = len(test_subjects)
    print(
        f"held out {test_subjects.tolist()}: "
        + " | ".join(f"{r} {np.mean(scores[r][-n:]):.2f}" for r in REGIMES)
    )

# %%
# Statistics over participants (one score each) and a figure with one dot per participant
for regime in REGIMES:
    s = np.array(scores[regime])
    p = wilcoxon(s - 0.5, alternative="greater").pvalue
    print(
        f"{regime:13s} mean {s.mean():.3f} | above chance in {(s > 0.5).sum()}/{len(s)} | Wilcoxon p = {p:.1e}"
    )
p = wilcoxon(scores["fine-tune"], scores["scratch"], alternative="greater").pvalue
print(f"fine-tune > scratch (paired over participants): p = {p:.1e}")

fig, ax = plt.subplots(figsize=(6, 4))
jitter = np.linspace(
    -0.2, 0.2, len(subjects)
)  # spread the dots; scores stay in participant order
for i, regime in enumerate(REGIMES):
    ax.bar(i, np.mean(scores[regime]), color="lightgray")
    ax.scatter(i + jitter, scores[regime], s=18, color="k", zorder=3)
ax.axhline(0.5, ls="--", color="gray")  # chance level
ax.set(
    xticks=range(len(REGIMES)),
    xticklabels=REGIMES,
    ylim=(0, 1),
    ylabel="Balanced accuracy per held-out participant",
)
plt.show()

# %%
# Read the final bars and design the next experiment
# --------------------------------------------------
#
# Each dot is one held-out participant, scored on that participant's windows
# alone; the bar is the mean across the eighteen participants and the dashed
# line is the two-class chance level. Random weights stay close to chance. The
# frozen published representation already separates the two eye states for most
# participants, and fine-tuning does best. The paired test between fine-tuning
# and scratch is the evidence that the pretrained weights matter, because the
# two regimes share folds, windows, budget and seed. A few participants sit near
# 0.5 under every regime; those recordings show little alpha reactivity in the
# selected channels, which is a data property to inspect, not a modeling
# failure to tune away.
#
# For a stronger comparison, add repeated seeds and prespecify longer budgets.
# Compare equal head initializations when estimating the effect of pretrained
# weights. Inspect per-participant confusion matrices and channel quality before
# attributing a change to the encoder. Save the selected weights together with
# the checkpoint revision, channel order, rate, window offsets, standardization
# rule and subject lists if you extend this page into a reusable trained model;
# weights alone omit the input contract.
