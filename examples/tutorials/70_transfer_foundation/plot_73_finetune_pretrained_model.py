"""How do I fine-tune a published pretrained EEG encoder?
======================================================

Adapt the published CBraMod checkpoint to real HBN eyes-open/closed cues.
The checkpoint is https://huggingface.co/braindecode/cbramod-pretrained
(documented by Braindecode's CBraMod model). It was pretrained on TUH EEG;
this example uses the distinct HBN R5 mini cohort. It downloads about 20 MB
of weights and six challenge recordings in small mode (roughly 100 MB).
Opt in to full mode for eighteen recordings (roughly 320 MB).

Small mode uses three folds with two test, two validation and two training
participants. Full mode uses six folds with three test, three validation and
twelve training participants. Each person is scored once per regime. Compare scratch, a frozen encoder with a learned linear
head, and fine-tuning. The small fixed budget demonstrates adaptation, not a
foundation-model benchmark.
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, Braindecode, PyTorch, NumPy, scikit-learn and Matplotlib.
# The public checkpoint requires network access on first use;
# ``from_pretrained`` caches its weights separately from ``EEGDASH_CACHE_DIR``,
# which caches the selected EEG recordings. The revision below pins the actual
# checkpoint rather than relying on a changing default branch.
#
# The executable comparison answers a narrow question: what happens when the
# same downstream task is learned from random weights, a frozen published
# representation, or an adaptable published representation? It does not repeat
# the checkpoint's large-scale pretraining. For the architecture and checkpoint
# contract, consult `Braindecode's pretrained-model example <https://braindecode.org/dev/auto_examples/model_building/plot_load_pretrained_models.html>`_.

import copy

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from braindecode.models import CBraMod
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
)
from sklearn.metrics import balanced_accuracy_score

from eegdash.paths import get_default_cache_dir

from eegdash import EEGChallengeDataset
from eegdash.const import SUBJECT_MINI_RELEASE_MAP

CACHE_DIR = get_default_cache_dir()
RESOURCE_MODE = "small"  # explicit opt-in: "full" uses all 18 retained participants
N_EPOCHS = 2 if RESOURCE_MODE == "small" else 6
N_FOLDS = 3 if RESOURCE_MODE == "small" else 6
N_VALID = 2 if RESOURCE_MODE == "small" else 3
torch.set_num_threads(2)
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
# ends and cannot supply the complete interval; MNE drops incomplete epochs. Labels
# represent the observed open/close instructions, not eye tracking.

# Load the resting-state recordings of the HBN R5 mini release, except two with
# flat or saturated posterior channels
BAD_RECORDINGS = ["NDARAP785CTE", "NDARCA740UC8"]
subjects = [
    s for s in sorted(SUBJECT_MINI_RELEASE_MAP["R5"]) if s not in BAD_RECORDINGS
]
if RESOURCE_MODE == "small":
    subjects = subjects[:6]  # about 100 MB; no outcome-based choice within this list
# Historical exclusions above are inherited, not reproduced by a QC threshold.
# They limit the estimand to this selected cohort; do not call it representative.
dataset = EEGChallengeDataset(
    release="R5", mini=True, task="RestingState", subject=subjects, cache_dir=CACHE_DIR
)


def standardize_recording(x):
    """Offline whole-recording normalization, including the held-out recording."""
    scale = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / scale


preprocess(
    dataset,
    [
        Preprocessor("pick", picks=CHANNELS),
        Preprocessor("resample", sfreq=SFREQ),
        # standardize each channel of each recording with its own mean and std (uses no labels)
        Preprocessor(standardize_recording),
    ],
)

# Cut seven 2 s epochs per eligible cue with explicit MNE sample origins.
# Native epochs reject BAD spans and incomplete trailing windows.
# Cue-specific offsets below avoid discarding valid open-eye windows based
# on the longer closed-eye interval.
epoch_arrays, metadata_tables = [], []
for recording in dataset.datasets:
    raw = recording.raw
    events = []
    for annotation in raw.annotations:
        cue = annotation["description"]
        if cue not in CUE_WINDOW:
            continue
        start, stop = CUE_WINDOW[cue]
        onset = annotation["onset"] - raw.first_time
        code = 1 if cue == "instructed_toOpenEyes" else 2
        for offset in range(start, stop, 2):
            sample = round((onset + offset) * SFREQ) + raw.first_samp
            events.append([sample, 0, code])
    events = np.asarray(events)
    epochs = mne.Epochs(
        raw,
        events[np.argsort(events[:, 0])],
        event_id={"eyes_open": 1, "eyes_closed": 2},
        tmin=0,
        tmax=2 - 1 / SFREQ,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )
    epoch_arrays.append(epochs.get_data())
    metadata_tables.append(
        pd.DataFrame(
            {
                "target": epochs.events[:, 2] - 1,
                "subject": str(recording.description["subject"]),
            }
        )
    )

# %%
# Reserve participants, not windows
# ---------------------------------
#
# ``X`` has shape ``(windows, 3, 400)`` and ``y`` contains the two integer
# instruction classes. Participants, rather than windows, determine the folds:
# group-disjoint folds use the counts in the parameter cell. Each participant
# is tested exactly once per regime, regardless of resource mode.
#
# All checkpoint selection uses validation balanced accuracy. Test labels are
# read only for the final score of each prespecified regime. Reporting several
# regimes is not permission to choose a winning configuration on test performance
# and call that a new independent result.

# Arrays for PyTorch: X = (windows, channels, samples), y = label, subject = participant of each window
X = np.concatenate(epoch_arrays).astype("float32")
metadata = pd.concat(metadata_tables, ignore_index=True)
y = metadata["target"].to_numpy()
subject = metadata["subject"].to_numpy()
print(f"X {X.shape} | classes {np.bincount(y)} | participants {len(subjects)}")

# Every participant is tested exactly once, in either resource mode
folds = np.array_split(np.array(subjects), N_FOLDS)
# Full-recording normalization uses each test recording's unlabeled distribution:
# this is offline/transductive preprocessing, not causal online deployment.

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
# ``1e-3`` for the linear head alone. The explicit small/full epoch budgets
# demonstrate execution, not recommended convergence settings. Each regime is
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
        encoder.requires_grad_(False)
    # Derive width from the actual encoder output, not a magic architecture constant.
    encoder.eval()
    with torch.inference_mode():
        width = encoder(torch.from_numpy(X[:1])).flatten(1).shape[1]
    torch.manual_seed(17)  # matched head initialization across regimes
    return torch.nn.Sequential(encoder, torch.nn.Flatten(), torch.nn.Linear(width, 2))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [model(batch).argmax(1).numpy() for batch in torch.from_numpy(X).split(16)]
        )


def fit(model, train, valid, lr, frozen, n_epochs=N_EPOCHS, batch_size=16):
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
                model(torch.from_numpy(X[idx])),
                torch.from_numpy(y[idx]),
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
    train = np.isin(subject, others[:-N_VALID])  # training people
    valid = np.isin(subject, others[-N_VALID:])  # validation people
    test = np.isin(subject, test_subjects)  # held-out people
    for regime in REGIMES:
        torch.manual_seed(0)
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
# alone; the bar is the mean across the selected participants and the dashed
# line is the two-class chance level. Any regime may do better on this cohort.
# Shared folds and a matched head do not isolate pretraining as a causal effect:
# learning rates and trainable parameters differ, and this is one random seed.
# No p-values are computed because overlapping training folds, multiple regime
# comparisons and the selected cohort need a prespecified inferential design.
# Accuracy alone does not diagnose alpha reactivity or signal quality.
#
# For a stronger comparison, add repeated seeds and prespecify longer budgets.
# Compare equal head initializations when estimating the effect of pretrained
# weights. Inspect per-participant confusion matrices and channel quality before
# attributing a change to the encoder. Save the selected weights together with
# the checkpoint revision, channel order, rate, window offsets, standardization
# rule and subject lists if you extend this page into a reusable trained model;
# weights alone omit the input contract.
