"""Does resting-state pretraining transfer to reaction time?
=========================================================

Load three R5 mini participants and run 1 of contrast-change detection.
Predict stimulus-to-response time in seconds from the preceding two seconds
of EEG. The split holds out a complete participant; selected samples end at
stimulus onset. Pretrain an EEGNet on observed eyes-open/closed cues from the
two training participants, then adapt its encoder to reaction-time regression. Compare with
an identically shaped network trained from scratch. No recording from the test
participant enters either training stage. Fixed two-epoch budgets illustrate
the operations; this is not evidence for general transfer gains.
"""

# %%
# Before you start
# ----------------
#
# Use an installed EEGDash environment with Braindecode, MNE, NumPy,
# scikit-learn and Matplotlib; this script runs on CPU. Keep a persistent
# ``EEGDASH_CACHE_DIR``: the three contrast-change run-1 recordings are downloaded
# in full on first access even though each example uses short windows. The
# transfer version also needs two resting-state recordings. Both tasks use the
# challenge's 100 Hz, 0.5–50 Hz filtered derivatives.
#
# Reaction time is a continuous observed latency, not a fast/slow category. The
# window ends at the stimulus anchor. This excludes poststimulus samples from
# the selected interval, but does not establish a causal online pipeline: the
# source release has already been filtered and its preprocessing must be audited
# separately before claiming real-time prediction.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from braindecode.preprocessing import create_windows_from_events
import copy
import torch
from braindecode.models import EEGNet
from braindecode import EEGClassifier, EEGRegressor
from sklearn.metrics import mean_absolute_error

from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
)

# %%
# Load the named participants and observed events
# -----------------------------------------------
#
# The explicit run filter prevents a subject query from pulling all three
# contrast-change runs. The first two subject IDs will train the model; the third
# is reserved for evaluation. The subject-coverage assertion catches a missing
# recording instead of silently changing that design.
#
# ``annotate_trials_with_target`` reads the recording's event sidecar and pairs
# contrast trials with actual stimulus and response times. Trials without the
# required events do not supply an observed latency. ``add_aux_anchors`` places
# annotations at those actual stimulus times; it does not create response labels.
# Inspect the printed annotation names and 100 Hz rate before windowing.

subjects = ["NDARDC843HHM", "NDAREC480KFA", "NDARAP785CTE"]
cache = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
dataset = EEGChallengeDataset(
    release="R5",
    mini=True,
    task="contrastChangeDetection",
    subject=subjects,
    run="1",
    cache_dir=cache,
)
print(dataset.description.to_string(index=False))
assert set(dataset.description.subject) == set(subjects)
for recording in dataset.datasets:
    raw = recording.raw
    raw.pick("eeg")
    print(
        recording.description.subject,
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
    assert raw.info["sfreq"] == 100
    annotate_trials_with_target(raw, target_field="rt_from_stimulus")
    add_aux_anchors(raw)
# %%
# Extract a two-second prestimulus predictor
# ------------------------------------------
#
# At 100 Hz, offsets ``-200`` and ``0`` select the interval immediately
# before the stimulus. The 200-sample size and stride produce one window for
# each usable anchor. ``mapping={"stimulus_anchor": 0}`` tells the windower
# which anchors to use; that zero is a selection code, not the regression target.
#
# ``add_extras_columns`` carries the measured ``rt_from_stimulus`` into the
# window metadata. Use that column for ``y``, in seconds. A finite array with
# shape ``(trials, EEG channels, 200)`` supplies predictors in volts. Positive,
# finite latency assertions expose malformed event pairings; they do not require
# a particular prediction error or favourable result.

windows = create_windows_from_events(
    dataset,
    mapping={"stimulus_anchor": 0},
    trial_start_offset_samples=-200,
    trial_stop_offset_samples=0,
    window_size_samples=200,
    window_stride_samples=200,
    preload=True,
)
windows = add_extras_columns(
    windows,
    dataset,
    desc="stimulus_anchor",
    keys=("target", "rt_from_stimulus", "stimulus_onset"),
)
metadata = windows.get_metadata().reset_index(drop=True)
X = np.stack([windows[i][0] for i in range(len(windows))])
y = metadata.rt_from_stimulus.to_numpy(dtype=float)
assert np.isfinite(X).all() and np.isfinite(y).all() and (y > 0).all()
train = metadata.subject.isin(subjects[:2]).to_numpy()
test = metadata.subject.eq(subjects[2]).to_numpy()
assert train.any() and test.any()
assert set(metadata.subject[train]).isdisjoint(metadata.subject[test])
# Fixed microvolt conversion does not estimate a transform on held-out data.
X = X.astype("float32") * 1e6

# %%
# Pretrain on an observed auxiliary task
# --------------------------------------
#
# Only the two training participants supply resting EEG. Checking their
# IDs against the held-out target participant closes a common transfer-learning
# leak: excluding a person from fine-tuning is insufficient if their recording
# was already used during pretraining.
#
# The source labels encode instructions to open (0) or close (1) the eyes. The
# one-second offset moves the two-second window away from the instruction onset;
# these labels reflect the protocol, not an independent measurement of eye
# position. This is supervised auxiliary-task pretraining, not self-supervision.
# Both source and target arrays use the identical channel order and 200-sample
# length. Multiplying volts by ``1e6`` gives microvolts without fitting any
# statistics on held-out participants.

source = EEGChallengeDataset(
    release="R5", mini=True, task="RestingState", subject=subjects[:2], cache_dir=cache
)
assert set(source.description.subject).isdisjoint(metadata.subject[test])
for recording in source.datasets:
    recording.raw.pick("eeg")
    assert recording.raw.ch_names == dataset.datasets[0].raw.ch_names
source_windows = create_windows_from_events(
    source,
    mapping={"instructed_toOpenEyes": 0, "instructed_toCloseEyes": 1},
    trial_start_offset_samples=100,
    trial_stop_offset_samples=300,
    window_size_samples=200,
    window_stride_samples=200,
    preload=True,
)
Xs = (
    np.stack([source_windows[i][0] for i in range(len(source_windows))]).astype(
        "float32"
    )
    * 1e6
)
ys = np.asarray([source_windows[i][1] for i in range(len(source_windows))])
assert np.isfinite(Xs).all() and set(ys) == {0, 1}
print(
    "Resting-state windows:",
    Xs.shape,
    "real eye-state counts:",
    np.unique(ys, return_counts=True),
)
torch.manual_seed(71)
torch.set_num_threads(2)


# %%
# Train the source encoder and replace its head
# ---------------------------------------------
#
# EEGNet learns temporal and spatial filters directly from the windows.
# The source head returns two logits for cross-entropy; the downstream head
# returns one real number for squared-error regression. Adam's ``1e-3`` step size,
# two epochs and 16-example batches (with a shorter final batch) are fixed teaching settings,
# not hyperparameters selected for this cohort. ``EEGClassifier`` and
# ``EEGRegressor`` manage batching, gradients and evaluation mode;
# ``train_split=None`` prevents an additional window-level validation split.
# Regression targets retain shape ``(trials, 1)`` to match the one-output head.
#
# Target standardization uses only training latencies. Predictions are later
# multiplied by that training standard deviation and shifted by the training mean
# to return to seconds. Copying all state except ``final_layer`` transfers the
# encoder while keeping the new one-output head. The missing-key assertion
# ensures that relaxed loading has not silently discarded unrelated parameters.
# Both downstream conditions begin with the same randomly initialized head.

encoder = EEGNet(n_chans=X.shape[1], n_outputs=2, n_times=200, sfreq=100)
source_trainer = EEGClassifier(
    encoder,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    batch_size=16,
    max_epochs=2,
    train_split=None,
    iterator_train__shuffle=False,
    classes=[0, 1],
    device="cpu",
)
source_trainer.fit(Xs, ys)
encoder = source_trainer.module_
mean, scale = y[train].mean(), y[train].std()
assert scale > 0
results = {}
initial = EEGNet(n_chans=X.shape[1], n_outputs=1, n_times=200, sfreq=100)
for regime in ["from scratch", "resting-state transfer"]:
    model = copy.deepcopy(initial)
    if regime == "resting-state transfer":
        state = {
            k: v
            for k, v in encoder.state_dict().items()
            if not k.startswith("final_layer")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("final_layer") for k in missing)
    regressor = EEGRegressor(
        model,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        batch_size=16,
        max_epochs=2,
        train_split=None,
        iterator_train__shuffle=False,
        device="cpu",
    )
    standardized_y = ((y[train] - mean) / scale).astype("float32")[:, None]
    regressor.fit(X[train], standardized_y)
    predicted = regressor.predict(X[test]).reshape(-1) * scale + mean
    results[regime] = mean_absolute_error(y[test], predicted)
print("Held-out reaction-time MAE (s):", results)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(results), list(results.values()))
ax.set(ylabel="Held-out participant trial MAE (s)")
plt.show()

# %%
# Interpret transfer without selecting on the test participant
# ------------------------------------------------------------
#
# Each bar is mean absolute error over usable trials from the same held-out
# person; lower is better and an error of 0.1 means 100 ms on average. The transfer
# condition receives extra source-task training, so the comparison is not matched
# for total optimization steps. One subject and one initialization cannot show a
# reliable general transfer benefit. The plot may favour either condition.
#
# To extend the experiment, reserve additional validation participants before
# either training stage. Select the epoch budget and learning rate there, then
# repeat the entire comparison over untouched test participants and seeds. Also
# compare a training-mean latency predictor and assess exclusions for missing
# responses. Do not tune source tasks after looking at these test bars.
#
# Related worked examples: `cross-dataset transfer
# <https://braindecode.org/dev/auto_examples/advanced_training/plot_transfer_learning.html>`_
# and `relative-positioning pretraining
# <https://braindecode.org/dev/auto_examples/advanced_training/plot_relative_positioning.html>`_.
