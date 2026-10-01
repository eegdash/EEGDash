"""EEG2025 Challenge 1: predict observed reaction time
===============================================================

Load three R5 mini participants and run 1 of contrast-change detection.
Predict stimulus-to-response time in seconds from the preceding two seconds
of EEG. The split holds out a complete participant; selected samples end at
stimulus onset. This compact ridge baseline is
an instructional subset, not the official competition split or score.
"""

# %%
# Before you start
# ----------------------------
#
# Use an installed EEGDash environment with Braindecode, MNE, NumPy,
# scikit-learn and Matplotlib; this script runs on CPU. Keep a persistent
# ``EEGDASH_CACHE_DIR``: the three contrast-change run-1 recordings are downloaded
# in full on first access even though each example uses short windows. The
# challenge uses 100 Hz, 0.5–50 Hz filtered derivatives.
#
# Reaction time is a continuous observed latency, not a fast/slow category. The
# window ends at the stimulus anchor. This excludes poststimulus samples from
# the selected interval, but does not establish a causal online pipeline: the
# source release has already been filtered and its preprocessing must be audited
# separately before claiming real-time prediction.

import matplotlib.pyplot as plt
import numpy as np
from braindecode.preprocessing import create_windows_from_events
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.paths import get_default_cache_dir

from eegdash import EEGChallengeDataset
from eegdash.features import signal_variance
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
)

# %%
# Load the named participants and observed events
# -----------------------------------------------------------
#
# The explicit run filter prevents a subject query from pulling all three
# contrast-change runs. The first two subject IDs will train the model; the third
# is reserved for evaluation. The subject-coverage check catches a missing
# recording instead of silently changing that design.
#
# ``annotate_trials_with_target`` reads the recording's event sidecar and pairs
# contrast trials with actual stimulus and response times. Trials without the
# required events do not supply an observed latency. ``add_aux_anchors`` places
# annotations at those actual stimulus times; it does not create response labels.
# Inspect the printed annotation names and 100 Hz rate before windowing.

subjects = ["NDARDC843HHM", "NDAREC480KFA", "NDARAP785CTE"]
cache = get_default_cache_dir()
dataset = EEGChallengeDataset(
    release="R5",
    mini=True,
    task="contrastChangeDetection",
    subject=subjects,
    run="1",
    cache_dir=cache,
)
print(dataset.description.to_string(index=False))
if not (set(dataset.description.subject) == set(subjects)):
    raise ValueError(
        "Unexpected cohort: check the query, missing recordings and duplicate participant rows."
    )
channels = None
for recording in dataset.datasets:
    raw = recording.raw
    raw.pick("eeg")
    if channels is None:
        channels = raw.ch_names.copy()
    if raw.ch_names != channels:
        raise ValueError(
            "Channel order differs across participants; align named EEG channels before fitting."
        )
    print(
        recording.description.subject,
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
    if not (raw.info["sfreq"] == 100):
        raise ValueError(
            "Unexpected sampling grid: inspect source timing and preprocessing before constructing windows."
        )
    annotate_trials_with_target(raw, target_field="rt_from_stimulus")
    add_aux_anchors(raw)
# %%
# Extract a two-second prestimulus predictor
# ------------------------------------------------------
#
# At 100 Hz, offsets ``-200`` and ``0`` select the interval immediately
# before the stimulus. The 200-sample size and stride produce one window for
# each usable anchor. ``mapping={"stimulus_anchor": 0}`` tells the windower
# which anchors to use; that zero is a selection code, not the regression target.
#
# ``add_extras_columns`` carries the measured ``rt_from_stimulus`` into the
# window metadata. Use that column for ``y``, in seconds. A finite array with
# shape ``(trials, EEG channels, 200)`` supplies predictors in volts. Positive,
# finite latency checks expose malformed event pairings; they do not require
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
if not (np.isfinite(X).all() and np.isfinite(y).all() and (y > 0).all()):
    raise ValueError(
        "Nonfinite signals, targets or predictions: inspect missing metadata and unusable channels before fitting."
    )
# Inspect the actual target distribution and one event/window relationship.
fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
for subject in subjects:
    axes[0].hist(y[metadata.subject.eq(subject)], bins=12, alpha=0.4, label=subject)
axes[0].set(xlabel="Observed stimulus-to-response latency (s)", ylabel="Trials")
axes[0].legend(fontsize="small")
axes[1].axvspan(-2, 0, alpha=0.2, label="EEG predictor")
axes[1].axvline(0, color="black", label="Stimulus")
axes[1].axvline(y[0], color="tab:orange", label="First retained response")
axes[1].set(xlabel="Time relative to stimulus (s)", yticks=[])
axes[1].legend()
plt.show()
# Log variance measures channel power; fit scaling on training participants.
# %%
# Reduce each trial to channel power
# ----------------------------------------------
#
# EEGDash's ``signal_variance`` reduces the time axis to one feature per channel. Its
# natural logarithm compresses the large range of power values; ``1e-30`` only
# prevents taking the logarithm of zero. It is not an artifact rejection threshold.
# This inexpensive representation discards the temporal waveform, which makes it
# a useful reference before introducing an EEG encoder.
#
# The Boolean masks split complete participants. Standardizing all rows before
# this split would expose the test distribution, so scaling remains inside the
# model pipeline. No separate validation set is used: ``alpha=10`` is fixed in
# advance and must not be adjusted in response to the final plot.

features = np.log(np.maximum(signal_variance(X), 1e-30))
train = metadata.subject.isin(subjects[:2]).to_numpy()
test = metadata.subject.eq(subjects[2]).to_numpy()
if not (train.any() and test.any()):
    raise ValueError(
        "Data do not satisfy the documented task contract; inspect the query, labels and retained windows before continuing."
    )
print(
    "Windows:", X.shape, "trials per participant:", metadata.groupby("subject").size()
)

# %%
# Fit a regularized baseline and compare errors
# ---------------------------------------------------------
#
# Ridge penalizes large coefficients, which is helpful when channel-power
# predictors are correlated and there are few training trials. ``StandardScaler``
# fits each feature's mean and spread on the training participants only. The dummy
# model predicts their mean latency and measures how much a signal-free predictor
# already explains.
#
# Mean absolute error averages absolute prediction errors in seconds over the
# held-out participant's trials. It weights trials equally, not participants,
# because this test set contains only one person. The scatter shows whether
# predictions vary with observed latency; a nearly constant prediction or a model
# worse than the dummy baseline is a valid outcome.

model = make_pipeline(StandardScaler(), Ridge(alpha=10))
predicted = model.fit(features[train], y[train]).predict(features[test])
baseline = DummyRegressor().fit(features[train], y[train]).predict(features[test])
print("Held-out trial MAE (s):", mean_absolute_error(y[test], predicted))
print("Training-mean MAE (s):", mean_absolute_error(y[test], baseline))
# %%
# Report the final starter kit metric on this subset
# --------------------------------------------------------------
#
# The final starter kit normalizes RMSE by the evaluated targets' population
# standard deviation (despite its obsolete range-based docstring). This is
# dimensionless; the small tutorial split is not the competition test cohort.
# See https://github.com/eeg2025/startkit/blob/f5c2f3fbccf5889bad904ecf145c12ca9c6c58c9/local_scoring.py.
target_spread = y[test].std(ddof=0)
if not (target_spread > 0):
    raise ValueError("NRMSE needs variation in the observed test targets")
print("Subset NRMSE:", root_mean_squared_error(y[test], predicted) / target_spread)
print(
    "Training-mean NRMSE:", root_mean_squared_error(y[test], baseline) / target_spread
)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y[test], predicted, label="Ridge")
ax.scatter(y[test], baseline, marker="x", label="Training mean")
ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], "k--")
ax.legend()
ax.set(xlabel="Observed reaction time (s)", ylabel="Predicted reaction time (s)")
plt.show()

# %%
# Extend to a defensible challenge evaluation
# -------------------------------------------------------
#
# The single held-out participant is a workflow check, not a population
# estimate or the official challenge evaluator. Ridge is unconstrained and can
# produce negative latencies; inspect such predictions instead of silently
# clipping them to improve a score. Missing-response exclusions also change the
# population being predicted.
#
# Add training and validation participants before comparing feature bands or
# ridge penalties. Keep final test participants out of every choice and report
# participant-level errors as well as pooled trial errors. For encoder transfer
# with the same observed target, continue with tutorial 71.
#
# Related evaluation example: `Braindecode train, test and tune
# <https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html>`_.
