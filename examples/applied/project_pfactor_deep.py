"""Regress observed p-factor with a neural EEG decoder
===================================================

Train a compact Braindecode EEGNet on recorded windows and aggregate
predictions for held-out participants. Two epochs exercise CPU training;
this is not a converged or clinically validated model.

Use six explicitly selected participants from the real
`HBN ds005505 release <https://openneuro.org/datasets/ds005505>`_. Their
RestingState signal files total approximately 595.8 MB, cached under
``EEGDASH_CACHE_DIR``. This applied example is larger than the introductory
21 MB SSVEP subset. Cropping after loading reduces computation, not download.
Targets come from the observed participant metadata. Six participants are
sufficient to exercise the workflow, not to support clinical conclusions.

Before you start
----------------
Install EEGDash with EEGPrep, Braindecode and PyTorch. Tutorial 02 explains minibatches;
tutorial 11 explains grouped evaluation. The feature-based p-factor project
provides a simpler comparison, but no saved output is required here. This
script trains a fresh compact network in each fold and reports participant
errors after aggregating its real window predictions.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
    Resampling,
    RemoveDrifts,
)

from eegdash import EEGDashDataset
import torch
from braindecode.models import EEGNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

# %%
# 1. Load the named cohort and inspect observed participant metadata
# ------------------------------------------------------------------
# Every window from a participant will share that participant's observed
# ``p_factor``. Repeating the score over windows is a supervised-learning
# representation of one measurement, not many independent clinical targets.
# Consequently all those windows must remain together during evaluation.
#
# The two assertions require one selected recording per named participant.
# ``description_fields`` makes the source attributes available alongside BIDS
# identifiers; the EEG still comes from ``EEGDashDataset``. These are the original
# OpenNeuro recordings, not the separately filtered/downsampled challenge
# release supplied by ``EEGChallengeDataset``.
subjects = [
    "NDARBH024NH2",
    "NDARAM704GKZ",
    "NDARAC904DMU",
    "NDARAN385MDH",
    "NDARAG143ARJ",
    "NDARAP359UM6",
]
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
dataset = EEGDashDataset(
    dataset="ds005505",
    task="RestingState",
    subject=subjects,
    cache_dir=cache_dir,
    description_fields=["subject", "task", "age", "sex", "p_factor"],
    n_jobs=1,
)
assert len(dataset.datasets) == len(subjects)
assert dataset.description["subject"].nunique() == len(subjects)
print(dataset.description[["subject", "age", "sex", "p_factor"]])

# %%
# 2. Prepare the first minute of recorded EEG
# -------------------------------------------
# The fixed first-minute interval and four named electrodes bound computation
# and establish a common channel order. They are not selected by test accuracy.
# The source reference channel can be flat; the chosen channels exclude that
# reference rather than pretending a zero-variance channel can be standardized.
#
# Braindecode's EEGPrep ``Resampling`` adapter reduces the rate to 100 Hz,
# with anti-alias filtering, before ``RemoveDrifts`` applies its high-pass
# transition from 0.5 to 1 Hz. Resampling first reduces the later filter's
# computation. These published components run on each recording separately;
# no custom filtering function or cross-participant fit is needed.
#
# A two-second window then contains 200 voltage samples. The source reference
# is retained: we do not average-reference an arbitrary four-channel subset.
# EEGPrep's forward-backward drift filter is offline, not causal streaming
# preprocessing. This small recipe neither detects every artifact nor invokes
# channel rejection or ASR. The neural decoder receives the resulting waveform.
#
# Fixed-length windows are used because the target belongs to a person, not an
# event. They do not relabel eyes-open/closed intervals or remove every
# instruction or artifact from the first minute. Equal size and stride avoid
# overlap; a final short remainder is discarded. The printed array has axes
# ``(windows, 4 channels, 200 samples)``. Check its actual row count rather than
# assuming every recording produces an identical number of windows.
channels = ["E11", "E62", "E75", "E22"]
for recording in dataset.datasets:
    raw = recording.raw
    print(
        recording.description["subject"],
        raw.info["sfreq"],
        raw.ch_names,
        "observed annotations:",
        sorted(set(raw.annotations.description)),
    )
    assert set(channels).issubset(raw.ch_names)
    raw.crop(tmax=59.99).load_data().pick(channels).reorder_channels(channels)
# Preserve annotation times in seconds across EEGPrep format conversions.
# Retain the measurement date too: it anchors annotations with absolute times.
annotations_before = [
    recording.raw.annotations.copy() for recording in dataset.datasets
]
measurement_dates = [recording.raw.info["meas_date"] for recording in dataset.datasets]
durations_before = [
    recording.raw.n_times / recording.raw.info["sfreq"]
    for recording in dataset.datasets
]
preprocess(
    dataset, [Resampling(sfreq=100), RemoveDrifts(transition=(0.5, 1.0))], n_jobs=1
)
for recording, annotations, duration, measurement_date in zip(
    dataset.datasets, annotations_before, durations_before, measurement_dates
):
    raw = recording.raw
    assert raw.ch_names == channels and raw.info["sfreq"] == 100
    assert abs(raw.n_times / 100 - duration) <= 1 / 100
    raw.set_meas_date(measurement_date)
    raw.set_annotations(annotations)
    assert raw.annotations.orig_time == annotations.orig_time
    np.testing.assert_array_equal(raw.annotations.onset, annotations.onset)
    np.testing.assert_array_equal(raw.annotations.description, annotations.description)
windows = create_fixed_length_windows(
    dataset,
    window_size_samples=200,
    window_stride_samples=200,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata().reset_index(drop=True)
X_windows = np.stack([item[0] for item in windows])
groups = metadata["subject"].astype(str).to_numpy()
assert X_windows.shape == (len(metadata), len(channels), 200)
assert np.isfinite(X_windows).all()
assert not metadata.duplicated(["subject", "i_start_in_trial"]).any()
print("Real two-second windows:", X_windows.shape)

# %%
# 3. Attach observed participant targets and define subject-disjoint folds
# ------------------------------------------------------------------------
# The metadata join maps each window's subject to its released score. That
# join is preferable to relying on recording order, which can change after
# filtering or concatenation. Missing/non-numeric targets stop the workflow.
# The random seed controls network initialization and minibatch shuffling;
# it does not create EEG, labels or a favourable target relationship.
participant_targets = dataset.description.set_index("subject")["p_factor"]
y = pd.to_numeric(
    metadata["subject"].map(participant_targets), errors="raise"
).to_numpy(dtype=np.float32)
assert np.isfinite(y).all()
torch.manual_seed(42)
torch.set_num_threads(1)
predictions = np.full(len(y), np.nan, dtype=np.float32)
baseline = np.full(len(y), np.nan, dtype=np.float32)

# %%
# 4. Fit signal scaling, target scaling and network inside each fold
# ------------------------------------------------------------------
# Three grouped folds reserve two participants at a time and train on the
# other four. Per-channel mean and standard deviation use only training windows
# and samples. Subtracting/dividing these arrays broadcasts over windows and
# time, producing dimensionless float32 network inputs. Target normalization
# uses one score per training participant, not one repeated value per window.
# The held-out scores never enter either normalization.
#
# ``EEGNet`` receives four channels, 200 time samples and one continuous output.
# ``F1=4`` and ``D=2`` keep the temporal/spatial filter bank small;
# ``kernel_length=32`` corresponds to 0.32 seconds at 100 Hz. These fixed choices
# are for a short CPU demonstration. Batch size 16 and two epochs expose the
# training operations without claiming that the network has converged.
#
# For each batch, zero old gradients, compute a prediction and squared-error
# loss on normalized targets, backpropagate, then update weights with Adam.
# The printed batch-mean training MSE measures optimization in normalized
# units; it is neither participant MAE nor held-out performance. ``eval()``
# switches dropout and normalization layers to inference behavior, while
# ``no_grad()`` disables gradient recording. Both are needed for a clear
# evaluation path; they serve different purposes.
#
# Finally, reverse the training target normalization so test predictions are
# back on the original score scale. The same fold's training mean supplies
# the non-EEG reference prediction.
for fold, (train, test) in enumerate(
    GroupKFold(n_splits=3).split(X_windows, y, groups)
):
    assert set(groups[train]).isdisjoint(groups[test])
    mean = X_windows[train].mean(axis=(0, 2), keepdims=True)
    scale = X_windows[train].std(axis=(0, 2), keepdims=True)
    assert (scale > 0).all()
    X = ((X_windows - mean) / scale).astype(np.float32)
    training_targets = participant_targets.loc[sorted(set(groups[train]))].astype(float)
    target_mean, target_scale = training_targets.mean(), training_targets.std(ddof=0)
    assert target_scale > 0
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X[train]),
            torch.from_numpy(
                ((y[train] - target_mean) / target_scale).astype(np.float32)
            ),
        ),
        batch_size=16,
        shuffle=True,
    )
    model = EEGNet(
        n_chans=len(channels), n_outputs=1, n_times=200, F1=4, D=2, kernel_length=32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2):
        model.train()
        losses = []
        for signals, targets in train_loader:
            optimizer.zero_grad()
            output = model(signals).reshape(-1)
            loss = torch.nn.functional.mse_loss(output, targets)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Fold {fold}, epoch {epoch + 1}, training MSE {np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        predictions[test] = (
            model(torch.from_numpy(X[test])).reshape(-1).numpy() * target_scale
            + target_mean
        )
    baseline[test] = target_mean
assert np.isfinite(predictions).all()

# %%
# 5. Score one prediction per held-out participant
# ------------------------------------------------
# Average the window predictions within each person, then compute MAE across
# those six people. Scoring all windows directly would treat repeated target
# measurements as independent observations and give extra weight to longer
# recordings. Here the target is constant within each group, so its grouped
# mean is still the original observed score.
#
# A prediction near the training mean can indicate that the short training run
# learned little person-specific information. A small MAE difference from the
# mean baseline on six people is not evidence of a reliable biomarker. The
# scatter plot and per-participant table show the actual outputs rather than a
# prescribed learning curve or minimum score.
results = (
    pd.DataFrame(
        {
            "subject": groups,
            "observed": y,
            "predicted": predictions,
            "training_mean": baseline,
        }
    )
    .groupby("subject")
    .mean()
)
assert len(results) == len(subjects)
print(results)
print("Participant MAE:", mean_absolute_error(results.observed, results.predicted))
print(
    "Training-mean MAE:", mean_absolute_error(results.observed, results.training_mean)
)
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
ax.scatter(results.observed, results.predicted)
limits = [results.observed.min(), results.observed.max()]
ax.plot(limits, limits, "k--")
ax.set(xlabel="Observed p-factor", ylabel="Held-out participant prediction")
plt.show()

# %%
# Develop a model using validation participants
# ---------------------------------------------
# To choose a longer epoch budget, reserve validation people inside each
# training fold and select the epoch there. Keep the outer test participants
# untouched until that choice is fixed. The feature-based p-factor project
# provides a useful simpler model, but its leave-one-out folds differ from
# these three grouped folds; use identical folds before making a paired model
# comparison.
