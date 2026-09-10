"""Track 3: predict time remaining until the first N2 epoch
====================================================================

Use three recorded Sleep-EDF participants from EEGDash ``nm000185``
(cassette63, cassette64, cassette65; night1), about 150 MB in total. Predict
once per five-second EEG window and evaluate on an unseen participant.
The `official tracks page <https://neural-interfaces26.github.io/tracks.html>`_
announces the exclusive wearable release for September 21, 2026. This page
uses the available PSG seed corpus, not an assumed Muse configuration.

The `NeuralBench Track 3 guide
<https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html>`_
defines a capped time-to-onset target and binned mean absolute error (bMAE).
We implement those target and metric conventions with real EEGDash data and
a small spectral Ridge baseline. Our three-person LOSO split and model are
not the competition's frozen split or official trained baseline.

Before running, install EEGDash and its Braindecode/scikit-learn dependencies.
No NeuralBench installation, GPU or prior feature file is required. Retain
signals with ``EEGDASH_CACHE_DIR``. The output contains each window's actual
target, held-out predictions, per-bin errors and their equally weighted mean.
"""

# %%
# 1. Load independent participants and inspect observed sleep stages
# ------------------------------------------------------------------------------
# A recording contains a sequence of scored stages. We need the full stage
# annotations to identify the first N2 onset, although only pre-onset voltage
# windows enter the model. This retrospective selection follows the task's
# annotation-based evaluation region; it is not a prospective onset detector
# that can choose its analysis region without knowing the reference onset.
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDashDataset
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)

subjects = ["cassette63", "cassette64", "cassette65"]
channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]
dataset = EEGDashDataset(
    dataset="nm000185",
    subject=subjects,
    session="night1",
    task="sleep",
    cache_dir=Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache")).expanduser(),
    n_jobs=1,
)
print(dataset.description.to_string(index=False))
assert len(dataset.datasets) == len(subjects)
assert set(dataset.description.subject) == set(subjects)

# %%
# 2. Tile the last twenty pre-onset minutes into five-second windows
# ------------------------------------------------------------------------------
# The implementation of `AddSleepOnsetTargets
# <https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/transforms.py>`_
# selects the earliest scored N2 start. It does not require a 60-second N2
# run: the guide's informal "stable" wording must not add a persistence rule.
# Its configured region starts at max(recording_start, N2_onset - 1200 s)
# and ends at N2 onset. No-N2 records produce no official task windows;
# this explicit subset instead fails visibly if a required onset is absent.
#
# Use the source 100 Hz sampling grid. A window is (2 bipolar derivations,
# 500 samples) in volts, covering [start, stop). Stop is the next sample
# boundary, not the timestamp of the last included sample. Target derivation
# therefore uses i_stop_in_trial / sfreq. Cropping after acquisition cannot
# reduce the full-file download, and trial metadata must retain both boundaries.
windowed_recordings = []
metadata_tables = []
for recording in dataset.datasets:
    raw = recording.raw
    print(
        recording.description.subject,
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
    assert raw.info["sfreq"] == 100 and set(channels).issubset(raw.ch_names)
    n2_onsets = (
        raw.annotations.onset[raw.annotations.description == "N2"] - raw.first_time
    )
    if not len(n2_onsets):
        raise ValueError(f"No observed N2 onset for {recording.description.subject}")
    onset = float(n2_onsets.min())
    region_start = max(0.0, onset - 1200.0)
    region_stop = min(onset, raw.n_times / raw.info["sfreq"])
    start_sample = int(np.ceil(region_start * raw.info["sfreq"]))
    stop_sample = int(np.floor(region_stop * raw.info["sfreq"]))
    assert stop_sample - start_sample >= 500, "No full pre-onset window"
    raw.pick(channels).reorder_channels(channels)
    windows = create_fixed_length_windows(
        BaseConcatDataset([recording]),
        start_offset_samples=start_sample,
        stop_offset_samples=stop_sample,
        window_size_samples=500,
        window_stride_samples=500,
        on_last_window="drop",
        preload=True,
    )
    metadata = windows.get_metadata().reset_index(drop=True)
    metadata["window_stop_s"] = metadata.i_stop_in_trial / raw.info["sfreq"]
    metadata["n2_onset_s"] = onset
    # This observed-annotation transformation matches SleepOnsetTargetExtractor.
    metadata["target"] = np.clip(onset - metadata.window_stop_s, 0.0, 600.0)
    assert (metadata.window_stop_s <= onset + 1e-9).all()
    assert (metadata.i_stop_in_trial - metadata.i_start_in_trial == 500).all()
    windowed_recordings.append(windows)
    metadata_tables.append(metadata)

# %%
# 3. Compute EEGDash band powers without using the onset as a predictor
# ---------------------------------------------------------------------------------
# The two source channels are recorded bipolar derivations, not independent
# Fpz/Cz/Pz/Oz electrodes. Preserve their reference. Five-second Hann-Welch
# segments give 0.2 Hz bins; one shared spectrum supplies four half-open bands.
# Multiplying summed PSD bins by their spacing approximates band power in V².
# Log power gives eight predictor columns. Neither window time, onset time,
# participant ID nor target enters X. No full-recording normalization or
# additional EEGPrep cleaning is fitted: this page isolates the target and
# evaluation contract rather than an artifact-removal protocol.
windows = BaseConcatDataset(windowed_recordings)
metadata = pd.concat(metadata_tables, ignore_index=True)
spectral = FeatureExtractor(
    {
        "power": partial(
            spectral_bands_power,
            bands={
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
            },
        )
    },
    preprocessor=partial(
        spectral_preprocessor,
        fs=100,
        nperseg=500,
        noverlap=0,
        window="hann",
        f_min=1,
        f_max=30,
    ),
)
feature_table = extract_features(
    windows, {"spectral": spectral}, batch_size=64, n_jobs=1
).to_dataframe()
X = np.log10(np.maximum(feature_table.to_numpy() * 0.2, 1e-30))
y = metadata.target.to_numpy(dtype=float)
groups = metadata.subject.astype(str).to_numpy()
assert X.shape == (len(metadata), 8) and np.isfinite(X).all()
assert np.isfinite(y).all() and ((0 <= y) & (y <= 600)).all()
assert not metadata.duplicated(["subject", "session", "i_start_in_trial"]).any()
print("Feature matrix:", X.shape)
print(
    metadata[["subject", "window_stop_s", "n2_onset_s", "target"]]
    .groupby("subject")
    .agg(["min", "max", "count"])
)

# %%
# 4. Predict every window from a participant held out of fitting
# --------------------------------------------------------------------------
# A fresh scaler and Ridge model fit only the other two participants. Ridge
# uses fixed alpha=10; the dummy reference predicts the training-window mean.
# We clip both predictions to the known target range as a fixed output choice.
# This is a small CPU regression demonstration; it does not reproduce the
# guide's neural model or regression-bin training sampler.
#
# Every window for a test person stays out of training, including neighboring
# windows from the same night. Thousands of windows would still be only three
# independent people. Hyperparameter selection needs additional validation
# participants within training, not feedback from these outer test predictions.
predicted, baseline = np.empty_like(y), np.empty_like(y)
test_counts = np.zeros(len(y), dtype=int)
for train, test in LeaveOneGroupOut().split(X, y, groups):
    assert set(groups[train]).isdisjoint(groups[test])
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    predicted[test] = np.clip(model.fit(X[train], y[train]).predict(X[test]), 0, 600)
    baseline[test] = np.clip(
        DummyRegressor().fit(X[train], y[train]).predict(X[test]), 0, 600
    )
    test_counts[test] += 1
assert (test_counts == 1).all() and np.isfinite(predicted).all()

# %%
# 5. Compute bMAE using ground-truth time-to-onset bins
# -----------------------------------------------------------------
# NeuralBench's `BinnedMAE implementation
# <https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/metrics.py>`_
# uses [0,40), [40,90), [90,300), and [300,600]. Interior edge values enter
# the higher bin; 600 belongs to the final bin. First compute mean absolute
# error within each ground-truth bin, then average the nonempty bin means
# equally. Thus the many capped targets do not dominate the headline score.
# The metric does not average absolute recording-onset estimates.
#
# Bin counts and per-person scores make the small evaluation inspectable.
# The pooled bMAE averages each bin's held-out window errors before averaging
# bins, matching the metric's aggregation. A mean of participant bMAEs is a
# different aggregation when their bin counts differ.
bin_edges = np.asarray([0.0, 40.0, 90.0, 300.0, 600.0])
bin_ids = np.searchsorted(bin_edges[1:-1], y, side="right")
assert set(bin_ids) == {0, 1, 2, 3}, "This subset should cover all four bins"
for edge, expected_bin in zip(bin_edges, [0, 1, 2, 3, 3], strict=True):
    assert (y == edge).any(), "The selected records should exercise each bin edge"
    assert (bin_ids[y == edge] == expected_bin).all()
errors = pd.DataFrame(
    {
        "subject": groups,
        "bin": bin_ids,
        "Ridge": np.abs(predicted - y),
        "training mean": np.abs(baseline - y),
    }
)
per_bin = errors.groupby("bin")[["Ridge", "training mean"]].mean()
bmae = per_bin.mean()
print("Held-out window counts by subject and bin:")
print(pd.crosstab(groups, bin_ids))
print("Per-bin MAE (seconds):\n", per_bin)
print("Pooled bMAE (seconds):\n", bmae)
print(
    "Per-participant bMAE (seconds):\n",
    errors.groupby(["subject", "bin"])[["Ridge", "training mean"]]
    .mean()
    .groupby("subject")
    .mean(),
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
per_bin.plot.bar(ax=axes[0], rot=0)
axes[0].set(
    xticklabels=["[0,40)", "[40,90)", "[90,300)", "[300,600]"],
    xlabel="Observed time-to-onset bin (s)",
    ylabel="Held-out MAE (s)",
)
axes[1].scatter(y, predicted, s=8, alpha=0.4)
axes[1].plot([0, 600], [0, 600], "k--")
axes[1].set(xlabel="Observed time remaining (s)", ylabel="Predicted time remaining (s)")
plt.show()

# %%
# 6. Extend the task without claiming a wearable benchmark
# --------------------------------------------------------------------
# A target of 600 means "at least ten minutes remaining," so adding that
# prediction to the window stop does not recover a unique onset timestamp.
# For uncapped pre-onset targets, window_stop + prediction can be interpreted
# as an onset estimate; choosing windows using the true onset remains a
# retrospective evaluation convention, not a deployment-time selection rule.
#
# Add sleepers and keep all nights of a person in one split. Verify the
# released wearable channels, timing, labels and official evaluation package
# before changing devices; no Muse dataset name or configuration is guessed
# here. For an official run, follow the linked NeuralBench guide and its
# released data/configuration rather than comparing this three-person result
# directly with a competition leaderboard.
