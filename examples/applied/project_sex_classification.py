"""Classify recorded sex metadata from real EEG
============================================

Use participant-level features and stratified participant folds. The F/M
labels describe this release metadata; they do not determine gender identity.

Use six explicitly selected participants from the real
`HBN ds005505 release <https://openneuro.org/datasets/ds005505>`_. Their
RestingState signal files total approximately 595.8 MB, cached under
``EEGDASH_CACHE_DIR``. This applied example is larger than the introductory
21 MB SSVEP subset. Cropping after loading reduces computation, not download.
Targets come from the observed participant metadata. Six participants are
sufficient to exercise the workflow, not to support clinical conclusions.

Before you start
----------------
Install EEGDash with its EEGPrep tutorial dependencies and scikit-learn. Tutorials 02,
11 and 40 introduce windows, grouped evaluation and spectral features. This
file runs independently: it predicts the recorded F/M sex field from one
feature row per participant and prints each held-out prediction.
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
from functools import partial
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
# 1. Load the named cohort and inspect observed participant metadata
# ------------------------------------------------------------------
# The six selected records contain three F and three M labels in the release's
# ``sex`` field. These are the observed categories used by this example; they
# must not be inferred from participant names, appearance or EEG. The cohort
# selection fixes this demonstration's class balance, not population prevalence.
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
# channel rejection or ASR. The feature models below use only 1–30 Hz.
#
# Fixed-length windows are used because the target belongs to a person, not an
# event. They do not relabel eyes-open/closed intervals or remove every
# instruction or artifact from the first minute. Equal size and stride avoid
# overlap; a final short remainder is discarded. EEGDash reads the windows
# in batches shaped ``(windows, 4 channels, 200 samples)``. Check the printed
# window count rather than assuming each recording retains the same number.
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
groups = metadata["subject"].astype(str).to_numpy()
assert len(windows) == len(metadata)
assert not metadata.duplicated(["subject", "i_start_in_trial"]).any()
print("Real two-second windows:", len(windows), "with", len(channels), "channels")

# %%
# 3. Average window band powers into one feature row per participant
# ------------------------------------------------------------------
# EEGDash's ``FeatureExtractor`` shares one Welch spectrum between the four
# band-power outputs. ``nperseg=200`` spans the two-second window, giving a
# 0.5 Hz grid; four bands times four electrodes produce 16 named columns.
# ``spectral_bands_power`` sums PSD values in each band. Its scale depends on
# that fixed grid, so preserve the rate and segment length when reusing this
# feature definition. No hand-written FFT or band-reduction function is needed.
#
# Log compression acts on each trial's feature values. We then group the
# resulting table by subject and average within each person before fitting.
# Thus ``X`` has six rows, and no participant gains weight merely because
# more windows were retained. The same sorted index selects the observed
# participant targets, preserving feature/target alignment.
bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
spectral = FeatureExtractor(
    {"power": partial(spectral_bands_power, bands=bands)},
    preprocessor=partial(
        spectral_preprocessor, fs=100, nperseg=200, noverlap=0, f_min=1, f_max=30
    ),
)
feature_table = extract_features(
    windows, spectral, batch_size=64, n_jobs=1
).to_dataframe()
assert feature_table.shape == (len(metadata), len(bands) * len(channels))
participant_features = (
    np.log10(feature_table.clip(lower=1e-30))
    .assign(subject=groups)
    .groupby("subject")
    .mean()
)
identities = participant_features.index.to_numpy()
X = participant_features.to_numpy()
participants = dataset.description.set_index("subject").loc[identities]
assert np.isfinite(X).all() and len(X) == len(subjects)

y = participants["sex"].to_numpy()
assert set(y) == {"F", "M"}
assert pd.Series(y).value_counts().min() >= 3

# %%
# 4. Evaluate each participant exactly once
# -----------------------------------------
# Because there is now exactly one row per person, ``StratifiedKFold`` operates
# at the participant level. Three folds with this 3F/3M cohort place one person
# from each class in each test fold. The group-disjoint assertion still checks
# that row identities remain separate. If you instead fit window rows, ordinary
# stratified folds would not provide that protection.
#
# Scaling and logistic-regression fitting restart in every fold. Balanced
# accuracy averages the recall for F and M, and the pooled confusion matrix
# counts held-out participants, not windows. Each person appears once, as
# checked by ``counts``. With only six people, one changed prediction has a
# large effect; the result should not be interpreted as a stable population
# estimate or as a statement about any individual's gender identity.
predicted = np.empty(len(y), dtype=object)
counts = np.zeros(len(y), dtype=int)
for train, test in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
    X, y
):
    assert set(identities[train]).isdisjoint(identities[test])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    predicted[test] = model.fit(X[train], y[train]).predict(X[test])
    counts[test] += 1
assert (counts == 1).all()
print(pd.DataFrame({"subject": identities, "observed": y, "predicted": predicted}))
print("Participant balanced accuracy:", balanced_accuracy_score(y, predicted))
ConfusionMatrixDisplay.from_predictions(y, predicted)
plt.title("Held-out participants: recorded sex field")
plt.show()

# %%
# Inspect confounds before expanding the classifier
# -------------------------------------------------
# In a larger cohort, examine age and site distributions alongside the sex
# field and keep acquisition-related groups separate where the deployment
# question requires it. Choose features and regularization using training
# validation participants; selecting them from this six-person result would
# reuse the test outcomes.
