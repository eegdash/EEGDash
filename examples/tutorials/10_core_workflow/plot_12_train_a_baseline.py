"""Train a baseline on real SSVEP trials
=====================================

Fit a spectral decoder on subjects 1 and 2 and evaluate subject 3.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 3 participant(s), about 21.1 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies, including scikit-learn. Tutorials 02
and 11 explain windows and subject-disjoint splits. This script repeats the
data preparation so it runs independently and returns actual predictions
for subject 3 from a model fitted to subjects 1 and 2.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The classification question is which of twelve flicker frequencies a trial
# was labelled with. Annotation names are converted to integer targets in
# numerical frequency order, and the same mapping is checked across subjects.
# This baseline keeps the source reference and source preprocessing; it does
# not load the optional average-referenced output from tutorial 10.
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
subjects = ["1", "2", "3"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="nm000118",
    subject=subjects,
    session="0",
    run="0",
    task="ssvep",
    n_jobs=1,
)
assert len(dataset.datasets) == len(subjects)
print(dataset.description[["subject", "session", "run"]])
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
assert len(mapping) == 12
for recording in dataset.datasets:
    assert recording.raw.ch_names == channel_names
    assert recording.raw.info["sfreq"] == sfreq
    assert set(recording.raw.annotations.description) == set(mapping)
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# 2. Window the observed trials
# -----------------------------
# Each four-second trial is an ``(8, 1024)`` voltage array. EEGDash will read
# these windows in batches, preserving row alignment with the metadata and
# target vector. The class-count table confirms what each subject contributes
# before a training split is applied.
# The source event lasts 4.15 seconds; its final 0.15 seconds are unused.
window_size = int(4 * sfreq)
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    window_size_samples=window_size,
    window_stride_samples=window_size,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata().reset_index(drop=True)
y = metadata["target"].to_numpy(dtype=int)
assert len(windows) == len(metadata)
assert set(y) == set(mapping.values())
assert not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
print(pd.crosstab(metadata["subject"], y))

print(
    "Windows:",
    len(windows),
    "with",
    len(channel_names),
    "channels and",
    window_size,
    "samples",
)

# %%
# 3. Summarize stimulus-band power without learning from other trials
# -------------------------------------------------------------------
# EEGDash's spectral preprocessor computes Welch power on each real window.
# A four-second segment gives a 0.25 Hz grid. The narrow bands below each
# contain the grid point at one of the twelve observed stimulus frequencies;
# their half-bin margins exclude neighbouring bins. The same definition is
# used for every trial, regardless of its label.
#
# ``FeatureExtractor`` shares the spectrum across all band/channel outputs.
# Twelve frequency bands times eight channels give 96 named columns. The
# power function sums PSD values in each band; keep the resolution fixed when
# comparing their scale. Log compression and a positive numerical floor act
# per value, without learning anything from other trials. The scaler below
# still needs to be fitted only on training participants.
stimulus_bands = {
    name: (float(name) - 0.125, float(name) + 0.125) for name in class_names
}
spectral = FeatureExtractor(
    {"power": partial(spectral_bands_power, bands=stimulus_bands)},
    preprocessor=partial(
        spectral_preprocessor,
        fs=sfreq,
        nperseg=window_size,
        noverlap=0,
        f_min=8,
        f_max=16,
    ),
)
feature_table = extract_features(
    windows, spectral, batch_size=64, n_jobs=1
).to_dataframe()
features = np.log(np.maximum(feature_table.to_numpy(), 1e-30))
assert features.shape == (len(metadata), len(mapping) * len(channel_names))
assert np.isfinite(features).all()
print("EEGDash spectral features:", features.shape)

# %%
# 4. Fit all learned transformations on training participants
# -----------------------------------------------------------
# The Boolean masks select complete participants. ``StandardScaler`` learns a
# mean and scale for each frequency feature from training rows only, then
# applies them to subject 3. Logistic regression learns twelve-class decision
# weights on those scaled features. ``max_iter=1000`` is an optimization limit,
# not a number of EEG training trials and not a promise of convergence.
#
# Balanced accuracy averages recall over the twelve classes. The displayed
# ``1/12`` is the expected value for uniform random guessing, not a simulated
# score. Rows of the normalized confusion matrix correspond to true frequencies;
# columns show predicted frequencies. Concentration near the diagonal indicates
# correct decisions; an off-diagonal pattern shows which frequencies are
# confused. The code accepts a low measured score as a valid outcome.
groups = metadata["subject"].astype(str).to_numpy()
train, test = groups != "3", groups == "3"
assert set(groups[train]).isdisjoint(groups[test])
assert set(y[train]) == set(y[test]) == set(mapping.values())
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(features[train], y[train])
predictions = model.predict(features[test])
score = balanced_accuracy_score(y[test], predictions)
print(f"Held-out subject 3 balanced accuracy: {score:.3f}; chance: {1 / 12:.3f}")
ConfusionMatrixDisplay.from_predictions(
    y[test],
    predictions,
    display_labels=class_names,
    normalize="true",
    include_values=False,
    xticks_rotation=90,
)
plt.title("Subject 3: observed frequency (Hz)")
plt.show()

# %%
# Extend the experiment without reusing the test set
# --------------------------------------------------
# A single subject can be unusually easy or difficult. Tutorial 51 evaluates
# all three subjects once with LOSO to expose that variation. Before trying a
# neural decoder or altering the spectral band, choose a validation protocol
# inside the training cohort; repeated inspection of subject 3 would turn it
# into development data.
