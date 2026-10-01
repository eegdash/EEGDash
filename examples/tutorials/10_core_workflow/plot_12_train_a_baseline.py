"""Train a baseline on real SSVEP trials
=====================================

Fit a first classical spectral baseline on subjects 1 and 2 and evaluate subject 3.
This uses scikit-learn, not a neural network; the same window/label contract applies.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
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
cache_dir = get_default_cache_dir()
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
if len(dataset.datasets) != len(subjects):
    raise ValueError(
        "Query did not return one recording per requested subject; inspect dataset.description"
    )
dataset.description[["subject", "session", "run"]]

# %%
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
for recording in dataset.datasets:
    other = recording.raw
    if (
        other.ch_names != channel_names
        or other.info["sfreq"] != sfreq
        or set(other.annotations.description) != set(mapping)
    ):
        raise ValueError(
            "Recordings must share channel order, sample rate and event vocabulary"
        )
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
if (
    len(windows) != len(metadata)
    or metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
):
    raise ValueError("Window rows must have aligned, unique recording/start identities")
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
        window="hamming",
        f_min=8,
        f_max=16,
    ),
)
feature_table = extract_features(
    windows, spectral, batch_size=64, n_jobs=1
).to_dataframe()
features = np.log(np.maximum(feature_table.to_numpy(), 1e-30))
if not (np.isfinite(features).all()):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )
print("EEGDash spectral features:", features.shape)
feature_table.head()

# %%
# Inspect only training rows before fitting; never choose bins from subject 3.
groups = metadata["subject"].astype(str).to_numpy()
train, test = groups != "3", groups == "3"
first_train = np.flatnonzero(train)[0]
freqs, density = spectral_preprocessor(
    windows[first_train][0][None],
    _metadata={"info": raw.info},
    fs=sfreq,
    nperseg=window_size,
    noverlap=0,
    window="hamming",
    f_min=8,
    f_max=16,
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].plot(freqs, density[0, 0] * 1e12)
for name in class_names:
    axes[0].axvline(float(name), color="gray", alpha=0.4)
axes[0].set(
    xlabel="Frequency (Hz)",
    ylabel="PSD (µV²/Hz)",
    title=f"Training trial: {channel_names[0]}",
)
# Select named columns rather than assuming the flattened feature order.
channel_columns = [f"power_{name}_{channel_names[0]}" for name in class_names]
class_means = np.array(
    [
        np.log10(
            np.maximum(feature_table.loc[train & (y == label), channel_columns], 1e-30)
        ).mean(axis=0)
        for label in range(len(class_names))
    ]
)
image = axes[1].imshow(class_means, aspect="auto")
axes[1].set(
    xticks=range(len(class_names)),
    xticklabels=class_names,
    yticks=range(len(class_names)),
    yticklabels=class_names,
    xlabel="Feature frequency (Hz)",
    ylabel="Training target (Hz)",
    title="Training-only mean log10 PSD-bin sum",
)
axes[1].tick_params(axis="x", rotation=90)
fig.colorbar(image, ax=axes[1], label="log10(sum PSD / (1 V²/Hz))")
plt.show()

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
# score. Rows of the count confusion matrix correspond to true frequencies;
# columns show predicted frequencies. Concentration near the diagonal indicates
# correct decisions; an off-diagonal pattern shows which frequencies are
# confused. The code accepts a low measured score as a valid outcome.
groups = metadata["subject"].astype(str).to_numpy()
train, test = groups != "3", groups == "3"
if not (set(groups[train]).isdisjoint(groups[test])):
    raise ValueError("Training and test participants overlap; fix the evaluation split")
if not (set(y[train]) == set(y[test]) == set(mapping.values())):
    raise ValueError(
        "Required event classes are missing; inspect annotation/retained-condition counts"
    )
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(features[train], y[train])
predictions = model.predict(features[test])
score = balanced_accuracy_score(y[test], predictions)
print(f"Held-out subject 3 balanced accuracy: {score:.3f}; chance: {1 / 12:.3f}")
ConfusionMatrixDisplay.from_predictions(
    y[test],
    predictions,
    display_labels=class_names,
    normalize=None,
    include_values=True,
    xticks_rotation=90,
)
plt.title(
    f"Subject 3: counts; balanced accuracy {score:.2f}; chance {1 / len(mapping):.2f}"
)
plt.show()

# %%
# Extend the experiment without reusing the test set
# --------------------------------------------------
# A single subject can be unusually easy or difficult. Tutorial 51 evaluates
# all three subjects once with LOSO to expose that variation. Before trying a
# neural decoder or altering the spectral band, choose a validation protocol
# inside the training cohort; repeated inspection of subject 3 would turn it
# into development data.
