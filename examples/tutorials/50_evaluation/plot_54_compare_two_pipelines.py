"""Compare pipelines on paired real subject folds
==============================================

Compare logistic regression and shrinkage LDA using identical held-out
participants. Pair scores by participant; trials are not independent
replicates for a population comparison.

Data: Nakanishi2015, NEMAR ``nm000118``, subjects 1–3, session 0,
run 0: approximately 21.1 MB on first download. Set ``EEGDASH_CACHE_DIR``
to reuse the cache. This processed release already includes filtering,
downsampling and latency handling; do not add another latency correction.
See the `source study <https://doi.org/10.1371/journal.pone.0140703>`_
and `NEMAR release <https://nemar.org/dataset/nm000118>`_.

Prerequisites: tutorial 51's LOSO loop and scikit-learn pipelines. This page
loads its own recordings and requires SciPy for the optional paired test.
The question is whether one fixed classifier improves the same participants'
scores. Neither classifier is selected or tuned using these test results.

"""

# %%
# 1. Select a small, explicit cohort
# ----------------------------------
# Filtering subjects, session and run bounds the download. Cropping after
# opening a recording would reduce computation but not its download size.
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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
assert len(dataset.datasets) == len(subjects), "Expected one recording per subject"
print(dataset.description[["subject", "session", "run"]])

# %%
# 2. Inspect real annotations and verify the signal contract
# ----------------------------------------------------------
# Accessing ``raw`` downloads that recording. Annotation names identify the
# attended stimulus frequency in Hz; they supply every classification label.
# All participants must have the same channel order and sampling frequency.
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
assert len(mapping) == 12, "Expected the twelve SSVEP stimulus frequencies"
for recording in dataset.datasets:
    recording_raw = recording.raw
    assert recording_raw.ch_names == channel_names
    assert recording_raw.info["sfreq"] == sfreq
    assert set(recording_raw.annotations.description) == set(mapping)
print(f"Channels: {channel_names}; sampling frequency: {sfreq} Hz")
print("Stimulus frequencies (Hz):", class_names)

# %%
# 3. Make one four-second window per annotated trial
# --------------------------------------------------
# Each annotated interval is 4.15 seconds. Keep its first four seconds and
# discard the remainder. Explicit size and stride avoid overlapping windows
# or extending the epoch beyond the recorded event duration.
# At 256 Hz, four seconds contain 1,024 samples. The resulting array has
# axes (540 trials, 8 EEG channels, 1,024 samples), with volt-valued data.
# The metadata has one row per array row. ``target`` is a class index, not a
# frequency in Hz; ``mapping`` is the explicit conversion between them.
# The source contains 15 trials of each of the 12 frequencies per person.
# A missing class is a data-contract failure, not a reason to relabel trials.
window_size = int(4 * sfreq)
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size,
    window_stride_samples=window_size,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata()
assert (metadata.i_window_in_trial == 0).all(), "Expected one window per trial"
assert not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
y = metadata["target"].to_numpy(dtype=int)
groups = metadata["subject"].astype(str).to_numpy()
X = np.stack([window[0] for window in windows])
assert X.shape == (len(metadata), len(channel_names), window_size)
assert set(groups) == set(subjects)
assert np.isfinite(X).all()
print(pd.crosstab(groups, y, rownames=["subject"], colnames=["class"]))

# %%
# 4. Extract spectral features from each window
# ---------------------------------------------
# SSVEP responses contain energy at the stimulus frequency. Use log spectral
# power around each stimulus frequency, retaining all eight posterior channels.
# This per-window transform learns nothing from other trials or subjects.
# The scaler below, in contrast, must be fitted only on training subjects.
# EEGDash's shared spectral preprocessor computes a Welch PSD with one
# four-second Hann segment and 0.25 Hz bins. Each narrow band is centered
# on a documented stimulus frequency; these centers define the task, not
# trial-specific predictors. Retaining eight channels gives 12 × 8 = 96
# features. The already processed SSVEP release does not need another EEGPrep
# cleaning pass or visual latency correction.
#
# spectral_bands_power sums selected PSD bins. Multiplying by their 0.25 Hz
# spacing converts V²/Hz to approximate band power in V². The log compresses
# that scale; the StandardScaler still fits only on training participants.
bands = {
    f"hz_{name}": (float(name) - 0.125, float(name) + 0.125) for name in class_names
}
spectral = FeatureExtractor(
    {"power": partial(spectral_bands_power, bands=bands)},
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
    windows, {"spectral": spectral}, batch_size=64, n_jobs=1
).to_dataframe()
assert feature_table.shape == (len(y), len(class_names) * len(channel_names))
features = np.log(np.maximum(feature_table.to_numpy() * sfreq / window_size, 1e-30))
assert np.isfinite(features).all()

# %%
# 5. Evaluate both classifiers on exactly the same LOSO folds
# -----------------------------------------------------------
# Both pipelines fit their scaler on training participants only. All choices
# are fixed before evaluation; tune alternatives within training folds.
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# %%
# Shrinkage LDA regularizes its class-shared covariance estimate, which helps
# when spectral bins are correlated. ``solver="lsqr"`` supports that shrinkage;
# ``shrinkage="auto"`` estimates its strength from the training fold. Logistic
# regression uses its default L2 penalty. Both consume the same 96 features
# and exactly the same 360/180 training/test trial assignment in each fold.
rows = []
for train, test in LeaveOneGroupOut().split(features, y, groups):
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(y[train]) == set(y[test]) == set(mapping.values())
    row = {"subject": groups[test][0]}
    for name, classifier in {
        "Logistic": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }.items():
        model = make_pipeline(StandardScaler(), classifier)
        model.fit(features[train], y[train])
        row[name] = balanced_accuracy_score(y[test], model.predict(features[test]))
    rows.append(row)
results = pd.DataFrame(rows).set_index("subject")
assert len(results) == len(subjects) and results.index.is_unique
print(results)
difference = results["LDA"] - results["Logistic"]
print("Paired differences (LDA minus Logistic):", difference.to_dict())
# With three participants a two-sided test has very little resolution. Exact
# signed-rank inference below requires nonzero differences with distinct ranks;
# report the measured differences alone when zeros or ties violate that case.
if (difference != 0).all() and difference.abs().is_unique:
    print("Exploratory exact Wilcoxon:", wilcoxon(difference, method="exact"))
else:
    print("Zeros or tied absolute differences: report paired differences only.")

# %%
# 6. Connect the two scores for each participant
# ----------------------------------------------
for subject, row in results.iterrows():
    plt.plot(["Logistic", "LDA"], row, "o-", label=f"Subject {subject}")
plt.ylabel("LOSO balanced accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()

# %%
# 7. Interpret the paired comparison
# ----------------------------------
# A connecting line slopes upward when LDA improves that participant's mean
# class recall; a downward line favors logistic regression. Pairing removes
# the misleading comparison that would arise from testing the two models on
# different people. The unit of replication remains the participant, not each
# of the 540 trials.
#
# The signed-rank calculation uses the magnitudes and signs of three paired
# differences and assumes a symmetric distribution of differences for its usual
# location interpretation. Distinct nonzero absolute differences permit the
# exact small-sample calculation used here. Three pairs cannot support strong
# evidence: even all differences in the same direction give a two-sided exact
# p-value of 0.25. Report effect sizes and additional independent participants
# before claiming a reliable advantage.
