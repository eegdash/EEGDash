"""Learning curves from real training subjects
===========================================

How does adding one training participant change validation performance?
Use nested training-subject subsets and keep subject 3 fixed for validation.
Two training sizes illustrate the procedure, not a scaling law.

Data: Nakanishi2015, NEMAR ``nm000118``, subjects 1–3, session 0,
run 0: approximately 21.1 MB on first download. Set ``EEGDASH_CACHE_DIR``
to reuse the cache. This processed release already includes filtering,
downsampling and latency handling; do not add another latency correction.
See the `source study <https://doi.org/10.1371/journal.pone.0140703>`_
and `NEMAR release <https://nemar.org/dataset/nm000118>`_.

Prerequisites: tutorial 51's subject-disjoint fit and evaluation, plus the
spectral baseline from tutorial 12. No saved model or earlier output file is
needed. The output is a two-point validation curve, with both participant
count and trial count reported so the horizontal axis is unambiguous.

"""

# %%
# 1. Select a small, explicit cohort
# ----------------------------------
# Filtering subjects, session and run bounds the download. Cropping after
# opening a recording would reduce computation but not its download size.
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.paths import get_default_cache_dir

from eegdash import EEGDashDataset
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)

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
if not (len(dataset.datasets) == len(subjects)):
    raise ValueError(
        "Expected one recording per requested participant; inspect the query results."
    )
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
for recording in dataset.datasets:
    recording_raw = recording.raw
    if (
        recording_raw.ch_names != channel_names
        or recording_raw.info["sfreq"] != sfreq
        or set(recording_raw.annotations.description) != set(mapping)
    ):
        raise ValueError(
            "Recordings must share channel order, sampling rate and cue vocabulary."
        )
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
if not ((metadata.i_window_in_trial == 0).all()):
    raise ValueError(
        "Multiple windows represent a trial; group by trial before splitting."
    )
if not (
    not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
):
    raise ValueError(
        "Duplicate recording/window identities; inspect metadata before splitting."
    )
y = metadata["target"].to_numpy(dtype=int)
groups = metadata["subject"].astype(str).to_numpy()
if not (set(groups) == set(subjects)):
    raise ValueError(
        "Some requested participants have no retained windows; inspect exclusions."
    )
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
if len(feature_table) != len(metadata):
    raise ValueError(
        "Feature rows no longer match window metadata; inspect extraction."
    )
features = np.log(np.maximum(feature_table.to_numpy() * sfreq / window_size, 1e-30))
if not (np.isfinite(features).all()):
    raise ValueError(
        "Nonfinite spectral features; inspect signals and extraction parameters."
    )

# %%
# 5. Grow a nested training cohort; keep validation fixed
# -------------------------------------------------------
# Show both possible starting participants at the same download budget.
# Repeated inspection makes subject 3 a validation set, not a final test set.
# A larger study needs additional untouched test subjects and repeated orders.
# Both training subsets contain every frequency class. The one-participant
# subset contributes 180 trials; the two-participant subset contains those
# same trials plus the other person's 180 trials. This nesting prevents a
# loss of the earlier trials; identity and sample size still change together.
# Subject 3 supplies the same 180 validation trials at both points.
#
# The feature extraction is fixed, but the scaler and classifier are refitted
# for each size. Reusing the larger fit would let information from the added
# participant enter the smaller-size result.
validation = np.flatnonzero(groups == "3")
rows = []
for training_order in [("1", "2"), ("2", "1")]:
    previous = set()
    for n_subjects in (1, 2):
        train = np.flatnonzero(np.isin(groups, training_order[:n_subjects]))
        if not (previous.issubset(set(train))):
            raise ValueError(
                "Training subsets are not nested; retain all earlier trials."
            )
        if not (set(groups[train]).isdisjoint(groups[validation])):
            raise ValueError(
                "Validation participants overlap training; repair the split."
            )
        if not (set(y[train]) == set(y[validation]) == set(mapping.values())):
            raise ValueError(
                "Every mapped class must occur in train and validation; inspect class counts."
            )
        previous = set(train)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features[train], y[train])
        prediction = model.predict(features[validation])
        rows.append(
            dict(
                order=" → ".join(training_order),
                n_subjects=n_subjects,
                n_trials=len(train),
                balanced_accuracy=balanced_accuracy_score(y[validation], prediction),
            )
        )
results = pd.DataFrame(rows)
print(results.to_string(index=False))

# %%
# 6. Display measured validation scores without extrapolation
# -----------------------------------------------------------
for order, curve in results.groupby("order", sort=False):
    plt.plot(curve.n_subjects, curve.balanced_accuracy, "o-", label=order)
plt.axhline(1 / len(mapping), color="black", linestyle="--", label="Chance (1/12)")
plt.xticks([1, 2])
plt.xlabel("Training participants")
plt.ylabel("Subject 3 validation balanced accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()

# %%
# 7. Read a small learning curve without over-interpreting it
# -----------------------------------------------------------
# A rising segment indicates improvement on this one validation participant
# for this particular order of training subjects. A flat or falling segment
# is equally valid: adding a participant changes both size and population
# composition. Both orders share the same two-person endpoint; they are not
# independent cohorts and do not supply population uncertainty.
#
# With a larger cohort, repeat several nested training-subject orders while
# keeping validation identities fixed, and report the spread at each size.
# Reserve additional untouched people for the final chosen model. Do not
# extrapolate a sample-efficiency law from these two points.
