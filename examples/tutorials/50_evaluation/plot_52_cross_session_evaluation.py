"""Transfer a decoder between recorded sessions
============================================

Train on one acquisition session and predict another for the same person.
Load subject 1, sessions ``0train`` and ``1train``, run 0 of real left/right
motor imagery from `NEMAR nm000135 <https://nemar.org/dataset/nm000135>`_
(BNCI2014-004). Each signal file is approximately 5.5 MB; the first run
needs internet. Set ``EEGDASH_CACHE_DIR`` to reuse recordings in CI.
The available catalogue subset supports a session demonstration for one
participant, not a population generalization claim.

Prerequisites: tutorial 11's distinction between trial and group splits,
and tutorial 12's scaler/classifier pipeline. Run this page independently
with EEGDash, Braindecode and scikit-learn installed. Session names are BIDS
acquisition identities. Their ``train`` suffix belongs to the source release;
it does not prevent us from reserving the second session for evaluation.

"""

# %%
# 1. Load two genuine session identifiers
# ---------------------------------------
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
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

sessions = ["0train", "1train"]
dataset = EEGDashDataset(
    cache_dir=get_default_cache_dir(),
    dataset="nm000135",
    subject="1",
    session=sessions,
    run="0",
    task="imagery",
    n_jobs=1,
)
if not (len(dataset.datasets) == 2):
    raise ValueError(
        "Invalid len(dataset.datasets) == 2; inspect cohort, windows and split before fitting."
    )
print(dataset.description[["subject", "session", "run"]])

# %%
# 2. Inspect observed hand labels and select EEG
# ----------------------------------------------
# The release was converted through MOABB. Preserve its preprocessing and
# event timing; selecting EEG excludes any non-EEG channels from features.
mapping = {"left_hand": 0, "right_hand": 1}
for recording in dataset.datasets:
    raw = recording.raw
    raw.pick("eeg")
    if not (set(mapping).issubset(raw.annotations.description)):
        raise ValueError(
            "Invalid set(mapping).issubset(raw.annotations.description); inspect cohort, windows and split before fitting."
        )
    print(
        recording.description["session"],
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
sfreq = dataset.datasets[0].raw.info["sfreq"]
channels = dataset.datasets[0].raw.ch_names
if not (
    all(
        r.raw.ch_names == channels and r.raw.info["sfreq"] == sfreq
        for r in dataset.datasets
    )
):
    raise ValueError(
        'Data contract failed: all(     r.raw.ch_names == channels and r.raw.info["sfreq"] == sfreq     for r in dataset.datasets ); inspect the selected recordings and metadata.'
    )

# %%
# Inspect a real cue before choosing the window: observed duration is not
# inferred from the requested three seconds.
cue_index = next(
    i for i, name in enumerate(raw.annotations.description) if name in mapping
)
cue_start = raw.annotations.onset[cue_index] - raw.first_time
print("Observed cue duration (s):", raw.annotations.duration[cue_index])
start = int(round(cue_start * sfreq))
excerpt = raw.get_data(start=start, stop=start + int(3 * sfreq))
fig, ax = plt.subplots(figsize=(8, 3))
for channel, trace in zip(channels, excerpt):
    ax.plot(np.arange(len(trace)) / sfreq, trace * 1e6, label=channel)
ax.set(
    xlabel="Seconds from imagery cue",
    ylabel="EEG (µV)",
    title=str(raw.annotations.description[cue_index]),
)
ax.legend()
plt.show()

# %%
# 3. Create one three-second window per actual imagery trial
# ----------------------------------------------------------
# The selected channels are C3, Cz and C4 over the motor area. At 250 Hz,
# three seconds are 750 samples; one window is (3 channels, 750 samples)
# in volts. The two sessions yield 120 labelled trials each in this subset.
# Keeping one window per cue avoids counting overlapping crops as independent
# trials. BAD_ACQ_SKIP annotations are not class labels; the explicit mapping
# selects only observed hand-imagery cues.
window_size = int(3 * sfreq)
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
y = metadata.target.to_numpy(dtype=int)
groups = metadata.session.astype(str).to_numpy()
if not (set(groups) == set(sessions)):
    raise ValueError(
        "Invalid set(groups) == set(sessions); inspect cohort, windows and split before fitting."
    )
print("Windows:", len(windows), "one window:", windows[0][0].shape)
print(pd.crosstab(groups, y))

# %%
# 4. Extract motor-band log power independently per trial
# -------------------------------------------------------
# EEGDash computes a shared Welch spectrum, then sums PSD bins in the
# 8–13 Hz and 13–30 Hz bands separately for C3, Cz and C4. This gives six
# features per trial. The public band function uses half-open intervals, so
# the 13 Hz bin belongs only to beta. Multiplication by the 1/3 Hz
# bin spacing approximates integrated power in V² before taking its log.
# The processed motor-imagery release is not cleaned a second time with EEGPrep.
bands = {"mu": (8, 13), "beta": (13, 30)}
spectral = FeatureExtractor(
    {"power": partial(spectral_bands_power, bands=bands)},
    preprocessor=partial(
        spectral_preprocessor,
        fs=sfreq,
        nperseg=window_size,
        noverlap=0,
        f_min=8,
        f_max=30,
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
# 5. Transfer in both directions with train-only scaling
# ------------------------------------------------------
# Training on the later session is a retrospective diagnostic. Only the
# 0train-to-1train direction represents forward session transfer.
# These session summaries are descriptive QC, not a test-set tuning criterion.
summary = pd.DataFrame(features, columns=feature_table.columns)
summary["session"] = groups
print(summary.groupby("session").median().T)
summary.groupby("session").median().T.plot.bar(figsize=(9, 4))
plt.ylabel("Median log band power (V² reference)")
plt.title("Session shift across named channel/band features")
plt.tight_layout()
plt.show()
rows = []
for train_session, test_session in [sessions, sessions[::-1]]:
    train = np.flatnonzero(groups == train_session)
    test = np.flatnonzero(groups == test_session)
    if not (set(groups[train]).isdisjoint(groups[test])):
        raise ValueError(
            "Training and test identities overlap; repair the group split."
        )
    if not (set(y[train]) == set(y[test]) == set(mapping.values())):
        raise ValueError(
            "Each train/test split must contain every mapped class; inspect retained class counts."
        )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[train], y[train])
    prediction = model.predict(features[test])
    if train_session == sessions[0]:
        ConfusionMatrixDisplay.from_predictions(
            y[test], prediction, display_labels=list(mapping), normalize="true"
        )
        plt.title("Forward transfer: subject 1, session 1train")
        plt.show()
    rows.append(
        dict(
            transfer=f"{train_session} → {test_session}"
            + (" (retrospective)" if train_session == sessions[1] else " (forward)"),
            balanced_accuracy=balanced_accuracy_score(y[test], prediction),
            n_test=len(test),
        )
    )
results = pd.DataFrame(rows)
print(results.to_string(index=False))

# %%
# 6. Plot measured session-transfer scores
# ----------------------------------------
results.plot.bar(x="transfer", y="balanced_accuracy", legend=False, rot=0)
plt.axhline(0.5, color="black", linestyle="--", label="Chance")
plt.ylim(0, 1)
plt.ylabel("Balanced accuracy")
plt.legend()
plt.show()

# %%
# 7. Decide what the transfer result supports
# -------------------------------------------
# Balanced accuracy is mean left/right recall, with chance 0.5. The printed
# n_test column is the number of actual reserved-session trials. Differences
# between directions can reflect training difficulty or session conditions;
# they do not isolate an electrode-drift mechanism.
#
# For deployment after calibration, reserve a later genuine session and tune
# only within earlier sessions. If you add target-session calibration trials,
# exclude those trials from its test set and report how many labels adaptation
# uses. That is a different protocol from the zero-calibration transfer here.
