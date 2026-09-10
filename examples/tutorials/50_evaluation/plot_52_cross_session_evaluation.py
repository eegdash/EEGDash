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
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDashDataset
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)

sessions = ["0train", "1train"]
dataset = EEGDashDataset(
    cache_dir=Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache")),
    dataset="nm000135",
    subject="1",
    session=sessions,
    run="0",
    task="imagery",
    n_jobs=1,
)
assert len(dataset.datasets) == 2
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
    assert set(mapping).issubset(raw.annotations.description)
    print(
        recording.description["session"],
        raw.ch_names,
        raw.info["sfreq"],
        np.unique(raw.annotations.description),
    )
sfreq = dataset.datasets[0].raw.info["sfreq"]
channels = dataset.datasets[0].raw.ch_names
assert all(
    r.raw.ch_names == channels and r.raw.info["sfreq"] == sfreq
    for r in dataset.datasets
)

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
assert (metadata.i_window_in_trial == 0).all(), "Expected one window per trial"
assert not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
X = np.stack([window[0] for window in windows])
y = metadata.target.to_numpy(dtype=int)
groups = metadata.session.astype(str).to_numpy()
assert set(groups) == set(sessions) and np.isfinite(X).all()
print("Windows:", X.shape)
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
assert feature_table.shape == (len(y), len(channels) * len(bands))
features = np.log(np.maximum(feature_table.to_numpy() * sfreq / window_size, 1e-30))
assert np.isfinite(features).all()

# %%
# 5. Transfer in both directions with train-only scaling
# ------------------------------------------------------
# Training on the later session is a retrospective diagnostic. Only the
# 0train-to-1train direction represents forward session transfer.
rows = []
for train_session, test_session in [sessions, sessions[::-1]]:
    train = np.flatnonzero(groups == train_session)
    test = np.flatnonzero(groups == test_session)
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(y[train]) == set(y[test]) == set(mapping.values())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[train], y[train])
    prediction = model.predict(features[test])
    rows.append(
        dict(
            transfer=f"{train_session} → {test_session}",
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
