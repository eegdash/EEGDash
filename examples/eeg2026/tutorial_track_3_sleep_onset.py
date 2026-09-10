"""Track 3: predict observed time to sustained N2 sleep
====================================================

Use three actual Sleep-EDF participants (nm000185), first night only, roughly
150 MB of signals. The target is seconds from recording start to the first
N2 interval lasting at least 60 seconds. Recording start is not lights-out:
this seed-corpus target is explicitly different from the wearable competition
endpoint and its weighted binned MAE. Only the first minute supplies predictors.

Source: https://nemar.org/dataset/nm000185
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, MNE, NumPy, scikit-learn and Matplotlib on CPU. The three
# first-night EDF recordings total roughly 150 MB and are cached under
# ``EEGDASH_CACHE_DIR``. Only two EEG derivations and the opening interval enter
# the predictor, but complete source annotations are needed to observe the later
# outcome. Cropping the predictor does not avoid the initial recording download.
#
# This is latency regression, not sleep-stage classification. The endpoint is
# the first observed N2 stretch lasting at least 60 seconds, measured from the
# file's recording origin. It is neither lights-out latency nor the official
# wearable challenge endpoint. In particular, the long initial wake period in a
# Sleep-EDF record must not be removed merely to produce a more appealing target.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDashDataset
from eegdash.features import spectral_preprocessor, spectral_bands_power

# %%
# Choose independent participant records
# --------------------------------------
#
# One first-night recording is requested for each of three named people.
# The subject-set assertion verifies coverage rather than allowing a missing
# record to change the cohort silently. A second night from one person would not
# be an additional independent participant; keep that identity when extending
# the query.
#
# The loader prints channel names, the 100 Hz rate and observed stage labels.
# Unlike a stage classifier, this model does not use stage labels as predictors.
# Annotations over the full recording establish the retrospective target; voltage
# features are restricted to the opening interval.

subjects = ["cassette63", "cassette64", "cassette65"]
dataset = EEGDashDataset(
    dataset="nm000185",
    subject=subjects,
    session="night1",
    task="sleep",
    cache_dir=Path(
        os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")
    ).expanduser(),
)
print(dataset.description.to_string(index=False))
assert set(dataset.description.subject) == set(subjects)
# %%
# Derive a sustained-N2 endpoint and a fixed predictor
# ----------------------------------------------------
#
# Consecutive N2 annotations are merged when their boundaries agree within
# 0.01 seconds. This accommodates both consolidated stage intervals and adjacent
# 30-second scoring epochs. The first merged interval of at least 60 seconds
# sets the observed latency. If no such interval exists, the script stops:
# no-event recordings would need an explicit censoring analysis, not a guessed
# latency at the file's end.
#
# The target must lie beyond the prespecified predictor horizon. The code then
# uses samples from zero through 59 seconds from the Fpz–Cz and Pz–Oz derivations,
# which keeps the feature interval before that target. These are recorded bipolar
# derivations; their names do not denote independently recoverable electrodes.
#
# EEGDash's ``spectral_preprocessor`` uses non-overlapping 200-sample Hamming
# segments at 100 Hz, giving 0.5 Hz bins. ``spectral_bands_power`` sums PSD bins
# in 1–4, 4–8, 8–13 and 13–30 Hz; multiplying by the bin spacing approximates
# power in V² before ``log10``. Each band includes its lower bound and excludes
# its upper bound, so adjacent bands do not share bins. This convention is the
# same for everyone. The floor only avoids log zero; it does not clean an artifact.

features, targets, identities = [], [], []
for recording in dataset.datasets:
    raw = recording.raw
    print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
    # Consolidate adjacent N2 annotations so ordinary 30-second scoring also works.
    intervals = []
    for ann in raw.annotations:
        if ann["description"] != "N2":
            continue
        start = float(ann["onset"] - raw.first_time)
        stop = start + float(ann["duration"])
        if intervals and np.isclose(start, intervals[-1][1], atol=0.01, rtol=0):
            intervals[-1][1] = stop
        else:
            intervals.append([start, stop])
    candidates = [start for start, stop in intervals if stop - start >= 60]
    if not candidates:
        raise ValueError(
            f"No sustained observed N2 interval: {recording.description.subject}"
        )
    onset = min(candidates)
    if onset <= 60:
        raise ValueError(
            "The fixed predictor minute includes N2 onset; choose a shorter prespecified horizon."
        )
    first_minute = (
        raw.copy().pick(["EEG Fpz-Cz", "EEG Pz-Oz"]).crop(tmax=59).load_data()
    )
    frequencies, psd = spectral_preprocessor(
        first_minute.get_data(),
        _metadata={"info": first_minute.info},
        f_min=1,
        f_max=30,
        nperseg=200,
        noverlap=0,
        window="hamming",
    )
    powers = spectral_bands_power(
        frequencies,
        psd,
        bands={"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)},
    )
    band_power = np.concatenate(list(powers.values())) * (
        frequencies[1] - frequencies[0]
    )
    features.append(np.log10(np.maximum(band_power, 1e-30)))
    targets.append(onset)
    identities.append(recording.description.subject)
# %%
# Verify one feature row per observed endpoint
# --------------------------------------------
#
# Two derivations times four bands yield eight features per participant,
# so ``X`` has shape ``(3, 8)``. ``y`` contains onset times in seconds, one per
# person. Identity and finiteness assertions verify that the design matrix did
# not silently duplicate participants or acquire invalid targets.
#
# The same fixed initial interval is used for everyone; choosing a different
# feature window according to each person's eventual onset would expose the
# target to preprocessing. This design can still be dominated by recording-start
# conventions rather than physiology. The target printout is therefore part of
# interpreting the task, not merely a debugging aid.

X, y = np.asarray(features), np.asarray(targets)
assert (
    len(set(identities)) == len(y) == 3
    and np.isfinite(X).all()
    and np.isfinite(y).all()
)
print("Participant features:", X.shape, "Observed onset seconds:", y)

# %%
# Evaluate with participant-level absolute error
# ----------------------------------------------
#
# Each leave-one-out fold trains on two people and predicts the third.
# The scaler is fitted inside the fold and ridge uses a fixed ``alpha=10``.
# A separately fitted dummy model predicts that fold's training-mean onset.
# Neither method sees the held-out participant's feature distribution while
# estimating its parameters.
#
# Mean absolute error is in seconds and weights the three people equally. It is
# not the competition's weighted binned MAE. With only two training people per
# fold, the example demonstrates the data and evaluation contract rather than
# estimating a clinically useful prediction rule. The scatter shows the three
# actual held-out predictions without requiring them to beat the mean baseline.

predicted, baseline = np.empty_like(y), np.empty_like(y)
for train, test in LeaveOneOut().split(X):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    predicted[test] = model.fit(X[train], y[train]).predict(X[test])
    baseline[test] = DummyRegressor().fit(X[train], y[train]).predict(X[test])
print("Held-out participant MAE (s):", mean_absolute_error(y, predicted))
print("Training-mean MAE (s):", mean_absolute_error(y, baseline))
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y, predicted)
ax.set(xlabel="Observed sustained N2 onset (s)", ylabel="Predicted onset (s)")
plt.show()

# %%
# Choose the endpoint before expanding the model
# ----------------------------------------------
#
# For a larger study, first verify whether recording start or a separate
# lights-out annotation is the intended origin. If lights-out is required, acquire
# that observed timestamp and redefine both predictor access and the outcome
# before fitting. Do not treat a metadata convention as a learned device effect.
#
# Then add participants, an inner validation scheme for penalties, and explicit
# handling of recordings without sustained N2. A neural sequence model would
# also need its accessible time horizon specified; feeding the whole night could
# turn prospective prediction into retrospective event detection.
#
# Compare with `Braindecode's U-Sleep walkthrough
# <https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_usleep.html>`_
# for a worked stage-classification pipeline. Its sequence labels and evaluation
# question differ from the participant-level latency regression here.
