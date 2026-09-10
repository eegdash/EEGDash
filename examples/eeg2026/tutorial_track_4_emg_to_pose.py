"""Track 4: predict real hand-joint trajectories from wrist EMG
========================================================================

Load three EMG2Pose recordings through EEGDash and regress 20 joint-angle
trajectories from 16-channel wrist EMG. The public dataset is NM000281.
The recordings retain their published train, validation and test assignments;
the test recording belongs to the held-out user-and-stage scenario.

This small linear baseline teaches the real pose target and evaluation path.
It does not reproduce the full VEMG2Pose baseline or hidden 2026 evaluation.

Sources:
https://neural-interfaces26.github.io/tracks.html
https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html
"""

# %%
# Before you start
# ----------------------------
#
# Install EEGDash, MNE, NumPy, SciPy, pandas, pooch, scikit-learn and Matplotlib.
# The three selected BDF files require about 24 MB, cached in
# ``EEGDASH_CACHE_DIR``. The full public release is hundreds of gigabytes;
# selecting exact recordings prevents an accidental full download.
#
# Track 4 predicts continuous hand pose, not typing. Its 20 targets are joint
# angles obtained from motion capture and inverse kinematics. They are stored
# as MISC channels in radians alongside the 16 EMG channels at 2000 Hz.
# Mean absolute angular error is reported in degrees, averaging over joints
# and time. This is neither a character error rate nor a fingertip distance.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pooch
from scipy.signal import lfilter
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDash, EEGDashDataset

# %%
# Read the published recording assignments
# ----------------------------------------------------
#
# These right-wrist recordings come from three different users. Training uses
# a free-style movement, validation a thumb movement, and testing a held-out
# movement stage. The pinned BIDS scans tables contain the original source-file
# identity, stage, split and generalization fields. Older dataset releases omit
# the last two fields, so we explicitly retrieve the version containing them.
# No split is inferred from recording order or generated from windows.
#
# The selected test scenario is ``user_stage``; the paper also evaluates held-out
# users and held-out stages separately. One recording per split cannot estimate
# population performance or reproduce those full benchmark comparisons.

cache = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
revision = "90cb29b450de36bb275b226cbc47918f6cbc3b09"
source = f"https://raw.githubusercontent.com/nemarDatasets/nm000281/{revision}"
selections = [
    ("train", "06", "01", "23"),
    ("val", "45", "01", "15"),
    ("test", "02", "02", "31"),
]
scans_hashes = {
    "06": "034c6c2e47db211d417ecb3f9ec1cd7e6e9b576e5f2e9114a3f8529f384f5210",
    "45": "51dc75aaa654868e3377085d88d82bb5a6051e6dc159117bfca6fd710bf9e607",
    "02": "d5dd75bb248ee5c7124e6d55d72c01bad6137797d5a3782814690828b74c70a0",
}
arrays, users, stages = {}, {}, {}
api = EEGDash()

# %%
# Load aligned EMG and pose, excluding failed tracking
# ----------------------------------------------------------------
#
# BIDS channel types keep the measured EMG separate from the pose labels. The
# source already high-pass filtered EMG at 40 Hz and rescaled its noise floor;
# its channel sidecar declares volts, but these are processed amplitudes, not
# untouched amplifier voltages. Pose targets were low-pass filtered at 15 Hz
# and interpolated upstream to the EMG sampling grid. No extra EEG reference,
# artifact-subspace reconstruction or target filtering is appropriate here.
#
# BDF files contain integer-second padding. Crop to the recorded duration from
# scans.tsv, then cut non-overlapping five-second windows. MNE rejects windows
# intersecting ``BAD_IK`` (inverse-kinematics failure) or other BAD annotations.
# Reject a complete window when any target is nonfinite as an additional guard;
# interpolating failed targets would create unobserved supervision.
#
# A causal 100 ms RMS envelope summarizes each EMG channel. SciPy's ``lfilter``
# computes the moving mean of squared samples, using only the current and past
# signal, before taking its square root. It starts with zero filter state at
# the recording boundary. Envelopes are computed continuously before windowing
# so later windows retain their preceding EMG context. Joint trajectories remain
# at 2 kHz; no temporal average replaces the dense pose target.

for split, subject, session, run in selections:
    filename = (
        f"sub-{subject}_ses-{session}_task-emg2pose_run-{run}_recording-right_emg.bdf"
    )
    relative = f"sub-{subject}/ses-{session}/emg/{filename}"
    scans_name = f"sub-{subject}_ses-{session}_scans.tsv"
    scans_path = pooch.retrieve(
        f"{source}/sub-{subject}/ses-{session}/{scans_name}",
        known_hash=scans_hashes[subject],
        path=cache / "emg2pose-splits" / revision,
        fname=scans_name,
    )
    scans = pd.read_csv(scans_path, sep="\t").set_index("filename")
    assignment = scans.loc[f"emg/{filename}"]
    assert assignment["split"] == split and assignment["side"] == "right"
    if split == "test":
        assert assignment["generalization"] == "user_stage"
    records = api.find(dataset="nm000281", subject=subject, session=session, run=run)
    records = [record for record in records if record["bids_relpath"] == relative]
    assert len(records) == 1
    users[split] = records[0]["participant_tsv"]["original_user"]
    stages[split] = assignment["stage"]
    dataset = EEGDashDataset(records=records, cache_dir=cache)
    raw = dataset.datasets[0].raw
    assert raw.info["sfreq"] == 2000 and raw.first_samp == 0
    assert raw.ch_names == [f"emg{i}" for i in range(16)] + [
        f"joint{i}" for i in range(20)
    ]
    assert raw.get_channel_types() == ["emg"] * 16 + ["misc"] * 20
    stop = int(round(float(assignment["duration"]) * raw.info["sfreq"]))
    raw.crop(tmax=(stop - 1) / raw.info["sfreq"])
    signal = raw.get_data(picks="emg")
    envelope = np.sqrt(lfilter(np.ones(200) / 200, [1.0], signal**2, axis=-1))
    epochs = mne.make_fixed_length_epochs(
        raw, duration=5.0, preload=True, reject_by_annotation=True
    )
    targets = epochs.get_data(picks="misc").transpose(0, 2, 1)
    inputs = np.stack(
        [envelope[:, start : start + 10000].T for start in epochs.events[:, 0]]
    )
    finite = np.isfinite(targets).all(axis=(1, 2)) & np.isfinite(inputs).all(
        axis=(1, 2)
    )
    inputs, targets = inputs[finite], targets[finite]
    assert len(targets) > 0 and inputs.shape[1:] == (10000, 16)
    assert targets.shape[1:] == (10000, 20)
    arrays[split] = (inputs, targets)
    print(split, users[split], stages[split], assignment["generalization"])
    print("EMG envelope / pose shapes:", inputs.shape, targets.shape)

assert len(set(users.values())) == 3
assert stages["test"] not in (stages["train"], stages["val"])

# %%
# Fit a dense regression baseline using training data only
# --------------------------------------------------------------------
#
# Each time point supplies 16 envelope features and 20 simultaneous angle
# targets. Flattening window and time axes allows Ridge to fit all targets with
# a shared regularization strength. Adjacent time samples remain strongly
# correlated: they are not independent participants or independent trials.
# StandardScaler is fitted on training EMG only. Angle units stay in radians.
#
# Choose the penalty from three values using the actual validation recording.
# The held-out test trajectory is evaluated once after selection. A training-mean
# pose provides a useful constant baseline: a complicated decoder should improve
# on predicting the same hand configuration at every time point.

X_train, y_train = arrays["train"]
X_valid, y_valid = arrays["val"]
X_test, y_test = arrays["test"]
models, validation_errors = [], []
for alpha in (0.1, 10.0, 1000.0):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train.reshape(-1, 16), y_train.reshape(-1, 20))
    prediction = model.predict(X_valid.reshape(-1, 16))
    validation_errors.append(mean_absolute_error(y_valid.reshape(-1, 20), prediction))
    models.append(model)
best = int(np.argmin(validation_errors))
prediction = models[best].predict(X_test.reshape(-1, 16)).reshape(y_test.shape)
dummy = DummyRegressor(strategy="mean").fit(
    X_train.reshape(-1, 16), y_train.reshape(-1, 20)
)
dummy_prediction = dummy.predict(X_test.reshape(-1, 16))
mae_degrees = np.rad2deg(
    mean_absolute_error(y_test.reshape(-1, 20), prediction.reshape(-1, 20))
)
dummy_degrees = np.rad2deg(
    mean_absolute_error(y_test.reshape(-1, 20), dummy_prediction)
)
print("Validation MAE (degrees):", np.rad2deg(validation_errors))
print("Selected alpha:", (0.1, 10.0, 1000.0)[best])
print("Test angular MAE (degrees), ridge / constant:", mae_degrees, dummy_degrees)

# %%
# Inspect an actual trajectory and understand the scope
# -----------------------------------------------------------------
#
# Plot joint0 from the first retained test window. The reported MAE above uses
# all 20 joints and all retained test windows, not just this illustrated joint.
# Predictions need not satisfy joint limits or produce anatomically consistent
# hand poses; forward kinematics and a temporal pose model are useful extensions.
#
# This small baseline preserves actual pose labels, five-second dense outputs
# and the paper's split assignments. Its linear envelope decoder, three users,
# single validation recording and partial movement coverage differ substantially
# from VEMG2Pose and the complete challenge protocol. Do not compare this score
# directly with the competition leaderboard or treat many time samples as evidence
# of broad user generalization. Expand recordings within each predefined split
# before comparing models, keeping test users and stages reserved throughout.
#
# For the full public NeuralBench baseline, follow the linked official guide:
# ``neuralbench emg pose -m vemg2pose --download``, then ``--prepare``, then
# ``--debug``, and finally the same command without an action flag. The full
# release requires substantially more disk and compute than this tutorial.

time = np.arange(10000) / 2000
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(time, np.rad2deg(y_test[0, :, 0]), label="Recorded joint0")
ax.plot(time, np.rad2deg(prediction[0, :, 0]), label="Predicted joint0")
ax.set(xlabel="Time within retained test window (s)", ylabel="Joint angle (degrees)")
ax.legend()
fig.tight_layout()
plt.show()
