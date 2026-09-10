"""Track 2: real cross-session motor-imagery decoding
==================================================

Load BNCI2014-004 subject 1, sessions 0train and 1train, run 0 (nm000135).
The two recordings total about 11 MB. This CPU baseline predicts observed
left-hand and right-hand cues with log channel power. Session 0train fits the
model; session 1train is held out without calibration. These are already
processed real recordings. This compact motor-imagery example has two classes
and its own session split, not the official Stieger/NeuralBench protocol.

Source: https://nemar.org/dataset/nm000135
"""

# %%
# Before you start
# ----------------
#
# Use an installed EEGDash environment with Braindecode, EEGPrep, MNE, NumPy,
# scikit-learn and Matplotlib. This two-recording CPU example needs about 11 MB
# of cached signal data. ``EEGDASH_CACHE_DIR`` chooses the persistent download
# location; a network connection is needed when either session is absent.
#
# The deployment question is prediction in another recorded session of the same
# person, without calibration on that session. The two labels are actual left-
# and right-hand imagery cues. This small BNCI corpus is an accessible example
# of that scientific task; it does not reproduce the official 2026 Stieger
# cohort, class inventory, preprocessing or frozen split.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from braindecode.preprocessing import create_windows_from_events, Resampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDashDataset
from eegdash.features import signal_variance

# %%
# Inspect both sessions before preprocessing
# ------------------------------------------
#
# The explicit subject, session, run and task query should return exactly
# two recordings. Keeping session identities intact is essential: relabelling
# two runs as sessions would not test session generalization. The annotation
# printout confirms the available class names before defining their numeric codes.
#
# C3, Cz and C4 cover the motor region and are the source's three EEG channels.
# The release is already processed; the additional 8–30 Hz filter selects the
# sensorimotor frequency range for this particular power baseline rather than
# claiming to redo acquisition preprocessing. EEGPrep's ``Resampling`` adapter
# converts to 100 Hz and reduces the
# number of samples while retaining that passband. These fixed operations do not
# learn their parameters from the held-out session, but the offline filtering
# here should not be mistaken for causal streaming preprocessing.

# The EEGPrep adapter converts through EEGLAB and can round event times or
# discard the measurement date. Because this step only resamples, we retain
# the original date and annotations, verify the origin, duration, channel order
# and new rate, then restore the observed event times in seconds. The final
# onset assertion prevents a silent cue shift from changing window labels.

dataset = EEGDashDataset(
    dataset="nm000135",
    subject="1",
    session=["0train", "1train"],
    run="0",
    task="imagery",
    cache_dir=Path(
        os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")
    ).expanduser(),
)
print(dataset.description.to_string(index=False))
assert len(dataset.datasets) == 2
for recording in dataset.datasets:
    raw = recording.raw.pick("eeg")
    print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
    # Limit RAM to three motor channels before filtering and resampling.
    raw.pick(["C3", "Cz", "C4"]).load_data().filter(8, 30)
    annotations = raw.annotations.copy()
    measurement_date = raw.info["meas_date"]
    recording_duration = raw.n_times / raw.info["sfreq"]
    assert raw.first_samp == 0
    Resampling(sfreq=100).apply(raw.load_data())
    assert raw.ch_names == ["C3", "Cz", "C4"] and raw.info["sfreq"] == 100
    assert raw.first_samp == 0
    assert abs(raw.n_times / raw.info["sfreq"] - recording_duration) <= 1 / 100
    # EEGPrep's file conversion can round annotation times; retain source seconds.
    raw.set_meas_date(measurement_date)
    raw.set_annotations(annotations)
    np.testing.assert_allclose(
        raw.annotations.onset, annotations.onset, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        raw.annotations.duration, annotations.duration, atol=1 / 100, rtol=0
    )
    np.testing.assert_array_equal(raw.annotations.description, annotations.description)
# %%
# Window the actual imagery intervals
# -----------------------------------
#
# ``left_hand`` and ``right_hand`` map explicitly to zero and one. After
# resampling, 300 samples correspond to three seconds. Equal stride and window
# size give non-overlapping complete windows within the labelled intervals;
# ``on_last_window="drop"`` discards an incomplete remainder rather than padding
# it. Zero start/stop offsets preserve the source event boundaries.
#
# The window metadata carries session and target through preprocessing. Before
# feature reduction, ``X`` has layout ``(windows, 3, 300)`` in volts. Natural-log
# variance over time reduces it to one channel-power feature per channel. The
# small numerical floor makes zero power finite, but does not perform artifact
# rejection or rescue a noninformative electrode.

classes = {"left_hand": 0, "right_hand": 1}
windows = create_windows_from_events(
    dataset,
    mapping=classes,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=300,
    window_stride_samples=300,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata()
X = np.stack([windows[i][0] for i in range(len(windows))])
y = metadata.target.to_numpy(dtype=int)
X = np.log(np.maximum(signal_variance(X), 1e-30))
# %%
# Hold out a genuine session without recalibration
# ------------------------------------------------
#
# Only ``0train`` fits the model; ``1train`` is the held-out session. Those
# strings are source identifiers, not instructions to include both in training.
# The two sessions share a participant intentionally. The masks and class-set
# assertions verify that both sessions contain the same observed two-class task.
#
# Feature standardization belongs inside the pipeline so the test-session mean
# and variance remain unseen during fitting. Logistic regression uses its default
# regularization; ``max_iter=1000`` gives the optimizer room to converge and is
# not a score-tuning parameter. This page has no validation set and therefore
# makes no data-driven hyperparameter selection.

train = metadata.session.eq("0train").to_numpy()
test = metadata.session.eq("1train").to_numpy()
assert train.any() and test.any() and np.isfinite(X).all()
assert set(metadata.session[train]).isdisjoint(metadata.session[test])
assert set(y[train]) == set(y[test]) == set(classes.values())
print("Features:", X.shape, "classes:", np.unique(y, return_counts=True))

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
predicted = model.fit(X[train], y[train]).predict(X[test])
print(
    "Held-out session balanced accuracy:", balanced_accuracy_score(y[test], predicted)
)
ConfusionMatrixDisplay.from_predictions(
    y[test], predicted, display_labels=list(classes)
)
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# %%
# Interpret session transfer and extend it
# ----------------------------------------
#
# Balanced accuracy is the mean of left-hand and right-hand recall, with
# 0.5 as the uniform two-class reference. The confusion matrix displays actual
# counts: inspect whether errors concentrate in one class instead of relying on
# a single score. A near-chance result can reflect a weak feature representation
# or real session differences; it should not trigger a favourable-score assertion.
#
# One person's two sessions do not establish population performance. To tune the
# band range, regularization or an EEGNet model, add a separate validation
# session and preserve the present test session. Then repeat the design across
# additional people where the catalogue actually supplies the required sessions.
# Compare pipelines using identical held-out windows and retain original trial
# IDs if extracting several windows from each trial.
#
# Related worked example: `Braindecode cross-session motor imagery
# <https://braindecode.org/dev/auto_examples/advanced_training/plot_moabb_benchmark.html>`_.
