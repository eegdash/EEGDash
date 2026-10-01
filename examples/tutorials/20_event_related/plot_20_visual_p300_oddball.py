"""Visual P300: from real events to held-out predictions
=====================================================

Load three visual-oddball recordings from OpenNeuro ``ds005863`` through
EEGDash, inspect their event codes, and decode targets on a new subject.
Subjects 054, 119 and 123 require about 69 MB of signal files in total.
Set ``EEGDASH_CACHE_DIR`` to reuse the download. CPU is sufficient; install ``eegprep[eeglabio]>=0.2.23,<0.3`` for the
EEGPrep stages.
The `dataset <https://openneuro.org/datasets/ds005863>`_ contains recorded
EEG; every label below comes from its stimulus markers.

An oddball task presents frequent standards and occasional targets. An
event-related potential (ERP) averages responses aligned to stimulus onset;
a decoder instead predicts the condition of each individual trial. Here you
will prepare both from the same recordings, then test whether a simple
amplitude-based classifier generalizes to a participant absent from training.

Run the page from top to bottom with EEGDash installed. Familiarity with NumPy
arrays and the first-recording tutorial is helpful. The outputs are an ERP
plot at Pz, a table of participant scores and a confusion matrix. No GPU or
previously prepared feature file is needed.
"""

# %%
# 1. Select the recordings before downloading
# -------------------------------------------
# The query describes three recordings, not three classification samples.
# Each recording contributes many stimulus trials after epoching. Inspect the
# description before accessing ``raw``, which acquires the signal file;
# ``load_data()`` below then brings its samples into memory.

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import RemoveCommonAverageReference, RemoveDCOffset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from eegdash.features import signal_mean

cache_dir = get_default_cache_dir()
subjects = ["054", "119", "123"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="ds005863",
    subject=subjects,
    task="visualoddball",
    n_jobs=1,
)
dataset.description[["subject", "task"]]

# %%
# 2. Map recorded events and prepare trial features
# -------------------------------------------------
# Consult the source task/event documentation before transferring this mapping:
# `ds005863 source tree <https://github.com/OpenNeuroDatasets/ds005863>`_.
# The mapping uses literal marker names; a catalogue task name alone
# cannot establish these semantics. In code XY, X is the block's target letter and Y is the presented
# letter, both coded 1..5. Matching digits denote targets. Explicitly
# exclude responses and other markers rather than calling all other
# annotations standards. Some readers prefix names with ``Stimulus/``.
# For example, ``S11`` maps to target (2), while ``S12`` maps to standard (1).
# Subtracting one after epoching gives classifier labels 0 and 1.
#
# MNE epochs align each trial to time zero at the recorded stimulus. The
# -100..0 ms baseline subtracts each channel's pre-stimulus mean from that
# trial. The 0.5–30 Hz filter reduces slow drift and faster activity; changing
# it can change ERP amplitude and should be decided before comparing scores.
# EEGPrep removes each channel's median offset, then its common-average
# reference subtracts the instantaneous EEG-channel mean. These operations
# do not remove samples. We verify the grid and restore source annotations
# explicitly because MNE↔EEGLAB conversions can round event latencies.
#
# Resampling epochs to 128 Hz reduces memory after events have been located
# on the original sample grid. ``X`` has axes (trials, channels, time) in volts.
# EEGDash's ``signal_mean`` over 300–450 ms produces one amplitude per
# channel and trial; multiplying
# by 1e6 expresses it in microvolts. This fixed interval is a compact baseline,
# not a claim that every participant's peak lies there. All channel amplitudes
# enter the decoder; Pz is used only for the illustrative ERP.
#
# ``reject_by_annotation`` respects existing bad spans. It does not detect
# every blink or noisy channel: inspect recordings and define any additional
# quality-control rules before extending this analysis.
features, labels, groups = [], [], []
first_epochs = None
first_subject = None
for recording in dataset.datasets:
    raw = recording.raw.copy().load_data().pick("eeg")
    mapping = {}
    for name in set(raw.annotations.description):
        code = name.split("/")[-1].replace(" ", "")
        if len(code) == 3 and code[0] == "S" and set(code[1:]) <= set("12345"):
            mapping[name] = 2 if code[1] == code[2] else 1
    # Preprocess, epoch and baseline-correct
    # Filtering is independent for each recording. Epochs span -0.1..0.8 s
    # relative to stimulus onset, irrespective of annotation duration.
    source_annotations = raw.annotations.copy()
    source_date = raw.info["meas_date"]
    source_grid = (raw.info["sfreq"], raw.n_times, raw.first_samp)
    RemoveDCOffset().apply(raw)
    RemoveCommonAverageReference().apply(raw)
    if not ((raw.info["sfreq"], raw.n_times, raw.first_samp) == source_grid):
        raise ValueError(
            "Preprocessing changed the sample grid; do not restore event times"
        )
    raw.set_meas_date(source_date)
    # With no absolute origin, set_annotations adds first_time itself.
    if source_annotations.orig_time is None:
        source_annotations.onset -= raw.first_time
    raw.set_annotations(source_annotations)
    raw.filter(0.5, 30.0)
    events, _ = mne.events_from_annotations(raw, event_id=mapping)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"standard": 1, "target": 2},
        tmin=-0.1,
        tmax=0.8,
        baseline=(-0.1, 0),
        preload=True,
        reject_by_annotation=True,
    )
    epochs.resample(128)
    if first_epochs is None:
        first_epochs = epochs
        first_subject = str(recording.description["subject"])
    epochs.reorder_channels(first_epochs.ch_names)
    X = epochs.get_data()
    y = epochs.events[:, 2] - 1

    # Use a fixed analysis interval; do not choose it from test accuracy.
    interval = (epochs.times >= 0.3) & (epochs.times <= 0.45)
    features.append(signal_mean(X[:, :, interval]) * 1e6)
    labels.append(y)
    groups.extend([str(recording.description["subject"])] * len(y))

# %%
# Inspect the first participant's condition averages before fitting.
# Whole-recording filtering/reference is offline, not causal online processing.
mne.viz.plot_compare_evokeds(
    {name: first_epochs[name].average() for name in ["standard", "target"]},
    picks="Pz",
    title=f"Subject {first_subject}: Pz",
    show=False,
)
plt.show()

# %%
# 3. Evaluate one held-out subject per fold
# -----------------------------------------
# StandardScaler is fitted inside each training fold. Balanced accuracy
# gives targets and standards equal weight despite the oddball imbalance.
# Concatenation stacks trials while ``groups`` keeps their participant identity.
# Read the class-count table before training: an always-standard prediction
# may have high ordinary accuracy, but binary balanced accuracy would be 0.5.
#
# Logistic regression learns a linear combination of channel amplitudes.
# Training class weights give the rarer targets more influence on its loss.
# This affects fitting; balanced accuracy separately averages the two class
# recalls at evaluation. Each LOSO fold trains on two complete participants
# and tests on the third. A random split of trials would answer a different
# question by allowing the same participant into training and test sets.
X = np.concatenate(features)
y = np.concatenate(labels)
groups = np.asarray(groups)
pd.crosstab(groups, y, rownames=["subject"], colnames=["class"])
predictions = np.full(len(y), -1)
rows = []
for train, test in LeaveOneGroupOut().split(X, y, groups):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    model.fit(X[train], y[train])
    predictions[test] = model.predict(X[test])
    rows.append(
        {
            "subject": groups[test][0],
            "balanced_accuracy": balanced_accuracy_score(y[test], predictions[test]),
        }
    )
pd.DataFrame(rows)

# %%
# 4. Inspect the measured ERP and decoding errors
# -----------------------------------------------
# The ERP describes the first recording in the printed cohort order; the
# confusion matrix uses held-out predictions from all three. A visible P300 or high score is not a test
# requirement. This small cohort demonstrates the workflow, not a benchmark.
# In the normalized confusion matrix, each row sums to one: the target-row
# diagonal is target recall and its off-diagonal cell represents missed
# targets. Compare both rows, since good standard recall can hide missed
# targets in an unbalanced task. An averaged ERP difference also need not
# imply that single-trial responses are reliably separable.
# This confusion pools trials; the score table above gives each person a row.
ConfusionMatrixDisplay.from_predictions(
    y, predictions, display_labels=["standard", "target"], normalize="true"
)
plt.show()

# %%
# 5. Extend the analysis without selecting on test results
# --------------------------------------------------------
# Add participants while retaining one complete participant per test fold.
# If you want to select channels, intervals or regularization strength, make
# those choices using additional validation participants inside the training
# fold. Keep the held-out participant untouched until the choice is fixed.
# The auditory oddball page uses the same Raw → Epochs → Evoked sequence for
# a descriptive response comparison; the P300 transfer project adds a separate
# adaptation participant to study a different deployment question.
