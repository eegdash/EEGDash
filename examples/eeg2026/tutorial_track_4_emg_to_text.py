"""Track 4: decode recorded cross-user typing events
=================================================

Load actual emg2qwerty wrist EMG and observed keystroke annotations from
nm000104. One recording from each of two users costs about 270 MB total.
Fit a simple log-power classifier on user 34527640 and test on user 70495563.

This baseline receives the true keystroke times and decodes lowercase letters
only. Its CER measures the concatenated lowercase-keystroke stream, not prompt
text, spaces, corrections or unconstrained sequence transduction. The official
NeuralBench track requires those additional sequence operations and frozen users.

Source: https://nemar.org/dataset/nm000104
"""

# %%
# Before you start
# ----------------
#
# Use EEGDash, MNE, NumPy, scikit-learn, RapidFuzz and Matplotlib on CPU. About
# 270 MB of signal data are downloaded for the two explicitly selected user
# recordings and retained in ``EEGDASH_CACHE_DIR``. This page consumes real EMG
# through EEGDash's recording interface; it does not reinterpret EEG channels as
# muscle activity.
#
# The decoder is given observed keystroke times and predicts lowercase letters
# at those times. This aligned offline task omits the difficult detection and
# sequence-alignment parts of full EMG-to-text transduction. Its output is the
# stream of lowercase keystrokes, not the displayed prompt or the user's final
# edited text.

import os
import string
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDash, EEGDashDataset
from eegdash.features import signal_root_mean_square

# %%
cache = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
# %%
# Select one recording from each real user
# ----------------------------------------
#
# User 34527640 provides training data from the named session; user
# 70495563 is held out. The latter has more than one catalogue entry, so the
# exact EDF relative path selects the intended recording. The one-recording
# assertion protects this bounded query against accidentally loading a user's
# entire session history.
#
# The 32 channels are explicitly ordered EMG0 through EMG31. Matching channel
# names is necessary for consistent feature columns, but cannot ensure identical
# sensor placement or muscle recruitment across users. That domain difference
# is part of the cross-user question.

queries = [{"subject": "34527640", "session": "1625302365"}, {"subject": "70495563"}]
# %%
# Extract observed lowercase keystrokes and channel power
# -------------------------------------------------------
#
# The mapping selects only ``keystroke_a`` through ``keystroke_z``.
# Prompt events, spaces, backspaces and other control keys are excluded, rather
# than converted into invented letter labels. This means the reference stream
# also retains lowercase letters subsequently erased by the user. The protocol
# must be understood before interpreting its error rate.
#
# Each epoch spans 100 ms before to 100 ms after the observed keystroke. At
# 2000 Hz this includes 401 samples because MNE includes both endpoints.
# ``event_repeated="drop"`` keeps one event when sample indices collide, and
# labels are read from the retained ``epochs.events`` so they remain aligned.
# The post-keystroke half is available in this offline task, not in prediction
# before a keypress.
#
# EEGDash's ``signal_root_mean_square`` computes each channel's RMS amplitude;
# squaring it gives 32 channel-power features in V² before
# the natural logarithm. A small floor avoids undefined logarithms; it is not
# artifact rejection or normalization across people. Closely spaced keypress
# windows may overlap within a user, which further motivates keeping complete
# users apart during evaluation.

features, targets, identities = [], [], []
for query in queries:
    records = EEGDash().find({"dataset": "nm000104", "task": "typing", **query})
    if query["subject"] == "70495563":
        records = [
            record
            for record in records
            if record["bids_relpath"]
            == "sub-70495563/emg/sub-70495563_task-typing_emg.edf"
        ]
    dataset = EEGDashDataset(records=records, cache_dir=cache)
    assert len(dataset.datasets) == 1
    print(dataset.description.to_string(index=False))
    raw = dataset.datasets[0].raw
    print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description)[:30])
    channels = [f"EMG{i}" for i in range(32)]
    raw.pick(channels)
    mapping = {f"keystroke_{char}": i for i, char in enumerate(string.ascii_lowercase)}
    # Keep real labels and known onsets; controls and prompt events are excluded.
    events, _ = mne.events_from_annotations(raw, event_id=mapping)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.1,
        tmax=0.1,
        baseline=None,
        picks=channels,
        preload=True,
        event_repeated="drop",
    )
    X = epochs.get_data()
    assert np.isfinite(X).all()
    features.append(np.log(np.maximum(signal_root_mean_square(X) ** 2, 1e-30)))
    targets.append(epochs.events[:, 2])
    identities.append(query["subject"])
assert identities[0] != identities[1]
assert len(np.unique(targets[0])) > 1
print("Training/test keystrokes:", [len(y) for y in targets])

# %%
# Fit without calibrating on the held-out user
# --------------------------------------------
#
# StandardScaler learns channel-feature means and spreads from the
# training user only. Logistic regression uses its default regularization and
# up to 1000 optimizer iterations. Those settings are fixed; the held-out user
# must not be used to choose penalties, window widths or feature scaling.
#
# Balanced accuracy averages recall over the observed test letter classes, so
# frequent letters cannot dominate the score solely by their prevalence. Inspect
# class support when extending to more recordings: a letter absent from training
# cannot be learned by this classifier. The current script does not relabel such
# a letter as a known one to make evaluation easier.

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
predicted = model.fit(features[0], targets[0]).predict(features[1])
print(
    "Held-out user letter balanced accuracy:",
    balanced_accuracy_score(targets[1], predicted),
)
alphabet = np.asarray(list(string.ascii_lowercase))
reference = "".join(alphabet[targets[1]])
hypothesis = "".join(alphabet[predicted])


# %%
# Measure errors on the defined character stream
# ----------------------------------------------
#
# Predictions are converted back to letters in temporal order. Levenshtein
# distance counts the minimum insertions, deletions and substitutions needed to
# turn the decoded string into the observed lowercase stream. Dividing by the
# number of reference letters gives character error rate; in general it can exceed
# one if a decoder inserts many characters. RapidFuzz provides the exact edit
# distance without a tutorial implementation of the metric.
#
# This classifier emits one letter per observed event, so it cannot measure
# missed keystrokes or spurious detections in continuous time. A constant-letter
# prediction, nearly unit CER or poor cross-user accuracy remains a valid result.
# The printed reference and decoded snippets help expose such failure modes
# instead of hiding them in an aggregate plot.

cer = Levenshtein.distance(reference, hypothesis) / len(reference)
print("Aligned lowercase-stream CER:", cer)
print("Observed:", reference[:100])
print("Decoded: ", hypothesis[:100])
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["held-out user"], [cer])
ax.set(ylabel="Lowercase keystroke-stream CER")
plt.show()

# %%
# Extend from aligned classification to text decoding
# ---------------------------------------------------
#
# First use additional training and validation users to inspect signal
# units, channel quality, class coverage and placement differences. Keep the
# current test user untouched while comparing normalization or temporal features;
# repeatedly adjusting the pipeline to this user's score turns that user into
# validation data.
#
# For a full text task, retain spaces and control-key semantics, define whether
# the target is keystrokes or final edited text, and replace supplied event times
# with a sequence decoder that handles insertions and deletions. Evaluate whole
# sequences with the official user split and evaluator before making a competition
# claim. Changing the output head alone does not implement that task.
#
# Related grouping and transfer example: `Braindecode cross-dataset transfer
# <https://braindecode.org/dev/auto_examples/advanced_training/plot_transfer_learning.html>`_.
