"""Pass EEGDash recordings into a MOABB paradigm
=============================================

Load real motor imagery with EEGDash, expose those same MNE recordings
through MOABB's dataset interface, and evaluate the resulting MOABB epochs.
This small adapter demonstrates signal and label interoperability; it is
not a full MOABB benchmark. Install the optional ``moabb`` dependency.

Use `NEMAR nm000135 <https://nemar.org/dataset/nm000135>`_ (BNCI2014-004),
subject 1, sessions ``0train`` and ``1train``, run 0: approximately 11 MB.
Set ``EEGDASH_CACHE_DIR`` to reuse the first download. The release is already
processed; MOABB applies the explicitly selected 8–30 Hz analysis filter.
Missing dependencies, downloads or labels raise errors instead of producing
replacement results.

Prerequisites: tutorial 52's real session split and familiarity with MNE Raw
and Epochs. Install EEGDash and the optional MOABB package in a compatible
Python environment. A missing MOABB import must be resolved before running;
the notebook has one acquisition path and one measured evaluation path.
The useful output is a MOABB-produced epoch object with retained session
identity, followed by predictions for the reserved session.

"""

# %%
# 1. Load and inspect the EEGDash signal source
# ---------------------------------------------
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from moabb.paradigms import LeftRightImagery
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash import EEGDashDataset
from eegdash.features import signal_variance

source = EEGDashDataset(
    cache_dir=Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache")),
    dataset="nm000135",
    subject="1",
    session=["0train", "1train"],
    run="0",
    task="imagery",
    n_jobs=1,
)
assert len(source.datasets) == 2
print(source.description[["subject", "session", "run"]])
for recording in source.datasets:
    raw = recording.raw
    print(raw.ch_names, raw.info["sfreq"], np.unique(raw.annotations.description))
    assert {"left_hand", "right_hand"}.issubset(raw.annotations.description)


# %%
# 2. Adapt the already loaded recordings to MOABB's public dataset contract
# -------------------------------------------------------------------------
# MOABB requires nested subject/session/run dictionaries. Keep genuine BIDS
# session and run names. Return copies because paradigm processing may mutate
# MNE objects. Integer event codes only encode the observed annotation names.
# This adapter is required by MOABB's external dataset interface; it is not
# a second acquisition or feature-extraction abstraction. It implements
# the two abstract methods required by BaseDataset.
# ``_get_single_subject_data`` is MOABB's dataset-provider hook despite its
# leading underscore. ``data_path`` intentionally has no second downloader:
# MOABB receives the EEGDash objects we already acquired. The equality check
# below verifies that copying preserves their actual samples before filtering.
#
# ``interval=[0, 3]`` declares our analysis interval relative to each existing
# cue. ``sessions_per_subject=2`` describes the actual two-session selection,
# not a request for MOABB to synthesize or fetch extra sessions.
class EEGDashImagery(BaseDataset):
    def __init__(self, recordings):
        self.recordings = recordings
        super().__init__(
            subjects=[1],
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2},
            code="EEGDashImagery",
            interval=[0, 3],
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        assert subject == 1
        return {
            str(recording.description["session"]): {
                str(recording.description["run"]): recording.raw.copy().load_data()
            }
            for recording in self.recordings.datasets
        }

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        raise NotImplementedError("EEGDash owns acquisition; use the loaded recordings")


adapter = EEGDashImagery(source)
# Check the actual signal handoff before MOABB filters or rescales anything.
handed_off = adapter._get_single_subject_data(1)
for recording in source.datasets:
    raw = handed_off[str(recording.description["session"])][
        str(recording.description["run"])
    ]
    np.testing.assert_array_equal(raw.get_data(), recording.raw.get_data())

# %%
# 3. Let MOABB create labelled epochs from those EEGDash recordings
# -----------------------------------------------------------------
# MNE Epochs retain volts. The explicit interval starts at the existing cue;
# no extra latency correction is applied. Channel order is fixed by name.
paradigm = LeftRightImagery(
    fmin=8, fmax=30, tmin=0, tmax=3, channels=["C3", "Cz", "C4"]
)
# %%
# Returning Epochs keeps MNE's volt units explicit. With inclusive endpoints,
# 0–3 seconds at 250 Hz yields 751 samples, unlike the fixed 750-sample
# Braindecode windows in tutorial 52. This endpoint and the extra 8–30 Hz filter
# mean scores across the two pages are not a paired pipeline comparison.
# MOABB supplies class-name strings in y and session identities in metadata.
epochs, y, metadata = paradigm.get_data(adapter, subjects=[1], return_epochs=True)
X = epochs.get_data()
assert len(X) == len(y) == len(metadata)
assert np.isfinite(X).all() and set(y) == {"left_hand", "right_hand"}
assert set(metadata.session) == {"0train", "1train"}
print("MOABB epochs:", X.shape, epochs.ch_names, "units: volts")
print(pd.crosstab(metadata.session, y))

# %%
# 4. Fit only on the first session and predict the second
# -------------------------------------------------------
# EEGDash signal_variance summarizes the band-filtered signal per channel
# in V²; taking its log gives three features per trial. Calling the public
# feature directly accepts the MOABB array without another dataset adapter.
# It is a fixed per-trial transform. Scaling and classification learn only from train.
features = np.log(np.maximum(signal_variance(X), 1e-30))
train = np.flatnonzero(metadata.session.to_numpy() == "0train")
test = np.flatnonzero(metadata.session.to_numpy() == "1train")
assert set(train).isdisjoint(test)
assert set(y[train]) == set(y[test]) == {"left_hand", "right_hand"}
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(features[train], y[train])
prediction = model.predict(features[test])
print(
    "Held-out session balanced accuracy:", balanced_accuracy_score(y[test], prediction)
)

# %%
# 5. Inspect predictions from the actual MOABB-produced epochs
# ------------------------------------------------------------
ConfusionMatrixDisplay.from_predictions(y[test], prediction, normalize="true")
plt.title("EEGDash → MOABB: subject 1, held-out session 1train")
plt.show()
# A full benchmark additionally needs more participants, a preregistered
# evaluation protocol, and explicit MOABB result-cache management.

# %%
# Inspect and extend the handoff
# ------------------------------
# The resulting array has 240 trials, three motor channels and 751 samples.
# Log variance reduces each trial to three features; the StandardScaler then
# uses only session 0train. The confusion matrix is row-normalized: diagonal
# entries are left- and right-hand recalls, whose average is balanced accuracy.
# A weak diagonal is a measured model limitation, not an integration failure.
#
# For another imagery subset, first check the annotation vocabulary, cue
# interval and named channels, then update the adapter declaration to match
# the queried recordings. Before using a MOABB evaluator, also configure its
# result storage and add enough participants for the claimed evaluation unit.
# This page proves the signal-to-paradigm boundary; it does not reproduce a
# published MOABB leaderboard.
