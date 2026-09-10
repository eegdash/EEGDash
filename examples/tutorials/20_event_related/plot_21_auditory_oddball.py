"""Inspect an auditory oddball response in real EEG
================================================

Use EEGDashDataset to load one auditory-P300 run from OpenNeuro
``ds003061``, inspect its actual labels and plot standard/oddball ERPs.
Subject 001, run 2 contains about 63.4 MB of signal data. Selecting the
run avoids downloading the other two recordings of this participant.
See the `dataset <https://openneuro.org/datasets/ds003061>`_. CPU is sufficient;
set ``EEGDASH_CACHE_DIR`` to retain downloads between runs.

The goal is to trace an observed stimulus marker into a measured response:
load the recording, select the two documented event types, create aligned
trials, and average each condition. An ERP is an average voltage time course;
it does not assign a predicted label to a new trial. This page therefore
ends with a descriptive comparison rather than a fitted classifier.

With EEGDash installed, run the blocks in order. Basic MNE Raw concepts from
the first-recording page are sufficient. You will inspect the retained trial
counts and array shape, a Cz waveform plot and a mean difference in microvolts.
"""

# %%
# 1. Load exactly one recording and inspect its annotations
# ---------------------------------------------------------
# Subject identifiers are strings: retain the leading zeros in ``001``.
# The task and run filters bound acquisition before any samples are read.
# Printing annotation counts makes the source vocabulary visible and avoids
# accidentally treating responses or unrelated markers as standard stimuli.
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import RemoveCommonAverageReference, RemoveDCOffset

from eegdash import EEGDashDataset
from eegdash.features import signal_mean

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="ds003061",
    subject="001",
    task="P300",
    run="2",
    n_jobs=1,
)
assert len(dataset.datasets) == 1
raw = dataset.datasets[0].raw.copy().load_data().pick("eeg")
print(pd.Series(raw.annotations.description).value_counts())
# Preserve the source marker spelling below, including its typo.
# These literal names are the label source, including the response qualification
# on the oddball class. Consequently the comparison concerns the selected
# response-associated oddballs, not every possible rare-stimulus outcome.
mapping = {"stimulus/standard": 1, "stimulus/oddball_with_reponse": 2}
assert set(mapping) <= set(raw.annotations.description)
assert "Cz" in raw.ch_names, "This ERP comparison requires Cz"

# %%
# 2. Filter and epoch relative to the actual stimulus onset
# ---------------------------------------------------------
# Keep the pre-stimulus baseline. We use MNE Epochs so the stated interval
# is independent of the annotation duration; resample after event extraction
# to preserve event timing on the original recording's sample grid.
# EEGPrep removes the per-channel median offset and applies its common
# average reference. Neither operation changes sample count or rate. We
# verify that grid and restore the original event annotations to avoid
# rounding their latencies through the MNE↔EEGLAB conversion. Filtering to 0.5–30 Hz attenuates drift and faster activity while keeping
# a broad ERP time course. These settings affect the measured waveform; hold
# them fixed when comparing recordings.
#
# Each epoch includes 100 ms before and 600 ms after the stimulus. Baseline
# correction subtracts the pre-stimulus channel mean within each trial.
# ``get_data()`` returns (trials, channels, time samples), with voltages in
# volts. Resampling to 128 Hz changes the last dimension, not trial identity.
# The printed condition counts refer to retained epochs, whereas the earlier
# counts describe all annotations. Epochs can be lost at recording boundaries
# or existing bad spans; missing conditions stop the example rather than
# silently producing an empty average.
source_annotations = raw.annotations.copy()
source_date = raw.info["meas_date"]
source_grid = (raw.info["sfreq"], raw.n_times, raw.first_samp)
RemoveDCOffset().apply(raw)
RemoveCommonAverageReference().apply(raw)
assert (raw.info["sfreq"], raw.n_times, raw.first_samp) == source_grid
raw.set_meas_date(source_date)
raw.set_annotations(source_annotations)
raw.filter(0.5, 30.0)
events, _ = mne.events_from_annotations(raw, event_id=mapping)
epochs = mne.Epochs(
    raw,
    events,
    event_id={"standard": 1, "oddball": 2},
    tmin=-0.1,
    tmax=0.6,
    baseline=(-0.1, 0),
    preload=True,
    reject_by_annotation=True,
)
epochs.resample(128)
assert np.isfinite(epochs.get_data()).all()
assert all(len(epochs[name]) > 1 for name in epochs.event_id)
print({name: len(epochs[name]) for name in epochs.event_id})
print("Epoch shape:", epochs.get_data().shape)

# %%
# 3. Plot the measured response at Cz
# -----------------------------------
# Compare conditions within this recording. A difference in this small
# sample does not by itself distinguish MMN, P3a and P3b generators.
# ``average()`` reduces the trial axis separately for each condition. MNE
# retains the time and channel information in an Evoked object and displays
# EEG amplitudes in microvolts. More trials can make an average less noisy;
# the two conditions need not have equal counts.
#
# EEGDash's ``signal_mean`` averages the oddball-minus-standard waveform from
# 250 to 400 ms at the preselected Cz channel. Positive values mean the
# oddball average is more positive in that interval under this reference;
# negative values are valid too. This interval mean is not a peak latency,
# a significance test or evidence that a new participant will show the effect.
evokeds = {name: epochs[name].average() for name in epochs.event_id}
mne.viz.plot_compare_evokeds(evokeds, picks="Cz", show=False)
difference = mne.combine_evoked([evokeds["oddball"], evokeds["standard"]], [1, -1])
interval = (difference.times >= 0.25) & (difference.times <= 0.4)
cz = difference.ch_names.index("Cz")
print(
    "Mean oddball-minus-standard at Cz, 250–400 ms (µV):",
    signal_mean(difference.data[cz, interval]) * 1e6,
)
plt.show()

# %%
# 4. Decide what evidence the next analysis needs
# -----------------------------------------------
# Inspect bad channels and artifacts before treating the difference as a
# physiological result: existing bad annotations are respected, but no new
# artifact detector is fitted here. To estimate between-participant variation,
# repeat the same fixed analysis for additional real participants and compare
# their interval means; individual trials are not independent participants.
# For a descriptive auditory/visual comparison, run the visual tutorial with
# matched reference, units and time axes. Different tasks and participants
# prevent that plot alone from establishing a modality effect.
