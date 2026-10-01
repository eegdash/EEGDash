"""Preprocess and window recorded EEG
==================================

Select EEG channels, apply an explicit reference, and save reusable windows.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash with its EEGPrep tutorial dependencies; tutorials 01 and 02 introduce Raw and
window indexing. This file independently loads the recording and writes an
average-referenced prepared dataset. Allow additional disk space for that
output as well as the original download.
"""

# %%

import json
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from braindecode.preprocessing import (
    Preprocessor,
    RemoveDCOffset,
    RemoveCommonAverageReference,
    preprocess,
)

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# Check the source rate, channel names and stimulus-frequency descriptions
# before applying transforms. This is already a processed SSVEP release, so a
# generic filtering recipe intended for unprocessed acquisition would not be an
# appropriate default. The following DC-offset and reference operations are explicit changes
# to the source representation, separate from its existing filtering.
cache_dir = get_default_cache_dir()
subjects = ["1"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="nm000118",
    subject=subjects,
    session="0",
    run="0",
    task="ssvep",
    n_jobs=1,
)
dataset.description[["subject", "session", "run"]]

# %%
(recording,) = dataset.datasets  # This lesson opens exactly one recording.
raw = recording.raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# Apply EEGPrep offset and reference components
# ---------------------------------------------
# ``preprocess`` runs the list in order. After selecting EEG channels, the
# Braindecode EEGPrep adapters convert to EEGPrep's representation and back.
# ``RemoveDCOffset`` subtracts each channel's temporal median, while
# ``RemoveCommonAverageReference`` subtracts the instantaneous channel mean.
# These are different axes: the first centers each channel over the recording;
# the second references each time sample across the selected EEG channels.
# Neither operation is a claim that this already-processed source needs a
# full artifact-cleaning pipeline. No ASR, channel rejection or extra filter
# is applied. The median uses the complete recording, so this preparation is
# appropriate for an offline demonstration, not causal online prediction.
#
# This cap contains eight posterior channels, so its average is the mean of
# those eight sensors, not a whole-head reference. The operation can change
# common SSVEP components as well as unwanted common signals. It illustrates a
# reproducible preprocessing decision, not a claim that this reference always
# improves decoding.
#
# Preserve the source montage and avoid filtering already filtered trials.
# Average reference needs voltage channels, not electrode coordinates.
annotations_before = raw.annotations.copy()
measurement_date = raw.info["meas_date"]
source_grid = (raw.info["sfreq"], raw.n_times, raw.first_samp)
excerpt = raw.get_data(start=0, stop=int(4 * sfreq))
median = np.median(raw.get_data(), axis=1, keepdims=True)
centered_excerpt = excerpt - median
common_average = centered_excerpt.mean(axis=0)
preprocess(
    dataset,
    [
        Preprocessor("pick", picks="eeg"),
        RemoveDCOffset(),
        RemoveCommonAverageReference(),
    ],
)
raw = dataset.datasets[0].raw  # Adapters may replace the Raw object.
channel_names = raw.ch_names
# These two components do not resample or cut data. The format conversion
# can quantize annotation latencies, so preserve original event timing once
# the sample-grid invariants have been verified. Restore the measurement date
# first because annotations may use it as their absolute time origin.
if (raw.info["sfreq"], raw.n_times, raw.first_samp) != source_grid:
    raise RuntimeError(
        "Offset/reference changed the sample grid; do not restore events"
    )
raw.set_meas_date(measurement_date)
# With no absolute origin, set_annotations adds first_time itself.
if annotations_before.orig_time is None:
    annotations_before.onset -= raw.first_time
raw.set_annotations(annotations_before)
after = raw.get_data(start=0, stop=excerpt.shape[1])
np.testing.assert_allclose(after.mean(axis=0), 0, atol=1e-10)
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
times = np.arange(excerpt.shape[1]) / sfreq
for values, label in [
    (excerpt[0], "Source"),
    (centered_excerpt[0], "Median removed"),
    (after[0], "Average referenced"),
]:
    axes[0].plot(times, values * 1e6, label=label, alpha=0.8)
axes[0].set_ylabel(f"{channel_names[0]} (µV)")
axes[0].legend()
axes[1].plot(times, common_average * 1e6)
axes[1].set_ylabel("Subtracted mean (µV)")
axes[2].plot(times, after.mean(axis=0) * 1e6)
axes[2].set(xlabel="Recording time (s)", ylabel="Residual mean (µV)")

# %%
# 2. Window the observed trials
# -----------------------------
# Size and stride are equal to 1,024 samples, giving a four-second input tensor
# at 256 Hz. The last incomplete portion of each event is discarded instead of
# creating overlapping examples. Each row in ``metadata`` must correspond to
# one item in ``windows``; labels remain the observed frequency classes.
#
# After stacking, ``X`` has axes ``(trials, channels, samples)`` and still uses
# volts. The channel-mean check verifies the reference operation at each time
# sample. It says nothing about classifier performance or artifact removal.
# The source event lasts 4.15 seconds; its final 0.15 seconds are unused.
window_size = int(4 * sfreq)
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    window_size_samples=window_size,
    window_stride_samples=window_size,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata().reset_index(drop=True)
y = metadata["target"].to_numpy(dtype=int)
if (
    len(windows) != len(metadata)
    or metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
):
    raise ValueError("Window rows must have aligned, unique recording/start identities")
print(pd.crosstab(metadata["subject"], y))

X = np.stack([window[0] for window in windows])
if not (np.isfinite(X).all()):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )
print("Windows:", X.shape)


# %%
# 3. Save the prepared data in the persistent cache
# -------------------------------------------------
# Braindecode saves signal data together with the information needed to index
# windows and recover their descriptions. Keep the whole output directory,
# not just a FIF file. The directory name distinguishes this average-referenced
# version from the source-reference windows in tutorial 13.
#
# The plotted channels should be read in the context of that reference: their
# instantaneous average is zero, but each waveform need not have zero temporal
# mean. Tutorial 13 demonstrates reloading and checking prepared datasets;
# pass this page's printed directory to ``load_concat_dataset`` to inspect this
# specific reference choice in a later session.
# This tutorial owns this named output. Re-running refreshes that output.
prepared_path = cache_dir / "tutorial_10_nm000118_average_reference"
prepared_path.mkdir(parents=True, exist_ok=True)
windows.save(str(prepared_path), overwrite=True)
prepared_path.with_suffix(".manifest.json").write_text(
    json.dumps(
        {
            "dataset": "nm000118",
            "subjects": subjects,
            "session": "0",
            "run": "0",
            "transforms": [
                "pick EEG",
                "temporal median removal",
                "posterior common average",
            ],
            "channels": channel_names,
            "sfreq": sfreq,
            "mapping": mapping,
            "window_samples": window_size,
            "stride_samples": window_size,
            "on_last_window": "drop",
            "versions": {
                name: version(name)
                for name in ["eegdash", "braindecode", "mne", "eegprep"]
            },
        },
        indent=2,
    )
)
print("Saved reusable windows:", prepared_path.resolve())
plt.show()
