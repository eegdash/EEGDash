"""From EEGDash to a PyTorch DataLoader
====================================

Create event-labelled Braindecode windows and inspect an actual minibatch.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash with Braindecode and PyTorch. Tutorial 01 explains the MNE
recording used here; no earlier output file is required. This page stops at a
checked minibatch so you can understand the data interface before fitting a
network.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from torch.utils.data import DataLoader

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The annotation strings supply the twelve classification targets. ``mapping``
# converts each observed frequency to a contiguous integer from 0 to 11,
# which is the label representation a multiclass neural loss expects. Retain
# ``class_names`` to translate predictions back to Hz: integer class 0 is an
# index, not a zero-Hz stimulus.
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
# 2. Window the observed trials
# -----------------------------
# Window size and stride are sample counts. Setting both to ``4 * sfreq``
# creates non-overlapping four-second windows. ``on_last_window="drop"`` avoids
# adding a shifted final window to cover the remaining fraction of each event;
# for these 4.15-second annotations, the result is one window per trial.
# ``preload=True`` makes subsequent indexing read prepared data in memory.
#
# ``get_metadata`` preserves subject, session, run and start-sample information
# alongside ``target``. Recording identifiers plus start samples identify windows.
# The crosstab reports observed counts, making missing or imbalanced classes
# visible before any model is constructed.
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
print(pd.crosstab(metadata["subject"], y))

# %%
# 3. Batch windows and their observed targets
# -------------------------------------------
# A window item contains signal, target and crop indices. The DataLoader stacks
# the signal into ``(batch, channels, samples)`` and the targets into one integer
# per example. Here the first signal batch is ``(16, 8, 1024)`` in volts.
# The crop-index output retains the window position information; it is not a
# third training target.
#
# Batch size 16 keeps inspection small. ``num_workers=0`` loads in the current
# process, which is straightforward in both scripts and notebooks. The
# non-shuffled order aligns batch labels with the first metadata rows.
# It is an inspection choice, not the recommended training
# order. The plots select up to four examples and the first channel without changing
# the arrays passed to a future network.
batch_size = 16  # Try 7 or 32; the final batch may contain fewer examples.
loader = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0)
X_batch, y_batch, crop_indices = next(iter(loader))
# With shuffle=True, batch labels still travel with their signals, but no longer
# match the first metadata rows by position.
if not np.isfinite(X_batch.numpy()).all():
    raise ValueError("Nonfinite batch voltages; inspect the selected windows")
print("Batch:", X_batch.shape, "labels:", y_batch.tolist())
print("First batch crop indices:", crop_indices)
fig, axes = plt.subplots(
    min(4, len(y_batch)) + 1, 1, figsize=(9, 8), layout="constrained"
)
for i, ax in enumerate(axes[:-1]):
    ax.plot(np.arange(window_size) / sfreq, X_batch[i, 0].numpy() * 1e6)
    ax.set(
        ylabel=f"{channel_names[0]} (µV)",
        title=f"Batch row {i}: class {int(y_batch[i])} = {class_names[int(y_batch[i])]} Hz",
    )
axes[-2].set_xlabel("Time in window (s)")
duration = raw.annotations.duration[0]
axes[-1].broken_barh(
    [(0, duration)], (0, 0.35), facecolors="lightgray", label="Source event"
)
axes[-1].broken_barh(
    [(0, window_size / sfreq)],
    (0.4, 0.35),
    facecolors="tab:blue",
    label="Retained window",
)
axes[-1].set(
    xlabel="Time from first event onset (s)",
    yticks=[],
    title=f"Unused event tail: {duration - window_size / sfreq:.2f} s",
)
axes[-1].legend()
plt.show()

# %%
# Use this batch contract in training
# -----------------------------------
# Change ``batch_size`` and confirm that only the leading dimension changes.
# For training, first follow tutorial 11 to obtain independent train/test data,
# then make a separate loader for each. A shuffled loader over the full cohort
# is not a substitute for a held-out split.
