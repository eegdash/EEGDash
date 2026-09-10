"""Save and reload prepared EEG windows
====================================

Persist real windows with their labels and verify the reloaded samples.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Tutorial 02 introduces the window
objects saved here. No earlier output is needed: the script loads real data,
writes a persistent prepared directory, then reloads it through a new handle.
Keep ``EEGDASH_CACHE_DIR`` stable across sessions to reuse the printed path.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash import EEGDashDataset
import json
from importlib.metadata import version
from braindecode.datautil import load_concat_dataset

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# An input cache and a prepared-data cache solve different problems. EEGDash's
# cache avoids fetching source files again; the prepared output captures the
# representation and indexing choices used by your analysis. Here we retain
# the source reference. Tutorial 10's average-referenced output deliberately
# has a different directory name.
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
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
assert len(dataset.datasets) == len(subjects)
print(dataset.description[["subject", "session", "run"]])
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
assert len(mapping) == 12
for recording in dataset.datasets:
    assert recording.raw.ch_names == channel_names
    assert recording.raw.info["sfreq"] == sfreq
    assert set(recording.raw.annotations.description) == set(mapping)
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# 2. Window the observed trials
# -----------------------------
# The serialization check covers four-second, non-overlapping windows with
# annotation-derived labels. Besides ``(trials, channels, samples)`` voltages,
# we retain start-sample and recording identifiers. Matching only the array
# shape after loading would miss reordered windows or targets.
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
assert len(windows) == len(metadata)
assert set(y) == set(mapping.values())
assert not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
print(pd.crosstab(metadata["subject"], y))

X = np.stack([window[0] for window in windows])
assert X.shape == (len(metadata), len(channel_names), window_size)
assert np.isfinite(X).all()
print("Windows:", X.shape)

# %%
# 3. Write a persistent Braindecode dataset and a provenance manifest
# -------------------------------------------------------------------
# ``save`` writes Braindecode's signal and indexing files into the prepared
# directory. Depending on the window dataset representation, serialization can
# store a backing Raw file and window metadata rather than one separate file
# for every trial. Copy the complete directory when moving the prepared data.
#
# The adjacent JSON manifest explains how to interpret it: the exact query,
# frequency-to-class mapping, channel order, sample rate and package versions.
# It lives outside the numbered recording directories so the loader does not
# mistake it for a recording. This is a record of local preparation settings,
# not a checksum-based guarantee that a future upstream release is identical.
# Only this tutorial's named output is overwritten on rerun. Nothing is
# deleted at exit: use this path in another process to reuse the windows.
prepared_path = cache_dir / "tutorial_13_nm000118_windows"
prepared_path.mkdir(parents=True, exist_ok=True)
windows.save(str(prepared_path), overwrite=True)
manifest = {
    "dataset": "nm000118",
    "subjects": subjects,
    "session": "0",
    "run": "0",
    "task": "ssvep",
    "window_samples": window_size,
    "mapping": mapping,
    "channels": channel_names,
    "sfreq": sfreq,
    "versions": {name: version(name) for name in ["eegdash", "braindecode", "mne"]},
}
prepared_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

# %%
# 4. Reload and check samples, labels and recording identity
# ----------------------------------------------------------
# ``load_concat_dataset`` opens the prepared files, not a new catalogue query.
# ``preload=True`` reads their signals into memory for repeated indexing. The
# checks compare every window numerically, every label exactly, and all window
# metadata. The small numerical tolerance permits float32 serialization without
# accepting reordered or differently referenced signals.
#
# The plotted waveform is read from the reloaded dataset, so its availability
# does not depend on the original Python object's lifetime. ``Saved bytes``
# measures this prepared directory on disk; it is not a network-download count.
reloaded = load_concat_dataset(str(prepared_path), preload=True)
assert len(reloaded) == len(windows)
np.testing.assert_allclose(
    np.stack([item[0] for item in reloaded]), X, rtol=1e-6, atol=1e-12
)
np.testing.assert_array_equal([item[1] for item in reloaded], y)
pd.testing.assert_frame_equal(reloaded.get_metadata().reset_index(drop=True), metadata)
print("Reusable path:", prepared_path.resolve())
print(
    "Saved bytes:",
    sum(p.stat().st_size for p in prepared_path.rglob("*") if p.is_file()),
)
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(np.arange(window_size) / sfreq, reloaded[0][0][0] * 1e6)
ax.set(xlabel="Time (s)", ylabel=f"{channel_names[0]} (µV)", title="Reloaded trial")
plt.show()

# %%
# Reuse the directory in a new process
# ------------------------------------
# In another script, import ``load_concat_dataset`` and call it with the
# printed prepared directory. Check the adjacent manifest before combining
# that data with another prepared dataset. For large recordings you can choose
# ``preload=False`` and load samples on access. If you change the query,
# reference or window duration, give the new preparation its own output name
# so an earlier analysis remains reproducible.
