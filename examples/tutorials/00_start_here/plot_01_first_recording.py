"""Load one EEG recording and inspect it
=====================================

What does a single EEG recording from EEGDash actually contain — how many
channels, sampling at what rate, with which annotations? In this tutorial we
load **one** recording from the Wakeman and Henson visual face perception
dataset on OpenNeuro (``ds002718``), inspect its `BIDS <https://bids.neuroimaging.io/>`_
metadata via the new ``EEGDashDataset.preview()`` helper, and look at the
first five seconds of signal. Why look at one record before scaling up?
"""

# %% [markdown]
# Learning objectives
# -------------------
#
# - Build an ``EEGDashDataset`` and read ``len(dataset)`` plus ``summary()``.
# - Use ``preview(index=0)`` to materialize one recording lazily.
# - Read sampling rate, channel count, duration, and annotations from ``raw``.
# - Plot the first 5-second snippet via ``RecordingPreview.plot()``.
# - Predict where data caches and why the first call is slow.
#
# See also the explanation page :doc:`/concepts/eegdash_objects` for the
# object model behind ``EEGDashDataset``.

# %% [markdown]
# Requirements
# ------------
#
# - Runtime: ~2 min on CPU on first run; <10 s once cached.
# - Network: ~80 MB on the first call (downloaded from OpenNeuro into the
#   cache directory). No network on subsequent runs.
# - Cache directory: parametrised via ``EEGDASH_CACHE_DIR`` (defaults to
#   ``~/.eegdash_cache``).
# - Prerequisite tutorial: ``plot_00_first_search`` for query basics.

# %% [markdown]
# Setup
# -----
#
# Imports follow the EEGDash convention: standard library, third-party
# (``numpy``, ``matplotlib``), then ``eegdash``. We seed NumPy for reproducibility.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eegdash
from eegdash import EEGDashDataset

np.random.seed(42)
cache_dir = os.environ.get("EEGDASH_CACHE_DIR", str(Path.home() / ".eegdash_cache"))
print(f"eegdash version: {eegdash.__version__}")
print(f"cache directory: {cache_dir}")

# %% [markdown]
# Step 1: Pick one recording
# --------------------------
#
# We pick subject ``002`` from ``ds002718`` (Wakeman and Henson's visual face
# perception study). One subject keeps the download small and the figure
# readable; the same code generalises by changing the BIDS entities.
#
# **Predict**: most consumer-grade EEG caps record 32 to 128 channels at 250 Hz
# to 1000 Hz. The Wakeman and Henson dataset uses 70 EEG channels at 1100 Hz.
# Note your prediction down to compare against the actual values.

# %%
DATASET = "ds002718"
SUBJECT = "002"

# %% [markdown]
# Step 2: Load it via ``EEGDashDataset`` and ``preview(index=0)``
# ---------------------------------------------------------------
#
# **Run**: building an ``EEGDashDataset`` queries the EEGDash metadata catalog
# for matching records — no EEG samples are read yet (lazy loading). The first
# call to ``preview(index=0)`` materializes the first recording: it downloads
# the BIDS file into ``cache_dir`` if missing, opens it with MNE-Python, and
# returns a compact ``RecordingPreview`` dataclass.

# %%
dataset = EEGDashDataset(cache_dir=cache_dir, dataset=DATASET, subject=SUBJECT)
print(f"matched records: {len(dataset)}")
assert len(dataset) >= 1, "Expected at least one recording for subject 002."

# %% [markdown]
# **Investigate**: ``dataset.summary()`` aggregates the records *without*
# touching the network. It is the safest first look at a query result.

# %%
report = dataset.summary(verbose=True)

# %% [markdown]
# Step 3: Inspect channels, sfreq, duration, annotations
# ------------------------------------------------------
#
# **Run** the recording-level loader: ``preview`` reads the file, slices the
# first 5 seconds for a quick glance, and exposes ``raw``, ``metadata``,
# ``annotations``, and a ``plot()`` shortcut.

# %%
preview = dataset.preview(index=0)
raw = preview.raw
sfreq = float(raw.info["sfreq"])
nchan = int(raw.info["nchan"])
duration = float(raw.times[-1] - raw.times[0])

assert sfreq > 0 and nchan > 0 and duration > 1.0, "Recording must have data."

print(f"subject:        {preview.metadata.get('subject', '?')}")
print(f"task:           {preview.metadata.get('task', '?')}")
print(f"sampling rate:  {sfreq:.1f} Hz")
print(f"channels:       {nchan}")
print(f"duration:       {duration:.1f} s")
print(f"annotations:    {len(preview.annotations)}")
if preview.annotations:
    print(f"first annot:    {preview.annotations[0]['description']!r}")

# %% [markdown]
# **Investigate**: BIDS surfaces five entities — ``dataset``, ``subject``,
# ``task``, ``session``, ``run`` — and EEGDash exposes them on
# ``preview.metadata``. The annotations come straight from the BIDS
# ``events.tsv`` file and inherit MNE's onset/duration/description schema.

# %% [markdown]
# Step 4: Plot a 5-second snippet
# -------------------------------
#
# **Run** ``preview.plot()``. It calls :meth:`mne.io.Raw.plot` under the hood;
# ``show=False`` returns a figure we can render without blocking on a viewer.
# We also plot the raw 5-second snippet array (shape
# ``(n_channels, n_samples)``) for the first channel.

# %%
fig_raw = preview.plot(
    duration=5.0,
    n_channels=min(20, nchan),
    show=False,
    title=f"{DATASET} sub-{SUBJECT} | {nchan} ch @ {sfreq:.0f} Hz",
)
fig_raw.suptitle(f"Source: OpenNeuro {DATASET}", y=0.02, fontsize=8)

snippet = np.asarray(preview.snippet)
times = np.arange(snippet.shape[1]) / sfreq
fig, ax = plt.subplots(figsize=(7.2, 2.4), constrained_layout=True)
ax.plot(times, snippet[0] * 1e6, color="#0072B2", linewidth=0.8)
ax.set_xlabel("time (s)")
ax.set_ylabel(f"{raw.ch_names[0]} (µV)")
ax.set_title(
    f"First 5 s, channel {raw.ch_names[0]} | sub-{SUBJECT}, {nchan} ch @ {sfreq:.0f} Hz"
)
ax.text(
    0.99,
    -0.35,
    f"Source: OpenNeuro {DATASET}",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
    color="#64748B",
)
plt.show()

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
# **Run.** Asking for a recording index past the end is the most common
# slip when iterating; we trigger it on purpose with ``try/except`` so
# the failure mode is visible.

# %%
try:
    _ = dataset.preview(index=999)
except (IndexError, KeyError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: clamp to a valid index.
    safe_idx = min(999, len(dataset) - 1)
    print(f"Recovery: preview(index={safe_idx}) instead of 999.")

# %% [markdown]
# Modify
# ------
#
# **Your turn**: change ``SUBJECT`` to ``"003"`` and re-run Steps 2 and 3.
# Compare the printed sampling rate, channel count, duration, and annotation
# count — they should match the previous subject because the acquisition
# protocol is shared, but small variations in run length per subject occur.

# %%
preview_other = EEGDashDataset(
    cache_dir=cache_dir, dataset=DATASET, subject="003"
).preview(index=0)
print(
    f"sub-003: sfreq={preview_other.raw.info['sfreq']:.1f} Hz, "
    f"nchan={preview_other.raw.info['nchan']}, "
    f"duration={preview_other.raw.times[-1]:.1f} s"
)

# %% [markdown]
# Try it yourself (Make / Extensions)
# -----------------------------------
#
# Pick a *different task* (or dataset) and repeat the inspection.
#
# - **Easier**: keep ``ds002718`` and pass ``task="FacePerception"`` explicitly.
# - **Same difficulty**: try another OpenNeuro dataset (e.g. ``ds002893``).
# - **Harder**: filter two subjects, then ``preview(index=1)`` for the second.

# %% [markdown]
# Result
# ------

# %%
print(
    f"loaded {len(dataset)} record(s) from {DATASET}; "
    f"sub-{SUBJECT} has {nchan} channels at {sfreq:.0f} Hz "
    f"for {duration:.1f} s with {len(preview.annotations)} annotations."
)

# %% [markdown]
# Wrap-up
# -------
#
# We loaded **one** EEG recording end-to-end without writing a download script:
# ``EEGDashDataset`` resolves the BIDS query against OpenNeuro,
# ``preview(index=0)`` lazily materializes one ``mne.io.Raw`` plus a 5-s
# snippet, and ``summary()`` reports counts without I/O. The next tutorial
# (``plot_02_dataset_to_dataloader``) wraps these recordings in a PyTorch
# ``DataLoader``. **Caveats**: the signal is unprocessed — alpha rhythms, line
# noise (50/60 Hz), and slow drifts coexist in this view; preprocessing
# belongs in ``plot_10_preprocess_and_window``.
#
# References
# ----------
#
# - Wakeman, D. G., and Henson, R. N. (2015). A multi-subject, multi-modal
#   human neuroimaging dataset. *Scientific Data* 2:150001. OpenNeuro
#   ``ds002718`` v1.0.5, doi:10.18112/openneuro.ds002718.v1.0.5.
# - Pernet, C. R. et al. (2019). EEG-BIDS: an extension to the brain imaging
#   data structure for electroencephalography. *Scientific Data* 6:103,
#   doi:10.1038/s41597-019-0104-8.
# - Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python.
#   *Frontiers in Neuroscience* 7:267, doi:10.3389/fnins.2013.00267.
#
# **See also**: ``how_to_work_offline`` (cache reuse without network) and
# ``how_to_handle_bad_records`` (drop integrity-failed recordings).
