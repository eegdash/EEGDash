"""Inspect your first EEG recording
================================

Open one recording and inspect its real voltage traces and spectrum.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Tutorial 00 introduces the catalogue
query, but this script runs independently. You will need basic NumPy indexing.
The useful output is an MNE ``Raw`` object whose channels, units and event
labels you have inspected before making training windows.
"""

# %%

import matplotlib.pyplot as plt

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The dataset constructor retrieves descriptions; accessing ``recording.raw``
# opens the signal file and acquires it if absent from the cache. Count
# ``dataset.datasets`` to count recordings. The concatenated dataset's length
# has a different meaning and should not be used as the recording count.
#
# SSVEP means a response to repeated visual stimulation. Here annotation
# strings name the attended flicker frequency in Hz.
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
raw = dataset.datasets[0].raw
raw.annotations.to_data_frame().head()

# %%
# 2. Browse voltage and annotations with Braindecode
# --------------------------------------------------
# EEGDashDataset inherits Braindecode's public ``BaseConcatDataset.plot``
# (available since Braindecode 1.8.0, already required by EEGDash). It opens
# the file-backed recording in the interactive EEGDash viewer: scroll through
# channels and time, and inspect the stimulus annotations beside the traces.
# This replaces manual per-channel plotting; no local server is needed.
#
# Run this cell in the downloadable notebook (trust saved notebook output).
# It returns HTML, loads the deployed viewer, and embeds the recording bytes
# in the output. The roughly 7 MB source fits its default 64 MiB base64 limit.
# The viewer reads the original file, not later in-memory preprocessing;
# use MNE for inspecting transformed data. Static pages may not execute the
# embedded script; the spectrum below remains a static scientific view.
dataset.plot(index=0)

# %%
# 3. Inspect the whole-recording spectrum
# -----------------------------------------------------
# The averaged power spectral density summarizes how signal power is distributed
# across frequency over the recording. The 40 Hz display limit focuses on the
# low-frequency range and does not apply another filter. A spectral peak can
# suggest a rhythm or stimulus response, but this whole-recording average mixes
# all twelve attended frequencies and cannot validate a class label by itself.
#
# Channel coordinates describe where sensors were placed; channel names and
# order determine which signal is which. Neither an attractive montage nor a
# smooth spectrum replaces inspection of the individual trial voltages.
raw.compute_psd(fmax=40, picks="eeg").plot(average=True, show=False)
plt.show()

# %%
# Browse another annotated trial in the viewer.
# The release concatenates trials; their order is not acquisition chronology.

# %%
# Continue with event-labelled windows
# ------------------------------------
# Tutorial 02 converts this same recording into four-second windows and shows
# where its annotation-derived label appears in a DataLoader batch. When
# inspecting another trial manually, derive its start from the annotation
# onset rather than assuming four-second trials are adjacent with no remainder.
