"""Read recorded EEG without a metadata or download connection
===========================================================

Stage a 5.6 MB motor-imagery recording online, then reopen the same EEGDash
query with ``download=False``. The offline stage compares actual voltage
samples and BIDS identities. Run the first stage where internet is available.

Prerequisites: familiarity with the download how-to; keep the signal and BIDS
sidecars together under a writable EEGDASH_CACHE_DIR. No prior execution
is required: this script executes
both stages to demonstrate equivalence. It is not intended to be launched
from its first line on a disconnected machine: the first stage is online.
The source is BNCI2014-004, distributed as processed motor imagery in
`nm000135 <https://nemar.org/dataset/nm000135>`_.

"""

# %%
# 1. Populate the cache once on an internet-connected machine
# -----------------------------------------------------------
import os
from pathlib import Path

import numpy as np

from eegdash import EEGDashDataset

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")

# %%
# The online query selects one recording, not all of subject 1. The one-second
# reference has shape (channels, 250 samples) in volts. It remains in memory
# for the verification below; it is not a second generated or transformed
# recording.
online = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
online.download_all(n_jobs=1)
reference = online.datasets[0].raw.get_data(start=0, stop=250)

# %%
# 2. Open the cached BIDS files on the offline machine
# ----------------------------------------------------
# This constructor reads local BIDS records rather than querying the registry.
# Use the same subset. No network fallback or substitute recording is used.
# Keep the same cache root and exact subject/session/run/task values after
# staging. ``download=False`` changes discovery to the local BIDS tree. It does
# not turn a partial cache into a complete dataset, and it must not be used to
# silence a failed download. A missing file requires repairing the staged copy.
#
# In a notebook, run stage 1, disconnect networking without restarting the
# kernel, then run stage 2: the reference array still exists. On a separate
# offline machine, copy the complete cache and execute the offline constructor
# and identity checks; the in-memory online comparison needs its saved
# reference or the original notebook kernel.
offline = EEGDashDataset(cache_dir=cache_dir, **query, download=False, n_jobs=1)
assert len(offline.datasets) == 1
raw = offline.datasets[0].raw
np.testing.assert_array_equal(reference, raw.get_data(start=0, stop=250))
for entity in ["subject", "session", "run", "task"]:
    assert str(offline.description.iloc[0][entity]) == query[entity]
assert {"left_hand", "right_hand"} <= set(raw.annotations.description)
print("Offline signal:", raw)
print(offline.description[["subject", "session", "run", "task"]])

# %%
# To verify network independence on your machine, stage first, disconnect
# networking, and run only stage 2. If opening fails, restore the missing
# original files before training; never create an empty stand-in file.

# 3. Decide when the offline job is ready
# ---------------------------------------
# The printed description should still identify subject 1, session 0train and
# run 0. The assertion checks observed left/right annotations, not arbitrary
# integer codes, so a successfully opened unrelated recording cannot satisfy
# that label contract alone together with the identity checks.
#
# Only data loading is tested offline here. Installing packages, loading an
# uncached model checkpoint, or making another registry query are separate
# network operations. Stage those requirements before disconnecting and keep
# the compute script's constructor in download=False mode.
