"""Download a real EEG subset and verify it can be reopened
========================================================

Prefetch one BNCI2014-004 motor-imagery recording through EEGDashDataset.
The explicit query downloads about 5.6 MB including sidecars. The same
operation can stage a larger cohort by expanding the subject/session query.
See `nm000135 <https://nemar.org/dataset/nm000135>`_. CPU is sufficient.

Before running, install EEGDash and verify that the chosen cache directory
is writable and persistent. This is an acquisition recipe: it prepares files
for a later analysis and does not train a classifier. Subject 1, session
0train, run 0 and task imagery identify one recording, rather than every
session belonging to that person.

"""

# %%
# 1. Specify the subset and persistent cache
# ------------------------------------------
import os
from pathlib import Path

import numpy as np

from eegdash import EEGDashDataset

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")

# %%
# The constructor resolves the query into recording metadata. Signal loading
# is lazy, so inspecting ``dataset.description`` does not by itself establish
# that the entire recording can be read. The recording-count assertion catches
# an unexpectedly broadened query before the explicit download starts.
dataset = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
assert len(dataset.datasets) == 1
print(dataset.description[["subject", "session", "run"]])

# %%
# 2. Download the selected recording and its dependencies
# -------------------------------------------------------
# The download is bounded by the query, not by cropping an already opened file.
# ``download_all`` stages the selected signals and their required BIDS
# sidecars. Events and channel metadata are part of a usable dataset; copying
# only a signal filename can leave an incomplete offline cache. One worker
# keeps this first acquisition easy to inspect. Increase concurrency only
# when the remote service and storage can sustain it.
dataset.download_all(n_jobs=1)
raw = dataset.datasets[0].raw
assert raw.n_times > 0 and raw.info["sfreq"] == 250
assert {"left_hand", "right_hand"} <= set(raw.annotations.description)

# %%
# 3. Reopen the same query from disk and compare real samples
# -----------------------------------------------------------
# The second constructor uses local discovery through ``download=False``.
# At 250 Hz, samples 0:250 cover one second. ``get_data`` returns a
# (channels, samples) array in volts; equality checks the reader result, not
# just the existence or size of a file. The observed hand annotation names
# check that the cache includes meaningful target information as well.
offline = EEGDashDataset(cache_dir=cache_dir, **query, download=False, n_jobs=1)
assert len(offline.datasets) == len(dataset.datasets)
np.testing.assert_array_equal(
    raw.get_data(start=0, stop=250), offline.datasets[0].raw.get_data(start=0, stop=250)
)
root = cache_dir / "nm000135"
print(
    "Cached bytes:",
    sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
)
print("Offline recording:", offline.datasets[0].raw)

# %%
# Keep the cache on a persistent volume for subsequent jobs. A nonempty
# filename alone is not proof of valid EEG; opening it and comparing samples
# checks the actual reader path. Expand the check to all queried recordings
# when staging a larger cohort.

# 4. Understand what was verified
# -------------------------------
# The byte count covers everything currently stored under nm000135, including
# other sessions from earlier runs. It is a measurement of this cache directory,
# not a fresh-download byte counter. The equality assertion verifies the first
# second; it is a quick read check rather than a checksum of every sample.
#
# Keep the BIDS directory tree intact when moving the cache. For a larger
# explicit query, repeat the read check for each returned recording before
# submitting an offline compute job. Use the offline how-to to separate that
# job's local loading from the acquisition stage.
