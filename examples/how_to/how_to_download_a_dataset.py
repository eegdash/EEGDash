"""Download a real EEG subset and inspect its cache
================================================

Stage one BNCI2014-004 motor-imagery recording (about 5.6 MB including
sidecars). CPU is sufficient; install EEGDash and choose writable persistent
storage. The source is `nm000135 <https://nemar.org/dataset/nm000135>`_.
This acquisition recipe does not train a model or require a waveform plot.
"""

# %%
# 1. Select one recording before downloading
# ------------------------------------------
# The resolver honours EEGDASH_CACHE_DIR, expands ``~`` and otherwise chooses
# a writable project cache. A query identifies recordings, not whole people.
from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

cache_dir = get_default_cache_dir()
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")
dataset = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
dataset.description[["subject", "session", "run", "task"]]

# %%
# 2. Stage signals and required BIDS sidecars
# -------------------------------------------
# Cropping after opening does not reduce download size. Keep events, channels
# and the complete BIDS directory structure with the signal files.
dataset.download_all(n_jobs=1)

# %%
# Next check that local discovery and the signal reader can reopen the subset.
# A successful open is not an integrity checksum of the entire recording.
offline = EEGDashDataset(cache_dir=cache_dir, **query, download=False, n_jobs=1)
raw = offline.datasets[0].raw
raw

# %%
# The reader confirms local loading, not file integrity. For cross-host
# transfers, keep the complete BIDS tree and verify file checksums separately.
