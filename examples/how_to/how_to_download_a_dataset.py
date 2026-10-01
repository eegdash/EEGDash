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
from time import perf_counter

import pandas as pd

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

cache_dir = get_default_cache_dir()
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")
dataset = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
if len(dataset.datasets) != 1:
    raise ValueError("Expected one recording; inspect the query before downloading.")
dataset.description[["subject", "session", "run", "task"]]

# %%
# 2. Stage signals and required BIDS sidecars
# -------------------------------------------
# Cropping after opening does not reduce download size. Keep events, channels
# and the complete BIDS directory structure with the signal files.
start = perf_counter()
dataset.download_all(n_jobs=1)
stage_seconds = perf_counter() - start
root = cache_dir / query["dataset"]
files = [path for path in root.rglob("*") if path.is_file()]
pd.DataFrame(
    [
        {
            "stage": "download_all completed",
            "cache": str(root),
            "files currently cached": len(files),
            "bytes currently cached": sum(path.stat().st_size for path in files),
            "stage seconds": stage_seconds,
        }
    ]
)

# %%
# These counts include pre-existing sessions: they are not download counters.
# Next check that local discovery and the signal reader can reopen the subset.
# A successful open is not an integrity checksum of the entire recording.
offline = EEGDashDataset(cache_dir=cache_dir, **query, download=False, n_jobs=1)
raw = offline.datasets[0].raw
pd.DataFrame(
    [
        {
            "status": "opened locally",
            "channels": len(raw.ch_names),
            "sampling Hz": raw.info["sfreq"],
            "duration seconds": raw.n_times / raw.info["sfreq"],
            "annotation labels": ", ".join(sorted(set(raw.annotations.description))),
        }
    ]
)

# %%
# 3. Optional reader-equivalence check
# ------------------------------------
# Set this flag when checking a transfer. Equality of the first second tests
# this reader path, not all samples or file integrity. For a larger query,
# inspect each recording; use file checksums for cross-host integrity checks.
verify_first_second = False
if verify_first_second:
    import numpy as np

    stop = round(raw.info["sfreq"])
    np.testing.assert_array_equal(
        dataset.datasets[0].raw.get_data(start=0, stop=stop),
        raw.get_data(start=0, stop=stop),
    )
    print("First-second reader equivalence: passed")
