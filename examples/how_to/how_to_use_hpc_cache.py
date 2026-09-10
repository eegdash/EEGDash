"""Stage a real EEG recording onto job-local storage
=================================================

Use a small EEGDash recording to exercise the same stage-in workflow used
on a cluster: persistent download, copy to local storage, then offline read.
The demonstration downloads about 5.6 MB and copies actual BIDS files.

Prerequisites: a writable persistent cache, enough local scratch space,
and EEGDash installed on the compute node. No Slurm allocation is needed
to run the demonstration; a workstation temporary directory provides the
same lifetime for the copied files. The recorded task is left/right imagery
from `nm000135 <https://nemar.org/dataset/nm000135>`_, subject 1,
session 0train, run 0. This page stages inputs, not trained model checkpoints.

"""

# %%
# 1. Populate persistent storage on a network-enabled node
# --------------------------------------------------------
import os
from pathlib import Path

import numpy as np

from eegdash import EEGDashDataset

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")

import shutil
import tempfile

# %%
# Download before entering the temporary-directory context so acquisition
# failure cannot be mistaken for a compute-node problem. The first 250 samples
# at 250 Hz form a one-second reference in volts, with one row per channel.
# The source remains on persistent storage after the local copy is removed.
persistent = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
persistent.download_all(n_jobs=1)
reference = persistent.datasets[0].raw.get_data(start=0, stop=250)

# %%
# 2. Stage into a private directory on job-local storage
# ------------------------------------------------------
# SLURM_TMPDIR identifies local scratch on some clusters. TemporaryDirectory
# makes this small demonstration runnable on a workstation as well; its
# lifecycle represents the job. The persistent source remains available.
# The dataset directory is copied with its relative BIDS structure, preserving
# channels and events beside the signals. This small example copies the whole
# cached nm000135 subtree: if earlier runs cached additional sessions, those
# files are copied too. Use a dedicated staging cache or select the required
# BIDS dependencies explicitly when that subtree no longer fits scratch.
#
# A private temporary directory avoids two jobs overwriting the same staged
# files. It does not benchmark filesystem throughput or synchronize concurrent
# downloads into the persistent source; complete acquisition first.
scratch = os.environ.get("SLURM_TMPDIR")
with tempfile.TemporaryDirectory(prefix="eegdash-stage-", dir=scratch) as job_dir:
    local_cache = Path(job_dir)
    shutil.copytree(cache_dir / "nm000135", local_cache / "nm000135")
    staged = EEGDashDataset(cache_dir=local_cache, **query, download=False, n_jobs=1)
    assert len(staged.datasets) == 1
    raw = staged.datasets[0].raw
    np.testing.assert_array_equal(reference, raw.get_data(start=0, stop=250))
    print("Staged recording:", raw)
    print("Local cache:", local_cache)
    # Execute the training/feature extraction step here, while local files exist.

# %%
# 3. Apply the same pattern in a batch job
# ----------------------------------------
# Pre-stage only the selected dataset before workers start. Point each worker's
# EEGDASH_CACHE_DIR at the staged root and pass download=False. Save derived
# results back to persistent storage before the scheduler removes job scratch.
# Storage performance varies by cluster; measure it instead of assuming a
# particular speedup from the path name.

# 4. Check the storage boundary before scaling up
# -----------------------------------------------
# The equality assertion verifies that offline reopening of the staged copy
# returns the same first-second samples. The displayed local path is valid only
# inside the with block. Put feature extraction or training there and copy its
# outputs to persistent storage before leaving the block, even if the shell
# job continues afterward.
#
# For a larger job, measure stage-in time separately from model time, include
# both source and destination space in your allocation, and record the exact
# query with the outputs. The Slurm EO/EC tutorial supplies a real training
# workload after this storage boundary has been verified.
