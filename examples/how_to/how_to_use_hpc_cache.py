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

import pandas as pd
from time import perf_counter

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

cache_dir = get_default_cache_dir()
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")

import shutil
import tempfile

# %%
# Download before entering the temporary-directory context so acquisition
# failure cannot be mistaken for a compute-node problem.
# The source remains on persistent storage after the local copy is removed.
persistent = EEGDashDataset(cache_dir=cache_dir, **query, n_jobs=1)
persistent.download_all(n_jobs=1)
source = cache_dir / "nm000135"
source_files = [path for path in source.rglob("*") if path.is_file()]
source_bytes = sum(path.stat().st_size for path in source_files)
pd.DataFrame(
    [
        {
            "storage": "persistent source",
            "path": str(source),
            "files": len(source_files),
            "bytes to copy": source_bytes,
            "lifetime": "survives this job",
        }
    ]
)

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
stage_rows = []
with tempfile.TemporaryDirectory(prefix="eegdash-stage-", dir=scratch) as job_dir:
    local_cache = Path(job_dir)
    if shutil.disk_usage(local_cache).free < source_bytes:
        raise OSError("Insufficient scratch space for the complete cached subtree.")
    start = perf_counter()
    shutil.copytree(source, local_cache / "nm000135")
    copy_seconds = perf_counter() - start
    staged = EEGDashDataset(cache_dir=local_cache, **query, download=False, n_jobs=1)
    if len(staged.datasets) != 1:
        raise RuntimeError("Expected one staged recording; inspect the BIDS copy.")
    raw = staged.datasets[0].raw
    stage_rows.append(
        {
            "storage": "private scratch",
            "path": str(local_cache),
            "bytes copied": source_bytes,
            "copy seconds": copy_seconds,
            "status": "opened locally; removed on context exit",
        }
    )
    print("Staged recording:", raw)
    print("Local cache:", local_cache)
    # Execute analysis here and return derived outputs to persistent storage.

pd.DataFrame(stage_rows)

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
# Opening verifies local discovery and the reader, not file integrity. Use
# checksums or the download recipe's optional sample comparison if needed.
# The displayed local path is valid only
# inside the with block. Put feature extraction or training there and copy its
# outputs to persistent storage before leaving the block, even if the shell
# job continues afterward.
#
# For a larger job, measure stage-in time separately from model time, include
# both source and destination space in your allocation, and record the exact
# query with the outputs. The Slurm EO/EC tutorial supplies a real training
# workload after this storage boundary has been verified.
