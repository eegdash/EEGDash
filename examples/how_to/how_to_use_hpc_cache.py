"""Place the EEGDash cache on shared or local cluster storage
==========================================================

On HPC clusters, *where* you put the EEGDash cache is often the difference
between a 30-minute and a 30-second epoch. Shared filesystems (Lustre, GPFS,
NFS) survive job restarts but throttle under metadata-heavy access; node-local
NVMe is fast but volatile; and ``$HOME`` is almost always too slow for
training. This how-to shows how to point :func:`eegdash.paths.get_default_cache_dir`
at the right tier, stage data once, and verify the cache before training.

The recipe assumes you already know how to populate the cache (see
``how_to_download_a_dataset``) and load offline (see ``how_to_work_offline``).
We follow the cluster-software best practices summarised by Cisotto and
Chicco (2024, doi:10.3389/fninf.2024.1338139): keep heavy IO on local-to-node
storage, stage in at job-start, and never read training data over the network
home.
"""

# %%
# Goal
# ----
#
# Resolve ``cache_dir`` from a SLURM environment variable, stage data from a
# shared persistent location to per-node fast scratch at job start, and verify
# the cache hit on subsequent runs without contacting S3.

# %%
# Prerequisites
# -------------
#
# - A SLURM/LSF/PBS account with one shared filesystem (e.g. ``/scratch`` or
#   ``$SCRATCH``) and one node-local fast disk (``$TMPDIR``,
#   ``/local/$SLURM_JOB_ID``, or an NVMe mount).
# - ``eegdash`` installed in the activated environment.
# - The dataset of interest already populated once on the shared filesystem
#   (head node with internet, or via ``how_to_download_a_dataset``).

# %%
# Recipe
# ------
#
# Step 1 -- Identify your storage tiers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Most schedulers expose three useful paths. ``$HOME`` is shared and slow;
# never put the cache there. ``$SCRATCH`` (or ``/scratch/$USER``) is shared
# and fast-ish but throttles under metadata-heavy reads. Per-job local scratch
# (``$TMPDIR`` on Slurm with ``--tmp``, or ``/local/$SLURM_JOB_ID``) is the
# fastest option but is wiped at job exit. Inspect them in your job script:
#
# .. code-block:: bash
#
#    # In your sbatch script
#    echo "HOME    = $HOME"
#    echo "SCRATCH = ${SCRATCH:-/scratch/$USER}"
#    echo "TMPDIR  = ${TMPDIR:-/tmp}"
#    df -h "$TMPDIR" "${SCRATCH:-/scratch/$USER}"

# %%
# Step 2 -- Set ``EEGDASH_CACHE_DIR`` to fast scratch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :func:`eegdash.paths.get_default_cache_dir` honours the ``EEGDASH_CACHE_DIR``
# environment variable first. Export it at the top of your sbatch script so
# every Python process in the job inherits the same path:
#
# .. code-block:: bash
#
#    export EEGDASH_CACHE_DIR="${TMPDIR:-/tmp}/eegdash_cache"
#    mkdir -p "$EEGDASH_CACHE_DIR"
#
# Verify from Python that the resolution works as expected.
import os
from pathlib import Path

from eegdash.paths import get_default_cache_dir

os.environ["EEGDASH_CACHE_DIR"] = str(Path.cwd() / ".eegdash_cache_local")
local_cache = get_default_cache_dir()
print(f"EEGDash will read/write under: {local_cache}")
assert local_cache == Path(os.environ["EEGDASH_CACHE_DIR"]).resolve()

# %%
# Step 3 -- Stage data from shared to node-local at job start
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The "stage-in" pattern is the workhorse of HPC IO: keep a single canonical
# copy of the dataset on shared scratch and ``rsync`` it to node-local disk
# at the start of each job. Reads during training then hit NVMe; the shared
# copy survives across jobs.
#
# .. code-block:: bash
#
#    SHARED_CACHE="${SCRATCH:-/scratch/$USER}/eegdash_cache"   # persistent
#    LOCAL_CACHE="${TMPDIR:-/tmp}/eegdash_cache"               # volatile
#
#    mkdir -p "$LOCAL_CACHE"
#    # -a archive, --info=progress2 quieter than -v on large trees
#    rsync -a --info=progress2 "$SHARED_CACHE"/ "$LOCAL_CACHE"/
#    export EEGDASH_CACHE_DIR="$LOCAL_CACHE"
#
#    # Optional: stage-out fresh artefacts back, so the next job benefits.
#    trap 'rsync -a --update "$LOCAL_CACHE"/ "$SHARED_CACHE"/' EXIT

# %%
# Step 4 -- Verify cache hit on subsequent runs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A correctly staged cache lets you instantiate the dataset with
# ``download=False`` and observe a non-zero record count without network IO.
# Use this as a smoke test at the top of your training script -- if it fails,
# stage-in did not complete and you should fail fast rather than silently
# re-downloading from S3.
print("\nSimulating a stage-in verification (no real cluster needed):")
local_cache.mkdir(parents=True, exist_ok=True)
fake_record_dir = local_cache / "ds_demo" / "sub-01"
fake_record_dir.mkdir(parents=True, exist_ok=True)
(fake_record_dir / "sub-01_task-rest_eeg.bdf").touch()

n_records = sum(1 for _ in local_cache.rglob("*_eeg.bdf"))
assert n_records >= 1, "stage-in copied 0 records; abort job before training"
print(f"  cache_dir   = {local_cache}")
print(f"  records on disk = {n_records}")
print("  --> training can proceed offline (download=False)")

# In a real script you would now do, e.g.:
#
#     ds = EEGDashDataset(cache_dir=local_cache, dataset="ds005514",
#                         task="RestingState", download=False)
#     assert len(ds.datasets) == n_records

# %%
# Common pitfalls
# ---------------
#
# - **Home directory is shared and slow.** Quotas on ``$HOME`` are tiny and
#   the filesystem is not designed for thousands of concurrent reads. Putting
#   the cache there is the most common cause of slow first-epoch IO.
# - **Node-local cache disappears between jobs.** ``$TMPDIR`` and
#   ``/local/$SLURM_JOB_ID`` are wiped at job exit. Always keep the canonical
#   copy on shared scratch and stage in fresh each job.
# - **Race conditions when multiple jobs hit one cache.** Two jobs writing
#   into the same ``EEGDASH_CACHE_DIR`` can produce truncated files. Either
#   give each job its own ``EEGDASH_CACHE_DIR`` (per-task subdirectory) or
#   pre-populate the shared cache once on a head node with internet access
#   and run all subsequent jobs with ``download=False``.
# - **Metadata-server contention on Lustre/GPFS.** Hundreds of small file
#   stats during dataloading can throttle the whole filesystem. If first-
#   epoch IO is slow but disk bandwidth is idle, the bottleneck is metadata,
#   not throughput -- move to node-local NVMe.

# %%
# See also
# --------
#
# - ``how_to_work_offline``: drives ``download=False`` once the cache exists.
# - ``how_to_run_preprocessing_on_slurm``: a full sbatch template wrapping
#   the stage-in/stage-out pattern shown here.

# %%
# References
# ----------
#
# Cisotto, G., & Chicco, D. (2024). *Ten quick tips for clinical
# electroencephalographic (EEG) data acquisition and signal processing.*
# Frontiers in Neuroinformatics, 18, 1338139.
# doi:10.3389/fninf.2024.1338139
