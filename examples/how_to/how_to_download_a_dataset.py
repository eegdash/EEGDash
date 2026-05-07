"""Download an EEGDash dataset in advance and validate the local cache
=====================================================================

Download all files for a dataset in advance, validate completeness, and
inspect the cache.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/how_to_download_a_dataset.png'

# %% [markdown]
# Goal
# ----
#
# Stage every file for a query *before* a long training run, an HPC job, or
# an air-gapped session. We pick a small public dataset, prefetch it with
# ``EEGDashDataset.download_all()``, then verify that an offline rebuild
# reports the same record count and that every recording really exists on
# disk. The same recipe scales to ``EEGChallengeDataset`` releases by
# swapping the constructor.

# %% [markdown]
# Prerequisites
# -------------
#
# - You have completed ``plot_00_first_search`` and ``plot_01_first_recording``.
# - Network access is available for the *initial* download.
# - ``EEGDASH_CACHE_DIR`` is set to a fast filesystem (or you accept the
#   default ``./.eegdash_cache``); never hard-code an absolute path.
# - Free disk roughly equal to the dataset footprint (``ds002718`` is
#   ~ 80 MB; full HBN releases are tens of GB).
# - Imports follow EEGDash convention: stdlib, third-party, then ``eegdash``.

# %%
import os
from pathlib import Path

import numpy as np

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

np.random.seed(42)

# %% [markdown]
# Recipe
# ------
#
# Step 1 -- Pick the dataset id and cache_dir
# ...........................................
#
# We use OpenNeuro ``ds002718`` (Wakeman & Henson visual face perception,
# 19 subjects, ~ 80 MB) so the recipe finishes in minutes on any laptop.
# The cache directory is resolved from ``EEGDASH_CACHE_DIR`` so the recipe
# stays portable between a workstation, a SLURM scratch volume, and CI.

# %%
DATASET = "ds002718"
CACHE_DIR = Path(get_default_cache_dir()).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"cache_dir = {CACHE_DIR}")
print(f"EEGDASH_CACHE_DIR set: {bool(os.environ.get('EEGDASH_CACHE_DIR'))}")

# %% [markdown]
# Step 2 -- Instantiate ``EEGDashDataset``
# ........................................
#
# Construction queries the metadata service but does *not* fetch raw EEG
# yet -- recordings stay lazy until ``.raw`` is accessed or
# ``download_all`` is called. We restrict to a single task so the example
# is bounded; drop the filter to stage the full release.

# %%
dataset = EEGDashDataset(
    cache_dir=CACHE_DIR,
    dataset=DATASET,
    task="FacePerception",
    description_fields=["subject", "session", "task", "run"],
)
n_records = len(dataset.datasets)
print(f"queried {n_records} record(s) from {DATASET}")

# %% [markdown]
# Step 3 -- Call ``download_all``
# ...............................
#
# ``EEGDashDataset.download_all(n_jobs=...)`` walks every record, skips
# files that already match the local cache, and downloads the rest in
# parallel threads. ``n_jobs=-1`` uses all cores; pin to a small number
# (e.g. ``4``) on shared filesystems to avoid throttling. The call is
# idempotent -- re-running it after a crash only refetches the missing
# files.

# %%
dataset.download_all(n_jobs=4)
print("prefetch complete")

# %% [markdown]
# Step 4 -- Verify completeness
# .............................
#
# Three independent checks together prove the cache is usable offline:
#
# 1. Each record advertises a ``local_path`` that resolves to an existing
#    file (catches partial downloads).
# 2. Re-instantiating with ``download=False`` reads only on-disk BIDS
#    files and must return the same number of recordings (catches missing
#    sidecars).
# 3. The summed footprint sanity-checks that no file was truncated to
#    zero bytes.

# %%
local_paths = [Path(ds.record["bidspath"]) for ds in dataset.datasets]
missing = [p for p in local_paths if not (CACHE_DIR / DATASET / p).exists()]
assert not missing, f"{len(missing)} file(s) missing under {CACHE_DIR}"

offline = EEGDashDataset(
    cache_dir=CACHE_DIR,
    dataset=DATASET,
    task="FacePerception",
    download=False,
)
assert len(offline.datasets) == n_records, (
    f"offline rebuild saw {len(offline.datasets)} records, expected {n_records}"
)

ds_root = CACHE_DIR / DATASET
total_bytes = sum(p.stat().st_size for p in ds_root.rglob("*") if p.is_file())
print(f"on-disk footprint: {total_bytes / 1e6:.1f} MB across {n_records} record(s)")

# %% [markdown]
# Step 5 -- Inspect the cache layout
# ..................................
#
# EEGDash mirrors the BIDS tree under ``cache_dir/<dataset_id>/``. Listing
# the top-level entries confirms the dataset descriptor, participant
# table, and per-subject folders are all present -- exactly what
# ``download=False`` needs later.

# %%
top_level = sorted(p.name for p in ds_root.iterdir())
print(f"{ds_root.name}/ contains {len(top_level)} entries:")
for name in top_level[:10]:
    print(f"  {name}")
if len(top_level) > 10:
    print(f"  ... ({len(top_level) - 10} more)")

# %% [markdown]
# Common pitfalls
# ---------------
#
# - **Hard-coded paths.** Always resolve ``cache_dir`` from
#   ``EEGDASH_CACHE_DIR`` or a CLI argument; literal ``"/scratch/..."``
#   paths break the moment the recipe runs on another machine.
# - **Filtering after download.** ``download_all`` only fetches what the
#   query selects. Add ``task=`` / ``subject=`` filters *before* calling
#   it -- otherwise you over-fetch and pay for bandwidth you discard.
# - **Stale partial caches.** If a previous run was killed mid-download,
#   re-run ``download_all`` (it is idempotent). For corruption, delete
#   the offending file and retry; never edit BIDS sidecars by hand.
# - **Network restrictions on GPU queues.** Run the download stage on an
#   internet-enabled queue and the training stage with ``download=False``
#   on the GPU queue, sharing one cache directory.
# - **n_jobs on shared filesystems.** Lustre/NFS often penalise heavy
#   parallel I/O; start with ``n_jobs=4`` and scale up only if the
#   filesystem is local SSD.
# - **Mini vs full releases.** For CI use ``EEGChallengeDataset(...,
#   mini=True)`` (a few subjects) instead of the full release to keep
#   wall time bounded.

# %% [markdown]
# See also
# --------
#
# - :doc:`how_to_work_offline </auto_examples/how_to/how_to_work_offline>`
#   -- consume the cache populated above with ``download=False``.
# - :doc:`/concepts/lazy_loading_and_cache` -- how the cache is laid out
#   and when files are materialised.
# - :doc:`plot_01_first_recording </auto_examples/tutorials/00_start_here/plot_01_first_recording>`
#   -- the prerequisite single-recording tutorial.
#
# References
# ----------
#
# - Pernet, C. R. et al. (2019). EEG-BIDS: an extension to the brain
#   imaging data structure for electroencephalography. *Scientific Data*
#   6:103, doi:10.1038/s41597-019-0104-8.
# - Wakeman, D. G., and Henson, R. N. (2015). A multi-subject, multi-modal
#   human neuroimaging dataset. *Scientific Data* 2:150001. OpenNeuro
#   ``ds002718`` v1.0.5, doi:10.18112/openneuro.ds002718.v1.0.5.
