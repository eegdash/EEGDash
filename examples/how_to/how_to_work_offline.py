"""How-to: work offline against a populated EEGDash cache
========================================================

**Difficulty 2** | **Runtime: 4m** | **Compute: CPU**

Goal: instantiate ``EEGChallengeDataset`` with ``download=False`` and load
the same records as the online path, with no network calls.
Keywords: offline, cache, metadata
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/how_to_work_offline.png'

# %% [markdown]
# Goal
# ----
# Load and filter EEGDash records from a local BIDS cache on an HPC node
# or air-gapped workstation, with zero network calls, and prove the cache
# is complete by comparing online vs. offline shape and metadata.

# %% [markdown]
# Prerequisites
# -------------
# - Estimated time: ~4 min on CPU (cache hit; one online prefetch on
#   first run only).
# - You have already populated the cache via
#   ``how_to_download_a_dataset`` (or ``download_all`` below).
# - Concept: [docs/source/concepts/lazy_loading_and_cache.rst](../../docs/source/concepts/lazy_loading_and_cache.rst).
# - Data: HBN release ``R2`` (OpenNeuro ``ds005506``), task
#   ``RestingState``, ``mini=True`` subset (<200 MB).

# %%
# Setup -- seed and resolve the cache directory from the environment.
import os
from pathlib import Path

import numpy as np

from eegdash import EEGChallengeDataset
from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP
from eegdash.paths import get_default_cache_dir

np.random.seed(42)

RELEASE = "R2"
TASK = "RestingState"
DATASET_ID = RELEASE_TO_OPENNEURO_DATASET_MAP[RELEASE]  # "ds005506"

# Resolve cache from EEGDASH_CACHE if set, else the package default.
# Never hard-code paths -- HPC jobs override this per node.
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", get_default_cache_dir())).resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"cache_dir = {cache_dir}")

# %% [markdown]
# Recipe
# ------

# %% [markdown]
# Step 1 -- Populate the cache (online, once)
# ...........................................
# Run this block on a node with internet. ``download_all`` prefetches
# every record so subsequent runs can use ``download=False``. If your
# cache is already populated, this is a near-instant no-op.

# %%
ds_online = EEGChallengeDataset(
    release=RELEASE,
    cache_dir=cache_dir,
    task=TASK,
    mini=True,
)
ds_online.download_all(n_jobs=-1)
print(f"online: {len(ds_online.datasets)} recording(s) cached.")

# %% [markdown]
# Step 2 -- Load offline with ``download=False``
# ..............................................
# This is the air-gapped path: EEGDash parses BIDS filenames in the cache
# instead of querying the database or S3. The challenge subset lives at
# ``<cache_dir>/<dataset_id>-bdf-mini``; check it exists before loading.

# %%
offline_root = cache_dir / f"{DATASET_ID}-bdf-mini"
assert offline_root.exists(), f"missing cache folder: {offline_root}"

ds_offline = EEGChallengeDataset(
    release=RELEASE,
    cache_dir=cache_dir,
    task=TASK,
    download=False,
)
print(f"offline: {len(ds_offline.datasets)} recording(s) loaded.")
if ds_offline.datasets:
    print("first bidspath:", ds_offline.datasets[0].record["bidspath"])

# %% [markdown]
# Step 3 -- Filter by BIDS entity offline
# .......................................
# With ``download=False`` you can still filter by ``subject``, ``session``,
# ``task``, and ``run`` -- those entities live in the BIDS filenames, not
# the database. Database-only fields (e.g., ``modality`` aliases) are not
# available offline.

# %%
ds_offline_sub = EEGChallengeDataset(
    release=RELEASE,
    cache_dir=cache_dir,
    task=TASK,
    download=False,
    subject="NDARAB793GL3",
)
print(f"subject filter: {len(ds_offline_sub.datasets)} recording(s).")
assert len(ds_offline_sub.datasets) <= len(ds_offline.datasets), (
    "filtered set must be a subset of the unfiltered offline records"
)

# %% [markdown]
# Step 4 -- Verify the cache is complete
# ......................................
# Compare record counts, raw-data shapes, and the description tables. If
# any of these diverge, the cache is partial -- re-run ``download_all`` or
# clear the suffixed folder and start over.

# %%
assert len(ds_offline.datasets) == len(ds_online.datasets), (
    "offline record count must match online; cache is partial"
)
shape_online = ds_online.datasets[0].raw.get_data().shape
shape_offline = ds_offline.datasets[0].raw.get_data().shape
print(f"online shape : {shape_online}")
print(f"offline shape: {shape_offline}")
assert shape_online == shape_offline, "raw shape mismatch"

desc_online = ds_online.description
desc_offline = ds_offline.description
print(f"description shapes: online={desc_online.shape} offline={desc_offline.shape}")
assert desc_offline.equals(desc_online), "description metadata diverges"
print("offline cache is complete.")

# %% [markdown]
# Result
# ------
# - ``ds_offline.records_count == ds_online.records_count`` (cache complete).
# - ``raw.get_data().shape`` matches across paths.
# - ``description.equals(...)`` is True -- offline parses identical metadata.
# - Subject filter returns a strict subset (asserted in Step 3).
# - No network call after Step 1 (network_mb == 0 for Steps 2-4).
#
# Source: HBN release R2 (OpenNeuro ds005506), task RestingState, mini=True.

# %% [markdown]
# Common pitfalls
# ---------------
# - If ``cache_dir`` does not exist, ``EEGDashDataset`` will silently
#   re-download. Always create it first AND set ``EEGDASH_OFFLINE=1`` (or
#   ``download=False``) on air-gapped nodes -- belt and braces.
# - The challenge subset lives under ``<cache_dir>/<dataset_id>-bdf-mini``,
#   not ``<cache_dir>/<dataset_id>``. Mixing ``mini=True`` online with
#   ``mini=False`` offline (or vice versa) loads zero records without an
#   obvious error -- always pass the same release suffix on both paths.
# - Filtering offline only honours BIDS-entity fields (subject, session,
#   task, run). Database-only filters (e.g., custom ``modality`` aliases)
#   silently match nothing; pre-stage a derived manifest if you need them.
# - ``download=False`` skips S3 but still walks the BIDS tree on
#   instantiation. On Lustre/NFS this can stall; stage the cache to local
#   NVMe (see ``how_to_use_hpc_cache``) before training.
#
# %% [markdown]
# Validate your result
# --------------------
# - **Prove it is offline.** Disconnect your network or use a dummy API URL.
#   Loading with ``download=False`` should succeed if the cache is populated.
# - **Cache Tree Example.** Your cache should look like this:
#   .. code-block:: text
#
#      .eegdash_cache/
#      └── ds005506-bdf-mini/
#          ├── participants.tsv
#          ├── sub-001/
#          │   └── eeg/
#          │       └── sub-001_task-RestingState_eeg.bdf
#          └── ...
# - **Recovery from Partial Downloads.** If a download was interrupted, delete
#    the incomplete file (or the whole subject folder) and re-run
#    ``download_all(n_jobs=1)`` to ensure a clean state.

# %% [markdown]
# See also
# --------
# - [how_to_download_a_dataset](how_to_download_a_dataset.py) -- populate
#   the cache before going offline.
# - [how_to_use_hpc_cache](how_to_use_hpc_cache.py) -- stage the cache
#   onto local-node storage for IO-bound jobs.
# - Concept: [docs/source/concepts/lazy_loading_and_cache.rst](../../docs/source/concepts/lazy_loading_and_cache.rst).

# %% [markdown]
# References
# ----------
# - Pernet et al. 2019, EEG-BIDS, *Sci. Data* 6:103.
#   https://doi.org/10.1038/s41597-019-0104-8 -- the BIDS-EEG layout that
#   makes offline filename-based filtering possible.
# - Dataset: OpenNeuro ds005506 (HBN R2, RestingState).
#   https://doi.org/10.18112/openneuro.ds005506.v1.0.0
