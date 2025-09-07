"""===========================
Working Offline with EEGDash
===========================

Many HPC clusters restrict or block network access. It's common to have
dedicated queues for internet-enabled jobs that differ from GPU queues.
This tutorial shows how to use EEGDash offline once a dataset is present
on disk.
"""

# %%
from pathlib import Path
import platformdirs

from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP
from eegdash.dataset.dataset import EEGChallengeDataset


# We'll use Release R2 as an example (HBN subset). EEGChallengeDataset uses a
# suffixed cache folder for the competition data (e.g., "-bdf-mini").
release = "R2"
dataset_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
task = "RestingState"
# Choose a cache directory. This should be on a fast local filesystem.
cache_dir = Path(platformdirs.user_cache_dir("EEGDash"))
cache_dir.mkdir(parents=True, exist_ok=True)

# Bucket string used to hint cache suffix ("-bdf-mini") for offline matching
challenge_bucket = f"s3://nmdatasets/NeurIPS25/{release}_mini_L100_bdf"

# %%
# Step 1: Populate the local cache (Online)
# -----------------------------------------
# This block downloads the dataset from S3 to your local cache directory.
# Run this part on a machine with internet access. If the dataset is already
# on your disk at the specified `cache_dir`, you can comment out or skip this section.
#
# To keep this example self-contained, we prefetch the data here.

ds_online = EEGChallengeDataset(
    release=release,
    cache_dir=cache_dir,
    task=task,
    mini=True,
)
# Optional prefetch of all recordings (downloads everything to cache).
from joblib import Parallel, delayed

_ = Parallel(n_jobs=-1)(delayed(lambda d: d.raw)(d) for d in ds_online.datasets)

# After this completes, the dataset will be available offline under:
offline_root = cache_dir / f"{dataset_id}-bdf-mini"
print(f"Local dataset folder exists: {offline_root.exists()}\n{offline_root}")

# Step 2: Basic Offline Usage
# ---------------------------
# Once the data is cached locally, you can interact with it without needing
# an internet connection. The key is to instantiate your dataset object
# with the `download=False` flag. This tells EEGDash to look for data in
# the `cache_dir` instead of trying to connect to the database or S3.
ds_offline = EEGChallengeDataset(
    release=release,
    cache_dir=cache_dir,
    task=task,
    download=False,
    # Hint for cache subfolder suffixing ("-bdf-mini")
    s3_bucket=challenge_bucket,
)

print(f"Found {len(ds_offline.datasets)} recording(s) offline.")
if ds_offline.datasets:
    print("First record bidspath:", ds_offline.datasets[0].record["bidspath"])

# Step 3: Filtering Entities Offline
# ----------------------------------
# Even without a database connection, you can still filter your dataset by
# BIDS entities like subject, session, or task. When `download=False`,
# EEGDash uses the BIDS directory structure and filenames to apply these
# filters. This example shows how to load data for a specific subject
# from the local cache.

ds_offline_sub = EEGChallengeDataset(
    cache_dir=cache_dir,
    release=release,
    download=False,
    subject="NDARAB793GL3",
    # pass a bucket string to imply the "-bdf-mini" cache suffix
    s3_bucket=challenge_bucket,
)

print(f"Filtered by subject=NDARAB793GL3: {len(ds_offline_sub.datasets)} recording(s).")
if ds_offline_sub.datasets:
    keys = ("dataset", "subject", "task", "run")
    print("Records (dataset, subject, task, run):")
    for idx, base_ds in enumerate(ds_offline_sub.datasets, start=1):
        rec = base_ds.record
        summary = ", ".join(f"{k}={rec.get(k)}" for k in keys)
        print(f"  {idx:03d}: {summary}")


# %%
# Notes and troubleshooting
# -------------------------
#
# - Working offline selects recordings by parsing BIDS filenames and directory
#   structure. Some DB-only fields are unavailable; entity filters (subject,
#   session, task, run) usually suffice.
# - If you encounter issues, please open a GitHub issue so we can discuss.

# Step 4: Comparing Online vs. Offline Data
# ------------------------------------------
# As a sanity check, you can verify that the data loaded from your local
# cache is identical to the data fetched from the online sources. This section
# compares the shape of the raw data from the online and offline datasets
# to ensure they match. This is a good way to confirm your local cache is
# complete and correct.
#
# If you have network access, you can uncomment the block below to download
# and compare shapes.
#

raw_online = ds_online.datasets[0].raw
raw_offline = ds_offline.datasets[0].raw
print("online shape:", raw_online.get_data().shape)
print("offline shape:", raw_offline.get_data().shape)
print("shapes equal:", raw_online.get_data().shape == raw_offline.get_data().shape)
