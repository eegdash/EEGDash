"""Which recordings belong to an EEG2025 mini release?
===================================================

Inspect the R5 challenge catalogue without reading signal payloads. The mini
list defines eligible participants; catalogue recording counts depend on task
availability. Challenge data are 100 Hz, 0.5–50 Hz filtered derivatives and
must not be confused with the original HBN OpenNeuro recordings.
"""

# %%
# Before you start
# ----------------
#
# Use an installed EEGDash environment and an internet connection to the
# catalogue. This page requests metadata only; it needs neither a GPU nor a
# signal download. ``EEGDASH_CACHE_DIR`` selects a reusable cache, defaulting to
# ``~/.eegdash_cache``. Reading a recording's ``.raw`` later is a separate step
# that acquires its signal and sidecars.
#
# The question is whether a proposed cohort contains the participants and tasks
# you expect. There is no prediction target or train/test score on this page.
# ``p_factor`` is requested to inspect available participant metadata, not to
# construct a new target or infer that every participant has a valid value.

import os
from pathlib import Path

from eegdash import EEGChallengeDataset
from eegdash.const import SUBJECT_MINI_RELEASE_MAP

# %%
# Select the challenge release
# ----------------------------
#
# ``release="R5"`` selects the challenge's release mapping; ``mini=True``
# restricts eligibility to its curated participant list. No task filter is used,
# so several tasks and runs may belong to the same person. The challenge loader
# also selects the preprocessed challenge storage location. Replacing it with an
# OpenNeuro query changes the data product even when participant IDs match.
#
# ``description_fields`` requests fields useful for checking cohort eligibility.
# The description can contain additional source metadata. Missing age, sex or
# p-factor values require an explicit exclusion policy in a later supervised
# analysis; this discovery step deliberately does not silently drop records.

dataset = EEGChallengeDataset(
    release="R5",
    mini=True,
    cache_dir=Path(
        os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")
    ).expanduser(),
    description_fields=["subject", "task", "run", "age", "sex", "p_factor"],
)
# %%
# Count participants separately from recordings
# ---------------------------------------------
#
# A row of ``description`` describes a recording, not an independent person.
# The eligibility list, unique matched subjects, and number of recording objects
# therefore answer different questions. Grouping by task exposes repeated runs:
# ``recordings`` can exceed ``participants`` without indicating duplicate data.
#
# The subset assertion checks that catalogue results respect mini eligibility.
# It does not require all eligible subjects to have every task, or freeze a live
# catalogue count into a test. Inspect the printed table before choosing a task
# for the next tutorial.

metadata = dataset.description
assert set(metadata.subject).issubset(SUBJECT_MINI_RELEASE_MAP["R5"])
print("Eligible mini participants:", len(SUBJECT_MINI_RELEASE_MAP["R5"]))
print("Matched participants:", metadata.subject.nunique())
print("Matched recordings:", len(dataset.datasets))
print(
    metadata.groupby("task").agg(
        recordings=("subject", "size"), participants=("subject", "nunique")
    )
)
print(metadata.head().to_string(index=False))

# %%
# Select participants explicitly before accessing .raw: the constructor above
# discovers metadata, whereas .raw triggers acquisition of signal payloads.
print("First recording provenance:", dataset.records[0]["bids_relpath"])

# %%
# Use the result to define a bounded cohort
# -----------------------------------------
#
# The printed ``bids_relpath`` identifies a concrete source recording. Keep
# that path, release, subject and task with any later window metadata. A model's
# sample count should come from its windows; a cohort's participant count should
# come from unique subject identifiers.
#
# Next, choose three listed subjects and a single task, add those filters to the
# constructor, and inspect their annotations through ``.raw``. Estimate the
# resulting download before widening the query. For regression, check observed
# p-factor availability before feature extraction; for reaction time, retain
# observed stimulus and response events. This metadata table alone cannot
# establish either model performance or data quality.
#
# Related data-loading example: `Braindecode BIDS Dataset Example
# <https://braindecode.org/dev/auto_examples/datasets_io/bids_dataset_example.html>`_.
