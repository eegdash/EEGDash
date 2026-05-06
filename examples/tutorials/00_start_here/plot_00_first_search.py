"""Find EEG datasets and records with EEGDash
=============================================

Before we ever load a single microvolt, we have to know *what* the public
EEGDash index contains. In this opening tutorial we ask a concrete question
about the BIDS-curated face-processing dataset ``ds002718`` (Wakeman & Henson,
2015): how many recordings, from how many subjects, at which sampling rate
does it expose, and is it a viable candidate for an eyes-open / eyes-closed
follow-up?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Build an ``EEGDash`` client and run a metadata-only query.
# - Interpret BIDS entity fields returned by ``EEGDash.find``.
# - Compare dataset-level documents against record-level documents.
# - Compute cohort statistics (subjects, sampling rates, hours).
# - Filter the catalogue down to candidate datasets matching a question.

# %% [markdown]
# ## Requirements
# - Estimated time ~1 minute on CPU; only JSON metadata (< 5 MB) is fetched.
# - Network required. Offline fallback handled for airgapped builds.
# - Prerequisites: none. The companion explanation page lives at
#   [docs/source/concepts/eegdash_objects.rst](../../docs/source/concepts/eegdash_objects.rst).

# %%
# Setup. The cache directory comes from EEGDASH_CACHE so the tutorial is
# portable across CI machines, laptops and Binder containers.
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

import eegdash
from eegdash import EEGDash

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"eegdash {eegdash.__version__}; cache_dir={cache_dir}")

# %% [markdown]
# ## Step 1: open a metadata-only client
# ``EEGDash()`` opens a REST connection to the public metadata index. No
# authentication is required for read-only access, and *no EEG bytes are
# transferred* — the client only ferries small JSON documents.
#
# **Predict**: what does ``EEGDash().find_datasets({}, limit=3)`` return —
# three full recordings, three thumbnails, or three metadata documents?
#
# **Run** the call. The try/except keeps the page rendering when the
# endpoint is unreachable (offline sphinx-gallery builds).

# %%
client = EEGDash()
try:
    datasets = client.find_datasets({}, limit=3)
    online = True
except Exception as exc:  # pragma: no cover -- offline fallback
    print(f"Metadata index unreachable ({exc}); using empty result set.")
    datasets, online = [], False
print(f"online={online}, retrieved {len(datasets)} dataset documents.")

# %% [markdown]
# **Investigate**: each ``dataset`` document is a *catalogue entry* — name,
# DOI, BIDS version, modality. It is not the raw signal. Confirm by
# looking at one document.

# %%
if datasets:
    keys = sorted(datasets[0].keys())
    print(f"dataset doc has {len(keys)} keys; first 12: {keys[:12]}")
    print(f"name : {datasets[0].get('name')!r}")
    print(f"DOI  : {datasets[0].get('dataset_doi')!r}")

# %% [markdown]
# ## Step 2: zoom in to a known dataset
# We pick the BIDS face-processing dataset ``ds002718`` (Wakeman & Henson,
# 2015, doi:10.1038/sdata.2015.1): small, well documented, textbook
# 70-channel montage. Switching from ``find_datasets`` to ``find`` moves
# us from *catalogue* documents to *record* documents — one per
# (subject, task, session, run) BIDS combination.
#
# **Run** the record query.

# %%
DATASET_ID = "ds002718"
try:
    records = client.find({"dataset": DATASET_ID}, limit=50)
except Exception as exc:  # pragma: no cover
    print(f"Query failed ({exc}); using empty record list.")
    records = []
print(f"Retrieved {len(records)} record documents for {DATASET_ID}.")

# Spec invariants: bounded by limit and every record carries BIDS keys.
assert len(records) <= 50
for r in records:
    assert {"dataset", "subject", "task"} <= set(r.keys())

# %% [markdown]
# **Investigate**: a record points to one BIDS file but does not contain
# its samples. Print one to confirm the BIDS entities, per the EEG-BIDS
# specification (Pernet et al., 2019, doi:10.1038/s41597-019-0104-8).

# %%
if records:
    sample = records[0]
    for field in (
        "dataset",
        "subject",
        "task",
        "session",
        "run",
        "sampling_frequency",
        "nchans",
        "ntimes",
    ):
        print(f"  {field:<20s}: {sample.get(field)!r}")

# %% [markdown]
# ## Step 3: shape a cohort statistic
# A research-grade question usually starts with "how many subjects, for
# how long, at what rate". We compute those over the records we have —
# still no signal download.


# %%
def cohort_stats(recs: list[dict]) -> dict:
    """Aggregate per-record metadata into a small summary dict."""
    rates = Counter(
        round(float(r["sampling_frequency"]), 1)
        for r in recs
        if r.get("sampling_frequency")
    )
    hours = (
        sum((r["ntimes"] / r["sampling_frequency"]) for r in recs if r.get("ntimes"))
        / 3600.0
    )
    return {
        "n_records": len(recs),
        "n_subjects": len({r["subject"] for r in recs}),
        "n_tasks": len({r.get("task") for r in recs}),
        "sampling_rates_Hz": dict(rates),
        "total_duration_h": round(hours, 3),
    }


print(cohort_stats(records) if records else {})

# %% [markdown]
# ## Step 4: filter by BIDS entity
# ``find`` accepts keyword filters that compose into a Mongo-style query.
# We narrow to the first three subjects as a sanity check before larger
# analyses.

# %%
if records:
    first_subjects = sorted({r["subject"] for r in records})[:3]
    subset = client.find(dataset=DATASET_ID, subject=first_subjects, limit=50)
    print(f"records for subjects {first_subjects}: {len(subset)}")
else:
    subset = []

# %% [markdown]
# ## Modify
# **Your turn**: change ``ALT_TASK`` and rerun. The face dataset has one
# task, but the catalogue also exposes resting-state and go/no-go families.

# %%
ALT_TASK = "RestingState"  # change me
try:
    alt_records = client.find(task=ALT_TASK, limit=20)
except Exception:  # pragma: no cover
    alt_records = []
print(f"records with task={ALT_TASK!r}: {len(alt_records)}")

# %% [markdown]
# ## Make
# **Mini-project**: build a query returning *at least 5 candidate datasets*
# for your own research question — pick a constraint that matters
# (datatype, minimum subject count, license) and surface the dataset ids.


# %%
def shortlist_datasets(
    min_subjects: int = 5, datatype: str = "eeg", limit: int = 30
) -> list[str]:
    """Return dataset ids with >= ``min_subjects`` for the given datatype."""
    try:
        docs = client.find_datasets({"datatypes": datatype}, limit=limit)
    except Exception:  # pragma: no cover
        docs = []
    return [
        doc.get("dataset_id")
        for doc in docs
        if (doc.get("demographics") or {}).get("subjects_count", 0) >= min_subjects
    ]


shortlist = shortlist_datasets(min_subjects=5, datatype="eeg", limit=30)
print(f"{len(shortlist)} EEG datasets pass the filter; first 10: {shortlist[:10]}")

# %% [markdown]
# ## Result
# We surveyed the EEGDash metadata index without pulling a single sample
# and printed a cohort summary above. Hedged claim, in line with Cisotto &
# Chicco (2024): the *index* answer ("how many records exist") does not
# yet tell us anything about signal quality.

# %% [markdown]
# ## Wrap-up
# We learned how to call ``find`` and ``find_datasets``, how to read
# record vs dataset documents, and how to spot a candidate dataset.
# Next: ``plot_01_first_recording.py`` downloads one record and inspects
# its raw signal, channel set and duration.

# %% [markdown]
# ## Try it yourself
# - Swap ``ds002718`` for a different OpenNeuro id and rerun Step 2.
# - Add a Mongo ``$gte`` filter on ``sampling_frequency`` (e.g. >= 250 Hz).
# - Persist the shortlist to ``cache_dir / 'shortlist.json'`` for reuse.
# - Compare ``EEGDash.count(...)`` to ``len(EEGDash.find(...))`` — when do
#   they diverge?

# %% [markdown]
# ## References
# - Pernet et al. 2019, EEG-BIDS, *Scientific Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - Wakeman & Henson 2015, *Scientific Data* 2:150001. https://doi.org/10.1038/sdata.2015.1
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Concept page: ``docs/source/concepts/eegdash_objects.rst`` (forthcoming).
