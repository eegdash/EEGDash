"""How do I find datasets in EEGDash?
======================================

EEGDash exposes a metadata index over hundreds of BIDS-curated EEG datasets,
served by the public REST API at https://data.eegdash.org. Before downloading
a single sample, the :class:`~eegdash.api.EEGDash` client lets us search,
filter, and summarise what is available — a full cohort survey on JSON only.

.. sphinx_gallery_thumbnail_path = '_static/thumbs/plot_00_first_search.png'
"""

# %% [markdown]
# Learning objectives
# -------------------
# - Open an :class:`~eegdash.api.EEGDash` client and discover its public methods.
# - Search records with multiple filter fields (BIDS entities + scientific metadata).
# - Convert query results into a :class:`pandas.DataFrame` for analysis.
# - Compute cohort statistics (subjects, sampling rates, modalities) from metadata only.

# %% [markdown]
# Requirements
# ------------
# - About 1 minute on CPU; metadata only (< 5 MB on the wire).
# - Network required; ``EEGDash`` consumes the public API at
#   ``https://data.eegdash.org``. No authentication is needed for read access.

# %%
# Setup. No randomness in this tutorial -- only a deterministic API survey,
# so no seed is required.
import matplotlib.pyplot as plt
import pandas as pd

import eegdash
from eegdash import EEGDash
from eegdash.viz import EEGDASH_BLUE, style_figure, use_eegdash_style

use_eegdash_style()
print(f"eegdash {eegdash.__version__}")

# %% [markdown]
# Step 1 -- Discover the EEGDash client
# -------------------------------------
# ``EEGDash()`` wraps the REST endpoint. Before reading the docs, *ask the
# object itself* what it can do.
#
# **Predict.** A read-only catalogue client should expose at least three
# verbs: count, find, get. How many of each does ``EEGDash`` actually have?
#
# **Run.** Print the public method list.

# %%
client = EEGDash()
public_methods = sorted(
    m for m in dir(client) if not m.startswith("_") and callable(getattr(client, m))
)
print(f"EEGDash() exposes {len(public_methods)} public methods:")
for m in public_methods:
    print(f"  - {m}")

# %% [markdown]
# **Investigate.** ``count``, ``exists``, ``find``, ``find_one``,
# ``find_datasets``, ``search_datasets``, ``get_dataset``, plus three
# admin verbs. The read paths -- ``find`` and ``find_datasets`` -- are the
# two we will use throughout the gallery.

# %% [markdown]
# Step 2 -- How big is the catalogue?
# -----------------------------------
# ``count`` runs server-side and is the cheapest call we can make.

# %%
n_records = client.count({})
print(f"records in the index: {n_records:,}")

# %% [markdown]
# Step 3 -- A broad scan vs. a targeted query
# -------------------------------------------
# Unfiltered ``find({})`` returns a lightweight projection -- BIDS entities
# only, no signal-shape fields. To get the full envelope (sampling rate,
# channel count, duration in samples), pass a dataset filter.
#
# **Run** both and compare the columns.

# %%
broad = pd.DataFrame(client.find({}, limit=200)).drop(columns=["_id"], errors="ignore")
focused = pd.DataFrame(
    client.find(
        {"dataset": {"$in": ["ds002718", "ds005514", "ds005863", "ds003061"]}},
        limit=200,
    )
).drop(columns=["_id"], errors="ignore")
print(f"broad   : {broad.shape[0]} rows x {broad.shape[1]} cols")
print(f"focused : {focused.shape[0]} rows x {focused.shape[1]} cols")
extra = sorted(set(focused.columns) - set(broad.columns))
print(f"extra columns when filtered: {extra}")

# %% [markdown]
# Step 4 -- What does one record actually contain?
# -------------------------------------------------
# A record is a *metadata document* -- one row per BIDS file -- not the
# samples themselves. The fields cover BIDS entities (subject, task,
# session, run), scientific metadata (sampling rate, channel count,
# duration in samples) and storage hints (relative path, datatype). Per
# the EEG-BIDS spec (Pernet et al. 2019, doi:10.1038/s41597-019-0104-8)
# every recording carries this minimal envelope.

# %%
sample = focused.iloc[0]
for f in [
    "dataset",
    "subject",
    "task",
    "session",
    "run",
    "sampling_frequency",
    "nchans",
    "ntimes",
    "datatype",
]:
    print(f"  {f:<22s}: {sample.get(f)!r}")
duration_s = sample["ntimes"] / sample["sampling_frequency"]
print(f"  duration (s)          : {duration_s:.1f}")

# %% [markdown]
# Step 5 -- Cohort analysis on the DataFrame
# ------------------------------------------
# Now we can ask the questions that matter for an ML project. ``pandas``
# carries the vocabulary -- ``value_counts``, ``groupby``, ``describe`` --
# so the EEGDash client stays minimal.

# %%
print("Records per dataset:")
print(focused["dataset"].value_counts().to_string())

print("\nSampling rates (Hz):")
print(focused["sampling_frequency"].value_counts().head(8).to_string())

print("\nTop tasks:")
print(focused["task"].value_counts().head(8).to_string())

# %% [markdown]
# Step 6 -- Visualise the cohort
# ------------------------------
# A horizontal bar of "records per dataset" reads at a glance. The
# pulse-style colour split mirrors the EEGDash Data Rail.

# %%
counts = focused["dataset"].value_counts().iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.barh(counts.index, counts.values, color=EEGDASH_BLUE)
ax.bar_label(bars, padding=4, fontsize=9, color="#102A43")
ax.set_xlabel("records (n)")
ax.set_ylabel("dataset")
ax.set_xlim(0, counts.max() * 1.15)
style_figure(
    fig,
    title="Records per dataset",
    subtitle=f"{len(focused)} records | 4 datasets queried with $in",
    source="EEGDash plot_00 | source: data.eegdash.org",
)
plt.show()

# %% [markdown]
# Step 7 -- Dataset-level documents
# ---------------------------------
# ``find_datasets`` returns the *catalogue* documents -- one per dataset --
# each carrying DOI, BIDS version, demographics, and citation counts.
# Promote them to a DataFrame and shortlist by subject count.

# %%
catalogue = pd.DataFrame(client.find_datasets({}, limit=300))
catalogue["n_subjects"] = catalogue["demographics"].apply(
    lambda d: (d or {}).get("subjects_count", 0) if isinstance(d, dict) else 0
)
shortlist = (
    catalogue.loc[catalogue["n_subjects"] >= 5, ["dataset_id", "n_subjects", "license"]]
    .sort_values("n_subjects", ascending=False)
    .head(10)
)
print("Top 10 EEG datasets with >= 5 subjects:")
print(shortlist.to_string(index=False))

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
# **Run.** Passing an unknown task name returns an empty list, not an
# error. The recovery is to surface what tasks the catalogue actually
# carries (Nederbragt et al. 2020).

# %%
unknown = client.find(task="FacePerceptionXYZ", limit=5)
print(f"records for unknown task: {len(unknown)}")
known_tasks = sorted(focused["task"].dropna().unique())[:8]
print(f"known tasks (first 8): {known_tasks}")

# %% [markdown]
# Modify
# ------
# **Your turn.** Combine filters: try ``client.find(task="RestingState",
# sampling_frequency=250, limit=20)``, then add ``subject="012"``. The
# query language is keyword-style for simple cases and Mongo-style
# (``{"$gte": 250}``) for ranges.

# %%
combined = client.find(task="RestingState", limit=20)
print(f"records for task=RestingState: {len(combined)}")

# %% [markdown]
# Make
# ----
# **Mini-project.** Build a candidate cohort: pick a sampling-rate range
# and a minimum subject count, then return a DataFrame with
# ``[dataset_id, n_subjects, sampling_rates_seen]``. We get you started.


# %%
def candidate_cohort(min_subjects: int = 5, sfreq_min: float = 200.0) -> pd.DataFrame:
    """Shortlist EEG datasets with >= ``min_subjects`` and at least one sampling rate >= ``sfreq_min``."""
    rates_per_dataset = (
        focused.groupby("dataset")["sampling_frequency"].agg(set).rename("rates")
    )
    out = catalogue.merge(
        rates_per_dataset,
        left_on="dataset_id",
        right_index=True,
        how="left",
    )
    out["max_sfreq"] = out["rates"].apply(
        lambda s: max(s) if isinstance(s, set) and s else 0.0
    )
    return (
        out.loc[
            (out["n_subjects"] >= min_subjects) & (out["max_sfreq"] >= sfreq_min),
            ["dataset_id", "n_subjects", "max_sfreq"],
        ]
        .sort_values(["n_subjects", "max_sfreq"], ascending=[False, False])
        .head(10)
    )


print(candidate_cohort(min_subjects=5, sfreq_min=200.0).to_string(index=False))

# %% [markdown]
# Result
# ------
# We surveyed the EEGDash index without pulling a single sample, surfaced
# the rich client API, ran multi-field queries, and turned the results
# into pandas DataFrames for cohort analysis.

# %% [markdown]
# Wrap-up
# -------
# Next: ``plot_01_first_recording.py`` actually downloads one record from
# the shortlist above and inspects its raw signal, channels, and duration.

# %% [markdown]
# Try it yourself
# ---------------
# - Swap ``client.count({})`` for ``client.count({"datatype": "eeg"})``.
# - Group ``df`` by ``dataset`` and report ``ntimes.sum() / sampling_frequency``
#   to get total recorded hours per dataset.
# - Persist your shortlist with ``shortlist.to_parquet("candidates.parquet")``.

# %% [markdown]
# References
# ----------
# - Pernet et al. 2019, EEG-BIDS, *Scientific Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Nederbragt et al. 2020, Ten simple rules for live coding tutorials, *PLOS Comp Bio*. https://doi.org/10.1371/journal.pcbi.1008090
