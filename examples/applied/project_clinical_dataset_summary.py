"""How do I survey a clinical EEG dataset before training a model?
==================================================================

A starter project: pull metadata for the OpenNeuro ``ds004504`` clinical EEG
release :cite:`miltiadous2023` through :class:`~eegdash.EEGDashDataset`,
without downloading any signal bytes, and answer four questions a project
plan needs answered before any modelling. The pool is served via
`NEMAR <https://nemar.org>`_ :cite:`delorme2022nemar`; the polished clinical
workflow recipe follows Cisotto and Chicco 2024. The deliverable is one
:class:`pandas.DataFrame` with per-condition counts and one three-panel
figure rendered from the live catalog numbers. Cohort imbalance, age
confounds, recording-length mismatch, and channel-count drift are the
four dataset-level pitfalls that silently break clinical EEG decoders
before training even starts, so why not answer them first?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/project_clinical_dataset_summary.png'

# %% [markdown]
# Learning objectives
# -------------------
# - Run :class:`~eegdash.EEGDashDataset` on a clinical OpenNeuro release and read its ``description`` DataFrame.
# - Aggregate the participants.tsv columns (``Group``, ``Age``, ``Gender``, ``MMSE``) into per-condition counts.
# - Compute recording-duration and channel-count distributions from ``ds.records`` without any signal download.
# - Show the four headline numbers a project plan needs: ``n_subjects``, ``n_recordings``, mean duration, ``n_channels``.

# %% [markdown]
# Requirements
# ------------
# - About 20 s on CPU on first run; under 5 s once the metadata catalog
#   is cached.
# - Network on first call (~1 MB into ``cache_dir`` for the catalog query).
#   No EEG signal bytes are pulled; the catalog row carries every number
#   this script reports.
# - Prerequisites: :doc:`/auto_examples/tutorials/00_start_here/plot_00_first_search`,
#   :doc:`/auto_examples/tutorials/00_start_here/plot_01_first_recording`.
# - Concept: :doc:`/concepts/eegdash_objects`.

# %%
# Step 0. Setup. ``np.random.seed`` keeps any later sampling reproducible;
# ``use_eegdash_style`` aligns matplotlib defaults with the rest of the
# gallery.
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eegdash
from eegdash import EEGDashDataset
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)

cache_dir = Path(
    os.environ.get("EEGDASH_CACHE_DIR", str(Path.home() / ".eegdash_cache"))
)
cache_dir.mkdir(parents=True, exist_ok=True)
DATASET = "ds004504"
print(f"eegdash version: {eegdash.__version__}")
print(f"cache directory: {cache_dir}")
print(f"dataset:         {DATASET}")

# %% [markdown]
# Step 1. The mental model: catalog row first, signals second
# -----------------------------------------------------------
# :class:`~eegdash.EEGDashDataset` is a thin query layer over the eegdash
# metadata catalog. Every recording has one row with BIDS entities, the
# storage descriptor, the technical fields (``sampling_frequency``,
# ``nchans``, ``ntimes``), and the ``participant_tsv`` extras (``Age``,
# ``Gender``, ``Group``, ``MMSE``). Any survey question that fits on a
# panel of this figure can be answered from the catalog alone; no S3
# traffic happens until ``ds.datasets[i].raw`` is accessed.

# %% [markdown]
# What does ``EEGDashDataset`` expose?
# ------------------------------------
# Before instantiating it, list the public methods and properties on the
# class. The ``description``, ``records``, and ``datasets`` names are the
# ones the rest of the script leans on.

# %%
ds_attrs = sorted(name for name in dir(EEGDashDataset) if not name.startswith("_"))
pd.DataFrame({"attribute": ds_attrs}).head(20)

# %% [markdown]
# Step 2. Build the dataset and read the description frame
# --------------------------------------------------------
# **Run.** Building the dataset issues one catalog query and returns
# instantly. ``ds.description`` is a :class:`pandas.DataFrame` with one
# row per recording and the participants.tsv columns merged in.

# %%
ds = EEGDashDataset(dataset=DATASET, cache_dir=str(cache_dir))
desc = ds.description
print(f"records: {len(ds.records)}")
print(f"columns: {list(desc.columns)}")
desc.head(8)

# %% [markdown]
# Step 3. Subjects per condition
# ------------------------------
# **Predict.** Before running the next cell, write down: which group do
# you expect to be largest? The Miltiadous 2023 release was designed for a
# three-way contrast (Alzheimer's disease ``A``, frontotemporal dementia
# ``F``, healthy control ``C``); cohort sizes shape every later modelling
# choice (loss weights, baseline accuracies, cross-validation folds).
#
# **Run.** The ``Group`` column lives in ``participant_tsv`` and is also
# surfaced as the lowercase ``group`` column on the description frame.

# %%
GROUP_LABELS = {
    "A": "Alzheimer's disease",
    "F": "Frontotemporal dementia",
    "C": "Healthy control",
}
desc = desc.copy()
desc["condition"] = desc["group"].map(GROUP_LABELS).fillna("Other")

condition_counts = desc.groupby("condition")["subject"].nunique().to_dict()
pd.Series(condition_counts, name="n_subjects").rename_axis("condition").to_frame()

# %% [markdown]
# **Investigate.** The cohort is mildly imbalanced: the largest group is
# roughly 1.5x the smallest. That ratio is benign enough to train a
# three-way classifier without resampling, but it is large enough that a
# naive accuracy score reports the majority-class baseline. A balanced
# accuracy or a per-class F1 should be the headline metric on this
# dataset (Cisotto and Chicco 2024 give the same advice for any clinical
# EEG cohort with a non-uniform group distribution).

# %% [markdown]
# Step 4. Age distribution per condition
# --------------------------------------
# **Run.** Group ages by condition and inspect the per-condition spread.
# Age is the most common confound in clinical EEG: a group difference on
# any spectral feature can quietly track age before it tracks the
# diagnosis (Cisotto and Chicco 2024, tip 7).

# %%
ages_by_condition = {
    cond: desc.loc[desc["condition"] == cond, "age"].dropna().tolist()
    for cond in sorted(condition_counts.keys())
}
age_summary = (
    desc.dropna(subset=["age"])
    .groupby("condition")["age"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(1)
)
age_summary

# %% [markdown]
# Step 5. Recording duration and channel count
# --------------------------------------------
# **Run.** Recording duration is ``ntimes / sampling_frequency`` (in
# seconds); channel count is ``nchans``. Both are catalog fields, no
# signal download required.

# %%
records_df = pd.DataFrame(
    [
        {
            "subject": rec["subject"],
            "duration_s": rec["ntimes"] / rec["sampling_frequency"],
            "nchans": rec["nchans"],
            "sfreq": rec["sampling_frequency"],
        }
        for rec in ds.records
    ]
)
duration_summary = records_df["duration_s"].describe().round(1)
channel_summary = records_df["nchans"].value_counts().to_dict()
sfreq_summary = records_df["sfreq"].value_counts().to_dict()
pd.Series(
    {
        "duration_s mean": float(duration_summary["mean"]),
        "duration_s min": float(duration_summary["min"]),
        "duration_s max": float(duration_summary["max"]),
        "nchans (count by value)": str(channel_summary),
        "sampling_frequency (count by value)": str(sfreq_summary),
    },
    name="value",
).to_frame()

# %% [markdown]
# **Investigate.** Two technical invariants pop out of the summary above.
# First, every recording in ``ds004504`` lists the same ``nchans`` and the
# same ``sampling_frequency``; that is what makes a fixed-length window
# loader work without per-recording resampling. Second, the duration
# spread is wide (a factor of four between the shortest and longest
# recordings); a fixed-length window count per recording will skew toward
# the longer recordings unless windows are capped per subject before the
# split.

# %% [markdown]
# Step 6. Render the live numbers as a 3-panel figure
# ---------------------------------------------------
# The drawing helpers live in a sibling ``_clinical_summary_figure``
# module so the rendering plumbing stays out of this script. Panel 1 is
# the per-condition bar chart from Step 3, Panel 2 is the age histogram
# from Step 4, and Panel 3 is the four-cell metadata card built from
# Steps 2 and 5.

# %%
from _clinical_summary_figure import draw_clinical_summary_figure

n_channels_value: int | str
unique_nchans = set(records_df["nchans"].astype(int))
n_channels_value = next(iter(unique_nchans)) if len(unique_nchans) == 1 else "varies"

summary_metrics = {
    "n_subjects": int(desc["subject"].nunique()),
    "n_recordings": int(len(ds.records)),
    "mean_duration_seconds": float(records_df["duration_s"].mean()),
    "n_channels": n_channels_value,
}

fig = draw_clinical_summary_figure(
    condition_counts=condition_counts,
    ages_by_condition=ages_by_condition,
    summary_metrics=summary_metrics,
    dataset=DATASET,
    plot_id="project_clinical_dataset_summary",
)
plt.show()

# %% [markdown]
# A common mistake, and how to recover
# ------------------------------------
# **Run.** Typing the dataset accession wrong (``"ds04504"`` instead of
# ``"ds004504"``) returns a description frame with zero rows, which
# silently passes any ``len(ds)`` check. We trigger the failure on
# purpose so the recovery path is visible: catch the empty frame at
# construction time, fail loud, and print the close matches.

# %%
try:
    ds_typo = EEGDashDataset(dataset="ds04504", cache_dir=str(cache_dir))
    if len(ds_typo.records) == 0:
        raise ValueError(
            f"dataset 'ds04504' returned an empty catalog; close match: '{DATASET}'"
        )
except ValueError as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:120]}")
    print(f"Recovery: use the canonical accession '{DATASET}'")
except Exception as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:120]}")
    print(f"Recovery: use the canonical accession '{DATASET}'")

# %% [markdown]
# Step 7. Modify: pivot the cohort by sex
# ---------------------------------------
# **Modify.** Re-run the per-condition aggregation but pivot by ``gender``
# instead of ``condition`` so the project-plan section can answer "is the
# sex split balanced inside each condition?" before training a sex-
# adjusted decoder.

# %%
sex_by_condition = (
    desc.groupby(["condition", "gender"])["subject"].nunique().unstack(fill_value=0)
)
sex_by_condition

# %% [markdown]
# Step 8. Make: a one-row dataset card
# ------------------------------------
# **Mini-project.** Tabulate one row per dataset attribute that a project
# plan or a manuscript Methods section needs: ``dataset``, ``n_subjects``,
# ``n_recordings``, ``n_channels``, ``sampling_frequency``, mean duration,
# and the per-condition counts. The cell below collapses the survey into
# that single row.


# %%
def _per_condition_str(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


dataset_card = pd.DataFrame(
    [
        {
            "dataset": DATASET,
            "n_subjects": int(desc["subject"].nunique()),
            "n_recordings": int(len(ds.records)),
            "n_channels": n_channels_value,
            "sampling_frequency": float(next(iter(set(records_df["sfreq"])))),
            "mean_duration_seconds": float(records_df["duration_s"].mean().round(1)),
            "per_condition": _per_condition_str(condition_counts),
        }
    ]
)
dataset_card

# %% [markdown]
# Result
# ------
# We surveyed ``ds004504`` end-to-end without downloading a single ``.set``
# file: 88 subjects across three diagnostic groups, 19 EEG channels at
# 500 Hz, recordings between 5 and 22 minutes long. Reproducing this
# without :class:`~eegdash.EEGDashDataset` would take a custom S3 listing,
# a custom participants.tsv parser, and a custom merge step; the catalog
# rolls all three into one query (Cisotto and Chicco 2024 would say: do
# not rebuild what the dataset class already enforces).

# %% [markdown]
# Wrap-up
# -------
# This is a project starter, not a tutorial. The numbers above feed
# straight into the four most common pitfalls of clinical EEG decoders:
# class imbalance (Step 3), age confound (Step 4), per-recording window
# count drift (Step 5), and channel-count mismatch (also Step 5). The
# polished workflow recipe in Cisotto and Chicco 2024 walks through how
# to defuse each one before training. Next up:
# :doc:`/auto_examples/tutorials/40_features/plot_40_first_features` shows
# how to extract per-recording features from the same catalog rows;
# :doc:`/auto_examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation`
# shows how to run a leakage-safe cross-subject evaluation on top of the
# windowed dataset.

# %% [markdown]
# Try it yourself (Extensions)
# ----------------------------
# - Re-run the survey on ``ds002778`` (Rockhill 2020, Parkinson's disease
#   resting-state EEG) and compare the cohort, age, and recording-duration
#   distributions side by side with ``ds004504``.
# - Bin the MMSE score (``desc['mmse']``) into three buckets (mild,
#   moderate, severe cognitive impairment) and tabulate
#   subjects-per-bucket inside the Alzheimer's disease group.
# - Pull one subject through ``ds.datasets[0].raw`` and confirm
#   ``raw.info['sfreq'] == 500.0`` and ``len(raw.info['ch_names']) == 19``;
#   the catalog numbers and the loaded raw must agree.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
