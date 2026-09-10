"""Summarize a real clinical EEG catalogue
=======================================

Inspect participant groups, age and recording duration in
`ds004504 <https://openneuro.org/datasets/ds004504>`_ without downloading
signal files. Metadata describe the released cohort, not a diagnostic test.

Before you start
----------------
Install EEGDash with pandas and Matplotlib. Tutorial 00 introduces catalogue
records; this project requires no windowing or model-training knowledge.
Network access retrieves metadata only. You will obtain participant-level
cohort summaries and separate recording-level acquisition summaries.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eegdash import EEGDashDataset

# %%
# 1. Query metadata through the same dataset loader used for EEG
# --------------------------------------------------------------
# ``description_fields`` requests the participant fields needed by the summary.
# The description table has recording rows, so a participant with several runs
# would otherwise be counted repeatedly. Deduplicating by subject defines the
# unit for the group and age summaries. In a new longitudinal dataset, first
# check whether age or other participant attributes vary across sessions before
# choosing one row per person.
#
# No ``.raw`` property is accessed. The constructor therefore supports cohort
# inspection before committing to signal downloads. This also means numerical
# metadata checks cannot establish whether individual recordings are usable.
dataset = EEGDashDataset(
    dataset="ds004504",
    cache_dir=Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache")),
    description_fields=["subject", "age", "group", "sex"],
)
metadata = dataset.description.copy()
assert not metadata.empty
# Count participants once even when they have several recordings.
participants = metadata.drop_duplicates("subject")
assert participants["subject"].notna().all()
print("Recordings:", len(dataset.datasets), "participants:", len(participants))
print(participants[["subject", "age", "group"]].head())

# %%
# 2. Inspect the actual group and age distributions
# -------------------------------------------------
# The release's A/F/C codes are mapped explicitly to readable group names.
# Unrecognized or missing codes stop execution instead of silently entering a
# new diagnostic category. Numeric age conversion similarly exposes malformed
# values rather than producing a misleading distribution.
#
# The bars count people. The table's age ``count`` is the number with a numeric
# age, while mean, standard deviation and range describe that observed sample.
# Histograms use eight bins within each group for an overview; their edges need
# not align between groups, so consult the numeric summaries for precise
# comparisons. Group differences here may reflect recruitment and age structure,
# not a specific EEG biomarker.
labels = {
    "A": "Alzheimer's disease",
    "F": "Frontotemporal dementia",
    "C": "Healthy control",
}
assert participants["group"].notna().all()
assert set(participants["group"]).issubset(labels)
participants = participants.assign(
    condition=participants["group"].map(labels),
    age=pd.to_numeric(participants["age"], errors="raise"),
)
print(
    participants.groupby("condition")["age"].agg(["count", "mean", "std", "min", "max"])
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
participants["condition"].value_counts().plot.barh(ax=axes[0])
axes[0].set(xlabel="Participants", ylabel="Recorded group")
for condition, group in participants.groupby("condition"):
    axes[1].hist(group["age"].dropna(), bins=8, alpha=0.5, label=condition)
axes[1].set(xlabel="Age (years)", ylabel="Participants")
axes[1].legend()
plt.show()

# %%
# 3. Summarize recording duration from catalogue fields
# -----------------------------------------------------
# Duration is the catalogue sample count divided by samples per second. Unlike
# the participant summaries above, each row here is a recording: multiple runs
# should contribute multiple durations. This helps plan later preprocessing
# and identify heterogeneous acquisition settings. It does not measure usable,
# artifact-free recording time.
records = pd.DataFrame(dataset.records)
duration = pd.to_numeric(records["ntimes"]) / pd.to_numeric(
    records["sampling_frequency"]
)
assert duration.notna().all() and (duration > 0).all()
print("Recording duration (seconds):", duration.describe())
print("Channel counts:", records["nchans"].value_counts())

# %%
# Turn the summary into an analysis plan
# --------------------------------------
# Before classification, decide how to handle age imbalance, select a defined
# recording subset, and inspect its EEG through tutorial 01's Raw workflow.
# Any train/test division should keep each participant's recordings together.
# These descriptive statistics alone neither fit nor validate a diagnostic
# model.
