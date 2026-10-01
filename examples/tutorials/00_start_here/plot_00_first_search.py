"""Find a small cohort with the EEGDash API
========================================

Search real catalogue records before downloading EEG. We select the same
three Nakanishi2015 SSVEP participants used by the following tutorials.
This page needs internet for metadata only; signal files are not opened.

Before you start
----------------
Install EEGDash with its dependencies and run this file in Python or a
notebook with network access. Familiarity with a pandas DataFrame is useful;
no MNE, PyTorch or model-training knowledge is needed. The result is a checked
cohort query that you can pass to ``EEGDashDataset`` in tutorial 01.
"""

# %%
import matplotlib.pyplot as plt
import pandas as pd
from eegdash import EEGDash

# %%
# 1. Query an explicit cohort
# ---------------------------
# A catalogue row describes one recording file. ``subject``, ``session`` and
# ``run`` are identifiers, so keep values such as ``"0"`` as strings. The
# ``$in`` expression accepts any of the three named participants; the other
# fields must all match. This bounds the later signal acquisition before we
# open a recording.
#
# ``limit=10`` is a result cap, not a request for ten participants. The check
# verifies that this query resolves to the three requested
# recordings. If it fails, inspect the returned identifiers before changing the
# query; silently taking the first three rows could select a different cohort.
client = EEGDash()
query = {
    "dataset": "nm000118",
    "subject": {"$in": ["1", "2", "3"]},
    "session": "0",
    "run": "0",
    "task": "ssvep",
}
records = pd.DataFrame(client.find(query, limit=10))
expected = {(subject, "0", "0") for subject in ["1", "2", "3"]}
actual = list(records[["subject", "session", "run"]].itertuples(index=False, name=None))
if set(actual) != expected or len(actual) != len(expected):
    raise ValueError(
        f"Inspect query: missing={expected - set(actual)}, unexpected={set(actual) - expected}; rows={actual}"
    )
records[["dataset", "subject", "session", "run", "task"]]

# %%
# 2. Inspect the signal metadata before committing to downloads
# -------------------------------------------------------------
# Read ``sampling_frequency`` as samples per second, ``nchans`` as recorded
# channels, and ``ntimes`` as time samples per channel. Thus ``ntimes`` divided
# by ``sampling_frequency`` estimates a file's signal duration. These fields
# help plan memory and window lengths, but they do not establish signal quality.
#
# The two counts deliberately answer different questions: a participant may
# contribute multiple runs. Here the duration chart shows the acquisition
# workload rather than repeating a one-recording-per-person count.
fields = ["subject", "sampling_frequency", "nchans", "ntimes"]
records["duration_minutes"] = records["ntimes"] / records["sampling_frequency"] / 60
records[fields + ["duration_minutes"]]

# %%
print("Recording count:", len(records))
print("Participant count:", records["subject"].nunique())
# This particular subset has approximately 21.1 MB of signal files. The
# next tutorial opens only subject 1 (about 7 MB) with EEGDashDataset.
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
records.plot.bar(x="subject", y="duration_minutes", ax=ax, rot=0, legend=False)
for i, row in enumerate(records.itertuples()):
    ax.text(
        i,
        row.duration_minutes,
        f"{row.sampling_frequency:g} Hz / {row.nchans} ch",
        ha="center",
        va="bottom",
    )
ax.margins(y=0.2)
ax.set(
    xlabel="Subject", ylabel="Recording duration (min)", title="Explicit SSVEP subset"
)
plt.show()

# %%
# 3. Broaden discovery without treating a limited sample as a census
# ------------------------------------------------------------------
# An unfiltered sample is useful for discovering dataset and task names. It
# cannot estimate how common a task is across the whole catalogue: the first
# 20 rows need not be randomly sampled or balanced across datasets. Choose a
# specific dataset from discovery, then query its relevant recordings before
# summarizing a cohort.
sample = pd.DataFrame(client.find({}, limit=20))
print("First 20 catalogue records (not representative cohort counts):")
print(sample[["dataset", "subject", "task"]])
# Change the dataset/task query to explore another paradigm, then inspect
# its real participant and event metadata before choosing a decoding target.
