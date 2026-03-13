# %% [markdown]
""".. _tutorial-api:

EEGDash API Tutorial
====================

This tutorial demonstrates how to use the *EEGDash* API to query and explore
EEG recording metadata without downloading any data files.

1. **Initializing EEGDash**: Create an :class:`~eegdash.EEGDash` client to
   connect to the metadata database.

2. **Finding Records**: Use :meth:`~eegdash.EEGDash.find` to retrieve recording
   metadata for a specific dataset.

3. **Exploring Record Keys**: Inspect the fields available in each record
   (e.g., ``subject``, ``task``, ``sampling_frequency``, ``ntimes``).

4. **Filtering Records**: Narrow down results by applying additional query
   filters such as task, subject, or session.

5. **Basic Statistics**: Compute summary statistics such as the number of
   subjects, recordings, and the total duration of a dataset.
"""

# %% [markdown]
# ## Initializing EEGDash
#
# Creating an :class:`~eegdash.EEGDash` instance opens a connection to the
# EEGDash metadata database. No credentials are required for read-only access.

# %%
from eegdash import EEGDash

eegdash = EEGDash()

# %% [markdown]
# ## Finding Records
#
# Use :meth:`~eegdash.EEGDash.find` to retrieve metadata records for all
# recordings in a dataset. The method accepts a MongoDB-style query dictionary.
# Only metadata is transferred at this stage — no EEG data is downloaded.

# %%
DATASET_ID = "ds003039"

records = eegdash.find({"dataset": DATASET_ID})
print(f"Found {len(records)} records for dataset {DATASET_ID}.")

# %% [markdown]
# ## Exploring Record Keys
#
# Each record is a dictionary containing metadata fields such as the subject
# identifier, task name, sampling frequency, and number of time points.
# Printing the keys of the first record gives an overview of available fields.

# %%
if records:
    print("Keys available in a record:")
    for key in sorted(records[0].keys()):
        print(f"  {key}: {records[0][key]!r}")

# %% [markdown]
# ## Filtering Records
#
# :meth:`~eegdash.EEGDash.find` supports a rich set of query operators.
# You can pass keyword arguments as a shorthand for simple equality filters,
# or combine a query dictionary with keyword filters.
#
# The examples below show how to select recordings by task or by a list of
# subject identifiers.

# %%
# Filter by task using keyword argument
task_records = eegdash.find({"dataset": DATASET_ID}, task="rest")
print(f"Records with task='rest': {len(task_records)}")

# Filter using the $in operator to select specific subjects
subjects_of_interest = [r["subject"] for r in records[:3]]
subject_records = eegdash.find(
    {"dataset": DATASET_ID, "subject": {"$in": subjects_of_interest}}
)
print(f"Records for subjects {subjects_of_interest}: {len(subject_records)}")

# %% [markdown]
# ## Computing Dataset Statistics
#
# Because each record contains ``ntimes`` (number of samples) and
# ``sampling_frequency`` (Hz), it is straightforward to compute the duration of
# every recording and derive summary statistics for the whole dataset.

# %%
durations = [r["ntimes"] / r["sampling_frequency"] for r in records]
subjects = set(r["subject"] for r in records)

print(
    f"{len(subjects)} subjects. "
    f"{len(records)} recordings. "
    f"{sum(durations) / 3600:.2f} hours."
)
