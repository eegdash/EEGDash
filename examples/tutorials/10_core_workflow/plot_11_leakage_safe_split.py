"""Split real windows without subject leakage
==========================================

Compare measured random-trial and subject-disjoint assignments.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 3 participant(s), about 21.1 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Familiarity with tutorial 02's window
metadata is useful; the script does not need a saved dataset. The output is
a pair of measured assignment tables and plots, not an accuracy comparison.
Use it to choose the evaluation unit before fitting the baseline in tutorial 12.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# Three participants make the difference between row-level and group-level
# splitting visible while keeping acquisition small. These recordings share
# channel order and sampling rate, not statistical interchangeability.
cache_dir = get_default_cache_dir()
subjects = ["1", "2", "3"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="nm000118",
    subject=subjects,
    session="0",
    run="0",
    task="ssvep",
    n_jobs=1,
)
dataset.description[["subject", "session", "run"]]

# %%
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}

# %%
# 2. Window the observed trials
# -----------------------------
# There is one four-second row per annotated trial. ``subject`` identifies the
# participant, while session, run and start sample identify the recording and
# trial. Keeping these columns is essential: once only the voltage array and
# class labels remain, there is no reliable way to reconstruct a valid group
# split from array positions.
# The source event lasts 4.15 seconds; its final 0.15 seconds are unused.
window_size = int(4 * sfreq)
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    window_size_samples=window_size,
    window_stride_samples=window_size,
    on_last_window="drop",
    preload=True,
)
metadata = windows.get_metadata().reset_index(drop=True)
y = metadata["target"].to_numpy(dtype=int)
pd.crosstab(metadata["subject"], y)

# %%
# 3. Compare what each split evaluates
# ------------------------------------
# ``train_test_split(..., stratify=y)`` preserves class proportions while
# sampling rows. It has no knowledge of participant identity. In contrast,
# ``GroupShuffleSplit`` assigns complete subject groups; its ``test_size=1/3``
# refers to the fraction of groups, not a guaranteed fraction of windows. Here
# the equal-sized participant recordings happen to make those fractions agree.
#
# The fixed seed makes the assignments repeatable. It does not protect against
# leakage; grouping defines the evaluation unit. Inspect the class counts,
# since group splitting does not promise class balance.
groups = metadata["subject"].astype(str).to_numpy()
indices = np.arange(len(y))
random_train, random_test = train_test_split(
    indices, test_size=1 / 3, stratify=y, random_state=42
)
train, test = next(
    GroupShuffleSplit(n_splits=1, test_size=1 / 3, random_state=42).split(
        indices, y, groups
    )
)
print("Held-out identities:", sorted(set(groups[test])))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, title, training, testing in [
    (axes[0], "Known-participant trial prediction", random_train, random_test),
    (axes[1], "New-participant prediction", train, test),
]:
    assignments = np.full(len(y), "train", dtype=object)
    assignments[testing] = "test"
    counts = pd.crosstab(groups, assignments)
    print(
        title, "shared subjects:", sorted(set(groups[training]) & set(groups[testing]))
    )
    counts.plot.bar(stacked=True, ax=ax, rot=0, title=title)
    ax.set(xlabel="Subject", ylabel="Real trial count")
plt.show()

# %%
# If you shorten windows and create several per trial, group those windows
# by trial for within-subject evaluation. Fit scalers only after splitting.

# %%
# Read the assignment tables before fitting
# -----------------------------------------
# In the random-trial plot, a participant can have both train and test colors.
# In the grouped plot, each participant should have only one. Both plots use
# the actual assignments. This demonstrates who is shared; it
# does not measure how much sharing would inflate a particular classifier.
#
# Participant overlap is appropriate for known-person trial prediction, but not
# for claims about new people. Tutorial 12 repeats the grouping principle with
# a different explicit split (subject 3 held out), not these seeded indices. If a revised
# experiment learns filters, feature selection or normalization across examples,
# fit those operations on its training indices. To tune choices, reserve
# validation participants inside training rather than changing settings after
# looking at the test participant.
