"""Measure parallel feature extraction on recorded EEG
===================================================

Compare one worker with two workers on two real Nakanishi2015 recordings
loaded through EEGDashDataset (about 14.1 MB). Check numerical equality and
report measured timings, including any slowdown from worker startup.
The nm000118 release is already processed SSVEP EEG.

Prerequisites: event-labelled Braindecode windows (tutorial 02) and the
feature table from tutorial 40. This recipe reconstructs its own windows,
so no saved CSV is required. EEGDash and its feature-extraction dependencies
must be installed in the worker environment. The two source participants
are 1 and 2, session 0, run 0, task ssvep; use EEGDASH_CACHE_DIR to reuse them.

"""

# %%
# 1. Load a bounded real cohort and window its events
# ---------------------------------------------------
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash import EEGDashDataset
from eegdash.features import extract_features, signal_variance

cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="nm000118",
    subject=["1", "2"],
    session="0",
    run="0",
    task="ssvep",
    n_jobs=1,
)
assert len(dataset.datasets) == 2
names = sorted(set(dataset.datasets[0].raw.annotations.description), key=float)
mapping = {name: i for i, name in enumerate(names)}
# %%
# The source sampling frequency is 256 Hz, so 1,024 samples represent four
# seconds. One window per annotated stimulus trial gives 360 windows with eight
# channels. Annotation strings identify the twelve attended frequencies;
# numeric class indices come from the explicit frequency-sorted mapping.
# No additional visual latency shift is applied to this processed release.
windows = create_windows_from_events(
    dataset,
    mapping=mapping,
    window_size_samples=1024,
    window_stride_samples=1024,
    on_last_window="drop",
    preload=True,
)

# %%
# 2. Compare execution time and feature values
# --------------------------------------------
# Respect scheduler allocation. A tiny feature mix can be slower in parallel;
# this measurement tells you whether more workers help your actual workload.
# Variance reduces the time axis of each (8 channels, 1,024 samples) window
# to eight channel features in squared volts. It does not fit a cohort-level
# transform, so changing worker count should not change feature values.
# The batch size of 64 controls how many windows extraction processes together;
# it is unrelated to minibatches for classifier training.
#
# We cap workers at two and respect SLURM_CPUS_PER_TASK. If only one CPU is
# allocated, the loop reports one configuration rather than oversubscribing it.
# More workers can increase RAM use through worker copies and in-flight batches;
# worker count is therefore a resource choice, not an accuracy parameter.
workers = max(
    1, min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
)
# %%
# Timing starts after acquisition and window creation. It covers feature
# extraction and conversion to a DataFrame, including worker startup when that
# occurs. The serial run always comes first, so this one-pass measurement is
# sensitive to filesystem and process warm-up. Numerical equality includes
# column names and row order, which must remain aligned with trial metadata.
rows = []
reference = None
for n_jobs in sorted({1, workers}):
    start = perf_counter()
    features = extract_features(
        windows, {"variance": signal_variance}, batch_size=64, n_jobs=n_jobs
    ).to_dataframe()
    elapsed = perf_counter() - start
    assert len(features) == len(windows)
    assert np.isfinite(features.to_numpy()).all()
    if reference is None:
        reference = features
    else:
        pd.testing.assert_frame_equal(reference, features)
    rows.append({"workers": n_jobs, "seconds": elapsed})
print(pd.DataFrame(rows).to_string(index=False))

# %%
# 3. Save the actual table with its window metadata
# -------------------------------------------------
metadata = windows.get_metadata().reset_index(drop=True)
table = pd.concat([metadata, reference.reset_index(drop=True)], axis=1)
output = cache_dir / "parallel_variance.csv"
table.to_csv(output, index=False)
print("Saved:", output, table.shape)

# %%
# 4. Read the measurement and choose the next experiment
# ------------------------------------------------------
# A smaller seconds value means this execution finished faster on this machine.
# No speedup is guaranteed: variance is cheap and worker startup can dominate.
# The saved CSV combines the serial reference features with actual window
# metadata so downstream analyses retain subject and trial identities.
#
# Before selecting a worker count for a larger feature set, repeat both
# configurations in alternating order on already cached recordings and measure
# peak memory as well as elapsed time. Keep BLAS/OpenMP threads within the
# scheduler allocation so each process does not launch a full node's threads.
# This recipe measures extraction time, not a memory/serialization benchmark.
