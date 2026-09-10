"""Extract features from real EEG trials
=====================================

Compute band powers and preserve the labels and identities needed by tutorial 42.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 3 participant(s), about 21.1 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Tutorial 02 explains windows, and
tutorial 11 explains why their participant identifiers matter. This page runs
independently, then writes ``plot_40_features.csv`` and a JSON schema for
tutorial 42. Keep both files in the same cache directory.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash import EEGDashDataset
from functools import partial
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)
import json

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The task is SSVEP frequency classification, but here the immediate goal is a
# readable feature table. The three recordings share eight posterior channels
# and a 256 Hz rate. Numeric ordering of annotation strings provides a stable
# class mapping; those labels describe the attended stimulus, not eye state.
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", ".eegdash_cache"))
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
assert len(dataset.datasets) == len(subjects)
print(dataset.description[["subject", "session", "run"]])
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
assert len(mapping) == 12
for recording in dataset.datasets:
    assert recording.raw.ch_names == channel_names
    assert recording.raw.info["sfreq"] == sfreq
    assert set(recording.raw.annotations.description) == set(mapping)
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# 2. Window the observed trials
# -----------------------------
# Four seconds gives enough samples for repeated one-second spectral segments.
# Equal size and stride, together with dropping the final remainder, retain
# one window per annotated trial. Thus feature-table rows can be traced back
# to a recording and start sample instead of becoming anonymous vectors.
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
assert len(windows) == len(metadata)
assert set(y) == set(mapping.values())
assert not metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
print(pd.crosstab(metadata["subject"], y))

# %%
# 3. Compute named band powers with one shared Welch spectrum
# -----------------------------------------------------------
# ``FeatureExtractor`` first runs ``spectral_preprocessor`` and passes its
# frequency grid and power values to the band-power function. ``partial`` fixes
# the chosen sampling rate and frequency range without defining a wrapper.
# ``nperseg=int(sfreq)`` uses one-second Welch segments, giving a 1 Hz grid.
# The function sums spectral values within each named band. Keep that grid
# fixed when comparing magnitudes across tables.
#
# Three bands times eight channels produce 24 feature columns per trial.
# ``batch_size=64`` controls extraction memory, not trial length or model
# training. Broad bands summarize power but discard the fine frequency detail
# that distinguishes nearby SSVEP classes; a weak downstream decoder is
# therefore possible even when the extraction is correct.
#
# The numerical feature columns are captured before adding metadata. Subject,
# session, run, start sample and target must accompany each row for later
# grouped evaluation; ``frequency_hz`` makes the target meaning human-readable.
# It must never be included as a predictor of that same target.
bands = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
spectral = FeatureExtractor(
    {"power": partial(spectral_bands_power, bands=bands)},
    preprocessor=partial(
        spectral_preprocessor, fs=sfreq, nperseg=int(sfreq), f_min=4, f_max=30
    ),
)
feature_dataset = extract_features(
    windows, {"spectral": spectral}, batch_size=64, n_jobs=1
)
feature_table = feature_dataset.to_dataframe()
feature_columns = list(feature_table.columns)
assert len(feature_columns) == len(bands) * len(channel_names)
assert np.isfinite(feature_table.to_numpy()).all()
# Window metadata carries participant and trial identity. Join explicitly:
# feature extraction's default DataFrame contains only the feature values.
feature_table = pd.concat(
    [
        metadata[["subject", "session", "run", "i_start_in_trial", "target"]],
        feature_table,
    ],
    axis=1,
)
feature_table["frequency_hz"] = [float(class_names[target]) for target in y]
assert not feature_table.duplicated(
    ["subject", "session", "run", "i_start_in_trial"]
).any()

# %%
# 4. Persist the exact table and column contract for tutorial 42
# --------------------------------------------------------------
# CSV makes the handoff inspectable without an additional storage engine.
# The JSON file supplies the authoritative feature-column order, mapping and
# extraction parameters. Read BIDS identifiers explicitly as strings: otherwise
# a CSV reader can reinterpret ``"0"`` or a zero-padded identifier as a number.
# The reload assertions verify the actual saved values and labels, rather than
# assuming that a successful write preserved the table contract.
output_path = cache_dir / "plot_40_features.csv"
feature_table.to_csv(output_path, index=False)
(output_path.with_suffix(".json")).write_text(
    json.dumps(
        {
            "dataset": "nm000118",
            "subjects": subjects,
            "session": "0",
            "run": "0",
            "task": "ssvep",
            "mapping": mapping,
            "feature_columns": feature_columns,
            "sfreq": sfreq,
            "window_samples": window_size,
            "bands": bands,
        },
        indent=2,
    )
)
roundtrip = pd.read_csv(output_path, dtype={"subject": str, "session": str, "run": str})
np.testing.assert_allclose(roundtrip[feature_columns], feature_table[feature_columns])
np.testing.assert_array_equal(roundtrip["target"], y)
print("Saved:", output_path.resolve(), feature_table.shape)

# %%
# 5. Inspect measured feature distributions
# -----------------------------------------
# The saved features remain linear power values. Only this plot takes log10,
# so a one-unit vertical difference represents a factor of ten in power.
# Boxplots summarize the trial distribution of each band/channel column;
# ``showfliers=False`` hides markers outside the whiskers but does not remove
# those trials from the table. Very low or constant columns deserve signal
# inspection before interpretation.
#
# Run tutorial 42 with the same cache to fit directly from these files. If you
# add a feature family, regenerate both CSV and JSON so the next page uses the
# new feature names and never accidentally trains on metadata.
log_power = np.log10(np.maximum(feature_table[feature_columns], 1e-30))
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.boxplot(log_power.to_numpy(), tick_labels=feature_columns, showfliers=False)
ax.tick_params(axis="x", rotation=90)
ax.set(ylabel="log10 band power", title="Recorded SSVEP trials: channel-wise features")
plt.show()
