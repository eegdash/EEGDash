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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import create_windows_from_events

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from functools import partial
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    spectral_bands_power,
    spectral_preprocessor,
)
import json
from importlib.metadata import version

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The task is SSVEP frequency classification, but here the immediate goal is a
# readable feature table. The three recordings share eight posterior channels
# and a 256 Hz rate. Numeric ordering of annotation strings provides a stable
# class mapping; those labels describe the attended stimulus, not eye state.
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
if len(dataset.datasets) != len(subjects):
    raise ValueError(
        "Query did not return one recording per requested subject; inspect dataset.description"
    )
dataset.description[["subject", "session", "run"]]

# %%
raw = dataset.datasets[0].raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
for recording in dataset.datasets:
    other = recording.raw
    if (
        other.ch_names != channel_names
        or other.info["sfreq"] != sfreq
        or set(other.annotations.description) != set(mapping)
    ):
        raise ValueError(
            "Recordings must share channel order, sample rate and event vocabulary"
        )
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
if (
    len(windows) != len(metadata)
    or metadata.duplicated(["subject", "session", "run", "i_start_in_trial"]).any()
):
    raise ValueError("Window rows must have aligned, unique recording/start identities")
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
        spectral_preprocessor,
        fs=sfreq,
        nperseg=int(sfreq),
        noverlap=0,
        window="hamming",
        detrend="constant",
        scaling="density",
        average="mean",
        f_min=4,
        f_max=30,
    ),
)
feature_dataset = extract_features(
    windows, {"spectral": spectral}, batch_size=64, n_jobs=1
)
feature_table = feature_dataset.to_dataframe()
if len(feature_table) != len(metadata) or not feature_table.index.equals(
    pd.RangeIndex(len(metadata))
):
    raise ValueError(
        "Expected extraction rows in window order with a default RangeIndex; inspect before joining"
    )
# Compare retained extraction metadata too: equal lengths alone cannot detect reorderings.
identity_columns = ["subject", "session", "run", "i_start_in_trial", "target"]
pd.testing.assert_frame_equal(
    feature_dataset.get_metadata()[identity_columns].reset_index(drop=True),
    metadata[identity_columns],
)
feature_table = feature_table.reset_index(drop=True)
# Integrate PSD sums: this 1 Hz grid has bin width 1, so values are unchanged.
feature_table *= sfreq / int(sfreq)
feature_columns = list(feature_table.columns)
if not (np.isfinite(feature_table.to_numpy()).all()):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )
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
if feature_table.duplicated(["subject", "session", "run", "i_start_in_trial"]).any():
    raise ValueError(
        "Duplicate recording/start identities; inspect row alignment before modelling"
    )

# %%
# 4. Persist the exact table and column contract for tutorial 42
# --------------------------------------------------------------
# CSV makes the handoff inspectable without an additional storage engine.
# The JSON file supplies the authoritative feature-column order, mapping and
# extraction parameters. Read BIDS identifiers explicitly as strings: otherwise
# a CSV reader can reinterpret ``"0"`` or a zero-padded identifier as a number.
# The reload assertions verify the actual saved values and labels, rather than
# assuming that a successful write preserved the table contract.
# This tutorial owns both fixed names; rerunning replaces CSV and schema together.
output_path = cache_dir / "plot_40_features.csv"
cache_dir.mkdir(parents=True, exist_ok=True)
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
            "schema_version": 1,
            "power_units": "V^2",
            "channels": channel_names,
            "nperseg": int(sfreq),
            "noverlap": 0,
            "welch_window": "hamming",
            "detrend": "constant",
            "scaling": "density",
            "average": "mean",
            "nfft": int(sfreq),
            "return_onesided": True,
            "f_min": 4,
            "f_max": 30,
            "band_interval": "left closed, right open",
            "bin_width_hz": sfreq / int(sfreq),
            "stored_log_transform": None,
            "display_log": "log10(power / 1 V^2), floor 1e-30 V^2",
            "window_stride_samples": window_size,
            "on_last_window": "drop",
            "reference": "source release unchanged",
            "versions": {
                name: version(name)
                for name in ["eegdash", "braindecode", "mne", "numpy", "scipy"]
            },
        },
        indent=2,
    )
)
roundtrip = pd.read_csv(output_path, dtype={"subject": str, "session": str, "run": str})
np.testing.assert_allclose(roundtrip[feature_columns], feature_table[feature_columns])
np.testing.assert_array_equal(roundtrip["target"], y)
print("Saved:", output_path.resolve(), feature_table.shape)

# %%
# 5. Inspect the representation, not only its shape
# -------------------------------------------------
# Stored values are integrated powers in V². The display uses log10 relative
# to 1 V² with a 1e-30 V² floor, not decibels. Broad bands discard SSVEP detail.
feature_table[["subject", "target", "frequency_hz"] + feature_columns[:6]].head()

# %%
freqs, density = spectral_preprocessor(
    windows[0][0][None],
    _metadata={"info": raw.info},
    fs=sfreq,
    nperseg=int(sfreq),
    noverlap=0,
    window="hamming",
    f_min=4,
    f_max=30,
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].semilogy(freqs, density[0, 0] * 1e12)
for name, (low, high) in bands.items():
    axes[0].axvspan(low, high, alpha=0.15, label=f"{name} [{low}, {high})")
axes[0].legend()
axes[0].set(
    xlabel="Frequency (Hz)",
    ylabel="PSD (µV²/Hz)",
    title=f"First trial: {channel_names[0]}",
)
values = np.array(
    [
        [feature_table.loc[0, f"spectral_power_{band}_{channel}"] for band in bands]
        for channel in channel_names
    ]
)
image = axes[1].imshow(np.log10(np.maximum(values, 1e-30)), aspect="auto")
axes[1].set(
    xticks=range(len(bands)),
    xticklabels=list(bands),
    yticks=range(len(channel_names)),
    yticklabels=channel_names,
    title="First trial: channel × band",
)
fig.colorbar(image, ax=axes[1], label="log10(power / 1 V²)")
plt.show()

# %%
# Distributions expose participant differences and retain visible outliers.
column = f"spectral_power_alpha_{channel_names[0]}"
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.boxplot(
    [
        np.log10(
            np.maximum(
                feature_table.loc[
                    feature_table["subject"].astype(str) == subject, column
                ],
                1e-30,
            )
        )
        for subject in subjects
    ],
    tick_labels=subjects,
    showfliers=True,
)
ax.set(
    xlabel="Subject",
    ylabel="log10(alpha power / 1 V²)",
    title=f"{channel_names[0]}: inspect outliers before modelling",
)
plt.show()
# Run tutorial 42 with the same cache; regenerate both files after changing features.
