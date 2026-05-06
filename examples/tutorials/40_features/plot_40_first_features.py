"""Extract a first feature table from EEG windows
=================================================

Berger first reported the parieto-occipital alpha bump in 1929, and every
textbook resting-state EEG dataset still shows it. Building on the windows
we produced in plot_10 (BIDS dataset ``ds005514`` HBN resting-state, 1-40 Hz
FIR band-pass, 128 Hz sample rate), this tutorial pins down a tiny
feature-engineering question that needs no model training at all.

Can a small set of band-power features distinguish eyes-open from eyes-closed?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Build a small feature dictionary mixing time-domain and spectral features.
# - Run :func:`eegdash.features.extract_features` and read the tidy DataFrame.
# - Plot a colorblind-safe channels-by-features heatmap of one window.
# - Save the feature table to parquet for plot_42.
# - Write a custom ``@univariate_feature`` of your own.
#
# ## Requirements
# - Estimated time ~30 s on CPU. No GPU. No network.
# - Prerequisite:
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`.
#   We mimic its windows below to stay runnable offline (E3.23).
# - Concept page: :doc:`/concepts/features_vs_deep_learning`.

# %%
# Setup -- imports, seed, parametrised cache directory (E3.24).
import os
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.preprocessing import create_fixed_length_windows
import eegdash
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    signal_root_mean_square,
    signal_variance,
    spectral_bands_power,
    spectral_preprocessor,
    univariate_feature,
)

np.random.seed(42)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"eegdash {eegdash.__version__}; cache_dir={cache_dir}")

# %% [markdown]
# ## Step 1 -- Reload (or mimic) the windows from plot_10
# In production you would reload windows saved by plot_10 with
# ``load_concat_dataset``. To stay reproducible offline we synthesise two short
# recordings at 128 Hz on a 4-channel parieto-occipital montage and inject a
# 10 Hz alpha sine into eyes-closed. The non-causal FIR (firwin) band-pass
# from 1 to 40 Hz applied below writes the pass-band into MNE's info, which
# ``spectral_preprocessor`` reads back later (Cisotto & Chicco 2024 Tips 4-5,
# doi:10.7717/peerj-cs.2256).

# %%
SFREQ, CH_NAMES, WINDOW_S = 128, ["O1", "Oz", "O2", "Cz"], 2.0
WINDOW_SAMPLES = int(WINDOW_S * SFREQ)


def _make_raw(eyes_open: bool, seed: int) -> mne.io.Raw:
    n_times = SFREQ * 16
    data = np.random.default_rng(seed).standard_normal((len(CH_NAMES), n_times)) * 1e-6
    if not eyes_open:
        data += 4e-6 * np.sin(2 * np.pi * 10.0 * np.arange(n_times) / SFREQ)
    raw = mne.io.RawArray(
        data, mne.create_info(CH_NAMES, SFREQ, ch_types="eeg"), verbose="ERROR"
    )
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")  # FIR firwin band-pass
    return raw


datasets = BaseConcatDataset(
    [
        RawDataset(
            _make_raw(True, 42),
            target_name="target",
            description={"subject": "sub-01", "condition": "eyes_open", "target": 0},
        ),
        RawDataset(
            _make_raw(False, 7),
            target_name="target",
            description={"subject": "sub-02", "condition": "eyes_closed", "target": 1},
        ),
    ]
)
windows = create_fixed_length_windows(
    datasets,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=WINDOW_SAMPLES,
    window_stride_samples=WINDOW_SAMPLES,
    drop_last_window=True,
    preload=True,
)
n_windows_total = sum(len(d) for d in windows.datasets)
print(
    f"windows: n_recordings={len(windows.datasets)} | n_windows={n_windows_total} | "
    f"window={WINDOW_SAMPLES} samples ({WINDOW_S:.1f} s @ {SFREQ} Hz)"
)

# %% [markdown]
# ## Step 2 -- Pick a small feature set
# Two time-domain summaries plus the canonical EEG band powers. The spectral
# features share a single ``spectral_preprocessor`` (Welch PSD) so the FFT
# runs once per window; that dependency tree is the topic of plot_41.
#
# **Predict.** In the alpha band (8-12 Hz), will eyes-closed windows show
# *higher* or *lower* power than eyes-open? Which channel should peak first?

# %%
BANDS = {"delta": (1, 4.5), "theta": (4.5, 8), "alpha": (8, 12), "beta": (12, 30)}
spectral = FeatureExtractor(
    {"band_power": partial(spectral_bands_power, bands=BANDS)},
    preprocessor=partial(spectral_preprocessor, window_size_in_sec=1.0),
)
features_dict = {
    "var": signal_variance,
    "rms": signal_root_mean_square,
    "spec": spectral,
}
print(f"feature kinds: {list(features_dict)} | bands: {list(BANDS)}")

# %% [markdown]
# ## Step 3 -- Run extract_features
# **Run #1.** ``extract_features`` walks every recording, applies the
# preprocessor once per window, then evaluates each feature; column names are
# ``<feature>_<channel>`` so a learner can grep ``alpha_O1``.

# %%
features_ds = extract_features(windows, features_dict, batch_size=64, n_jobs=1)
feature_table = features_ds.to_dataframe(include_target=True)
n_rows, n_cols = feature_table.shape
print(f"feature table: n_rows={n_rows} | n_cols={n_cols}")

# %% [markdown]
# **Investigate.** Each row is one window; column names embed the channel
# name. The asserts below are the structural invariants from the spec.

# %%
print(feature_table.head(3))
non_meta_cols = [c for c in feature_table.columns if c != "target"]
assert n_rows == n_windows_total
assert len(non_meta_cols) >= 5  # RMS, variance, and at least three band powers
assert all(any(ch in c for ch in CH_NAMES) for c in non_meta_cols)
alpha_cols = [c for c in feature_table.columns if "alpha" in c]
print(
    "alpha mean per condition (0=open, 1=closed):",
    feature_table.groupby("target")[alpha_cols].mean().to_dict(),
)

# %% [markdown]
# ## Step 4 -- Channels-by-features heatmap
# **Run #2.** Pick the first eyes-closed window and reshape its row into a
# (channels x features) matrix. The heatmap uses ``viridis`` -- a colorblind-
# safe perceptually-uniform palette appropriate for a continuous scale
# (data-viz-design.md). Both axes are directly labelled and a source line is
# footed below the figure.

# %%
example = feature_table[feature_table["target"] == 1].iloc[0]
feat_names = ["var", "rms", "alpha", "beta", "delta", "theta"]
heat = np.empty((len(CH_NAMES), len(feat_names)))
for j, fn in enumerate(feat_names):
    cols = (
        [f"{fn}_{ch}" for ch in CH_NAMES]
        if fn in {"var", "rms"}
        else [f"spec_band_power_{fn}_{ch}" for ch in CH_NAMES]
    )
    heat[:, j] = np.log10(np.maximum(example[cols].to_numpy(dtype=float), 1e-30))
fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=120)
im = ax.imshow(heat, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(feat_names)), feat_names)
ax.set_yticks(range(len(CH_NAMES)), CH_NAMES)
ax.set_title("First eyes-closed window: log10(feature value)")
ax.set_xlabel("feature")
ax.set_ylabel("channel")
fig.colorbar(im, ax=ax, label="log10 value")
fig.text(
    0.01,
    0.01,
    f"source: ds005514 mock | n_chans={len(CH_NAMES)} | sfreq={SFREQ} Hz | "
    "filter: 1-40 Hz FIR (firwin)",
    fontsize=6,
    ha="left",
)
fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
plt.show()

# %% [markdown]
# ## Step 5 -- Save the feature table for plot_42
# Parquet keeps dtypes (float64) and is what scikit-learn / LightGBM expect.

# %%
parquet_path = cache_dir / "plot_40_features.parquet"
feature_table.to_parquet(parquet_path)
roundtrip = pd.read_parquet(parquet_path)
assert roundtrip.shape == feature_table.shape
assert dict(roundtrip.dtypes) == dict(feature_table.dtypes)
print(f"saved {parquet_path.name}; reload shape={roundtrip.shape}")

# %% [markdown]
# ## Modify
# **Your turn.** Add Hjorth complexity to ``features_dict``: import
# ``signal_hjorth_complexity``, register ``"hjorth_comp": signal_hjorth_complexity``,
# and re-run Steps 3-5.
#
# ## Make
# **Mini-project.** Write a univariate feature with the
# :func:`eegdash.features.univariate_feature` decorator -- the cell below
# defines a peak-to-peak / std ratio. Plug it into ``features_dict``.


# %%
@univariate_feature
def signal_peak_to_peak_ratio(x, /):
    """Peak-to-peak amplitude divided by std along the last axis."""
    return (x.max(axis=-1) - x.min(axis=-1)) / (np.std(x, axis=-1) + 1e-12)


extended = extract_features(
    windows,
    {**features_dict, "p2p_ratio": signal_peak_to_peak_ratio},
    batch_size=64,
).to_dataframe()
print(f"extended n_cols={extended.shape[1]} (was {n_cols - 1})")

# %% [markdown]
# ## Result
# - Feature table ``n_rows`` matches the number of windows; columns embed
#   channel names so downstream code can group by feature family.
# - Eyes-closed alpha is roughly two orders of magnitude higher than
#   eyes-open here -- a sanity check, not a claim about real data.
# - Parquet at ``cache_dir / "plot_40_features.parquet"`` hands the table to
#   plot_42.

# %% [markdown]
# ## Try it yourself / Extensions
# - Swap ``BANDS`` for a clinical mu-band split and rerun Step 3.
# - Reload your real plot_10 windows from disk and rerun ``extract_features``.
# - Add ``include_metadata=["subject"]`` and check a leakage-safe split (plot_11).
# - Wire ``spectral_entropy`` plus ``spectral_normalized_preprocessor`` (plot_41).

# %% [markdown]
# ## Wrap-up and links
# - Concept: :doc:`/concepts/features_vs_deep_learning`.
# - API: :func:`eegdash.features.extract_features`, :class:`eegdash.features.FeatureExtractor`, :func:`eegdash.features.spectral_bands_power`.
# - Cisotto & Chicco 2024, *PeerJ CS* 10:e2256. https://doi.org/10.7717/peerj-cs.2256
# - Gramfort et al. 2013, *Front. Neurosci.* 7:267. https://doi.org/10.3389/fnins.2013.00267
# - Berger 1929. https://doi.org/10.1007/BF01797193
# - HBN resting-state ds005514 (Release 9). https://doi.org/10.18112/openneuro.ds005514.v1.0.0
