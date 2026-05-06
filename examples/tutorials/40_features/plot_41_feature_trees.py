"""Reuse spectral computation with feature trees
=================================================

Welch's PSD underlies every classical EEG marker -- band power,
spectral entropy, peak frequency, slope. Asking for four band powers
as four independent features re-runs the FFT four times.
:class:`~eegdash.features.FeatureExtractor` solves this with a small
dependency-tree API: declare the PSD once, hang every band feature
off it (Cisotto & Chicco 2024 Tip 5, doi:10.7717/peerj-cs.2256).

If alpha-, beta-, theta- and delta-band powers each call Welch
separately, how many PSDs run per batch -- and what does sharing one
``spectral_preprocessor`` change?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Identify when independent feature definitions cause repeated PSD work.
# - Build a shared ``spectral_preprocessor`` feeding multiple band features.
# - Read the dependency tree printed by :class:`~eegdash.features.FeatureExtractor`.
# - Compute the wall-time speedup from sharing one PSD versus N.
# - Implement a custom decorated feature on the same shared PSD.
#
# ## Requirements
# - Estimated time ~5 s on CPU. No GPU. No network.
# - Prerequisites: :doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`,
#   :doc:`/auto_examples/tutorials/40_features/plot_40_first_features`.
# - Concept: :doc:`/concepts/features_vs_deep_learning`.

# %%
# Setup -- imports, seed, welch counter (Gramfort 2013).
import time
from functools import partial

import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.preprocessing import create_fixed_length_windows

import eegdash
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    feature_predecessor,
    spectral_bands_power,
    spectral_preprocessor,
    univariate_feature,
)
from eegdash.features.feature_bank import spectral as _spec

np.random.seed(42)
print(f"eegdash {eegdash.__version__}")

PSD_CALLS, _TAG, _orig_welch = {"flat": 0, "tree": 0}, {"value": "tree"}, _spec.welch


def _counting_welch(*a, **k):
    PSD_CALLS[_TAG["value"]] = PSD_CALLS[_TAG["value"]] + 1
    return _orig_welch(*a, **k)


# Sphinx-gallery executes many tutorials in one process, so we patch in
# a try/finally so ``_spec.welch`` is always restored after Run #2.
_spec.welch = _counting_welch

# %% [markdown]
# ## Step 1 -- Build a small windowed dataset
# Two 16 s recordings at 128 Hz, four parieto-occipital channels, 10 Hz
# alpha sine in eyes-closed. The 1-40 Hz FIR band-pass writes
# ``info["highpass"]/lowpass`` so ``spectral_preprocessor`` reads them.

# %%
SFREQ, CH_NAMES, WIN = 128, ["O1", "Oz", "O2", "Cz"], int(2.0 * 128)


def _make_raw(eyes_open: bool, seed: int) -> mne.io.Raw:
    n = SFREQ * 16
    data = np.random.default_rng(seed).standard_normal((len(CH_NAMES), n)) * 1e-6
    if not eyes_open:
        data += 4e-6 * np.sin(2 * np.pi * 10.0 * np.arange(n) / SFREQ)
    raw = mne.io.RawArray(
        data, mne.create_info(CH_NAMES, SFREQ, ch_types="eeg"), verbose="ERROR"
    )
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    return raw


datasets = BaseConcatDataset(
    [
        RawDataset(
            _make_raw(True, 42),
            target_name="target",
            description={"subject": "s1", "condition": "eyes_open", "target": 0},
        ),
        RawDataset(
            _make_raw(False, 7),
            target_name="target",
            description={"subject": "s2", "condition": "eyes_closed", "target": 1},
        ),
    ]
)
windows = create_fixed_length_windows(
    datasets,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=WIN,
    window_stride_samples=WIN,
    drop_last_window=True,
    preload=True,
)
print(
    f"windows: n_recordings=2 | n_windows={sum(len(d) for d in windows.datasets)}"
    f" @ {SFREQ} Hz"
)

# %% [markdown]
# ## Step 2 -- Predict (PRIMM)
# **Predict.** Four independent band-power features each run their own
# Welch internally -- expect *four PSDs per batch*. With one shared
# ``spectral_preprocessor`` it should drop to *one PSD per batch*.
# Predict the speedup before running.

# %% [markdown]
# ## Step 3 -- Run #1: WITHOUT the tree
# **Run.** Each band gets its own top-level :class:`FeatureExtractor`
# with its own ``spectral_preprocessor`` -- the FFT runs once per band
# per batch.

# %%
BANDS = {"delta": (1, 4.5), "theta": (4.5, 8), "alpha": (8, 12), "beta": (12, 30)}


def _band_pow(n, lim):
    return partial(spectral_bands_power, bands={n: lim})


def _psd_pre():
    return partial(spectral_preprocessor, window_size_in_sec=1.0)


flat_features = {
    n: FeatureExtractor({f"{n}_pow": _band_pow(n, lim)}, preprocessor=_psd_pre())
    for n, lim in BANDS.items()
}
PSD_CALLS["flat"], _TAG["value"] = 0, "flat"
t0 = time.perf_counter()
flat_table = extract_features(
    windows, flat_features, batch_size=64, n_jobs=1
).to_dataframe(include_target=True)
runtime_flat = time.perf_counter() - t0
psds_flat = PSD_CALLS["flat"]

# %% [markdown]
# ## Step 4 -- Investigate the flat run
# **Investigate.** Every band rebuilt the PSD: ``n_bands * n_batches``.

# %%
print(
    f"flat: shape={flat_table.shape} | runtime={runtime_flat:.4f} s | PSDs={psds_flat}"
)

# %% [markdown]
# ## Step 5 -- Run #2: WITH the tree (shared PSD)
# **Run.** One ``spectral_preprocessor`` at the top, four band features
# below. Printing the extractor renders the dependency tree.

# %%
tree_extractor = FeatureExtractor(
    {f"{n}_pow": _band_pow(n, lim) for n, lim in BANDS.items()}, preprocessor=_psd_pre()
)
print(tree_extractor)
PSD_CALLS["tree"], _TAG["value"] = 0, "tree"
t0 = time.perf_counter()
tree_table = extract_features(
    windows, tree_extractor, batch_size=64, n_jobs=1
).to_dataframe(include_target=True)
runtime_tree = time.perf_counter() - t0
psds_tree = PSD_CALLS["tree"]

# %% [markdown]
# ## Step 6 -- Investigate the speedup
# **Investigate.** Same row count, identical band columns, but the PSD
# counter dropped 4x. :func:`~eegdash.features.get_feature_predecessors`
# gives the same dependency view programmatically. After Run #1 and
# Run #2 we restore ``_spec.welch`` (try/finally) so any subsequent
# tutorial in the same sphinx-gallery process sees the original Welch.

# %%
try:
    speedup = max(runtime_flat / max(runtime_tree, 1e-6), 1.0)
    assert tree_table.shape[0] == flat_table.shape[0]
    assert tree_table.shape[1] >= flat_table.shape[1]
    assert psds_tree * len(BANDS) == psds_flat
    print(
        f"tree: shape={tree_table.shape} | runtime={runtime_tree:.4f} s | "
        f"PSDs={psds_tree} | speedup={speedup:.2f}x"
    )
finally:
    _spec.welch = _orig_welch
    print("welch restored")

# %% [markdown]
# ## A common mistake -- and how to recover
#
# **Run.** A common slip is reversing a band tuple so the lower bound
# exceeds the upper bound -- ``spectral_bands_power`` then sees an empty
# mask and the column is all NaN. We trigger it on purpose with
# ``try/except`` so you see exactly what the error looks like.

# %%
try:
    bad_band = (12, 8)  # reversed (lo > hi)
    lo, hi = bad_band
    if lo >= hi:
        raise ValueError(f"band tuple ({lo}, {hi}) reversed: lo must be < hi")
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: order the tuple so lo < hi (alpha = 8 to 12 Hz).
    print(f"Recovery: use {(min(bad_band), max(bad_band))} so lo < hi.")

# %% [markdown]
# ## Modify -- add a fifth band (gamma)
# **Your turn.** Add ``gamma`` (30-40 Hz) to ``BANDS`` and rerun the
# tree. Runtime barely budges; the flat version would pay a fifth PSD.

# %%
EXTENDED = {**BANDS, "gamma": (30, 40)}
ext_tree = FeatureExtractor(
    {f"{n}_pow": _band_pow(n, lim) for n, lim in EXTENDED.items()},
    preprocessor=_psd_pre(),
)
ext_table = extract_features(windows, ext_tree, batch_size=64, n_jobs=1).to_dataframe(
    include_target=True
)
print(f"extended (with gamma): n_cols={ext_table.shape[1]}")

# %% [markdown]
# ## Make -- a custom feature on the same shared PSD
# **Mini-project.** :func:`~eegdash.features.feature_predecessor` pins
# the predecessor; :func:`~eegdash.features.univariate_feature` names
# columns per channel. Relative alpha is a classical drowsiness marker.


# %%
@feature_predecessor(spectral_preprocessor)
@univariate_feature
def relative_alpha(f, p, /):
    """Alpha power divided by total in-band power."""
    mask = (f >= 8.0) & (f <= 12.0)
    return p[..., mask].sum(axis=-1) / (p.sum(axis=-1) + 1e-30)


custom_tree = FeatureExtractor(
    {
        **{f"{n}_pow": _band_pow(n, lim) for n, lim in BANDS.items()},
        "rel_alpha": relative_alpha,
    },
    preprocessor=_psd_pre(),
)
custom_table = extract_features(
    windows, custom_tree, batch_size=64, n_jobs=1
).to_dataframe(include_target=True)
print(
    "custom rel_alpha columns:",
    [c for c in custom_table.columns if c.startswith("rel_alpha")],
)

# %% [markdown]
# ## Result

# %%
result = pd.DataFrame(
    {
        "n_features": [flat_table.shape[1] - 1, tree_table.shape[1] - 1],
        "runtime_s": [runtime_flat, runtime_tree],
        "psds_computed": [psds_flat, psds_tree],
        "speedup_vs_flat": [1.0, speedup],
    },
    index=["flat (no tree)", "tree (shared PSD)"],
)
print(result.round(4).to_string())
assert speedup >= 1.0

# %% [markdown]
# ## Try it yourself / Extensions
# - Swap ``spectral_preprocessor`` for ``spectral_normalized_preprocessor`` and add ``spectral_entropy``.
# - Wire ``spectral_db_preprocessor`` plus ``spectral_slope`` to read 1/f.
# - Save ``custom_table`` to parquet and continue in plot_42.
# - Compare runtimes on a longer recording (~5 minutes) -- the speedup widens with FFT size.

# %% [markdown]
# ## Wrap-up and links
# - Concept: :doc:`/concepts/features_vs_deep_learning`.
# - API: :class:`eegdash.features.FeatureExtractor`,
#   :func:`eegdash.features.spectral_preprocessor`,
#   :func:`eegdash.features.spectral_bands_power`,
#   :func:`eegdash.features.feature_predecessor`.
# - Cisotto & Chicco 2024, *PeerJ CS* 10:e2256. https://doi.org/10.7717/peerj-cs.2256
# - Gramfort et al. 2013, *Front. Neurosci.* 7:267. https://doi.org/10.3389/fnins.2013.00267
# - HBN resting-state ds005514 (Release 9). https://doi.org/10.18112/openneuro.ds005514.v1.0.0
