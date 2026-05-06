"""Save and reuse prepared windows and features
=============================================

Preprocessing EEG is *expensive*. Filtering, resampling and windowing a
single subject can take seconds; doing it for every notebook restart
wastes hours over the lifetime of a project. This tutorial answers a
practical question for both artifact types — windowed signals (via
Braindecode's ``BaseConcatDataset.save`` / ``load_concat_dataset``) and
tabular features (via Apache Parquet) — and treats cached files as
first-class research outputs versioned with the code. The motivating
question: once we have spent CPU time turning raw recordings into
model-ready windows or a feature table, how do we save that work to disk
and reload it in the next session without rerunning the pipeline?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Save a Braindecode windowed dataset with ``windows.save``.
# - Load the saved artifact and verify shape and sample parity.
# - Write a per-window feature table to Parquet and reload it.
# - Verify provenance so the cache stays FAIR (Wilkinson et al. 2016).

# %% [markdown]
# ## Requirements
# - **Estimated time**: ~3 s on CPU, no GPU.
# - **Network**: none — we synthesise a tiny signal locally so the lesson
#   is reproducible offline. Every artifact is routed through
#   ``tempfile.mkdtemp`` so we never write to a host path (E3.24).
# - **Prerequisites**: ``plot_10_preprocess_and_window`` (how a windowed
#   ``BaseConcatDataset`` is built) and ``plot_40_first_features`` (how
#   a feature row maps to a window). Concept reference:
#   :doc:`/concepts/eegdash_objects`.

# %%
# Setup. ``np.random.seed(42)`` makes the synthetic signal — and every
# computed feature — byte-identical across runs (E3.21).
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import braindecode
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_fixed_length_windows
import eegdash

SEED = 42
np.random.seed(SEED)
mne.set_log_level("ERROR")
print(f"eegdash {eegdash.__version__} | braindecode {braindecode.__version__}")

# %% [markdown]
# ## Step 1: build a small windowed dataset
# We simulate one subject of resting EEG (2 channels, 4 s at 100 Hz) and
# slice it into two non-overlapping 2-second windows. A miniature dataset
# keeps runtime under a second yet exercises every code path of the real
# preprocessing pipeline.

# %%
SFREQ, WIN_S = 100, 2
signal = np.random.randn(2, 4 * SFREQ).astype("float32") * 1e-6
info = mne.create_info(["Cz", "Pz"], sfreq=SFREQ, ch_types="eeg")
recording = RawDataset(
    mne.io.RawArray(signal, info), description={"subject": "S01", "task": "rest"}
)
windows = create_fixed_length_windows(
    BaseConcatDataset([recording]),
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=WIN_S * SFREQ,
    window_stride_samples=WIN_S * SFREQ,
    drop_last_window=True,
    preload=True,
)
n_windows = len(windows)
sample_shape = windows[0][0].shape
print(f"built {n_windows} window(s) of shape {sample_shape}")

# %% [markdown]
# **Predict**: the FIF roundtrip below converts float32 samples to FIF
# and reads them back with MNE. Will the reloaded array be bit-exact, or
# will you see floating-point drift on the order of 1e-9?
#
# ## Step 2: save the windows to disk
# ``BaseConcatDataset.save(path, overwrite=True)`` writes one subdirectory
# per child dataset, each holding ``-raw.fif`` (or ``-epo.fif``) plus JSON
# sidecars for description, target name, and preprocessing kwargs. Caching
# the full bundle — not just the array — carries the metadata a downstream
# tutorial needs.
#
# **Run**: write to a fresh temporary directory (E3.24).

# %%
cache_root = Path(tempfile.mkdtemp(prefix="eegdash_save_"))
windows_path = cache_root / "windows"
windows.save(str(windows_path), overwrite=True)
saved_files = sorted(
    p.relative_to(cache_root).as_posix() for p in windows_path.rglob("*")
)
print(f"saved: {windows_path}")
print(f"artifact tree (first 6): {saved_files[:6]}")

# %% [markdown]
# ## Step 3: reload the windows in a fresh handle
# In a new kernel you would call ``load_concat_dataset(windows_path,
# preload=True)`` exactly like below. ``preload=True`` returns float32 in
# RAM — what almost every learner wants the second time around.
#
# **Run**: rehydrate the artifact and confirm we got a BaseConcatDataset.

# %%
reloaded = load_concat_dataset(windows_path, preload=True)
print(f"reload OK: type={type(reloaded).__name__}, n={len(reloaded)}")

# %% [markdown]
# **Investigate**: shape and sample-level parity. The spec asserts
# ``reloaded.shape == original.shape`` and that the metadata frame survives
# the round-trip. FIF rounds samples through 4-byte floats, so we use
# ``np.allclose`` rather than ``array_equal``.

# %%
x_orig, _, _ = windows[0]
x_re, _, _ = reloaded[0]
assert len(reloaded) == n_windows, "reloaded window count differs"
assert x_re.shape == sample_shape, "reloaded window shape differs"
assert np.allclose(x_re, x_orig, atol=1e-7), "samples drifted beyond float32 tol"
assert list(reloaded.description.columns) == list(windows.description.columns)
print(f"shapes match: original={sample_shape}, reloaded={x_re.shape}")

# %% [markdown]
# ## Step 4: save and reload a tabular feature table
# Many downstream notebooks consume a ``(n_windows, n_features)`` table
# rather than raw signals. Parquet fits that need: columnar, typed,
# compressed, readable from R, Julia, Python, and DuckDB. We compute one
# feature per channel per window (per-channel mean) and assert the round
# trip preserves dtypes — the property a feature store relies on.

# %%
features = pd.DataFrame(
    [
        {
            "Cz_mean": float(windows[i][0][0].mean()),
            "Pz_mean": float(windows[i][0][1].mean()),
            "window_idx": i,
        }
        for i in range(n_windows)
    ]
)
features_path = cache_root / "features.parquet"
features.to_parquet(features_path, index=False)
features_back = pd.read_parquet(features_path)
pd.testing.assert_frame_equal(features_back, features)
print(f"feature table dtype:\n{features.dtypes.to_string()}")
print(
    f"feature reload OK: rows={len(features_back)}, cols={list(features_back.columns)}"
)

# %% [markdown]
# ## Modify
# **Your turn**: serialise the windows as **Zarr** instead of FIF. Zarr is
# chunked, supports parallel reads, and can sit on cloud object stores
# (Wilkinson et al. 2016 — FAIR ``A2`` accessibility). Try
# ``windows.save(str(cache_root / "windows.zarr"), overwrite=True)`` in
# your own kernel and compare disk usage with ``du -sh``.

# %% [markdown]
# ## Make
# **Mini-project**: write a tiny ``cache_or_compute`` wrapper that calls
# ``compute_fn`` only on a cache miss — the smallest useful step toward the
# FAIR provenance manifest (Wilkinson et al. 2016).


# %%
def cache_or_compute(path: Path, compute_fn, *, force: bool = False):
    """Return a cached BaseConcatDataset, computing it once on cache miss."""
    if path.exists() and not force:
        return load_concat_dataset(path, preload=True)
    result = compute_fn()
    result.save(str(path), overwrite=True)
    return result


demo_path = cache_root / "demo_cache"
first = cache_or_compute(demo_path, lambda: windows)
second = cache_or_compute(demo_path, lambda: windows)
assert len(first) == len(second) == n_windows
print(f"cache_or_compute OK: hits={demo_path.exists()}")

# %% [markdown]
# ## Result
# We turned a synthetic recording into a 2-window dataset, persisted both
# signals and a derived feature table, and reloaded each artifact with
# shape, dtype, and value parity. Caching is never automatic: it costs
# disk and demands provenance. Following Wilkinson et al. 2016 (FAIR) we
# pair every cache with a version stamp (``eegdash``, ``braindecode``,
# ``SEED``) so a future reader knows which code wrote it.

# %%
shutil.rmtree(cache_root, ignore_errors=True)  # keep the cache in real projects
print("cleanup OK")

# %% [markdown]
# ## Wrap-up
# You can now stop paying the preprocessing cost on every kernel restart;
# downstream tutorials consume the saved artifact directly.

# %% [markdown]
# ## Try it yourself
# - Re-save with ``overwrite=False`` and observe the ``FileExistsError``.
# - Add a ``manifest.json`` recording package versions, ``SEED``, and DOI.
# - Compare Parquet vs CSV size on the same table (Parquet is ~10x smaller).

# %% [markdown]
# ## References
# - Schirrmeister et al. 2017, *Deep learning with convolutional neural
#   networks for EEG decoding* (Braindecode), Hum Brain Mapp.
#   https://doi.org/10.1002/hbm.23730
# - Wilkinson et al. 2016, *The FAIR Guiding Principles for scientific
#   data management and stewardship*, Sci Data 3:160018.
#   https://doi.org/10.1038/sdata.2016.18 — why we cache *and* version
#   data rather than treating intermediates as throwaway.
# - API: :func:`braindecode.datautil.load_concat_dataset`,
#   :meth:`braindecode.datasets.BaseConcatDataset.save`.
