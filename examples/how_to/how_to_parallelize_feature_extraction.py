"""Parallelize EEGDash feature extraction
==========================================

**Difficulty 2** | **Runtime: 30s** | **Compute: CPU (Multi-core)**

Goal: scale :func:`eegdash.features.extract_features` across multiple cores
on one node by tuning ``n_jobs`` and ``batch_size``, then persist the
# result so re-runs are free.
#
# Validate your result
# --------------------
# - **Wall-clock Speedup.** Using ``n_jobs=4`` should significantly reduce
#   extraction time compared to ``n_jobs=1``.
# - **Memory Usage.** Monitor your system's RAM; parallel jobs increase
#   memory pressure linearly with ``n_jobs``.
# - **Consistency Check.** The resulting feature table should be identical
#   to a single-core run (assert with ``pd.testing.assert_frame_equal``).
#
# Keywords: parallel, feature-extraction, joblib
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/how_to_parallelize_feature_extraction.png'

# %% [markdown]
# Goal
# ----
# Cut wall-clock for feature extraction on a single node, keep memory below
# the cgroup limit, and avoid recomputing the Welch PSD across job restarts.
# Tied to Cisotto and Chicco (2024,
# `doi:10.7717/peerj-cs.2256 <https://doi.org/10.7717/peerj-cs.2256>`_) Tip 6
# (reuse cached spectra) and the joblib parallelism documented in
# scikit-learn (Pedregosa et al., 2011).
#
# Learning objectives
# -------------------
# - Choose ``n_jobs`` from ``$SLURM_CPUS_PER_TASK`` instead of ``-1``.
# - Pick a ``batch_size`` that keeps each worker busy without OOMing.
# - Persist the feature table once and reload it across jobs.
# - Read a small scaling table and stop adding workers when it pays nothing.
#
# Prerequisites
# -------------
# - Completed :doc:`/auto_examples/tutorials/40_features/plot_40_first_features`.
# - Read :doc:`/auto_examples/how_to/how_to_use_hpc_cache` so you know where
#   ``EEGDASH_CACHE`` and ``EEGDASH_FEATURES_CACHE`` should point.
# - Local Python with :mod:`braindecode`, :mod:`mne`, :mod:`joblib`, :mod:`pyarrow`.

# %%
import os

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # quiet joblib workers
import time
import warnings
from functools import partial
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.preprocessing import create_fixed_length_windows

from eegdash.features import (
    FeatureExtractor,
    complexity_multiscale_entropy,
    extract_features,
    signal_variance,
    spectral_bands_power,
    spectral_preprocessor,
)

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
np.random.seed(0)
CACHE = Path(os.environ.get("EEGDASH_FEATURES_CACHE", Path.cwd() / "feat_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
N_CORES = os.cpu_count() or 1


# %% [markdown]
# Synthetic dataset (mimics plot_10 windows)
# ------------------------------------------
# 16 short 6-channel resting-state recordings at 128 Hz; half get a 10 Hz
# alpha bump so the feature table is non-degenerate.
def _make_raw(seed: int, eyes_closed: bool, secs: int = 240) -> mne.io.Raw:
    n = 128 * secs
    data = np.random.default_rng(seed).standard_normal((6, n)) * 1e-6
    if eyes_closed:
        data += 4e-6 * np.sin(2 * np.pi * 10.0 * np.arange(n) / 128)
    info = mne.create_info(["O1", "Oz", "O2", "Cz", "Pz", "POz"], 128, "eeg")
    raw = mne.io.RawArray(data, info)
    raw.filter(1.0, 40.0)
    return raw


N_REC = 16
ds = BaseConcatDataset(
    [
        RawDataset(
            _make_raw(i, bool(i % 2)),
            target_name="target",
            description={"subject": f"sub-{i:02d}", "target": int(i % 2)},
        )
        for i in range(N_REC)
    ]
)
windows = create_fixed_length_windows(
    ds,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=256,
    window_stride_samples=256,
    drop_last_window=True,
    preload=True,
)
N_WIN = sum(len(d) for d in windows.datasets)
print(f"recordings={N_REC} windows={N_WIN} cores_available={N_CORES}")

# %% [markdown]
# Feature mix: multiscale sample entropy dominates and is what makes
# parallelism pay; ``spectral_preprocessor`` is shared so the Welch PSD runs
# once per window, not once per band.
features = {
    "var": signal_variance,
    "mse": complexity_multiscale_entropy,
    "spec": FeatureExtractor(
        {
            "bp": partial(
                spectral_bands_power, bands={"alpha": (8, 12), "beta": (12, 30)}
            )
        },
        preprocessor=partial(spectral_preprocessor, window_size_in_sec=1.0),
    ),
}


# %% [markdown]
# Step 1 -- profile the single-threaded baseline
# ----------------------------------------------
# Run once with ``n_jobs=1`` and assert no rows are silently dropped.
def _run(nj: int, bs: int = 64) -> float:
    t = time.perf_counter()
    out = extract_features(windows, features, batch_size=bs, n_jobs=nj)
    df = out.to_dataframe(include_target=True)
    assert df.shape[0] == N_WIN, "rows dropped silently"
    return time.perf_counter() - t


t1 = _run(nj=1)
print(f"baseline n_jobs=1: {t1:.2f}s")

# %% [markdown]
# Step 2 -- scale with ``n_jobs``
# -------------------------------
# Read ``n_jobs`` from ``$SLURM_CPUS_PER_TASK`` (or your scheduler's
# equivalent). Never hard-code ``-1`` on a shared node.
sched = int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, N_CORES)))
sweep = sorted({1, 2, sched})
scaling = [{"n_jobs": 1, "wall_s": round(t1, 2), "speedup": 1.0}]
for nj in sweep:
    if nj == 1:
        continue
    sec = _run(nj=nj)
    scaling.append(
        {"n_jobs": nj, "wall_s": round(sec, 2), "speedup": round(t1 / sec, 2)}
    )
print(pd.DataFrame(scaling).to_string(index=False))
# On a 14-core macOS dev box: n_jobs=1 ~15.3 s, n_jobs=2 ~11.4 s (1.35x),
# n_jobs=4 ~8.2 s (1.86x). Joblib's process-spawn cost (~3 s on macOS,
# <1 s on Linux SLURM) eats part of the n_jobs=2 win on a laptop; on a
# dedicated SLURM node you typically reach >=1.5x at n_jobs=2 and the
# curve flattens once n_jobs equals the number of recordings.

# %% [markdown]
# Step 3 -- tune ``batch_size``
# -----------------------------
# ``batch_size`` controls how many windows each worker holds in memory.
# Too small and Python per-batch overhead dominates; too large and you OOM.
batch_report = [
    {"batch_size": bs, "wall_s": round(_run(sched, bs), 2)} for bs in (16, 64, 256)
]
print(pd.DataFrame(batch_report).to_string(index=False))

# %% [markdown]
# Step 4 -- persist intermediate results
# --------------------------------------
# Write the feature table to parquet once; reload on every subsequent call.
parquet = CACHE / "features.parquet"
if parquet.exists():
    t = time.perf_counter()
    df = pd.read_parquet(parquet)
    reload_s = time.perf_counter() - t
else:
    out = extract_features(windows, features, batch_size=64, n_jobs=sched)
    df = out.to_dataframe(include_target=True)
    df.to_parquet(parquet)
    reload_s = float("nan")
assert parquet.exists() and len(df) == N_WIN
print(f"persisted={parquet.name} rows={len(df)} reload_s={reload_s:.3f}")

# %% [markdown]
# Step 5 (optional) -- ``joblib.dump`` the extractor
# --------------------------------------------------
# When the pipeline contains a fitted CSP or trainable feature, pickling the
# extractor lets you reapply it to held-out data without retraining.
extractor_path = CACHE / "extractor.joblib"
joblib.dump(features, extractor_path)
reused = joblib.load(extractor_path)
print(f"checkpoint -> {extractor_path.name} (keys={list(reused)})")

# %% [markdown]
# Common pitfalls
# ---------------
# - **Oversubscribed cores.** ``n_jobs=-1`` on a shared SLURM node steals
#   cores from other jobs. Read ``$SLURM_CPUS_PER_TASK``.
# - **Large batches OOMing.** Each worker holds
#   ``batch_size * n_channels * window_samples`` floats plus spectra.
#   On a 4 GB cgroup, keep ``batch_size <= 256``.
# - **Joblib caching gotchas.** ``joblib.Memory`` keys on argument hashes;
#   non-picklable lambdas in ``features`` break caching. Use
#   :func:`functools.partial` (as above) or named module-level functions.
# - **Process spawn cost.** macOS ``loky`` spawns fresh Pythons (2-3 s
#   fixed); Linux ``fork`` is sub-second.
#
# See also
# --------
# - :doc:`/auto_examples/tutorials/40_features/plot_41_feature_trees` --
#   shared-preprocessor pipelines that amplify the speedup here.
# - :doc:`/auto_examples/how_to/how_to_use_hpc_cache` -- placing
#   ``EEGDASH_CACHE`` on local NVMe.
#
# References
# ----------
# - Cisotto, G. and Chicco, D. (2024). Ten quick tips for clinical
#   electroencephalographic (EEG) data acquisition and signal processing.
#   *PeerJ Computer Science*, 10, e2256.
#   `doi:10.7717/peerj-cs.2256 <https://doi.org/10.7717/peerj-cs.2256>`_
# - Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
#   *JMLR*, 12, 2825-2830;
#   https://scikit-learn.org/stable/computing/parallelism.html
