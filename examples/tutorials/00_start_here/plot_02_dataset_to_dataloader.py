"""From EEGDashDataset to a PyTorch DataLoader
===========================================

How do we go from a single EEG recording on disk to a tensor batch a deep
learning model can ingest? In this third Start-Here lesson we take one
subject of the BIDS face-processing dataset ``ds002718`` (Wakeman & Henson,
2015, doi:10.1038/sdata.2015.1), apply two safe preprocessors via
Braindecode (Schirrmeister et al., 2017, doi:10.1002/hbm.23730), cut the
continuous signal into fixed-length windows, and wrap the result in a
``torch.utils.data.DataLoader``. We do not train a model -- the goal is to
print one batch's shape and confirm it matches what the spec asserts: what
shape will that first batch be?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Build a single-subject ``EEGDashDataset`` and chain two safe preprocessors with :func:`braindecode.preprocessing.preprocess`.
# - Convert the preprocessed recording into fixed-length windows using :func:`braindecode.preprocessing.create_fixed_length_windows`.
# - Predict the shape of one window from ``(n_channels, window_seconds, sfreq)`` and verify it.
# - Use :class:`torch.utils.data.DataLoader` to read one batch's shape and dtype without training any model.
# - Set seeds for both ``numpy`` and ``torch`` so a teammate gets the same first batch.

# %% [markdown]
# ## Requirements
#
# - Estimated time: about 2 minutes on CPU (the slow step is the first-time cache fetch).
# - Data downloaded: ~30-60 MB of EEG from OpenNeuro, cached locally.
# - Network: required on the first run; subsequent runs are offline.
# - Prerequisites: ``plot_00_first_search.py`` and ``plot_01_first_recording.py``.
#
# Deeper conceptual material lives in
# [docs/source/concepts/eegdash_objects.rst](../../docs/source/concepts/eegdash_objects.rst);
# this page stays in the *tutorial* quadrant of Diataxis.

# %%
# Setup: imports stdlib -> third-party -> eegdash/braindecode (E1.5); RNG
# seeds (E3.21); EEGDASH_CACHE env var for portable cache dir (E3.24).
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import eegdash
from eegdash import EEGDashDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"eegdash {eegdash.__version__}, torch {torch.__version__}")
print(f"cache_dir={cache_dir}")

# %% [markdown]
# ## Step 1: instantiate one-subject EEGDashDataset
# We reuse ``ds002718`` from the previous lessons and ask EEGDash for one
# subject (``sub-002``) so the pipeline stays inside the 60-second budget.
# ``EEGDashDataset`` is a Braindecode ``BaseConcatDataset``, so it composes
# directly with ``preprocess``.

# %%
DATASET_ID = "ds002718"
SUBJECT_ID = "002"  # one subject only -- E3.23 data minimality
dataset = EEGDashDataset(cache_dir=cache_dir, dataset=DATASET_ID, subject=SUBJECT_ID)
print(f"len(dataset)={len(dataset)} recording(s) for subject {SUBJECT_ID}")
assert len(dataset) >= 1, "Expected at least one recording for the chosen subject."

# %% [markdown]
# **Predict**: before peeking at the loaded recording, write down what you
# expect ``raw.info['sfreq']`` and ``len(raw.ch_names)`` to be. The
# face-processing dataset uses a 70-channel BioSemi montage at 250 Hz.

# %%
raw = dataset.datasets[0].raw  # triggers the lazy download on the first run
n_channels_raw = len(raw.ch_names)
sfreq_raw = float(raw.info["sfreq"])
print(
    f"raw: n_channels={n_channels_raw}, sfreq={sfreq_raw} Hz, dur={raw.times[-1]:.1f}s"
)

# %% [markdown]
# **Run**: that read confirms the dataset shape. The recording sits at
# 250 Hz with EEG plus EOG channels mixed in -- we keep only EEG below.

# %% [markdown]
# ## Step 2: apply two safe preprocessors
# ``pick_types(eeg=True)`` keeps the EEG channels (dropping EOG/MISC), and
# ``resample(sfreq=100)`` downsamples to 100 Hz, keeping the CPU runtime
# budget tight without distorting slower cortical rhythms.
# ``plot_10_preprocess_and_window`` revisits filtering, montage, and
# reference choices in detail.

# %%
TARGET_SFREQ = 100  # Hz; documented and asserted below
preprocessors = [
    Preprocessor("pick_types", eeg=True, eog=False, misc=False),
    Preprocessor("resample", sfreq=TARGET_SFREQ),
]
preprocess(dataset, preprocessors)
raw_pp = dataset.datasets[0].raw
n_channels = len(raw_pp.ch_names)
sfreq = float(raw_pp.info["sfreq"])
print(f"preprocessed: n_channels={n_channels}, sfreq={sfreq} Hz")
assert sfreq == TARGET_SFREQ, "Resampling did not hit the target sfreq."

# %% [markdown]
# **Investigate**: we now have a homogeneous EEG-only recording at 100 Hz.
# ``n_channels`` and ``sfreq`` are the two numbers that determine every
# tensor shape downstream.

# %% [markdown]
# ## Step 3: cut into fixed-length windows
# ``create_fixed_length_windows`` slides a non-overlapping window across
# the recording. We pick 2-second windows with stride equal to the window
# size, so each sample appears in exactly one window.

# %%
WINDOW_SECONDS = 2.0
window_size_samples = int(WINDOW_SECONDS * TARGET_SFREQ)
stride_samples = window_size_samples
windows = create_fixed_length_windows(
    dataset,
    window_size_samples=window_size_samples,
    window_stride_samples=stride_samples,
    drop_last_window=True,
    preload=True,
)
print(
    f"len(windows)={len(windows)} of {window_size_samples} samples ({WINDOW_SECONDS}s)"
)
X_one, y_one, _idx = windows[0]
print(f"one window X.shape={tuple(X_one.shape)}, y={y_one}")
assert X_one.shape == (n_channels, window_size_samples)

# %% [markdown]
# **Run**: the first window is a NumPy array shaped
# ``(n_channels, window_size_samples)``. Stacking ``batch_size`` of those
# along a new axis gives the 3-D tensor a model expects.

# %% [markdown]
# ## Step 4: wrap windows in a DataLoader
# ``shuffle=False`` plus ``num_workers=0`` makes the first batch
# bit-for-bit reproducible; we assert ``len(windows) >= batch_size`` because
# a smaller dataset would silently change the returned shape.

# %%
BATCH_SIZE = 8
assert len(windows) >= BATCH_SIZE, f"Need >= {BATCH_SIZE} windows; got {len(windows)}."
loader = DataLoader(windows, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
X_batch, y_batch, _idx_batch = next(iter(loader))
print(f"batch: X.shape={tuple(X_batch.shape)}, X.dtype={X_batch.dtype}")
print(
    f"       y.shape={tuple(y_batch.shape)}, unique y={torch.unique(y_batch).tolist()}"
)

# %% [markdown]
# **Investigate**: the batch tensor is shaped
# ``(batch_size, n_channels, window_size_samples)`` with floating-point
# dtype. That is exactly what Braindecode models such as ``ShallowFBCSPNet``
# and ``EEGNetv4`` consume.

# %% [markdown]
# ## Modify
# **Your turn**: edit ``WINDOW_SECONDS`` to 4 seconds, rerun Step 3 and
# Step 4, and predict before you run: how should the per-window time axis
# change, and how should ``len(windows)`` change?

# %% [markdown]
# ## Make
# **Mini-project**: write a tiny custom collate function that returns only
# the signal tensor (drops ``y`` and the index), and rebuild the loader
# with it. This is the kind of helper you write when feeding pretext tasks
# such as masked reconstruction.


# %%
def signal_only_collate(batch):
    """Stack only the signal tensor; drop labels and indices."""
    return torch.stack([torch.as_tensor(item[0]) for item in batch], dim=0)


loader_signal = DataLoader(
    windows,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=signal_only_collate,
)
X_only = next(iter(loader_signal))
print(f"custom collate: X.shape={tuple(X_only.shape)}, X.dtype={X_only.dtype}")
assert X_only.ndim == 3 and X_only.shape[0] == BATCH_SIZE

# %% [markdown]
# ## Result
# We turned one subject of ``ds002718`` (BIDS entities ``dataset``,
# ``subject``, ``task``) into a reproducible PyTorch ``DataLoader``. The
# first batch shape is ``(batch_size, n_channels, window_size_samples)``.
# Hedged claim, in line with Cisotto & Chicco (2024): a clean batch shape
# only confirms *plumbing* -- not signal quality or task design.

# %% [markdown]
# ## Wrap-up
# We chained ``EEGDashDataset`` -> ``preprocess`` ->
# ``create_fixed_length_windows`` -> ``DataLoader`` for one subject, seeded
# both RNGs, and read one batch's shape. Next:
# ``plot_10_preprocess_and_window.py`` replaces these safe defaults with
# intentional preprocessing choices.

# %% [markdown]
# ## Try it yourself
#
# - Re-run with ``shuffle=True`` after seeding ``torch`` and observe which positions land in the first batch.
# - Set ``num_workers=2`` and confirm the batch shape is identical while wall-time changes; explain why on a 1-2-core CPU.
# - Replace ``stride_samples = window_size_samples`` with a 50% overlap (``window_size_samples // 2``) and predict the new ``len(windows)``; verify your prediction.

# %% [markdown]
# ## References
#
# - Wakeman & Henson 2015, multi-subject, multi-modal face-processing dataset, *Scientific Data* 2:150001. https://doi.org/10.1038/sdata.2015.1
# - Schirrmeister et al. 2017, Deep learning with convolutional neural networks for EEG decoding and visualization, *Human Brain Mapping* 38(11). https://doi.org/10.1002/hbm.23730
# - Gramfort et al. 2013, MEG and EEG data analysis with MNE-Python, *Frontiers in Neuroscience* 7. https://doi.org/10.3389/fnins.2013.00267
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Concept page: ``docs/source/concepts/eegdash_objects.rst`` (forthcoming) -- explains the ``EEGDash`` / ``EEGDashDataset`` object split.
