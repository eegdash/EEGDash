"""Preprocess EEG and create reusable windows
==========================================

Raw EEG is rarely model-ready: wrong sampling rate, drift and line
noise, no fixed reference, continuous timeline instead of fixed epochs.
Cisotto & Chicco (2024, PeerJ CS, doi:10.7717/peerj-cs.2256) say it
plainly in their ten quick tips for clinical EEG: a decoding result is
only as trustworthy as the *named* preprocessing choices behind it. So
how do we turn one OpenNeuro recording into a deterministic, reusable
windows dataset that we can audit later?

In this tutorial we walk the canonical EEGDash preprocessing recipe on
one resting-state subject from ``ds002718`` (Wakeman & Henson,
OpenNeuro). Set the montage, average-reference, band-pass filter,
resample, cut into 2 s windows. We name every choice and print every
parameter. The result is the ``windows`` object the next four
core-workflow tutorials reuse.

Learning objectives
-------------------

- Set a 10-20 montage with ``set_montage`` and surface BIDS entities.
- Set an average reference with ``set_eeg_reference`` (Cisotto Tip 5).
- Apply a band-pass filter and report pass-band, stop-band, filter type.
- Resample to a target sampling rate and verify the new ``sfreq``.
- Create fixed-length windows and check the count from duration.

Requirements
------------

- Prereq: :ref:`plot_02_dataset_to_dataloader`. CPU only, ~3 minutes.
- Cached on first run via :func:`eegdash.paths.get_default_cache_dir`;
  re-runs are offline. Concept: ``docs/source/concepts/preprocessing_decisions.rst``.

References: Pernet et al. 2019 (BIDS-EEG, doi:10.1038/s41597-019-0104-8);
Gramfort et al. 2013 (MNE, doi:10.3389/fnins.2013.00267); Cisotto &
Chicco 2024 (Tips 4-5, doi:10.7717/peerj-cs.2256). Why care about *this*
recording in particular?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_10_preprocess_and_window.png'

# %%
# Setup
# -----
#
# We pin the RNG, import only what each later step needs, and parameterise
# the cache directory so the example runs on any host.

import numpy as np

np.random.seed(42)

from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

CACHE_DIR = get_default_cache_dir()
TARGET_SFREQ = 100.0  # Hz, see Step 5
WINDOW_SIZE_S = 2.0  # seconds, see Step 6

# %%
# Step 1 -- Load one recording (lazy)
# -----------------------------------
#
# The Pernet et al. 2019 BIDS-EEG schema names *exactly* one file:
# dataset, subject, task, run. ``EEGDashDataset.preview`` materialises
# only that recording -- a cheap inspection step before the full
# pipeline. ``ds002718`` (Wakeman & Henson) is small, public, 10-20.

dataset = EEGDashDataset(
    cache_dir=CACHE_DIR, dataset="ds002718", subject="012", task="FacePerception"
)
preview = dataset.preview(0)
raw = preview.raw.load_data().copy()
print(
    f"BIDS: subject={preview.metadata.get('subject')}, "
    f"task={preview.metadata.get('task')}, run={preview.metadata.get('run')}"
)
print(
    f"sfreq={raw.info['sfreq']} Hz, n_channels={len(raw.ch_names)}, "
    f"duration={raw.times[-1]:.1f} s"
)

# %% [markdown]
# **Predict.** Before the next cell, what do you think the average
# reference does to the per-channel mean of a 5 s snippet -- move it
# closer to zero, leave it unchanged, or grow it?

# %%
# Step 2 -- Set the montage
# -------------------------
#
# A montage tells MNE where each electrode sits in 3-D space. EEGDash
# surfaces channels via the BIDS ``channels.tsv`` (Pernet 2019), but we
# still attach the 10-20 positions explicitly so downstream tools know
# the geometry. ``on_missing="ignore"`` keeps non-EEG sensors untouched.

raw.set_montage("standard_1020", on_missing="ignore")
print(f"montage attached: {raw.get_montage().kind if raw.get_montage() else None}")

# %%
# Step 3 -- Set an average reference
# ----------------------------------
#
# Cisotto & Chicco 2024 (Tip 5) want the reference scheme reported
# explicitly: every voltage in EEG is a *difference*, so the choice
# changes the signal. The common-average reference subtracts the mean
# across channels at each time point and is a defensible default for
# whole-head montages. ``projection=False`` applies it immediately.

raw.set_eeg_reference("average", projection=False)
print(f"custom_ref_applied={raw.info['custom_ref_applied']}")

# %% [markdown]
# **Run.**
#
# **Investigate.** The per-channel mean of the snippet is now near
# zero -- the average reference at work. That makes downstream
# filtering and ICA more numerically stable.

# %%
# Step 4 -- Band-pass filter (1-40 Hz, FIR, zero-phase)
# -----------------------------------------------------
#
# Filter type, pass-band, and stop-band are reported here per Cisotto &
# Chicco 2024 (Tip 4). We use a non-causal **FIR** filter with the
# ``firwin`` design (Hamming window, the MNE default). Pass-band:
# **1-40 Hz** -- drift below 1 Hz removed, line noise above 40 Hz
# attenuated. Stop-band edges are set automatically by MNE from the
# transition bandwidth. ``method="fir"``, ``fir_design="firwin"``, and
# ``phase="zero"`` are named explicitly so the recipe is reproducible.

L_FREQ, H_FREQ = 1.0, 40.0  # pass-band edges in Hz
print(f"filter type=FIR, pass-band={L_FREQ}-{H_FREQ} Hz, design=firwin, phase=zero")
raw.filter(
    l_freq=L_FREQ,
    h_freq=H_FREQ,
    method="fir",
    fir_design="firwin",
    phase="zero",
    verbose=False,
)
print(f"highpass={raw.info['highpass']:.2f} Hz, lowpass={raw.info['lowpass']:.2f} Hz")

# %%
# Step 5 -- Resample to 100 Hz
# ----------------------------
#
# The original sampling rate is far above what 1-40 Hz content needs.
# 100 Hz keeps comfortable headroom above Nyquist (80 Hz) and shrinks
# memory; window arithmetic also stays clean.

raw.resample(TARGET_SFREQ, verbose=False)
assert raw.info["sfreq"] == TARGET_SFREQ
print(f"resampled sfreq={raw.info['sfreq']} Hz")

# %%
# Step 6 -- Create 2 s fixed-length windows
# -----------------------------------------
#
# We replay the same five Preprocessor steps on the *dataset* (so the
# braindecode wrapper carries the metadata) and hand it to
# :func:`braindecode.preprocessing.create_fixed_length_windows`. Window
# size is ``WINDOW_SIZE_S * TARGET_SFREQ = 200`` samples; stride equals
# the window size for **0% overlap**.

window_samples = int(WINDOW_SIZE_S * TARGET_SFREQ)
preprocess(
    dataset,
    [
        Preprocessor("set_montage", montage="standard_1020", on_missing="ignore"),
        Preprocessor("set_eeg_reference", ref_channels="average"),
        Preprocessor(
            "filter", l_freq=L_FREQ, h_freq=H_FREQ, method="fir", fir_design="firwin"
        ),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
    ],
)
windows = create_fixed_length_windows(
    dataset,
    window_size_samples=window_samples,
    window_stride_samples=window_samples,
    drop_last_window=True,
)

# %% [markdown]
# **Predict.** The recording is around ``raw.times[-1]`` seconds long.
# With 2 s non-overlapping windows we expect roughly
# ``floor(duration / 2)`` windows. Write that number down.
#
# **Run.**
#
# **Investigate.** The first window is ``windows[0][0]``: a NumPy
# array shaped ``(n_channels, 200)``. Each window also carries
# metadata columns (``i_start_in_trial``, ``target``, ...) that the
# next tutorial (:ref:`plot_11_leakage_safe_split`) uses for
# subject-aware splits.

# %%
# Result
# ------

x0, _, _ = windows[0]
print(f"n_windows={len(windows)}, window_shape={tuple(x0.shape)} (channels x samples)")
print(f"sfreq={TARGET_SFREQ} Hz, n_channels={x0.shape[0]}")
assert x0.shape[1] == window_samples
assert raw.info["sfreq"] == TARGET_SFREQ

# %% [markdown]
# **A common mistake -- and how to recover.**
# **Run.** Swapping the high-pass and low-pass cutoffs is the most
# common slip when tuning this filter. MNE catches it with a
# ``ValueError``; we trigger it with ``try/except`` so you see the
# error first-hand.

# %%
try:
    raw.copy().filter(l_freq=H_FREQ, h_freq=L_FREQ, verbose=False)
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: l_freq must be below h_freq.
    print(f"Recovery: l_freq={L_FREQ} < h_freq={H_FREQ}.")

# %% [markdown]
# **Modify.** Re-run Step 4 with ``L_FREQ, H_FREQ = 1.0, 8.0`` to
# isolate the delta-theta band. The alpha bump in the PSD should
# disappear; explain why in one sentence.

# %% [markdown]
# **Make.** Apply the same six steps to ``subject="013"`` and confirm
# the window shape matches. The pipeline drops in unchanged.

# %% [markdown]
# Try it yourself
# ---------------
#
# - Set ``L_FREQ, H_FREQ = 1.0, 8.0`` and watch the alpha bump leave the PSD; explain why in one sentence.
# - Set ``window_stride_samples = window_samples // 2`` for 50% overlap and note the window count double; weigh that against leakage-safe splits.
# - Apply the same six steps to a new subject (``subject="013"``) and confirm the window shape matches.
# - Save ``windows`` with ``windows.save(...)`` and reload next session without re-running preprocessing.
# - Read the [preprocessing decisions concept](../../docs/source/concepts/preprocessing_decisions.rst) page and pick one default to question on your own data.
