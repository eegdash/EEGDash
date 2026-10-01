"""Inspect your first EEG recording
================================

Open one recording and inspect its real voltage traces and spectrum.

These real Nakanishi2015 SSVEP recordings are distributed as the processed
`nm000118 release <https://nemar.org/dataset/nm000118>`_
(`study <https://doi.org/10.1371/journal.pone.0140703>`_).
Filtering, downsampling and latency handling were already applied; do not
shift the event onsets again. CPU is sufficient. Internet is needed for the
first download; ``EEGDASH_CACHE_DIR`` keeps downloads across runs.

The explicit subset uses 1 participant(s), about 7.0 MB of signal files.

Before you start
----------------
Install EEGDash and its dependencies. Tutorial 00 introduces the catalogue
query, but this script runs independently. You will need basic NumPy indexing.
The useful output is an MNE ``Raw`` object whose channels, units and event
labels you have inspected before making training windows.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset

# %%
# 1. Load and inspect the selected recordings
# -------------------------------------------
# The dataset constructor retrieves descriptions; accessing ``recording.raw``
# opens the signal file and acquires it if absent from the cache. Count
# ``dataset.datasets`` to count recordings. The concatenated dataset's length
# has a different meaning and should not be used as the recording count.
#
# SSVEP means a response to repeated visual stimulation. Here annotation
# strings name the attended flicker frequency in Hz. Sorting with ``key=float``
# keeps numerical frequency order; sorting strings or relying on automatic
# integer event codes need not do that. The channel and rate checks establish
# the array contract that the next tutorials reuse.
cache_dir = get_default_cache_dir()
subjects = ["1"]
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
(recording,) = dataset.datasets  # This lesson opens exactly one recording.
raw = recording.raw
sfreq = raw.info["sfreq"]
channel_names = raw.ch_names
class_names = sorted(set(raw.annotations.description), key=float)
mapping = {name: index for index, name in enumerate(class_names)}
print(f"Channels: {channel_names}; sampling rate: {sfreq} Hz")
print("Observed stimulus frequencies (Hz):", class_names)

# %%
# 2. Inspect voltage and annotations
# ----------------------------------
# ``get_data`` returns ``(channels, samples)``. At 256 Hz, four seconds contain
# 1,024 samples, and ``stop`` is exclusive. Separate labelled panels share
# the time-from-event axis. Multiplying by
# ``1e6`` changes display units from volts to microvolts, not the cached signal.
#
# Inspect the annotations table alongside the trace: an event description tells
# you the stimulus condition, whereas a waveform shows the recorded response.
# A finite-array check catches invalid numerical values; it does not rule
# out artifacts, clipping or a poorly connected electrode.
print(raw)
raw.annotations.to_data_frame().head()

# %%
trial = 0  # Change the annotation index, not an assumed concatenation offset.
onset = raw.annotations.onset[trial] - raw.first_time
start = raw.time_as_index(onset, use_rounding=True)[0]
signal = raw.get_data(picks="eeg", start=start, stop=start + int(4 * sfreq))
if signal.size == 0 or not np.isfinite(signal).all():
    raise ValueError("Selected trial is empty or nonfinite; inspect source samples")
fig, axes = plt.subplots(
    len(channel_names), 1, figsize=(9, 8), sharex=True, layout="constrained"
)
for ax, name, voltage in zip(axes, channel_names, signal):
    ax.plot(np.arange(signal.shape[1]) / sfreq, voltage * 1e6, linewidth=0.6)
    ax.axvline(0, color="black", linestyle=":")
    ax.set_ylabel(f"{name}\nµV")
axes[0].set_title(
    f"Annotation {trial}: {raw.annotations.description[trial]} Hz; onset {onset:.3f} s"
)
axes[-1].set_xlabel("Time from annotated onset (s)")

# %%
# 3. Inspect the supplied channel geometry and spectrum
# -----------------------------------------------------
# The averaged power spectral density summarizes how signal power is distributed
# across frequency over the recording. The 40 Hz display limit focuses on the
# low-frequency range and does not apply another filter. A spectral peak can
# suggest a rhythm or stimulus response, but this whole-recording average mixes
# all twelve attended frequencies and cannot validate a class label by itself.
#
# Channel coordinates describe where sensors were placed; channel names and
# order determine which signal is which. Neither an attractive montage nor a
# smooth spectrum replaces inspection of the individual trial voltages.
montage = raw.get_montage()
if montage is not None:
    positions = np.array(list(montage.get_positions()["ch_pos"].values()))
    if positions.size and np.isfinite(positions).all() and np.any(positions):
        raw.plot_sensors(show_names=True, show=False)
raw.compute_psd(fmax=40, picks="eeg").plot(average=True, show=False)
# A separate selected-trial spectrum does not mix conditions.
trial_raw = raw.copy().crop(tmin=onset, tmax=onset + (signal.shape[1] - 1) / sfreq)
spectrum = trial_raw.compute_psd(fmax=40, picks="eeg")
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.semilogy(spectrum.freqs, spectrum.get_data().mean(axis=0) * 1e12)
frequency = float(raw.annotations.description[trial])
for harmonic in np.arange(frequency, 40, frequency):
    ax.axvline(harmonic, color="tab:orange", linestyle="--")
ax.set(
    xlabel="Frequency (Hz)",
    ylabel="PSD (µV²/Hz)",
    title=f"Selected {frequency:g} Hz trial: stimulus and harmonics",
)
plt.show()

# %%
# Try another observed trial by changing the start and stop sample indices.
# The release concatenates trials; their order is not acquisition chronology.

# %%
# Continue with event-labelled windows
# ------------------------------------
# Tutorial 02 converts this same recording into four-second windows and shows
# where its annotation-derived label appears in a DataLoader batch. When
# inspecting another trial manually, derive its start from the annotation
# onset rather than assuming four-second trials are adjacent with no remainder.
