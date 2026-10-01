"""Decode eyes open and eyes closed from recorded resting EEG
==========================================================

Load three Healthy Brain Network participants from ``ds005514``, identify
recorded eyes-open/closed instruction markers, and evaluate alpha-power
features on held-out participants. These recordings are approximately
95 MB each; this task-specific example is larger than the SSVEP basics.
CPU is sufficient; retain downloads with ``EEGDASH_CACHE_DIR``. Install
``eegprep[eeglabio]>=0.2.23,<0.3``. EEGPrep cleaning of the full-density
recordings takes several minutes and is outside the small CI selection.
See the `HBN dataset <https://openneuro.org/datasets/ds005514>`_.

The task is to predict the instructed eye condition of a two-second EEG
window from a participant absent from training. We use integrated spectral power
in a fixed 8–13 Hz band as a small feature set, keeping participant identity
through windowing and evaluation. This tests whether those features carry
useful condition information in the selected recordings.

Run from top to bottom with EEGDash installed; the preprocessing and
leakage-safe split tutorials introduce the Braindecode objects used here.
The outputs are a subject-by-condition count table, a descriptive spectrum
and one held-out score per participant. Cropping these source recordings
would reduce computation but would not avoid their initial download.
"""

# %%
# 1. Query a bounded cohort
# -------------------------
# Explicit IDs make the three-recording workload reproducible. Keep the EGI
# channel names supplied by the source; replacing them with arbitrary 10–20
# names would misrepresent electrode identity. This small five-channel subset
# is fixed before evaluation, and the plot uses E70, its first channel.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.paths import get_default_cache_dir
from eegdash import EEGDashDataset
from eegdash.features import spectral_bands_power, spectral_preprocessor
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation

cache_dir = get_default_cache_dir()
subjects = ["NDARAE710YWG", "NDARAH239PGG", "NDARAL897CYV"]
dataset = EEGDashDataset(
    cache_dir=cache_dir,
    dataset="ds005514",
    task="RestingState",
    subject=subjects,
    n_jobs=1,
)
if len(dataset.datasets) != len(subjects):
    raise ValueError(
        "Query did not return one recording per requested subject; inspect dataset.description"
    )
channels = ["E70", "E62", "E92", "E96", "Cz"]
for recording in dataset.datasets:
    raw = recording.raw
    if not (set(channels) <= set(raw.ch_names)):
        raise ValueError(
            "Required EEG channels or their order differ; inspect channel metadata"
        )
    if not (
        {"instructed_toCloseEyes", "instructed_toOpenEyes"}
        <= set(raw.annotations.description)
    ):
        raise ValueError("Roundtrip length changed; inspect saved windows")
    print(
        recording.description["subject"],
        pd.Series(raw.annotations.description).value_counts(),
    )

# %%
# 2. Clean with EEGPrep and retain the recorded instruction times
# ---------------------------------------------------------------
# EEGPrep performs offset/drift removal, bad-channel detection, ASR burst
# reconstruction, channel interpolation and common-average referencing. Run it
# on the complete EEG montage before selecting five predictors: spatial channel
# checks need the available electrode coverage. Resampling to 128 Hz reduces
# this stage's computation. A 40 Hz low-pass follows the cleaning pipeline.
#
# The fixed ASR cutoff of 20 is a demonstration choice, not a guarantee of
# artifact-free EEG. Cleaning calibrates independently on each entire recording,
# including an unlabelled held-out recording. This is an offline, per-recording
# calibration protocol; it is not a fixed cleaner learned only on training
# subjects. The classifier below still receives labels only from training people.
#
# Whole-window rejection is disabled so cleaning retains the original timeline.
# EEGPrep's MNE/EEGLAB conversions can change annotation timing and measurement
# dates. Save the source annotations, verify that only the sample rate changed,
# then restore their physical times. Do not use this restoration if enabling
# any operation that removes time segments.
before = dataset.datasets[0].raw.copy().pick(["E70"]).crop(tmin=0, tmax=30).load_data()
annotations = [recording.raw.annotations.copy() for recording in dataset.datasets]
measurement_dates = [recording.raw.info["meas_date"] for recording in dataset.datasets]
durations = [
    recording.raw.n_times / recording.raw.info["sfreq"]
    for recording in dataset.datasets
]
if not (all(recording.raw.first_samp == 0 for recording in dataset.datasets)):
    raise ValueError(
        "This restoration recipe requires first_samp=0; inspect source time origins"
    )
preprocess(
    dataset,
    [
        EEGPrep(
            resample_to=128,
            burst_removal_cutoff=20,
            bad_window_max_bad_channels=None,
            max_mem_mb=128,
        ),
        Preprocessor("filter", l_freq=None, h_freq=40),
    ],
    n_jobs=1,
)
for recording, annotation, measurement_date, duration in zip(
    dataset.datasets, annotations, measurement_dates, durations
):
    raw = recording.raw
    if (
        raw.info["sfreq"] != 128
        or raw.first_samp != 0
        or abs(raw.n_times / raw.info["sfreq"] - duration) > 1 / raw.info["sfreq"]
    ):
        raise RuntimeError(
            "Cleaning changed the expected 128 Hz timeline; do not restore instructions"
        )
    raw.set_meas_date(measurement_date)
    raw.set_annotations(annotation)
    np.testing.assert_allclose(raw.annotations.onset, annotation.onset, atol=1e-12)
    np.testing.assert_array_equal(raw.annotations.description, annotation.description)
    # The HBN helper replaces annotations; retain original BAD spans explicitly.
    bad_spans = annotation[
        np.char.startswith(np.char.lower(annotation.description), "bad")
    ]
    hbn_ec_ec_reannotation().apply(raw)
    raw.set_annotations(raw.annotations + bad_spans)
    raw.pick(channels).reorder_channels(channels)
    if raw.ch_names != channels:
        raise ValueError("Predictor channels must retain the declared order")
    if np.any(raw.annotations.duration < 0):
        raise ValueError("Invalid annotation durations after conversion")

# %%
# Inspect cleaning on matched physical time axes
# ----------------------------------------------
# This comparison includes resampling, filtering and referencing as well as
# reconstruction; it cannot attribute a change uniquely to ASR. MNE bad-channel
# lists are observable diagnostics, not a full EEGPrep reconstruction log.
after = dataset.datasets[0].raw.copy().pick(["E70"]).crop(tmin=0, tmax=30)
print("Post-cleaning MNE bad-channel list:", dataset.datasets[0].raw.info["bads"])
fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
for signal, label in [(before, "Source"), (after, "Cleaned")]:
    axes[0].plot(signal.times, signal.get_data()[0] * 1e6, label=label, alpha=0.75)
    spectrum = signal.compute_psd(fmin=1, fmax=40, picks="eeg")
    axes[1].semilogy(spectrum.freqs, spectrum.get_data()[0] * 1e12, label=label)
axes[0].set(
    xlabel="Recording time (s)",
    ylabel="E70 (µV)",
    title="First participant: matched first 30 seconds",
)
axes[1].set(xlabel="Frequency (Hz)", ylabel="E70 PSD (µV²/Hz)")
for ax in axes:
    ax.legend()
plt.show()

# %%
# 3. Window stable periods following the actual instructions
# ----------------------------------------------------------
# The HBN helper places starts at 15..27 seconds after a close instruction
# and 5..17 seconds after an open instruction, every two seconds. The 256-sample
# stop offset extends these zero-duration markers into two-second windows at
# 128 Hz, covering 15..29 and 5..19 seconds. Equal size and stride avoid overlap.
# The target is the instructed condition, without independent verification of
# compliance. Existing BAD spans are respected when creating windows.
#
# Stacking Braindecode windows creates (windows, 5 channels, 256 samples) in
# volts. Metadata row order supplies the observed condition and participant
# group for each window. Inspect the actual retained counts before modelling.
windows = create_windows_from_events(
    dataset,
    mapping={"eyes_open": 0, "eyes_closed": 1},
    trial_start_offset_samples=0,
    trial_stop_offset_samples=256,
    window_size_samples=256,
    window_stride_samples=256,
    on_last_window="drop",
    use_mne_epochs=True,  # MNE rejects epochs overlapping preserved BAD annotations.
    preload=True,
)
metadata = windows.get_metadata()
X = np.stack([window[0] for window in windows])
y = metadata["target"].to_numpy(dtype=int)
groups = metadata["subject"].astype(str).to_numpy()
if not (X.shape[1:] == (len(channels), 256) and np.isfinite(X).all()):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )
if not (set(groups) == set(subjects)):
    raise ValueError(
        "Expected cohort identities are missing; inspect query and retained windows"
    )
print(pd.crosstab(groups, y, rownames=["subject"], colnames=["condition"]))

# %%
# Inspect proposed versus retained windows and preserved BAD spans
# ----------------------------------------------------------------
first_raw = dataset.datasets[0].raw
first_subject = str(dataset.datasets[0].description["subject"])
proposed = first_raw.annotations[
    np.isin(first_raw.annotations.description, ["eyes_open", "eyes_closed"])
]
retained = metadata.loc[metadata["subject"].astype(str) == first_subject]
fig, ax = plt.subplots(figsize=(12, 3), layout="constrained")
for ann in annotations[0]:
    onset = ann["onset"] - first_raw.first_time
    if ann["description"].lower().startswith("bad"):
        ax.axvspan(onset, onset + ann["duration"], color="red", alpha=0.2)
    elif ann["description"].startswith("instructed_"):
        ax.axvline(onset, color="black", alpha=0.4)
        ax.text(
            onset,
            1.3,
            ann["description"].replace("instructed_to", ""),
            rotation=90,
            va="top",
            fontsize=7,
        )
ax.broken_barh(
    [(onset - first_raw.first_time, 2) for onset in proposed.onset],
    (0, 0.3),
    facecolors="lightgray",
    label="Proposed (gray-only spans excluded)",
)
ax.broken_barh(
    [(sample / 128, 2) for sample in retained["i_start_in_trial"]],
    (0.4, 0.3),
    facecolors="tab:blue",
    label="Retained",
)
ax.set(
    xlabel="Recording time (s)",
    yticks=[],
    ylim=(0, 1.4),
    title=f"{first_subject}: instructions, 2-second windows, BAD spans (red)",
)
ax.legend(loc="upper right")
plt.show()

# %%
# 4. Compute alpha power with EEGDash's spectral functions
# --------------------------------------------------------
# ``spectral_preprocessor`` computes a Welch PSD shared by the features and
# the plot. A 256-sample Hamming segment at 128 Hz gives 0.5 Hz bins. The
# PSD retains shape (windows, channels, frequencies) and units V²/Hz.
# EEGDash's band function sums bins in the half-open interval [8, 13) Hz;
# multiplying by the bin width integrates density into power in V².
#
# One alpha-power value per channel yields (windows, 5) features. The log
# compresses their range, and a small floor avoids log(0). These per-window
# operations require no fit; the scaler is fitted only within training folds.
frequencies, psd = spectral_preprocessor(
    X,
    _metadata={"info": dataset.datasets[0].raw.info},
    fs=128,
    nperseg=256,
    noverlap=0,
    window="hamming",
    f_min=1,
    f_max=40,
)
alpha_power = spectral_bands_power(frequencies, psd, bands={"alpha": (8, 13)})["alpha"]
alpha_power *= frequencies[1] - frequencies[0]
features = np.log10(np.maximum(alpha_power, 1e-30))
if not (
    features.shape == (len(metadata), len(channels)) and np.isfinite(features).all()
):
    raise ValueError(
        "Unexpected shape or nonfinite values; inspect input signals and extraction settings"
    )

# %%
# 5. Fit a fresh scaler and classifier in every LOSO fold
# -------------------------------------------------------
# The held-out subject supplies no scaling statistics. Every subject is
# held out once, and all windows from that participant stay together.
# Logistic regression combines the five log-band features into a binary
# decision. Balanced accuracy averages eyes-open and eyes-closed recall,
# with a 0.5 chance reference when both conditions are present. The checks
# verify those conditions and exactly-once test coverage; they do not require
# the model to beat chance. With two training participants in each fold, there
# is little support for model selection, so parameters are fixed in advance.
rows = []
counts = np.zeros(len(y), dtype=int)
for train, test in LeaveOneGroupOut().split(features, y, groups):
    if not (set(groups[train]).isdisjoint(groups[test])):
        raise ValueError(
            "Training and test participants overlap; fix the evaluation split"
        )
    if not (set(y[train]) == set(y[test]) == {0, 1}):
        raise ValueError("Roundtrip length changed; inspect saved windows")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[train], y[train])
    rows.append(
        {
            "subject": groups[test][0],
            "balanced_accuracy": balanced_accuracy_score(
                y[test], model.predict(features[test])
            ),
        }
    )
    counts[test] += 1
if not (len(rows) == len(subjects) and np.all(counts == 1)):
    raise ValueError("Every trial must have exactly one held-out prediction")
results = pd.DataFrame(rows)
print(results.to_string(index=False))

# %%
# 6. Compare the measured spectra and held-out scores
# ---------------------------------------------------
# Average within each subject before averaging across subjects. Differences
# in this small cohort need not reproduce a textbook effect or exceed chance.
# PSD is calculated in V²/Hz; multiplying by 1e12 converts the displayed
# density to µV²/Hz. The spectrum is descriptive and includes all three
# participants. The bars instead summarize strictly held-out predictions.
# A difference in the averaged spectrum does not guarantee separation of
# individual windows, and repeated windows do not increase the number of
# independent participants beyond three. Check whether the largest peaks lie
# inside the selected 8–13 Hz band before calling them alpha activity. A large
# out-of-band peak is a reason to inspect the raw channels and recording
# quality; its size alone does not identify a neural source.
e70 = dataset.datasets[0].raw.ch_names.index("E70")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
for label, name in [(0, "eyes open"), (1, "eyes closed")]:
    subject_psds = [
        psd[(groups == subject) & (y == label), e70].mean(axis=0)
        for subject in subjects
    ]
    axes[0].semilogy(frequencies, np.mean(subject_psds, axis=0) * 1e12, label=name)
axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD at E70 (µV²/Hz)")
axes[0].axvspan(8, 13, color="gray", alpha=0.15, label="[8, 13) Hz")
axes[0].legend()
axes[1].bar(range(len(results)), results["balanced_accuracy"])
axes[1].set_xticks(range(len(results)), results["subject"], rotation=45, ha="right")
axes[1].axhline(0.5, color="black", linestyle="--", label="Chance")
axes[1].set(ylabel="Balanced accuracy", xlabel="Held-out subject", ylim=(0, 1))
axes[1].legend()
for subject in subjects:
    means = [
        features[(groups == subject) & (y == label), e70].mean() for label in [0, 1]
    ]
    axes[2].plot([0, 1], means, "o-", label=subject)
axes[2].set(
    xticks=[0, 1],
    xticklabels=["Open", "Closed"],
    ylabel="Mean log10(E70 alpha power / 1 V²)",
    title="Paired participant summaries",
)
axes[2].legend(fontsize=7)
plt.show()

# %%
# 7. Extend the participant-level evaluation
# ------------------------------------------
# Inspect which channels and bursts EEGPrep changes before interpreting
# the spectra as physiological evidence. Then inspect the timing of retained
# instruction intervals. Then add participants under the same fixed protocol
# and examine the distribution of participant scores. If you change channel
# selection or the frequency band based on performance, choose them using
# separate validation participants inside each training fold. Randomly mixing
# this participant's windows into training would instead measure performance
# on an already observed person.
