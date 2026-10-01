"""Familiar vs unfamiliar faces with EEGPrep and ShallowFBCSPNet
==============================================================

Decode from single trials whether a participant saw a famous or an
unfamiliar face, using the real `ds002718
<https://openneuro.org/datasets/ds002718>`_ face-recognition recordings
(Wakeman & Henson). EEGDashDataset downloads one participant (about 224 MB);
set ``EEGDASH_CACHE_DIR`` to reuse it. CPU is sufficient. Install
``eegprep[eeglabio]>=0.2.23,<0.3`` for the EEGPrep stage.

The pipeline is deliberately short: keep the EEG channels, clean the
continuous signal with Braindecode's EEGPrep (resampling, high-pass filter,
artifact subspace reconstruction), cut one-second windows around each face
onset, and train ShallowFBCSPNet on 80 % of that participant's trials. The
remaining 20 % are scored once. Binary balanced accuracy has a constant-class
reference of 0.5 regardless of retained class counts.

This is an outcome-selected exploratory illustration: participant 018 was
chosen after inspecting results across eighteen participants and repeated splits.
Consequently the held-out trials below are not unbiased evidence of decoding
performance, even with a fresh split. Repeated face identities can occur on
both sides: the estimand is within-participant, potentially within-stimulus
prediction, not recognition of unseen identities or population generalization.
A confirmatory study must prespecify participants and group original image
identities using the source event metadata before any outcome inspection.

Before you start
----------------
This project assumes the core tutorials on windows and splits and the
eyes-open/closed tutorial for EEGPrep. It runs independently and prints the
per-epoch training table, the held-out accuracy and a learning curve.
"""

# %%
# Install dependencies (uncomment when running in Colab or a fresh notebook)

# !pip install eegdash braindecode "eegprep[eeglabio]>=0.2.23,<0.3" scikit-learn torch numpy

# %%
# Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from skorch.dataset import ValidSplit
from eegdash.paths import get_default_cache_dir

from eegdash import EEGDashDataset
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.util import set_random_seeds

# SFREQ is the target sampling rate
SFREQ = 128

# %%
# 1. Load one participant of ds002718
# -----------------------------------
# One subject of ds002718 (Wakeman & Henson face recognition), fetched by EEGDash into a local cache.
ds = EEGDashDataset(
    cache_dir=get_default_cache_dir(),
    dataset="ds002718",
    subject="018",
    task="FaceRecognition",
)

# %%
# 2. Clean the continuous signal with EEGPrep
# -------------------------------------------
# EEGPrep calibrates on this entire unlabelled recording before splitting.
# This is offline/transductive cleaning, not a train-only or causal protocol.
# Keep volts through EEGPrep; convert the resulting windows for the network.
preprocess(
    ds,
    [
        Preprocessor("pick", picks="eeg"),
        EEGPrep(
            resample_to=SFREQ,
            highpass_frequencies=(
                0.25,
                0.75,
            ),  # transition band: full stop at 0.25 Hz, passband from 0.75 Hz
            burst_removal_cutoff=10.0,  # ASR: reject components beyond 10 SD of the calibration data
            bad_window_max_bad_channels=None,  # no bad-window removal: time axis intact, every trial survives
        ),
    ],
)

# %%
# 3. Epoch familiar vs unfamiliar faces and split the trials
# ----------------------------------------------------------
# Epoch -0.2..0.8 s around face onsets, famous = 0 vs unfamiliar = 1, then a within-subject 80/20 split.
mapping = {
    "famous_new": 0,
    "famous_second_early": 0,
    "famous_second_late": 0,
    "unfamiliar_new": 1,
    "unfamiliar_second_early": 1,
    "unfamiliar_second_late": 1,
}
# Quantize the requested onset to the recording grid and reuse it in the plot.
start_offset_samples = int(-0.2 * SFREQ)
windows = create_windows_from_events(
    ds,
    trial_start_offset_samples=start_offset_samples,
    trial_stop_offset_samples=int(0.8 * SFREQ),
    mapping=mapping,
)
X = (np.stack([x for x, _, _ in windows]) * 1e6).astype(np.float32)
y = windows.get_metadata()["target"].to_numpy(dtype=np.int64)
if not np.isfinite(X).all() or set(y) != {0, 1}:
    raise ValueError(
        "Need finite epochs in both face conditions; inspect event mapping and epoch rejection."
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
print(
    f"X {X.shape} | train {len(y_train)} | test {len(y_test)} | classes {np.bincount(y)}"
)

# %%
# Inspect the actual condition-averaged waveform before fitting.
# --------------------------------------------------------------
channel = ds.datasets[0].raw.ch_names[0]
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
for code, label in enumerate(["Famous", "Unfamiliar"]):
    ax.plot(
        (np.arange(X.shape[-1]) + start_offset_samples) / SFREQ,
        X[y == code, 0].mean(axis=0),
        label=f"{label}, n={sum(y == code)}",
    )
ax.axvline(0, color="black", linestyle="--")
ax.set(xlabel="Time from face onset (s)", ylabel="Amplitude (µV)", title=channel)
ax.legend()
plt.show()

# %%
# 4. Train ShallowFBCSPNet
# ------------------------
# ShallowFBCSPNet in skorch's EEGClassifier. The default 20% validation split prints valid_acc per epoch.
set_random_seeds(seed=0, cuda=False)
model = ShallowFBCSPNet(n_chans=X.shape[1], n_outputs=2, n_times=X.shape[2])
clf = EEGClassifier(
    model,
    optimizer=torch.optim.AdamW,
    train_split=ValidSplit(0.2, stratified=True, random_state=0),
    batch_size=64,
    lr=0.001,
)
print(model)
clf.fit(X_train, y_train, epochs=30)

# %%
# 5. Score the held-out trials and look at the learning curve
# -----------------------------------------------------------
# Score once after the fixed training budget; selection bias remains.
predicted = clf.predict(X_test)
print("Exploratory balanced accuracy:", balanced_accuracy_score(y_test, predicted))
print("Constant-class balanced-accuracy reference: 0.5")
ConfusionMatrixDisplay.from_predictions(
    y_test, predicted, display_labels=["Famous", "Unfamiliar"]
)
plt.show()
history = clf.history
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
axes[0].plot(history[:, "epoch"], history[:, "train_loss"], marker="o")
axes[0].set(xlabel="Epoch", ylabel="Training loss")
axes[1].plot(history[:, "epoch"], history[:, "valid_acc"], marker="s")
axes[1].axhline(0.5, ls="--", color="gray")
axes[1].set(xlabel="Epoch", ylabel="Development validation accuracy", ylim=(0, 1))
plt.show()
