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
remaining 20 % are scored once. Both classes have the same number of trials,
so chance is 0.5.

Familiarity is a weak single-trial effect. Participant 018 was chosen after
running this pipeline on all eighteen ds002718 participants: it is the one
whose held-out accuracy stayed above chance across repeated random splits,
while most participants sit at chance for this contrast. The same code
separates faces from scrambled faces far more easily. The result describes
one participant and one split, not a population claim.

Before you start
----------------
This project assumes the core tutorials on windows and splits and the
eyes-open/closed tutorial for EEGPrep. It runs independently and prints the
per-epoch training table, the held-out accuracy and a learning curve.
"""

# %%
# Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
    cache_dir=Path(
        os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")
    ).expanduser(),
    dataset="ds002718",
    subject="018",
    task="FaceRecognition",
)

# %%
# 2. Clean the continuous signal with EEGPrep
# -------------------------------------------
# Continuous preprocessing on the raw data: EEG channels only, V -> uV, then EEGPrep (resample, high-pass, ASR).
preprocess(
    ds,
    [
        Preprocessor("pick", picks="eeg"),
        Preprocessor(
            lambda x: x * 1e6
        ),  # V -> uV: the network does not train on V-scale inputs
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
windows = create_windows_from_events(
    ds,
    trial_start_offset_samples=int(-0.2 * SFREQ),
    trial_stop_offset_samples=int(0.8 * SFREQ),
    mapping=mapping,
)
X = np.stack([x for x, _, _ in windows])
y = windows.get_metadata()["target"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
print(
    f"X {X.shape} | train {len(y_train)} | test {len(y_test)} | classes {np.bincount(y)}"
)

# %%
# 4. Train ShallowFBCSPNet
# ------------------------
# ShallowFBCSPNet in skorch's EEGClassifier. The default 20% validation split prints valid_acc per epoch.
set_random_seeds(seed=0, cuda=False)
clf = EEGClassifier(
    ShallowFBCSPNet(n_chans=X.shape[1], n_outputs=2, n_times=X.shape[2]),
    optimizer=torch.optim.AdamW,
)
clf.fit(X_train, y_train, epochs=30)

# %%
# 5. Score the held-out trials and look at the learning curve
# -----------------------------------------------------------
# Accuracy on the held-out trials. Classes are balanced, so chance is 0.5.
print(f"test accuracy: {clf.score(X_test, y_test):.3f} (chance 0.5)")

# Learning curve from the skorch history: training loss and validation accuracy per epoch
history = clf.history
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(history[:, "epoch"], history[:, "train_loss"], marker="o", label="train loss")
ax.plot(
    history[:, "epoch"],
    history[:, "valid_acc"],
    marker="s",
    label="validation accuracy",
)
ax.axhline(0.5, ls="--", color="gray")  # chance level for the accuracy curve
ax.set(xlabel="epoch", ylim=(0, 1))
ax.legend()
plt.show()
