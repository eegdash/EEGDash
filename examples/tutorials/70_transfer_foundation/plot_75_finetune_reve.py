"""How do I fine-tune a published EEG foundation model (REVE)?
=============================================================

**Difficulty 3** | **Runtime: ~2m (CPU)** | **Compute: CPU (GPU Optional)**

Adapt the published REVE-Base checkpoint to a motor-imagery-vs-rest task from
OpenNeuro dataset ``ds003810``, served through EEGDash and trained with
Braindecode's ``EEGClassifier``. The checkpoint is
https://huggingface.co/brain-bzh/reve-base (documented by Braindecode's ``REVE``
model). It is gated: accepting the authors' terms once and logging in to the
Hugging Face Hub is part of this tutorial. It downloads about 280 MB of weights
and 7 MB of EEG (one participant, four runs).

Two ways to adapt a pretrained encoder are shown with the same code path:
a **linear probe** (encoder frozen, only the classification head trains -- runs
on a laptop CPU) and **full fine-tuning** (everything trains at a lower learning
rate -- enabled automatically on a GPU). Runs 1 and 2 train, run 3 chooses when
to stop, and run 4 is scored exactly once. The small budget demonstrates
adaptation, not a foundation-model benchmark.

Keywords: foundation-model, fine-tuning, linear-probe, REVE, motor-imagery, transfer-learning
"""

# %%
# Before you start
# ----------------
# REVE (Representation for EEG with Versatile Embeddings) is a transformer
# pretrained with a masked-autoencoder objective on about 60,000 hours of EEG
# from 92 datasets. Its distinctive part is a 4-D positional encoding: every
# electrode is described by its 3-D position on a template head plus time, so
# one model accepts any montage. A 15-channel low-cost headset recording goes
# straight into a model pretrained mostly on 19- to 128-channel data, with no
# common-channel subset to pick.
#
# The weights are published under a gated license. One-time setup:
#
# 1. Open https://huggingface.co/brain-bzh/reve-base and accept the terms
#    (also for https://huggingface.co/brain-bzh/reve-positions, the electrode
#    position table the model needs).
# 2. Log in once from a terminal: ``hf auth login`` (or set ``HF_TOKEN``).
#
# The check below turns a missing login into a readable message instead of an
# HTTP traceback further down. Without network it simply continues and relies
# on the local cache.

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from huggingface_hub import auth_check
from huggingface_hub.errors import HfHubHTTPError
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from sklearn.metrics import balanced_accuracy_score

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset
from braindecode.models import REVE
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash import EEGDashDataset

MODEL_ID = "brain-bzh/reve-base"
DATASET, SUBJECT, RUNS = "ds003810", "02", ["1", "2", "3", "4"]
SFREQ = 200  # REVE was pretrained at 200 Hz; it does not check, so we must
WINDOW_S = 4  # cue -> end of trial in this paradigm
CACHE_DIR = Path(os.environ.get("EEGDASH_CACHE_DIR", "~/.eegdash_cache")).expanduser()
DEVICE = os.environ.get("REVE_TUTORIAL_DEVICE") or (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Full fine-tuning updates all 69M parameters; on CPU that is minutes per epoch,
# so it is switched on only where it is cheap. Force it with REVE_TUTORIAL_FINETUNE=1.
RUN_FINETUNE = (
    torch.cuda.is_available() or os.environ.get("REVE_TUTORIAL_FINETUNE") == "1"
)
FT_EPOCHS = int(os.environ.get("REVE_TUTORIAL_FT_EPOCHS", 5))
torch.manual_seed(0)
print(
    f"device: {DEVICE} | full fine-tuning: {'on' if RUN_FINETUNE else 'off (linear probe only)'}"
)

# Keep the output readable: the libraries are chatty at INFO level.
mne.set_log_level("WARNING")  # filter-design reports, "legacy function" notes
for name in ("httpx", "huggingface_hub", "eegdash", "braindecode"):
    logging.getLogger(name).setLevel(logging.WARNING)
try:
    auth_check(MODEL_ID)
    print("Hugging Face access to", MODEL_ID, "OK")
except HfHubHTTPError as err:  # gated repo without an accepted license / token
    raise SystemExit(  # a plain message, not a traceback
        f"\nCannot access {MODEL_ID} ({err.__class__.__name__}). The REVE weights are gated:\n"
        f"  1) open https://huggingface.co/{MODEL_ID} and accept the terms\n"
        f"  2) run `hf auth login` (or export HF_TOKEN=...) and re-run this example.\n"
    ) from None
except Exception as err:  # no network: fine if the weights are already cached
    print(
        f"Could not reach the Hub ({err.__class__.__name__}); continuing with the local cache."
    )

# %%
# Match the real signal to the encoder input
# ------------------------------------------
# Load one participant's motor-imagery runs from EEGDash and bring them to the
# format REVE was pretrained on.
#
# ``ds003810`` ("Motor Imagery vs Rest -- Low-Cost EEG System") has ten
# participants, 15 electrodes at 125 Hz and five runs each. Run 0 is a warm-up
# in which the participant *really* squeezes their hand -- with the same event
# codes as the imagery runs -- so loading it would train on executed movement
# labelled as imagery. We load runs 1-4 only.
#
# The recording software (OpenViBE) wrote its own cue codes: ``OVTK_GDF_Right``
# is the imagery cue and ``OVTK_GDF_Tongue`` the rest cue (the dataset's
# ``events.json`` documents this). Each cue is an instantaneous marker and the
# trial runs 4 s from cue to feedback, so one 4 s window per cue fits exactly.
#
# REVE's input contract -- none of it enforced at runtime: 200 Hz, EEG channels
# only, and per-channel z-scored input clipped at 15 standard deviations.

ds = EEGDashDataset(cache_dir=CACHE_DIR, dataset=DATASET, subject=SUBJECT, run=RUNS)
print(ds.description[["subject", "run", "task"]].to_string(index=False))


def zscore_clip(x, clip=15.0):
    x = (x - x.mean()) / (x.std() + 1e-8)
    return np.clip(x, -clip, clip)


preprocess(
    ds,
    [
        Preprocessor("pick", picks="eeg", apply_on_array=False),
        Preprocessor("filter", l_freq=0.5, h_freq=45.0, apply_on_array=False),
        Preprocessor("resample", sfreq=SFREQ, apply_on_array=False),
        Preprocessor(zscore_clip, apply_on_array=True),
    ],
)
chs_info = ds.datasets[0].raw.info["chs"]
print(len(chs_info), "EEG channels:", [c["ch_name"] for c in chs_info])

# The cue markers have zero duration, so the trial would be empty with the
# default stop offset; extend each trial to exactly one window from the cue.
windows = create_windows_from_events(
    ds,
    mapping={"OVTK_GDF_Right": 0, "OVTK_GDF_Tongue": 1},  # 0 = motor imagery, 1 = rest
    trial_start_offset_samples=0,
    trial_stop_offset_samples=WINDOW_S * SFREQ,
    window_size_samples=WINDOW_S * SFREQ,
    window_stride_samples=WINDOW_S * SFREQ,
    preload=True,
)
print(
    len(windows),
    "windows of",
    WINDOW_S * SFREQ,
    "samples;",
    "labels per run:",
    windows.get_metadata().groupby("run")["target"].value_counts().unstack().to_dict(),
)

# %%
# Reserve runs, not windows
# -------------------------
# Windows from the same run share slow drifts, electrode impedance and the
# participant's state, so a random window-level split leaks. Split by run:
# runs 1-2 train, run 3 decides when to stop training, run 4 is scored once at
# the end and never looked at before that.
#
# This is a *within-subject* design, chosen so the page runs on a CPU in a
# minute. To claim the model works on *new people* you need to hold out whole
# participants -- and one held-out participant is a single draw from a wide
# spread, so the reference number at the end is a leave-one-subject-out mean
# over all ten participants, computed with the same recipe on a GPU.

by_run = windows.split("run")
train_set = BaseConcatDataset([by_run["1"], by_run["2"]])
valid_set, test_set = by_run["3"], by_run["4"]
print(f"train {len(train_set)} windows | valid {len(valid_set)} | test {len(test_set)}")

# %%
# Load the checkpoint and understand the head
# -------------------------------------------
# ``REVE.from_pretrained`` reads the checkpoint's ``config.json`` to build
# Braindecode's ``REVE`` module at the right depth and width, loads the weights,
# and swaps the pretraining head for a fresh classification head with
# ``n_outputs`` classes. Electrode positions are resolved from channel *names*
# against the published position table (543 standard names -> x, y, z). That
# lookup is exact and case-sensitive and unknown names are dropped silently, so
# check that every channel resolved before training.

model = REVE.from_pretrained(
    MODEL_ID,
    n_outputs=2,
    n_chans=len(chs_info),
    n_times=WINDOW_S * SFREQ,
    sfreq=SFREQ,
    chs_info=chs_info,
)
missing = [
    c["ch_name"] for c in chs_info if c["ch_name"] not in model._position_bank.mapping
]
assert not missing, f"channels missing from REVE's position bank: {missing}"
n_total = sum(p.numel() for p in model.parameters())
print(
    f"REVE-Base: {n_total / 1e6:.1f}M parameters | positions resolved: {tuple(model.default_pos.shape)}"
)
# The head flattens every token: 15 channels x 4 patches (200-sample patches with
# 20 overlap across 800 samples) x 512 dims = 30,720 features -> 2 classes.
print("head:", model.final_layer)

# %%
# Two ways to adapt the encoder
# -----------------------------
# *Linear probe*: freeze the encoder and train only ``final_layer``. Cheap,
# stable with little data, and a fair test of what the pretrained features
# already contain.
#
# *Full fine-tuning*: unfreeze everything and continue from the probed model at
# a lower learning rate. More capacity, more data needed, and it overfits fast
# on a few hundred windows -- which is why both use early stopping on run 3.
#
# Both share one ``EEGClassifier`` recipe. A note on the learning rate: REVE's
# paper probes at 1e-3, but with this 30,720-wide head Adam saturates at 1e-3
# and predicts one class for the first epochs -- it looks like a broken model.
# 1e-4 behaves from epoch 1.


def set_trainable(model, head_only):
    for name, p in model.named_parameters():
        p.requires_grad = (not head_only) or name.startswith("final_layer")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"trainable parameters: {n_train:,} of {n_total:,} ({'head only' if head_only else 'all'})"
    )


def make_classifier(model, lr, patience=5):
    stop = EarlyStopping(
        monitor="valid_balanced_accuracy",
        lower_is_better=False,
        patience=patience,
        load_best=True,
    )
    return EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=1e-2,
        train_split=predefined_split(valid_set),  # run 3 is scored after every epoch
        batch_size=32,
        callbacks=["balanced_accuracy", stop],
        classes=[0, 1],
        device=DEVICE,
        iterator_train__shuffle=True,
    )


def score_on_test(clf):
    y_true = test_set.get_metadata()["target"].to_numpy()
    return balanced_accuracy_score(y_true, clf.predict(test_set))


def valid_curve(clf):
    return clf.history[:, "valid_balanced_accuracy"]


# %%
# Linear probe
# ------------
# Skorch prints one line per epoch: training loss, validation balanced
# accuracy on run 3, and timing. Early stopping ends the run five epochs after
# the last improvement and restores the best epoch's weights.

results, curves = {}, {}
set_trainable(model, head_only=True)
probe = make_classifier(model, lr=1e-4)
probe.fit(train_set, y=None, epochs=30)
results["linear probe"] = score_on_test(probe)
curves["linear probe"] = valid_curve(probe)
print(
    f"\nlinear probe: test balanced accuracy on run 4 = {results['linear probe']:.3f}"
)

# %%
# Full fine-tuning (GPU)
# ----------------------
# Continue from the probed model with every parameter trainable, for a few
# epochs at the same learning rate. On CPU this cell only reports what it would
# do; on a GPU it runs (about 1.5 s per epoch on a V100 for this participant).

if RUN_FINETUNE:
    model = probe.module_
    set_trainable(model, head_only=False)
    finetune = make_classifier(model, lr=1e-4, patience=2)
    finetune.fit(train_set, y=None, epochs=FT_EPOCHS)
    results["fine-tune"] = score_on_test(finetune)
    curves["fine-tune"] = valid_curve(finetune)
    print(
        f"\nfull fine-tune: test balanced accuracy on run 4 = {results['fine-tune']:.3f}"
    )
else:
    print(
        "Skipped on CPU. Same recipe with set_trainable(model, head_only=False), lr 1e-4,",
        f"{FT_EPOCHS} epochs, early stopping; set REVE_TUTORIAL_FINETUNE=1 to run it here.",
    )

# %%
# Result
# ------
# Left: validation balanced accuracy per epoch, with the epoch whose weights were
# kept. Right: the number that matters, balanced accuracy on run 4, scored once.

fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(10, 3.6))
for name, curve in curves.items():
    epochs = np.arange(1, len(curve) + 1)
    ax_curve.plot(epochs, curve, marker="o", ms=3, label=name)
    ax_curve.scatter(
        epochs[np.argmax(curve)],
        max(curve),
        s=90,
        facecolors="none",
        edgecolors="k",
        zorder=3,
    )
ax_curve.axhline(0.5, color="grey", ls=":", lw=1)
ax_curve.set(
    xlabel="epoch",
    ylabel="validation balanced accuracy (run 3)",
    title="early stopping on run 3",
)
ax_curve.legend()
ax_bar.bar(
    list(results), list(results.values()), color=["#4a8", "#c44"][: len(results)]
)
ax_bar.axhline(0.5, color="grey", ls=":", lw=1)
ax_bar.set(
    ylim=(0.4, 1.0),
    ylabel="test balanced accuracy (run 4)",
    title=f"participant {SUBJECT}, 40 test trials",
)
fig.tight_layout()

print("Summary -- balanced accuracy on run 4 (chance = 0.50):")
for name, acc in results.items():
    print(f"  {name:14s} {acc:.3f}")
plt.show()

# %%
# Read the result and design the next experiment
# ----------------------------------------------
# * **This is one participant and 40 test trials.** One trial is 2.5 points of
#   balanced accuracy, so the standard error is about 8 points. Differences of
#   that size are noise; report participants, not windows.
# * **Cross-subject is the real question.** The same recipe, looped over the ten
#   participants of ``ds003810`` on a GPU node (train on eight, stop on a ninth,
#   test on the tenth, 40 max epochs, patience 5), gives REVE-Base a
#   leave-one-subject-out mean of **0.72 +- 0.08** balanced accuracy (range
#   0.57-0.81; about 50 s per fold on a V100). A single favourable split had
#   scored 0.825 -- one more reason to average over participants.
# * **More encoder is not more accuracy.** REVE-Large (390M parameters) probed on
#   the same 1,270 training windows scored *lower* (0.688 on the same split):
#   it memorised them. Early stopping is not optional at this data size.
# * **Full fine-tuning needs thousands of labelled trials** to beat the probe
#   reliably; the paper's benchmark for that is BCI-IV-2a via MOABB (0.64
#   fine-tuned vs 0.52 probed with REVE-Base).
#
# References
# ----------
# El Ouahidi et al. (2025). REVE: A Foundation Model for EEG -- Adapting to Any
# Setup with Large-Scale Pretraining on 25,000 Subjects. NeurIPS 2025.
# https://arxiv.org/abs/2510.21585
#
# Peterson et al. (2021). Motor Imagery vs Rest -- Low-Cost EEG System. OpenNeuro
# ds003810. https://openneuro.org/datasets/ds003810
