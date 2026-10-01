"""How do I fine-tune a published EEG foundation model (REVE)?
=============================================================

**Difficulty 3** | **Compute: CPU; optional accelerator**
Runtime depends on hardware; full fine-tuning is an explicit opt-in.

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
rate -- explicitly opt in before running). Runs 1 and 2 train, run 3 chooses when
to stop, and run 4 is scored once per prespecified regime. The small budget demonstrates
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
# The weights are published under a gated license (the electrode position table,
# ``brain-bzh/reve-positions``, is public). One-time setup:
#
# 1. Open https://huggingface.co/brain-bzh/reve-base and accept the terms.
# 2. Log in once from a terminal: ``hf auth login`` (or set ``HF_TOKEN``).
#
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from eegdash.paths import get_default_cache_dir

from eegdash import EEGDashDataset

MODEL_ID = "brain-bzh/reve-base"
MODEL_REVISION = (
    "fa9a2163a4b7c0a42c8e28b56077ef9c368944dc"  # pinned published checkpoint
)
DATASET, SUBJECT, RUNS = "ds003810", "02", ["1", "2", "3", "4"]
SFREQ = 200  # REVE was pretrained at 200 Hz; it does not check, so we must
WINDOW_S = 4  # cue -> end of trial in this paradigm
CACHE_DIR = get_default_cache_dir()
# Changing hardware never changes which experiment runs.
DEVICE = "cpu"  # explicitly choose "cuda" or "mps" when available
RUN_FINETUNE = False  # opt in before inspecting any test results
PROBE_EPOCHS, FT_EPOCHS = 5, 2  # small workflow budget, not converged performance
PROBE_LR, FINETUNE_LR = 1e-4, 1e-5
torch.manual_seed(0)
print(f"device={DEVICE}; probe epochs={PROBE_EPOCHS}; fine-tune={RUN_FINETUNE}")
# from_pretrained reports gated access/download errors directly. Complete the
# one-time login above before running; no exception is silently treated as offline.

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
    """Per-channel z-score over time, clipped at ``clip`` SD (REVE's pretraining normalisation).

    ``x`` is the whole (channels, times) array -- braindecode's ``apply_on_array`` is not
    channel-wise by default -- so reduce over the time axis explicitly.
    """
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
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
# runs 1-2 train, run 3 selects the epoch, run 4 is scored once per prespecified
# regime and never used to select settings.
#
# This is within-person, across-run evaluation, not generalization to new people.
# Complete-recording filtering and normalization also use each held-out run's
# unlabeled signal: an offline protocol, not causal online prediction.
# Verify that this dataset/participant was absent from the checkpoint pretraining
# corpus before claiming a contamination-free transfer estimate; it is not checked here.

by_run = windows.split("run")
train_set = BaseConcatDataset([by_run["1"], by_run["2"]])
valid_set, test_set = by_run["3"], by_run["4"]
print(f"train {len(train_set)} windows | valid {len(valid_set)} | test {len(test_set)}")
counts = windows.get_metadata().groupby(["run", "target"]).size().unstack(fill_value=0)
counts.plot.bar(stacked=True, figsize=(7, 3))
plt.ylabel("Retained trials")
plt.title("Runs 1–2 train; run 3 validation; run 4 test (0 imagery, 1 rest)")
plt.tight_layout()
plt.show()

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
    revision=MODEL_REVISION,
    n_outputs=2,
    n_chans=len(chs_info),
    n_times=WINDOW_S * SFREQ,
    sfreq=SFREQ,
    chs_info=chs_info,
)
# Public resolved positions must contain one finite xyz coordinate per input channel.
positions = model.get_positions([channel["ch_name"] for channel in chs_info])
if positions.shape[-2:] != (len(chs_info), 3) or not torch.isfinite(positions).all():
    raise ValueError(
        "REVE did not resolve all input channels to finite xyz positions; inspect chs_info."
    )
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
# Both use fixed, explicit rates: 1e-4 for the probe and 1e-5 for continuation.
# Fine-tuning starts from the validation-selected probe, so this is sequential
# adaptation, not two independently initialized regimes or equal-compute training.


def set_trainable(model, head_only):
    # Freezing only touches gradients. That is enough for REVE: it has no dropout or
    # batch-norm, so the frozen encoder returns the same features in train and eval mode.
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
probe = make_classifier(model, lr=PROBE_LR)
probe.fit(train_set, y=None, epochs=PROBE_EPOCHS)
results["linear probe"] = score_on_test(probe)
curves["linear probe"] = valid_curve(probe)
print(
    f"\nlinear probe: test balanced accuracy on run 4 = {results['linear probe']:.3f}"
)

# %%
# Optional full fine-tuning
# -------------------------
# Continue from the selected probe at a tenfold smaller learning rate. Enable
# RUN_FINETUNE in the parameter cell before scoring either regime.

if RUN_FINETUNE:
    model = probe.module_
    set_trainable(model, head_only=False)
    finetune = make_classifier(model, lr=FINETUNE_LR, patience=2)
    finetune.fit(train_set, y=None, epochs=FT_EPOCHS)
    results["fine-tune"] = score_on_test(finetune)
    curves["fine-tune"] = valid_curve(finetune)
    print(
        f"\nfull fine-tune: test balanced accuracy on run 4 = {results['fine-tune']:.3f}"
    )
else:
    print(
        "Probe-only protocol; set RUN_FINETUNE=True before running for sequential adaptation."
    )

# %%
# Result
# ------
# Left: validation balanced accuracy per epoch, with the epoch whose weights were
# kept. Right: balanced accuracy on run 4, scored once per prespecified regime.

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
    ylim=(0.0, 1.0),
    ylabel="test balanced accuracy (run 4)",
    title=f"participant {SUBJECT}, {len(test_set)} retained test trials",
)
fig.tight_layout()

print("Summary -- balanced accuracy on run 4 (chance = 0.50):")
for name, acc in results.items():
    print(f"  {name:14s} {acc:.3f}")
plt.show()

# %%
# Read the result and design the next experiment
# ----------------------------------------------
# Either regime may be better on this participant. The test runs do not select
# the regime, epoch budget, rate or checkpoint. No external benchmark result is
# reproduced here, and test-trial variation is not population uncertainty.
# To study new-person transfer, prespecify participant-disjoint train/validation/
# test cohorts and repeated seeds, audit pretraining overlap, and retain a
# simple baseline. This five-epoch workflow is not evidence of convergence.
#
# References
# ----------
# El Ouahidi et al. (2025). REVE: A Foundation Model for EEG -- Adapting to Any
# Setup with Large-Scale Pretraining on 25,000 Subjects. NeurIPS 2025.
# https://arxiv.org/abs/2510.21585
#
# Peterson et al. (2021). Motor Imagery vs Rest -- Low-Cost EEG System. OpenNeuro
# ds003810. https://openneuro.org/datasets/ds003810
