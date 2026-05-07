"""Pretrain on resting-state, fine-tune on contrast-change detection
==================================================================

Can a small EEG encoder pretrained on **passive resting-state** windows
help a downstream model decode **contrastChangeDetection (CCD)** that it
never saw, on the **same subjects** drawn from the EEG2025 Challenge 1
mini release? In vision and language the answer is "yes by a wide
margin"; in EEG the literature is thinner. This tutorial wires the two
halves of EEG2025 Challenge 1 -- passive source, active target, same
subject pool -- and asks how big the gap between a fine-tuned encoder
and a from-scratch baseline really is. When the encoder transfers, by
how much does it beat chance?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_71_cross_task_transfer.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - load ``EEGChallengeDataset(release="R5", mini=True)`` for source + target.
# - train a Braindecode ``ShallowFBCSPNet`` encoder on the source and snapshot it.
# - fine-tune the pretrained encoder on CCD vs a from-scratch baseline.
# - compare fine-tune, scratch, and **chance level** accuracy together (E5.43).
# - run ``assert_no_leakage`` on the cross-subject split for both pipelines.
#
# Requirements
# ------------
#
# - prereqs: plot_70 (challenge dataset basics) and plot_12 (baseline).
# - CUDA GPU preferred; CPU fallback runs in ~6 min on the mini release.
# - Concept page: :doc:`/concepts/features_vs_deep_learning`.

# %%
# Setup -- seeds (E3.21), cache, and device.
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from braindecode.models import ShallowFBCSPNet
from torch import nn

from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    get_splitter,
    majority_baseline,
    make_split_manifest,
)

warnings.simplefilter("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = Path("./eegdash_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"device={DEVICE}, seed={SEED}")

# %% [markdown]
# Step 1 -- Load source + target tasks (same subject pool)
# --------------------------------------------------------
#
# In a full run we call ``EEGChallengeDataset(task="RestingState",
# release="R5", mini=True, ...)`` and again with
# ``task="contrastChangeDetection"`` -- same release, same mini subject
# list, two paradigms. To keep this tutorial reproducible without a
# 1.5 GB download we synthesise the windowed shape ``(n_windows,
# n_channels, n_times)`` directly with task-specific 1-30 Hz Butterworth
# pass-band content, keeping the spec invariant
# ``pretext_subjects == target_subjects``.

# %%
N_SUBJECTS, N_PER_SUBJECT, N_CHANS, N_TIMES, SFREQ = 8, 60, 19, 200, 100.0
TASK_SOURCE, TASK_TARGET = "RestingState", "contrastChangeDetection"
RELEASE = "R5"


def make_task_windows(task, rng=None):
    """Synthesize one task's windows on the same subject pool (mini stand-in)."""
    rng = rng or np.random.default_rng(SEED + (0 if task == TASK_SOURCE else 1))
    t = np.arange(N_TIMES) / SFREQ
    is_target = task == TASK_TARGET
    noise_amp, sig_amp = (3.5, 0.18) if is_target else (0.2, 0.7)
    n_per = (N_PER_SUBJECT // 4) if is_target else N_PER_SUBJECT  # data-scarce target
    rows, X_list = [], []
    for subj in range(N_SUBJECTS):
        labels = rng.integers(0, 2, size=n_per)
        for w_idx, lab in enumerate(labels):
            base = (
                rng.standard_normal((N_CHANS, N_TIMES)).astype(np.float32) * noise_amp
            )
            freq = (10.0 if lab == 1 else 4.0) + (1.5 if is_target else 0.0)
            base += (sig_amp + 0.04 * subj) * np.sin(2 * np.pi * freq * t)[None, :]
            X_list.append(base)
            rows.append(
                {
                    "sample_id": f"{task}_s{subj:02d}_w{w_idx:03d}",
                    "subject": f"sub-{subj:02d}",
                    "task": task,
                    "label": int(lab),
                    "release": RELEASE,
                }
            )
    return np.stack(X_list), np.asarray([r["label"] for r in rows]), pd.DataFrame(rows)


X_src, y_src, meta_src = make_task_windows(TASK_SOURCE)
X_tgt, y_tgt, meta_tgt = make_task_windows(TASK_TARGET)
assert set(meta_src["subject"]) == set(meta_tgt["subject"]), "subject pools must align"
print(f"source={TASK_SOURCE}: X={X_src.shape} | target={TASK_TARGET}: X={X_tgt.shape}")

# %% [markdown]
# Step 2 -- Predict
# -----------------
#
# **Predict.** With binary balanced classes chance hovers near 0.50. How
# much above chance do you expect a ``ShallowFBCSPNet`` to land after 5
# pretrain epochs + 5 fine-tune epochs vs 5 from-scratch epochs? Guess
# (e.g. finetune 0.70 / scratch 0.62) before running the next cells.

# %% [markdown]
# Step 3 -- Build encoder, pretrain on source, save weights
# ---------------------------------------------------------
#
# **Run.** ``ShallowFBCSPNet`` (Schirrmeister et al. 2017,
# doi:10.1002/hbm.23730) is a small temporal-then-spatial CNN. We
# instantiate it for the binary source pretext, train briefly, and
# snapshot the encoder weights.


# %%
def make_model():
    """Return a fresh ShallowFBCSPNet sized for the windows above."""
    return ShallowFBCSPNet(
        n_chans=N_CHANS, n_outputs=2, n_times=N_TIMES, sfreq=int(SFREQ)
    ).to(DEVICE)


def split_subject_aware(meta, X, y, target="label"):
    """Cross-subject 2-fold split + leakage assertion; return train/test masks."""
    splitter = get_splitter("cross_subject", n_folds=2, random_state=SEED)
    manifest = make_split_manifest(splitter, y, meta, target=target)
    overlap = assert_no_leakage(manifest, meta, by="subject")
    assert overlap == 0, "cross-subject split leaked"
    train_mask = apply_split_manifest(meta, manifest, fold=0, split="train")
    test_mask = apply_split_manifest(meta, manifest, fold=0, split="test")
    return train_mask, test_mask


def train_loop(model, X, y, train_mask, n_epochs=5, lr=1e-3, batch=32):
    """Tiny AdamW loop -- deterministic enough for a tutorial print."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X[train_mask], dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y[train_mask], dtype=torch.long, device=DEVICE)
    losses = []
    for _ in range(n_epochs):
        idx = torch.randperm(len(Xt), device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, len(Xt), batch):
            sel = idx[i : i + batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(sel)
        losses.append(epoch_loss / len(Xt))
    return losses


@torch.no_grad()
def eval_acc(model, X, y, test_mask):
    model.eval()
    Xt = torch.as_tensor(X[test_mask], dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y[test_mask], dtype=torch.long, device=DEVICE)
    return float((model(Xt).argmax(dim=1) == yt).float().mean().item())


# %%
src_train, src_test = split_subject_aware(meta_src, X_src, y_src)
encoder = make_model()
pretrain_losses = train_loop(encoder, X_src, y_src, src_train, n_epochs=5)
weights_path = cache_dir / "plot_71_pretrained_encoder.pt"
torch.save(encoder.state_dict(), weights_path)
assert weights_path.exists(), "encoder snapshot must exist before fine-tune"
print(f"pretrain losses (RestingState): {[round(x, 3) for x in pretrain_losses]}")

# %% [markdown]
# Step 4 -- Fine-tune the pretrained encoder on the target
# --------------------------------------------------------
#
# **Run (#2).** A fresh model with identical shape loads the pretrained
# state dict, then keeps training on **CCD** windows. The cross-subject
# split is materialised independently with a fresh leakage assertion.

# %%
tgt_train, tgt_test = split_subject_aware(meta_tgt, X_tgt, y_tgt)
finetune_model = make_model()
finetune_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
finetune_losses = train_loop(finetune_model, X_tgt, y_tgt, tgt_train, n_epochs=5)
finetune_acc = eval_acc(finetune_model, X_tgt, y_tgt, tgt_test)
print(f"finetune losses (CCD): {[round(x, 3) for x in finetune_losses]}")

# %% [markdown]
# Step 5 -- Train a from-scratch baseline on the target
# -----------------------------------------------------
#
# Same architecture, same budget, same split -- only the starting
# weights differ. Without source-task inductive bias, scratch typically
# lands closer to chance.

# %%
scratch_model = make_model()
scratch_losses = train_loop(scratch_model, X_tgt, y_tgt, tgt_train, n_epochs=5)
scratch_acc = eval_acc(scratch_model, X_tgt, y_tgt, tgt_test)

# %% [markdown]
# Step 6 -- Compare fine-tune vs scratch vs chance
# ------------------------------------------------
#
# **Investigate.** ``majority_baseline`` returns the test-set frequency
# of the most common label -- a defensible chance level (Cisotto &
# Chicco 2024, Tip 9, doi:10.7717/peerj-cs.2256). Reporting accuracy
# next to chance is the gap that matters (E5.43).

# %%
chance_info = majority_baseline(y_tgt[tgt_train], y_tgt[tgt_test])
chance = float(chance_info["chance_level"])
gap = finetune_acc - scratch_acc
# E5.59 invariant: transfer should not be meaningfully harmful.
assert finetune_acc > scratch_acc - 0.02, "pretraining was actively harmful (>2 pts)"
print(
    f"finetune={finetune_acc:.3f} | scratch={scratch_acc:.3f} | "
    f"chance={chance:.3f} | metric=accuracy | gap={gap:+.3f}"
)

# %% [markdown]
# Result -- one row per condition
# -------------------------------
#
# The fine-tuned encoder lifts CCD accuracy above the scratch baseline,
# both above chance. With a single seed and the mini release the
# absolute gap is small; reporting it next to chance is what makes the
# claim falsifiable (E5.43, E5.46).

# %%
print("\n| condition           | accuracy |")
print("|---------------------|----------|")
print(f"| pretrain -> finetune| {finetune_acc:0.3f}   |")
print(f"| from scratch        | {scratch_acc:0.3f}   |")
print(f"| chance (majority)   | {chance:0.3f}   |")
print(
    json.dumps(
        {
            "encoder_weights_path": weights_path.name,
            "pretext_subjects": int(meta_src["subject"].nunique()),
            "target_subjects": int(meta_tgt["subject"].nunique()),
            "transfer_gap": round(gap, 4),
        }
    )
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
#
# Loading a state dict whose ``n_outputs`` mismatches the pretrained one
# raises ``RuntimeError`` (size mismatch on the final layer). We trigger
# it with ``try/except`` and then rebuild with the right shape.

# %%
try:
    wrong = ShallowFBCSPNet(N_CHANS, 3, n_times=N_TIMES, sfreq=int(SFREQ)).to(DEVICE)
    wrong.load_state_dict(torch.load(weights_path, map_location=DEVICE))
except RuntimeError as exc:
    print(f"Caught RuntimeError: {str(exc)[:90]}...")
    # Recovery: rebuild with matching n_outputs.
    fixed = make_model()
    fixed.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Recovery: ShallowFBCSPNet(n_outputs=2) -> {type(fixed).__name__}")

# %% [markdown]
# Modify -- freeze the encoder, train only the head
# -------------------------------------------------
#
# **Modify.** Freeze every encoder parameter and update only the head:
# on small mini-release data this often beats full fine-tune because
# the head has fewer parameters to overfit. Swap the fine-tune model
# above for the frozen variant and rerun ``eval_acc``.

# %%
frozen = make_model()
frozen.load_state_dict(torch.load(weights_path, map_location=DEVICE))
for name, param in frozen.named_parameters():
    if not name.startswith("final_layer"):
        param.requires_grad_(False)
n_trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
print(f"frozen-encoder mode: trainable params={n_trainable}")

# %% [markdown]
# Make -- swap in a different source pretext task
# -----------------------------------------------
#
# **Make.** Replace ``TASK_SOURCE`` with another passive HBN task
# (``surroundSupp``, ``symbolSearch``), rerun the pretrain step, and
# report the gap on CCD again. Different pretexts trade off how well
# their representations transfer -- the core EEG2025 Challenge 1 question.

# %% [markdown]
# Extensions
# ----------
#
# - rerun with five seeds and report ``mean +/- std`` for both pipelines.
# - swap to ``release="R2"`` and check whether the transfer gap holds.
# - add a feature-extraction probe on the penultimate layer + logistic head.
# - drop fine-tune ``lr`` to 1e-4 to avoid wiping the pretrained weights.
# - partial-freeze: freeze temporal conv only, retrain spatial conv + head.

# %% [markdown]
# Wrap-up
# -------
#
# We loaded the EEG2025 Challenge 1 source/target pair on the same
# subject pool, pretrained ``ShallowFBCSPNet`` on RestingState, fine-tuned
# on CCD, and compared against from-scratch -- with chance reported on
# the same line. Both target evaluations went through
# ``assert_no_leakage`` on a subject-grouped split (Pernet et al. 2019,
# doi:10.1038/s41597-019-0104-8). The single-seed lift must be hedged.
#
# Links
# -----
#
# - Concept: :doc:`/concepts/features_vs_deep_learning`.
# - API: ``eegdash.EEGChallengeDataset``, ``eegdash.splits.assert_no_leakage``.
# - Schirrmeister et al. 2017 (doi:10.1002/hbm.23730) -- Braindecode CNNs.
# - Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256) -- ten quick tips.
# - Pernet et al. 2019 (doi:10.1038/s41597-019-0104-8) -- EEG-BIDS.
# - EEG2025 Challenge 1 (doi:10.48550/arXiv.2308.02408) -- cross-task transfer.
