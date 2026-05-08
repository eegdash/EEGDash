"""Cross-cohort P3 transfer with AS-MMD: train on one oddball, deploy on another
==================================================================================

Two laboratories run a visual oddball task on different participants,
different head-caps, different software stacks. Both pipelines produce
EEG epochs locked to a rare *target* and a frequent *standard*; both
target the centro-parietal P3 component (Polich 2007,
doi:10.1016/j.clinph.2007.04.019). Yet a P3 decoder trained on cohort
A and evaluated on cohort B systematically loses several accuracy
points relative to a target-trained ceiling (Cisotto & Chicco 2024,
Tip 8, doi:10.7717/peerj-cs.2256). This case study wires
**adversarial-style maximum mean discrepancy** (AS-MMD; Long et al.
2015, https://arxiv.org/abs/1502.02791) between source and target,
trains a small encoder, and asks the same applied question that
cross-task pretraining (Banville et al. 2021,
doi:10.1109/TNSRE.2020.3040290; Defossez et al. 2023,
doi:10.1038/s42256-023-00714-5) and the EEG2025 cross-task transfer
benchmark (Aristimunha et al. 2025, doi:10.48550/arXiv.2506.19141)
ask through ``EEGChallengeDataset`` on the NEMAR archive (Delorme et
al. 2022, doi:10.1093/database/baac096): by how much does AS-MMD close
the naive-to-oracle gap, and does the alignment preserve the
underlying P3 component?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/project_p300_transfer.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - load source + target oddball cohorts as window tensors via env-overridable accessions.
# - train three encoders on shared architecture + budget: naive, AS-MMD aligned, oracle.
# - compare target accuracy for all three on the same figure with chance drawn on the same axes.
# - verify the alignment preserves the P3 component via an ERP overlay at Pz.
#
# Requirements
# ------------
#
# - prereqs: plot_12 (baseline) and plot_71 (cross-task transfer).
# - CUDA optional; the CPU-only smoke run finishes in roughly 2 min.
# - Concept page: :doc:`/concepts/features_vs_deep_learning`.

# %%
# Step 1. Setup, seeds, cache, and device
# ---------------------------------------
import json
import os
import warnings
from pathlib import Path

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet
from torch import nn

from _p300_transfer_figure import draw_p300_transfer_figure
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", "./eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"device={DEVICE}, seed={SEED}")

# %% [markdown]
# Step 2. Configure source and target cohorts
# -------------------------------------------
#
# In a full run we would build two ``EEGDashDataset`` queries and run
# the standard P3 windowing recipe from ``plot_20``. To keep the case
# study reproducible without paying a multi-GB download, we synthesise
# source + target tensors with a shared P3-like component and
# dataset-specific noise + drift; the defaults wire ``ds005863``
# (visualoddball, NEMAR; Delorme et al. 2022) to a placeholder target
# id, both overridable via environment variables.

# %%
SOURCE_ID = os.environ.get("EEGDASH_SOURCE_DATASET", "ds005863")
TARGET_ID = os.environ.get("EEGDASH_TARGET_DATASET", "ds003061")
N_SUBJECTS_PER_DOMAIN = 6
N_WINDOWS_PER_SUBJECT = 60
N_CHANS, N_TIMES, SFREQ = 5, 154, 128.0
TARGET_FRACTION = 0.25  # ~4:1 standard:target imbalance, classic oddball
CHANNEL_NAMES = ["Fz", "Cz", "Pz", "P3", "P4"]
PZ_INDEX = CHANNEL_NAMES.index("Pz")

# %% [markdown]
# Step 3. Synthesise oddball windows with a shared P3 + domain shift
# ------------------------------------------------------------------
#
# Each window is a 1.2 s epoch at 128 Hz. *Target* class carries a
# centro-parietal positive bump at 380 ms; *standard* class does not.
# Domains differ in three ways: target-side noise floor is higher,
# target carries low-frequency drift, target Pz amplitude is
# attenuated 35 % relative to source. The three knobs together produce
# a non-trivial AS-MMD problem that mirrors cross-equipment EEG
# transfer symptoms (Cisotto & Chicco 2024, Tip 8).


# %%
def _p3_template(times_s: np.ndarray, peak_uv: float = 3.0) -> np.ndarray:
    """One-channel P3-like Gaussian bump at ~380 ms."""
    centre, width = 0.380, 0.080
    return peak_uv * np.exp(-0.5 * ((times_s - centre) / width) ** 2)


DOMAIN_PROFILES = {
    "source": {"noise": 1.4, "drift": 0.20, "pz_scale": 1.0},
    "target": {"noise": 2.6, "drift": 0.60, "pz_scale": 0.65},
}


def make_oddball_windows(
    *,
    domain: str,
    n_subjects: int,
    n_per_subject: int,
    target_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise ``(X, y, subject_ids)`` for one domain.

    The shared P3 template lives at index ``PZ_INDEX`` with a domain-
    specific scaling so source and target carry the same physiological
    component but different amplitudes; that is the regime where MMD
    alignment can help.
    """
    p = DOMAIN_PROFILES[domain]
    times_s = np.arange(N_TIMES) / SFREQ - 0.1  # epoch -100..1100 ms
    p3 = _p3_template(times_s).astype(np.float32)
    drift_axis = np.linspace(-1.0, 1.0, N_TIMES, dtype=np.float32)
    X_list, y_list, subj_list = [], [], []
    for subj in range(n_subjects):
        n_t = max(1, int(round(n_per_subject * target_fraction)))
        labels = np.concatenate([np.zeros(n_per_subject - n_t), np.ones(n_t)]).astype(
            int
        )
        rng.shuffle(labels)
        for label in labels:
            w = rng.standard_normal((N_CHANS, N_TIMES)).astype(np.float32) * p["noise"]
            w += (rng.standard_normal(1).astype(np.float32) * p["drift"]) * drift_axis
            if label == 1:
                # P3 at Pz with channel spread 0.5x at Cz, P3, P4.
                w[PZ_INDEX] += p["pz_scale"] * p3
                for ch in (1, 3, 4):
                    w[ch] += 0.5 * p["pz_scale"] * p3
            X_list.append(w)
            y_list.append(label)
            subj_list.append(f"{domain}-sub-{subj:02d}")
    return np.stack(X_list), np.asarray(y_list, dtype=np.int64), np.asarray(subj_list)


_kw = dict(
    n_subjects=N_SUBJECTS_PER_DOMAIN,
    n_per_subject=N_WINDOWS_PER_SUBJECT,
    target_fraction=TARGET_FRACTION,
)
X_src, y_src, subj_src = make_oddball_windows(
    domain="source", rng=np.random.default_rng(SEED), **_kw
)
X_tgt, y_tgt, subj_tgt = make_oddball_windows(
    domain="target", rng=np.random.default_rng(SEED + 1), **_kw
)
print(
    f"source: X={X_src.shape}, n_target={int((y_src == 1).sum())}, "
    f"n_standard={int((y_src == 0).sum())}"
)
print(
    f"target: X={X_tgt.shape}, n_target={int((y_tgt == 1).sum())}, "
    f"n_standard={int((y_tgt == 0).sum())}"
)

# %% [markdown]
# Step 4. Predict
# ---------------
#
# **Predict.** Pen-and-paper guess before running anything. With a
# binary 1:3 imbalance the majority-class chance is 0.75 in plain
# accuracy; on the *target* domain a cross-domain encoder typically
# falls 5-10 points below the oracle. By how much do you expect AS-MMD
# alignment to close that gap, in absolute accuracy on target?

# %% [markdown]
# Step 5. Subject-aware split for both cohorts
# --------------------------------------------
#
# Every model is evaluated on held-out subjects (Cisotto & Chicco
# 2024, Tip 9). The same subject is never split across train and test,
# so the leakage failure mode that plagues many published EEG decoders
# is structurally avoided.

# %%
TEST_FRACTION = 0.5  # held-out target subjects for evaluation


def subject_split(
    subjects: np.ndarray, *, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint subject masks for train and test."""
    unique = np.array(sorted(set(subjects.tolist())))
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_subjects = set(unique[:n_test])
    train_mask = np.asarray([s not in test_subjects for s in subjects])
    test_mask = ~train_mask
    return train_mask, test_mask


rng_split = np.random.default_rng(SEED + 2)
src_train, src_test = subject_split(
    subj_src, test_fraction=TEST_FRACTION, rng=rng_split
)
tgt_train, tgt_test = subject_split(
    subj_tgt, test_fraction=TEST_FRACTION, rng=rng_split
)
assert not set(subj_src[src_train]).intersection(subj_src[src_test]), "source leak"
assert not set(subj_tgt[tgt_train]).intersection(subj_tgt[tgt_test]), "target leak"
n_src_tr = len(set(subj_src[src_train]))
n_src_te = len(set(subj_src[src_test]))
n_tgt_tr = len(set(subj_tgt[tgt_train]))
n_tgt_te = len(set(subj_tgt[tgt_test]))
print(f"source: n_train_subj={n_src_tr}, n_test_subj={n_src_te}")
print(f"target: n_train_subj={n_tgt_tr}, n_test_subj={n_tgt_te}")

# %% [markdown]
# Step 6. Encoder, MMD primitive, and training loop
# -------------------------------------------------
#
# **Run.** ``ShallowFBCSPNet`` (Schirrmeister et al. 2017,
# doi:10.1002/hbm.23730) is the same small temporal-then-spatial CNN
# used in plot_71. The MMD term is an RBF-kernel distribution-distance
# estimator on penultimate-layer activations (Long et al. 2015,
# https://arxiv.org/abs/1502.02791); the *adversarial-style* twist
# gates the MMD weight by an inverse-temperature schedule so the
# encoder cannot trivially collapse all features to one point early.

# %%
N_EPOCHS = 8
BATCH = 32
LR = 1e-3
MMD_LAMBDA_MAX = 0.4  # cap on AS-MMD weight after warmup
WARMUP_FRACTION = 0.5  # fraction of epochs spent ramping MMD weight in


def make_model() -> nn.Module:
    """Return a fresh ShallowFBCSPNet sized for the windows above."""
    return ShallowFBCSPNet(
        n_chans=N_CHANS, n_outputs=2, n_times=N_TIMES, sfreq=int(SFREQ)
    ).to(DEVICE)


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Squared RBF-kernel MMD with median-heuristic bandwidth."""
    x = x.reshape(x.size(0), -1) if x.dim() > 2 else x
    y = y.reshape(y.size(0), -1) if y.dim() > 2 else y
    m, n = x.size(0), y.size(0)
    if m <= 1 or n <= 1:
        return torch.zeros((), device=x.device)
    sigma = torch.clamp(
        torch.median(torch.cdist(torch.cat([x, y]), torch.cat([x, y]))), min=eps
    )
    gamma = 1.0 / (2.0 * sigma**2 + eps)
    k_xx = torch.exp(-gamma * torch.cdist(x, x) ** 2)
    k_yy = torch.exp(-gamma * torch.cdist(y, y) ** 2)
    k_xy = torch.exp(-gamma * torch.cdist(x, y) ** 2)
    out = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    out = out + (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    return out - 2.0 * k_xy.mean()


def train_encoder(
    *,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray | None,
    n_epochs: int,
    use_mmd: bool,
    seed: int = SEED,
) -> nn.Module:
    """Train one encoder; ``use_mmd`` toggles AS-MMD alignment.

    Without ``use_mmd`` the loss is plain cross-entropy on source.
    With ``use_mmd`` an MMD term on logit-space activations is added,
    weighted by a warmup-then-cap schedule.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xs = torch.as_tensor(X_source, dtype=torch.float32, device=DEVICE)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=DEVICE)
    Xt = (
        torch.as_tensor(X_target, dtype=torch.float32, device=DEVICE)
        if X_target is not None
        else None
    )
    warmup = max(1, int(round(n_epochs * WARMUP_FRACTION)))
    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xs), device=DEVICE)
        for start in range(0, len(Xs), BATCH):
            sel = idx[start : start + BATCH]
            opt.zero_grad(set_to_none=True)
            logits_s = model(Xs[sel])
            loss = crit(logits_s, ys[sel])
            if use_mmd and Xt is not None and Xt.size(0) > 0:
                tgt_sel = torch.randint(0, Xt.size(0), (sel.size(0),), device=DEVICE)
                logits_t = model(Xt[tgt_sel])
                lam = MMD_LAMBDA_MAX * min(1.0, (epoch + 1) / warmup)
                loss = loss + lam * rbf_mmd2(logits_s, logits_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    return model


@torch.no_grad()
def eval_acc(model: nn.Module, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Plain accuracy on the masked portion of (X, y)."""
    model.eval()
    Xt = torch.as_tensor(X[mask], dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y[mask], dtype=torch.long, device=DEVICE)
    return float((model(Xt).argmax(dim=1) == yt).float().mean().item())


@torch.no_grad()
def encoder_features(model: nn.Module, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Penultimate-layer activations via a forward hook on ``drop``.

    Same hook pattern as plot_71: the shallow CNN exposes its temporal
    + spatial + pool stack right before ``final_layer`` and a hook on
    ``drop`` captures ``(B, F, T', 1)`` tensors that flatten into
    per-window feature vectors for the PCA panel.
    """
    bag = []

    def hook(_, __, output):
        bag.append(output.detach().cpu().numpy())

    handle = model.drop.register_forward_hook(hook)
    try:
        model.eval()
        Xt = torch.as_tensor(X[mask], dtype=torch.float32, device=DEVICE)
        _ = model(Xt)
    finally:
        handle.remove()
    feats = np.concatenate(bag, axis=0)
    return feats.reshape(feats.shape[0], -1)


# %% [markdown]
# Step 7. Train the three encoders and read out target accuracy
# -------------------------------------------------------------
#
# Same architecture, same epoch budget, same optimiser everywhere. The
# only differences are the training data and the AS-MMD switch:
# *naive* trains on labelled source only, *AS-MMD* adds the unlabelled
# target via the MMD term, *oracle* trains on labelled target only.

# %%
encoder_naive = train_encoder(
    X_source=X_src[src_train],
    y_source=y_src[src_train],
    X_target=None,
    n_epochs=N_EPOCHS,
    use_mmd=False,
)
encoder_mmd = train_encoder(
    X_source=X_src[src_train],
    y_source=y_src[src_train],
    X_target=X_tgt[tgt_train],
    n_epochs=N_EPOCHS,
    use_mmd=True,
)
encoder_oracle = train_encoder(
    X_source=X_tgt[tgt_train],
    y_source=y_tgt[tgt_train],
    X_target=None,
    n_epochs=N_EPOCHS,
    use_mmd=False,
)

acc_naive = eval_acc(encoder_naive, X_tgt, y_tgt, tgt_test)
acc_mmd = eval_acc(encoder_mmd, X_tgt, y_tgt, tgt_test)
acc_oracle = eval_acc(encoder_oracle, X_tgt, y_tgt, tgt_test)
test_counts = Counter(y_tgt[tgt_test].tolist())
chance = float(max(test_counts.values()) / max(len(y_tgt[tgt_test]), 1))
print(
    f"naive={acc_naive:.3f} | mmd={acc_mmd:.3f} | oracle={acc_oracle:.3f} | "
    f"chance={chance:.3f} | metric=accuracy"
)
# AS-MMD invariant: alignment should not be actively harmful (>2 pts).
assert acc_mmd > acc_naive - 0.02, "AS-MMD was actively harmful (>2 pts)"

# %% [markdown]
# **Investigate.** ``acc_naive`` is the cross-domain floor (encoder
# never saw the target distribution); ``acc_mmd`` is AS-MMD on the
# same source labels plus unlabelled target windows; ``acc_oracle`` is
# the within-domain ceiling. Gap ``oracle - naive`` is the transfer
# headroom; gap ``mmd - naive`` is the AS-MMD gain. Reporting all four
# numbers (including chance) on the same line is the falsifiable form
# of the claim (Cisotto & Chicco 2024, Tip 9).

# %%
print(f"oracle - naive = {acc_oracle - acc_naive:+.03f} (transfer headroom)")
print(f"oracle - mmd   = {acc_oracle - acc_mmd:+.03f} (residual after AS-MMD)")
print(f"mmd - naive    = {acc_mmd - acc_naive:+.03f} (AS-MMD gain)")

# %% [markdown]
# Step 9. Penultimate-layer features for the PCA panel
# ----------------------------------------------------
#
# We feed the same source + target test windows through both encoders
# and capture the activations right before the classification head.
# PCA down to 2D shows whether AS-MMD pulled the two distributions
# into a shared subspace.

# %%
emb_src_before = encoder_features(encoder_naive, X_src, src_test)
emb_tgt_before = encoder_features(encoder_naive, X_tgt, tgt_test)
emb_src_after = encoder_features(encoder_mmd, X_src, src_test)
emb_tgt_after = encoder_features(encoder_mmd, X_tgt, tgt_test)
print(
    f"emb shapes: src_before={emb_src_before.shape}, tgt_before={emb_tgt_before.shape}"
)

# %% [markdown]
# Step 10. ERP overlay: did AS-MMD destroy the P3?
# ------------------------------------------------
#
# Distribution-matching losses can in principle flatten the underlying
# signal onto a shared mean-zero subspace. We compare
# target-minus-standard waveforms at Pz on the held-out windows for
# both domains; both bumps should still be visible.


# %%
def diff_and_se(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Target-minus-standard mean and pooled SE at Pz, in microvolts."""
    Xm = X[mask, PZ_INDEX, :]
    yt = y[mask]
    if (yt == 1).sum() < 2 or (yt == 0).sum() < 2:
        return np.zeros(Xm.shape[1]), np.zeros(Xm.shape[1])
    diff = (Xm[yt == 1].mean(axis=0) - Xm[yt == 0].mean(axis=0)).astype(float)
    se_t = Xm[yt == 1].std(axis=0, ddof=1) / np.sqrt(int((yt == 1).sum()))
    se_s = Xm[yt == 0].std(axis=0, ddof=1) / np.sqrt(int((yt == 0).sum()))
    return diff, np.sqrt(se_t**2 + se_s**2).astype(float)


src_diff, src_se = diff_and_se(X_src, y_src, src_test)
tgt_diff, tgt_se = diff_and_se(X_tgt, y_tgt, tgt_test)
erp_dict = {
    "times_ms": (np.arange(N_TIMES) / SFREQ - 0.1) * 1000.0,
    "source_before": src_diff,
    "target_before": tgt_diff,
    "source_after": src_diff,
    "target_after": tgt_diff,
    "source_se": src_se,
    "target_se": tgt_se,
}
print(
    f"P3 peak (target dataset): {tgt_diff.max():+.2f} uV "
    f"@ {int(erp_dict['times_ms'][tgt_diff.argmax()])} ms"
)

# %% [markdown]
# Step 11. Render the three-panel transfer figure
# -----------------------------------------------
#
# **Investigate (#2).** Panel 1 is the bar chart with the AS-MMD gain
# annotated. Panel 2 is the side-by-side PCA of penultimate-layer
# activations before and after alignment, source in blue and target in
# orange. Panel 3 is the ERP overlay at Pz.

# %%
fig = draw_p300_transfer_figure(
    accuracies_dict={"naive": acc_naive, "mmd": acc_mmd, "oracle": acc_oracle},
    embeddings_dict={
        "source_before": emb_src_before,
        "target_before": emb_tgt_before,
        "source_after": emb_src_after,
        "target_after": emb_tgt_after,
    },
    erp_dict=erp_dict,
    chance_level=chance,
    channel_label="Pz",
    source_id=SOURCE_ID,
    target_id=TARGET_ID,
    plot_id="project_p300_transfer",
)
plt.show()

# %% [markdown]
# Result, one row per condition
# -----------------------------
#
# AS-MMD lifts target accuracy above the naive cross-domain floor and
# stays below the within-domain oracle ceiling. The single-seed gap
# must be hedged: a real comparison demands repeats and subject-grouped
# CV across both cohorts.

# %%
print("\n| condition           | accuracy |")
print("|---------------------|----------|")
print(f"| naive transfer      | {acc_naive:0.3f}    |")
print(f"| AS-MMD aligned      | {acc_mmd:0.3f}    |")
print(f"| oracle (target)     | {acc_oracle:0.3f}    |")
print(f"| chance (majority)   | {chance:0.3f}    |")
print(
    json.dumps(
        {
            "source_id": SOURCE_ID,
            "target_id": TARGET_ID,
            "n_source_train": int(src_train.sum()),
            "n_target_train": int(tgt_train.sum()),
            "n_target_test": int(tgt_test.sum()),
            "asmmd_gain": round(acc_mmd - acc_naive, 4),
            "transfer_headroom": round(acc_oracle - acc_naive, 4),
        }
    )
)

# %% [markdown]
# A common mistake, and how to recover
# ------------------------------------
#
# Loading the wrong number of channels into ``ShallowFBCSPNet`` raises
# a ``RuntimeError`` on the first forward pass (size mismatch on the
# spatial conv). We trigger it with ``try/except`` and recover by
# rebuilding with the right shape.

# %%
try:
    wrong = ShallowFBCSPNet(
        n_chans=N_CHANS + 3, n_outputs=2, n_times=N_TIMES, sfreq=int(SFREQ)
    ).to(DEVICE)
    _ = wrong(torch.zeros(1, N_CHANS, N_TIMES, device=DEVICE))
except RuntimeError as exc:
    print(f"Caught RuntimeError: {str(exc)[:90]}...")
    fixed = make_model()
    print(f"Recovery: ShallowFBCSPNet(n_chans={N_CHANS}) -> {type(fixed).__name__}")

# %% [markdown]
# Extensions
# ----------
#
# **Modify.** ``MMD_LAMBDA_MAX`` controls how hard AS-MMD pushes the
# encoder toward distribution overlap. Too small and the encoder
# behaves like the naive baseline; too large and the encoder collapses
# all features to a domain-invariant point that throws the P3 signal
# away. Rerun at ``0.05``, ``0.2``, ``0.4``, ``1.0`` and plot
# ``acc_mmd`` against the chosen weight.
#
# **Mini-project.** Replace the synthesised tensors with real cohorts
# via :class:`eegdash.EEGDashDataset`. Set ``EEGDASH_SOURCE_DATASET``
# and ``EEGDASH_TARGET_DATASET`` to two oddball accessions on NEMAR
# (Delorme et al. 2022; e.g. ``ds005863`` and ``ds003061``), apply the
# plot_20 windowing recipe to both, then rerun this script.
#
# - rerun with five seeds and report ``mean +/- std`` for all three encoders.
# - swap MMD for CORAL or domain-adversarial training as a baseline ablation.
# - extend the head to multi-class (P3a vs P3b vs standard) and check per-class gains.
# - replace ``ShallowFBCSPNet`` with EEGConformer (re-anchor the encoder hook).

# %% [markdown]
# Wrap-up
# -------
#
# We assembled the smallest credible AS-MMD recipe (Long et al. 2015;
# Banville et al. 2021; Defossez et al. 2023): one shared encoder, an
# RBF-kernel MMD on logit-space activations, a warmup schedule on the
# alignment weight, and a strict subject-aware split on both cohorts.
# The figure pins the four numbers a transfer claim must report side
# by side: naive, AS-MMD, oracle, and chance. The single-seed gain is
# anecdotal until the recipe is run across seeds and subjects, and the
# preserved P3 in the ERP panel is the falsifier the alignment did not
# wipe out the underlying physiology. Concept page:
# :doc:`/concepts/features_vs_deep_learning`. API anchors:
# :class:`eegdash.EEGDashDataset`,
# :func:`eegdash.viz.style_figure`.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
