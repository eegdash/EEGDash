"""Predict p-factor from EEG with a Braindecode model (deep-learning regime)
==============================================================================

This is the deep-learning case study for the p-factor regression
project. The companion script ``project_pfactor_features.py`` covers
the feature-based regime; this one trains
:class:`braindecode.models.EEGConformer` end-to-end on raw resting-state
windows from the Healthy Brain Network release ds005505 (Alexander et
al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme
et al. 2022, doi:10.1093/nargab/lqac023). The p-factor is a
transdiagnostic score from the Child Behavior Checklist (Caspi et al.
2014, doi:10.1177/2167702613497473) and the modelling contract is the
clinical-cautious one Cisotto and Chicco 2024
(doi:10.7717/peerj-cs.2256) ask for: cross-subject split, baseline
alongside score, no diagnostic claim. Three regimes shape the framing,
mirroring cousin tutorial plot_73: train from scratch, fine-tune a
pretrained Braindecode encoder (Schirrmeister et al. 2017,
doi:10.1002/hbm.23730), and read back where the network looks. The
deliverable is a 3-panel figure plus printed metrics. So can a small
EEGConformer beat the train-mean predictor on held-out subjects?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/project_pfactor_deep.png'

# %% [markdown]
# Learning objectives
# -------------------
# After this case study you will be able to:
#
# - load ds005505 with :class:`eegdash.EEGDashDataset` and surface p_factor.
# - build a strict cross-subject split before any windowing happens.
# - train :class:`braindecode.models.EEGConformer` end-to-end with AdamW.
# - read a 3-panel figure: curves, predicted-vs-true scatter, saliency.

# %% [markdown]
# Requirements
# ------------
# - **Estimated time**: ~25 s on CPU, ~10 s on GPU (synthetic surrogate).
# - **Data downloaded**: 0 MB on the synthetic path; ~1.5 GB to fetch the
#   real ds005505 windows on the HBN path (cached after first run).
# - **Prerequisites**: :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`,
#   :doc:`/auto_examples/tutorials/70_transfer_foundation/plot_72_subject_invariant_regression`
#   (the feature analogue of this script).
# - **Concept page**: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Step 1. Setup, seeding (E3.21), parametrised cache (E3.24)
# -----------------------------------------------------------
# Numpy and torch seeds make the printed metrics and the rendered curves
# byte-stable across reruns. The cache dir is parametrised through the
# ``EEGDASH_CACHE`` env var so HPC and local runs share one prepared
# windowed dataset.
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGConformer

torch.manual_seed(SEED)
# %% [markdown]
# Step 2. The mental model and the production data path
# ------------------------------------------------------
# The case study fixes the data pipeline (8 GSN 128 channels, 100 Hz,
# 2 s windows) and trains EEGConformer end-to-end with ``n_outputs=1``
# on a strict cross-subject split. from-scratch (this script) is the
# no-transfer baseline that fine-tuned regimes (linear-probe,
# full-finetune; cousin tutorial plot_73) must beat. The p-factor
# signal in EEG is faint, so the honest question is "does the network
# beat the train-set mean on never-seen subjects?"
#
# The canonical production call (run on real ds005505) is:
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#    from braindecode.preprocessing import (
#        Preprocessor, create_fixed_length_windows, preprocess,
#    )
#    ds = EEGDashDataset(dataset="ds005505", cache_dir=cache_dir,
#        description_fields=["subject", "session", "run", "task", "p_factor"])
#    # HydroCel GSN 128 midline: Fz=E11, Cz, Pz=E62, Oz=E75, C3=E36,
#    # C4=E104, P3=E52, P4=E92.
#    ch_names = ["E11", "Cz", "E62", "E75", "E36", "E104", "E52", "E92"]
#    preprocess(ds, [
#        Preprocessor("pick_channels", ch_names=ch_names, ordered=True),
#        Preprocessor("resample", sfreq=100),
#        Preprocessor("filter", l_freq=1, h_freq=30),
#    ])
#    windows = create_fixed_length_windows(ds,
#        window_size_samples=200, window_stride_samples=200,
#        drop_last_window=True, preload=False)
#
# Below we synthesise a Challenge-shaped windowed table with the same
# column layout so the gallery runs offline (E3.24).

# %%
# Step 3. Synthesise a Challenge-shaped windowed table
# -----------------------------------------------------
# 18 subjects, 24 windows each, 8 channels, 200 samples per 2 s window
# at 100 Hz. p_factor lives at subject level (one score per subject,
# replicated per window). A faint 10 Hz signal hides in Cz and E62 (the
# central and parietal midline); everything else is i.i.d. Gaussian
# noise plus a per-subject offset that the cross-subject split must NOT
# let the network memorise.
N_SUBJECTS, N_WINDOWS = 18, 24
N_CHANS, N_TIMES = 8, 200
SFREQ = 100.0
CH_NAMES = ("E11", "Cz", "E62", "E75", "E36", "E104", "E52", "E92")

subject_p = rng.normal(0.0, 1.0, size=N_SUBJECTS)
t = np.arange(N_TIMES) / SFREQ
X_list, rows = [], []
for s in range(N_SUBJECTS):
    p = float(subject_p[s])
    bias = 0.10 * (s - N_SUBJECTS / 2)
    for w in range(N_WINDOWS):
        base = rng.standard_normal((N_CHANS, N_TIMES)).astype(np.float32)
        # Faint p-factor signal at 10 Hz on Cz (ch 1) and E62 (ch 2).
        carrier = 0.30 * p * np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        base[1] += carrier
        base[2] += 0.85 * carrier
        # Subject identity offset that a leaky split would memorise.
        base += np.float32(bias)
        X_list.append(base)
        rows.append(
            {
                "subject": f"sub-{s:02d}",
                "sample_id": f"sub-{s:02d}__w{w:03d}",
                "p_factor": p,
            }
        )
X_all = np.stack(X_list)
meta = pd.DataFrame(rows)
y_all = meta["p_factor"].to_numpy(dtype=np.float32)
print(
    f"X_all={X_all.shape} | y_all={y_all.shape} | "
    f"subjects={meta['subject'].nunique()} | "
    f"p_factor mean={y_all.mean():+.3f} std={y_all.std():.3f}"
)
assert pd.api.types.is_float_dtype(meta["p_factor"]), "p_factor must be float"

# %% [markdown]
# Step 4. Predict: what r should chance look like?
# -------------------------------------------------
# **Predict.** A constant predictor that returns the train-set mean has
# Pearson r exactly zero on the validation cohort. With 14 train
# subjects and 4 held-out subjects, what r do you expect a small
# EEGConformer to reach: 0.05? 0.20? 0.50? Write your guess; the
# scatter panel will overwrite it.

# %% [markdown]
# Step 5. Cross-subject split before windowing
# ---------------------------------------------
# The split operates on unique subjects so no subject's data appears in
# both train and validation. Subject leakage is the most common failure
# mode for clinical EEG regression; an honest validation r demands
# subject-disjoint cohorts (Cisotto and Chicco 2024 Tip 9,
# doi:10.7717/peerj-cs.2256).
unique_subj = sorted(meta["subject"].unique())
n_val = max(2, int(round(0.20 * len(unique_subj))))
val_subj = set(unique_subj[-n_val:])
train_subj = set(unique_subj[:-n_val])
assert train_subj.isdisjoint(val_subj), "train/val subject overlap"
train_mask = meta["subject"].isin(train_subj).to_numpy()
val_mask = meta["subject"].isin(val_subj).to_numpy()
X_tr, y_tr = X_all[train_mask], y_all[train_mask]
X_va, y_va = X_all[val_mask], y_all[val_mask]
subj_va = meta.loc[val_mask, "subject"].tolist()
n_train_subjects = int(meta.loc[train_mask, "subject"].nunique())
n_val_subjects = int(meta.loc[val_mask, "subject"].nunique())
print(
    f"train: n_windows={X_tr.shape[0]} n_subjects={n_train_subjects} | "
    f"val: n_windows={X_va.shape[0]} n_subjects={n_val_subjects}"
)

# %% [markdown]
# Step 6. Configure EEGConformer for regression
# ----------------------------------------------
# :class:`braindecode.models.EEGConformer` (Song et al. 2023,
# doi:10.1109/TNSRE.2022.3230250) pairs convolutional embeddings with a
# transformer encoder. ``n_outputs=1`` makes the final layer a scalar
# regression head; ``num_layers=3`` keeps a CPU smoke run under a minute.


# %%
def build_model() -> "nn.Module":
    """Return a fresh EEGConformer regression head (n_outputs=1)."""
    return EEGConformer(
        n_chans=N_CHANS,
        n_outputs=1,
        n_times=N_TIMES,
        sfreq=SFREQ,
        num_layers=3,
        num_heads=4,
        final_fc_length="auto",
    )


def normalize_batch(x: "torch.Tensor") -> "torch.Tensor":
    """Per-channel z-score within each window (B, C, T)."""
    return (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)


# %% [markdown]
# Step 7. Run: train across seeds, record per-epoch curves
# ---------------------------------------------------------
# **Run.** A single-seed training run on a small target is noisy. We
# fit the model with ``N_SEEDS=2`` seeds and ``NUM_EPOCHS=4`` and
# record train MSE, val MSE, and val Pearson r per epoch. The figure
# folds the seeds into a mean +/- SE band.
NUM_EPOCHS = 4
N_SEEDS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

EPOCH_AXIS = np.arange(1, NUM_EPOCHS + 1)
train_loss_curve = np.full((N_SEEDS, NUM_EPOCHS), np.nan, dtype=float)
val_loss_curve = np.full((N_SEEDS, NUM_EPOCHS), np.nan, dtype=float)
val_r_curve = np.full((N_SEEDS, NUM_EPOCHS), np.nan, dtype=float)
last_model = None
last_val_pred = np.zeros_like(y_va, dtype=float)


def epoch_loop(model, X_tr_t, y_tr_t, X_va_t, y_va_t, *, epochs, lr, batch):
    """:class:`torch.optim.AdamW` epoch loop with per-epoch val metrics."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    train_mse, val_mse, val_r = [], [], []
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        running, count = 0.0, 0
        for i in range(0, len(idx), batch):
            sel = idx[i : i + batch]
            opt.zero_grad()
            pred = model(normalize_batch(X_tr_t[sel])).squeeze(-1)
            loss = F.mse_loss(pred, y_tr_t[sel])
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(sel)
            count += len(sel)
        train_mse.append(running / max(count, 1))
        model.eval()
        with torch.no_grad():
            v_pred = model(normalize_batch(X_va_t)).squeeze(-1).cpu().numpy()
        v_true = y_va_t.cpu().numpy()
        val_mse.append(float(np.mean((v_pred - v_true) ** 2)))
        if v_pred.std() > 1e-9 and v_true.std() > 1e-9:
            val_r.append(float(np.corrcoef(v_pred, v_true)[0, 1]))
        else:
            val_r.append(0.0)
    return train_mse, val_mse, val_r, v_pred


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
X_tr_t = torch.from_numpy(X_tr).float().to(device)
y_tr_t = torch.from_numpy(y_tr).float().to(device)
X_va_t = torch.from_numpy(X_va).float().to(device)
y_va_t = torch.from_numpy(y_va).float().to(device)
for s in range(N_SEEDS):
    torch.manual_seed(SEED + s)
    np.random.seed(SEED + s)
    model = build_model().to(device)
    tr_mse, va_mse, va_r, last_val_pred = epoch_loop(
        model,
        X_tr_t,
        y_tr_t,
        X_va_t,
        y_va_t,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch=BATCH_SIZE,
    )
    train_loss_curve[s, :], val_loss_curve[s, :], val_r_curve[s, :] = (
        tr_mse,
        va_mse,
        va_r,
    )
    last_model = model
    print(
        f"seed {s}: epoch {NUM_EPOCHS} | train_mse={tr_mse[-1]:.3f} | "
        f"val_mse={va_mse[-1]:.3f} | val_r={va_r[-1]:+.3f}"
    )

# %% [markdown]
# Step 8. Investigate: per-subject scatter and saliency
# ------------------------------------------------------
# **Investigate.** Mean MSE hides which subjects the model misses and
# which channel-time region it reads. The scatter panel aggregates val
# predictions to one point per held-out subject; the saliency panel
# averages ``|grad x|`` over the highest-confidence test windows, so
# the heatmap tells you whether the network read the carrier we hid in
# Cz/E62 or just memorised the per-subject offset (Schirrmeister et
# al. 2017, doi:10.1002/hbm.23730).


# %%
def compute_saliency(model, X_va_t, y_va_t, *, top_frac=0.5):
    """|grad x|, averaged over the top-confidence half of val windows."""
    if model is None:
        return np.zeros((N_CHANS, N_TIMES), dtype=float)
    model.eval()
    X = X_va_t.clone().detach().requires_grad_(True)
    pred = model(normalize_batch(X)).squeeze(-1)
    err = (pred - y_va_t) ** 2
    n_top = max(1, int(round(top_frac * X.shape[0])))
    keep = torch.topk(-err, n_top).indices  # smallest squared error == highest conf
    out = pred[keep].sum()
    out.backward()
    grad = X.grad.detach().abs().cpu().numpy()
    return grad[keep.cpu().numpy()].mean(axis=0)


saliency_map = np.zeros((N_CHANS, N_TIMES), dtype=float)
saliency_map = compute_saliency(last_model, X_va_t, y_va_t, top_frac=0.5)
print(
    f"saliency: peak channel={CH_NAMES[int(saliency_map.mean(axis=1).argmax())]} | "
    f"peak time idx={int(saliency_map.mean(axis=0).argmax())} | max={saliency_map.max():.3e}"
)

# %% [markdown]
# Step 9. Result: the 3-panel diagnostic
# ---------------------------------------
# Drawing helpers live in a sibling ``_pfactor_deep_figure`` module so
# the plumbing stays out of the rendered case study. The subtitle
# threads live runtime values: train/val subject counts, epochs, and
# best val r averaged across seeds.

# %%
from _pfactor_deep_figure import draw_pfactor_deep_figure

fig = draw_pfactor_deep_figure(
    train_curves={
        "epochs": EPOCH_AXIS,
        "train_loss": train_loss_curve,
        "val_loss": val_loss_curve,
        "val_r": val_r_curve,
    },
    y_true_subj=y_va,
    y_pred_subj=last_val_pred,
    saliency_map=saliency_map,
    channel_names=CH_NAMES,
    subject_ids=subj_va,
    sfreq=SFREQ,
    n_train_subjects=n_train_subjects,
    n_val_subjects=n_val_subjects,
    plot_id="project_pfactor_deep",
)
plt.show()

fig_metrics = fig._eegdash_pfactor_deep_metrics
print(
    f"figure metrics: r={fig_metrics['pearson_r']:+.3f} | "
    f"R^2={fig_metrics['r2']:+.3f} | MAE={fig_metrics['mae']:.3f} | "
    f"best_val_r={fig_metrics['best_val_r']:+.3f} | n_subjects={fig_metrics['n_subjects']}"
)

# %% [markdown]
# Step 10. A common mistake, and how to recover
# ----------------------------------------------
# **Run.** Wiring p_factor into a regression head as strings is a
# frequent slip when CSV loaders skip dtype hints; the model then
# refuses to backprop. We trigger it with try/except so the failure
# mode is visible (Nederbragt et al. 2020, doi:10.1371/journal.pcbi.1008090).

# %%
try:
    bad_y = meta["p_factor"].astype(str).to_numpy()
    torch.tensor(bad_y[:BATCH_SIZE], dtype=torch.float32)
except (ValueError, TypeError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:90]}")
    fixed = pd.to_numeric(meta["p_factor"], errors="coerce").to_numpy()
    fixed_t = torch.from_numpy(fixed[:BATCH_SIZE]).float()
    print(
        f"Recovery: cast p_factor to float (dtype={fixed.dtype}); shape={tuple(fixed_t.shape)}."
    )

# %% [markdown]
# Step 11. Modify: from-scratch -> linear-probe -> full-finetune
# ----------------------------------------------------------------
# **Modify (concept).** This script ran the from-scratch regime; cousin
# tutorial plot_73 carries the recipe for linear-probe and full-finetune.
# To extend: pretrain the same EEGConformer on a related task, reload
# the encoder with ``strict=False``, toggle ``requires_grad`` on the
# head only (linear-probe) or on every parameter (full-finetune), keep
# the rest constant, and compare best_val_r and gap-vs-from-scratch on
# the same 3-panel figure.

# %% [markdown]
# Wrap-up
# -------
# We trained a small :class:`braindecode.models.EEGConformer` end-to-end
# on a Challenge-shaped windowed table, kept the cross-subject split
# strict, recorded train/val curves and val r across two seeds, scored
# the held-out subjects, and pulled a saliency map back through the
# network. The p-factor signal in EEG is genuinely faint and
# ``p_factor`` is a derived score, not a diagnosis; any clinical framing
# belongs in a follow-up study with much larger N.

# %% [markdown]
# Try it yourself
# ---------------
# - Swap :class:`braindecode.models.EEGConformer` for
#   :class:`braindecode.models.ShallowFBCSPNet` and rerun.
# - Bump ``N_SEEDS`` to 5 and report the 5-seed mean r +/- SE.
# - Replace the synthetic windows with a real ds005505 query
#   (:doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`).
# - Pretrain on eyes-open vs eyes-closed and run linear-probe;
#   compare best_val_r with the from-scratch run from this script.

# %% [markdown]
# References
# ----------
# - Alexander et al. 2017, *Sci. Data* 4:170181, HBN. https://doi.org/10.1038/sdata.2017.181
# - Caspi et al. 2014, *Clin. Psychol. Sci.* 2:119, p-factor. https://doi.org/10.1177/2167702613497473
# - Delorme et al. 2022, *NAR Genom. Bioinform.* 4:lqac023, NEMAR. https://doi.org/10.1093/nargab/lqac023
# - Cisotto and Chicco 2024, *PeerJ CS* 10:e2256, clinical-EEG tips. https://doi.org/10.7717/peerj-cs.2256
# - Schirrmeister et al. 2017, *Hum. Brain Mapp.* 38:5391, Braindecode. https://doi.org/10.1002/hbm.23730
# - Song et al. 2023, *IEEE TNSRE* 31:710, EEGConformer. https://doi.org/10.1109/TNSRE.2022.3230250
# - Nederbragt et al. 2020, *PLOS Comput. Biol.* 16:e1008090, teaching-coding rules. https://doi.org/10.1371/journal.pcbi.1008090
