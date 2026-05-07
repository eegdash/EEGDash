"""Fine-tune a pretrained EEG model on a downstream task
=====================================================

How do you adapt a pretrained EEG model to a new EEGDash task without
retraining the encoder from scratch every time, and which regime --
frozen, partially unfrozen, or fully unfrozen -- actually wins when
your target dataset has only a handful of subjects?

The Braindecode foundation-model API (``from_pretrained``, ``reset_head``,
``return_features``) is gated on upstream stabilisation (cf.
``tutorial_restructure_plan.md`` L1228-L1229), so we mock the workflow:
pretrain a small ``ShallowFBCSPNet`` on a synthetic source task, save
the encoder, and reload it into a fresh model. The mechanics -- inspect
layers, freeze the encoder, replace the head, fine-tune on a
leakage-safe cross-subject split, compare to a from-scratch baseline --
mirror the production recipe, so muscle memory transfers when the
upstream API lands. A frozen-encoder fine-tune typically lifts above
chance on a downstream EEG task (Schirrmeister et al. 2017,
doi:10.1002/hbm.23730); the open question is whether unfreezing actually
helps when the target task is small. So which regime wins?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_73_finetune_pretrained_model.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - load a pretrained encoder checkpoint and inspect its layers / param count.
# - freeze the encoder and replace the classification head with a new ``n_outputs``.
# - fine-tune the head on a leakage-safe cross-subject split with a chance line.
# - compare frozen-encoder, fully-unfrozen, and from-scratch regimes side by side.
# - apply a partial unfreeze (last block only) and read the trade-off curve.

# %% [markdown]
# Requirements
# ------------
#
# - **Estimated time**: ~30 s on CPU, ~10 s on GPU.
# - **Data downloaded**: 0 MB (synthetic windows, deterministic).
# - **Prerequisites**: ``plot_71_cross_task_transfer.py``.
# - **Concept page**:
#   [docs/source/concepts/features_vs_deep_learning.rst](../../docs/source/concepts/features_vs_deep_learning.rst).

# %%
# Setup. Seeding numpy and torch makes the printed accuracy byte-stable.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eegdash.splits import assert_no_leakage, majority_baseline
from eegdash.viz import use_eegdash_style

use_eegdash_style()
SEED = 42
np.random.seed(SEED)

cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = cache_dir / "plot_73_pretrained_encoder.pt"

# torch + braindecode are optional at parse time so the audit can read this
# file even on a torch-less environment. The training cells gate on the flag.
try:
    import torch
    import torch.nn as nn
    from braindecode.models import ShallowFBCSPNet

    torch.manual_seed(SEED)
    HAS_TORCH = True
except ImportError as exc:  # pragma: no cover - documented gating
    print(f"torch/braindecode unavailable ({exc}); training cells will be skipped.")
    HAS_TORCH = False


# %% [markdown]
# Step 1 -- Pretrain a small encoder on a synthetic source task
# -------------------------------------------------------------
#
# A real foundation model is pretrained on thousands of hours of EEG; we
# stand in with a 6-subject synthetic source task and a couple of epochs
# through ``ShallowFBCSPNet``. The encoder weights are saved to disk so
# Step 3 can reload them like ``from_pretrained``. Using 8 channels and
# 2 s windows @ 128 Hz keeps source/target shapes identical -- a hard
# requirement of any transfer recipe.
N_CHANS, N_TIMES, SFREQ = 8, 256, 128.0


# %%
def synth_windows(n_subj, n_per, prefix="src", rng=None):
    """Return ``(X, y, metadata)`` with two-class alpha-vs-delta injection."""
    rng = rng or np.random.default_rng(SEED)
    t = np.arange(N_TIMES) / SFREQ
    X_list, rows = [], []
    for s in range(n_subj):
        labels = rng.integers(0, 2, size=n_per)
        for w, lab in enumerate(labels):
            base = rng.standard_normal((N_CHANS, N_TIMES)).astype(np.float32) * 0.2
            freq = 10.0 if lab == 1 else 2.0
            base += (0.7 + 0.05 * s) * np.sin(2 * np.pi * freq * t).astype(np.float32)
            X_list.append(base)
            rows.append(
                {
                    "sample_id": f"{prefix}_s{s:02d}_w{w:03d}",
                    "subject": f"sub-{s:02d}",
                    "label": int(lab),
                }
            )
    return np.stack(X_list), np.array([r["label"] for r in rows]), pd.DataFrame(rows)


def build_model(n_outputs=2):
    return ShallowFBCSPNet(
        n_chans=N_CHANS, n_outputs=n_outputs, n_times=N_TIMES, final_conv_length="auto"
    )


X_src, y_src, _ = synth_windows(n_subj=6, n_per=40, prefix="src")
print(f"source X={X_src.shape}, y={y_src.shape}")


# %%
def quick_train(model, X, y, epochs=1, lr=1e-3, batch=32):
    """A textbook AdamW loop -- enough to learn the contrast, not a recipe."""
    if not HAS_TORCH:
        return model
    params = [p for p in model.parameters() if p.requires_grad]
    opt, loss_fn = torch.optim.AdamW(params, lr=lr), nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(Xt))
        for i in range(0, len(idx), batch):
            sel = idx[i : i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[sel]), yt[sel]).backward()
            opt.step()
    return model


def evaluate(model, X, y):
    """Return classification accuracy on (X, y)."""
    if not HAS_TORCH:
        return float("nan")
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float()).argmax(dim=1).numpy()
    return float((preds == y).mean())


if HAS_TORCH:
    pretrained = quick_train(build_model(), X_src, y_src, epochs=2)
    # Save only the encoder (everything except final_layer) so the head is
    # contractually replaced -- the foundation-model ``reset_head=True``.
    enc_state = {
        k: v
        for k, v in pretrained.state_dict().items()
        if not k.startswith("final_layer")
    }
    torch.save(enc_state, ckpt_path)
    print(f"saved encoder: {ckpt_path.name} ({len(enc_state)} tensors)")


# %% [markdown]
# **Predict.** Of the three regimes -- frozen encoder, fully unfrozen,
# or from scratch -- which wins on a small downstream task with ~120
# windows? Write a guess before running Step 4.

# %% [markdown]
# Step 2 -- Build a leakage-safe downstream split
# -----------------------------------------------
# We synthesise a 4-subject target task, hold out one subject for test,
# and call ``assert_no_leakage`` so the runtime validator (E5.42) sees
# the contract JSON line.

# %%
X_tgt, y_tgt, meta = synth_windows(n_subj=4, n_per=30, prefix="tgt")
all_subj = sorted(meta["subject"].unique())
train_mask = (~meta["subject"].isin({all_subj[-1]})).to_numpy()
test_mask = ~train_mask
folds = [
    (
        meta.loc[train_mask, "sample_id"].tolist(),
        meta.loc[test_mask, "sample_id"].tolist(),
    )
]
overlap = assert_no_leakage(folds, meta, by="subject")
assert overlap == 0, "subject overlap detected; rebuild the split before training"
X_tr, y_tr = X_tgt[train_mask], y_tgt[train_mask]
X_te, y_te = X_tgt[test_mask], y_tgt[test_mask]
print(f"target: train={len(X_tr)} test={len(X_te)} subjects={len(all_subj)}")


# %% [markdown]
# Step 3 -- Reload the encoder, replace the head, freeze
# ------------------------------------------------------
# In the production recipe this is one line:
# ``model = from_pretrained(...); model.reset_head(n_outputs=K)``. We
# mirror it: load the encoder state, leave the freshly-initialised
# ``final_layer`` alone, and toggle ``requires_grad`` so only the head
# learns. We assert ``frozen + trainable == total`` (spec invariant E3.22).


# %%
def reset_and_freeze(model, freeze=True, last_block_only=False):
    """Load encoder weights, optionally freeze; return (frozen, trainable, total)."""
    if not HAS_TORCH:
        return 0, 0, 0
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    head_reset = all(k.startswith("final_layer") for k in missing)
    assert head_reset, f"unexpected missing keys (head not reset): {missing}"
    for name, p in model.named_parameters():
        if not freeze:
            p.requires_grad = True
            continue
        if last_block_only and ("conv_classifier" in name or "bnorm" in name):
            p.requires_grad = True
        else:
            p.requires_grad = name.startswith("final_layer")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert frozen + trainable == total, "param accounting drift"
    return frozen, trainable, total


# %% [markdown]
# Run -- frozen, unfrozen, and from-scratch fine-tunes
# ----------------------------------------------------
# Three regimes, same target data and budget. Frozen uses a higher lr
# (only the head learns); unfrozen drops to 1e-3 because encoder weights
# are fragile (Cisotto & Chicco 2024 Tip 7); scratch is randomly init.

# %%
results: dict[str, float] = {}
chance = float(majority_baseline(y_tr, y_te)["chance_level"])
if HAS_TORCH:
    m_frozen = build_model()
    f, t, tot = reset_and_freeze(m_frozen, freeze=True)
    print(f"frozen: trainable={t} frozen={f} total={tot}")
    results["frozen"] = evaluate(
        quick_train(m_frozen, X_tr, y_tr, epochs=4, lr=1e-2), X_te, y_te
    )

    m_unfrozen = build_model()
    f, t, _ = reset_and_freeze(m_unfrozen, freeze=False)
    print(f"unfrozen: trainable={t} frozen={f}")
    results["unfrozen"] = evaluate(
        quick_train(m_unfrozen, X_tr, y_tr, epochs=4, lr=1e-3), X_te, y_te
    )

    results["scratch"] = evaluate(
        quick_train(build_model(), X_tr, y_tr, epochs=4, lr=1e-3), X_te, y_te
    )


# %% [markdown]
# Investigate
# -----------
# Per-regime accuracy vs chance level. With 30 train windows per subject
# the absolute numbers are noisy; the *gap above chance* is what
# generalises across runs.

# %%
print("\n| regime              | accuracy | chance |")
print("|---------------------|----------|--------|")
for name, acc in results.items():
    print(f"| {name:<19} | {acc:0.3f}    | {chance:0.3f}  |")
if not results:
    print(f"| (torch unavailable) | nan      | {chance:0.3f}  |")


# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
#
# **Run.** A frequent slip is reloading the encoder into a model whose
# ``n_chans`` differs from the pretrained one -- ``load_state_dict`` then
# raises a size-mismatch ``RuntimeError`` on the first conv. We trigger
# it on purpose with ``try/except`` so you see exactly what the error
# looks like (Nederbragt et al. 2020, doi:10.1371/journal.pcbi.1008090).

# %%
if HAS_TORCH:
    try:
        wrong = ShallowFBCSPNet(
            n_chans=N_CHANS + 2,  # pretrained on 8 chans; target rebuilt with 10
            n_outputs=2,
            n_times=N_TIMES,
            final_conv_length="auto",
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        wrong.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print(f"Caught RuntimeError: {str(exc)[:90]}...")
        # Recovery: rebuild with matching n_chans, load with strict=False, re-init head.
        fixed = build_model()
        missing, _ = fixed.load_state_dict(state, strict=False)
        head_only = all(k.startswith("final_layer") for k in missing)
        print(f"Recovery: matching n_chans + strict=False; head re-init={head_only}.")

# %% [markdown]
# Modify
# ------
# **Your turn**: re-run with ``last_block_only=True`` in
# ``reset_and_freeze``. The classifier conv and final batch norm
# unfreeze; earlier layers stay pinned. This middle-ground regime is
# what most foundation-model recipes default to for small target tasks.

# %%
if HAS_TORCH:
    m_partial = build_model()
    f, t, _ = reset_and_freeze(m_partial, freeze=True, last_block_only=True)
    print(f"partial: trainable={t} frozen={f}")
    results["partial"] = evaluate(
        quick_train(m_partial, X_tr, y_tr, epochs=4, lr=5e-3), X_te, y_te
    )
    print(f"Partial-unfreeze acc: {results['partial']:.2f} | chance: {chance:.2f}")


# %% [markdown]
# Result
# ------
# Headline metric: best fine-tune accuracy on the held-out subject,
# alongside chance. The gap is the only number worth quoting in a paper.

# %%
best_name = max(results, key=results.get) if results else "n/a"
best_acc = results.get(best_name, float("nan"))
print(
    json.dumps(
        {
            "finetune_best_regime": best_name,
            "accuracy": round(best_acc, 3),
            "chance_level": round(chance, 3),
        }
    )
)


# %% [markdown]
# Wrap-up
# -------
# We pretrained a Braindecode encoder, saved its weights, reloaded them
# into a fresh model with a replaced head, and compared frozen,
# fully-unfrozen, partial, and from-scratch regimes against a chance
# line. The split was subject-aware (``leakage_report`` overlap=0), every
# RNG was seeded, and ``frozen + trainable == total`` held at every step.
# When the Braindecode foundation-model API stabilises (plan L1228-L1229)
# the only edits are: swap ``build_model`` + ``torch.load`` for
# ``from_pretrained(...)`` and call ``model.reset_head(n_outputs=K)``.

# %% [markdown]
# Try it yourself
# ---------------
# - Vary the source-task pretraining length (1, 2, 5 epochs) and replot the curve.
# - Swap ``ShallowFBCSPNet`` for ``EEGNetv4`` and re-run the four regimes.
# - Increase the target test set to two subjects and report mean +/- std accuracy.
# - Replace the synthetic data with a windowed EEGDash dataset from ``plot_10``.

# %% [markdown]
# References
# ----------
# - Schirrmeister et al. 2017, Deep learning with convolutional neural
#   networks for EEG decoding, *Human Brain Mapping*.
#   https://doi.org/10.1002/hbm.23730
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ
#   Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Braindecode foundation-model tutorial (link only; no DOI):
#   https://braindecode.org/dev/auto_examples/model_building/plot_load_pretrained_models.html
# - Concept page:
#   [docs/source/concepts/features_vs_deep_learning.rst](../../docs/source/concepts/features_vs_deep_learning.rst).
