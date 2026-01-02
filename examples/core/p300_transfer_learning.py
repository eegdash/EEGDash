""".. _tutorial-p3-transfer-learning:

EEG P3 Transfer Learning with AS-MMD
====================================

This tutorial demonstrates how to train a domain-adaptive deep learning model for
EEG P3 component classification across two different datasets using Adaptive
Symmetric Maximum Mean Discrepancy (AS-MMD).

**Paper:** Chen, W., Delorme, A. (2025). Adaptive Split-MMD Training for Small-Sample
Cross-Dataset P300 EEG Classification. arXiv: `2510.21969 <https://arxiv.org/abs/2510.21969>`_

Key Concepts
============

This tutorial covers:

- **Domain Adaptation**: Training on multiple datasets with different recording setups
- **Deep Learning**: Using EEGConformer, a transformer-based model for EEG
- **AS-MMD**: A technique that aligns feature distributions across datasets
- **Cross-Validation**: Robust evaluation using nested stratified folds

By the end, you'll understand how to:

1. Load and preprocess multi-dataset EEG recordings
2. Build a domain-adaptive classifier
3. Evaluate performance across domains
4. Apply the method to your own datasets
"""

# %%
# Part 1: Loading and Preprocessing Data
# ========================================
#
# First, we load two oddball datasets from the EEGDash API. You can override the
# dataset IDs with EEGDASH_SOURCE_DATASET and EEGDASH_TARGET_DATASET.
from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_USE_NUMBA", "false")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path.cwd()))
(Path(os.environ["_MNE_FAKE_HOME_DIR"]) / ".mne").mkdir(exist_ok=True)

from eegdash import EEGDash, EEGDashDataset

cache_folder = Path(
    os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")
).resolve()
cache_folder.mkdir(parents=True, exist_ok=True)
eegdash = EEGDash()
ODDBALL_TASK = os.getenv("EEGDASH_ODDBALL_TASK", "visualoddball")
record_limit = int(os.getenv("EEGDASH_RECORD_LIMIT", "60"))


def _fetch_records(dataset_id):
    if ODDBALL_TASK:
        records = eegdash.find(
            {"dataset": dataset_id, "task": ODDBALL_TASK}, limit=record_limit
        )
        if records:
            return records
    return eegdash.find({"dataset": dataset_id}, limit=record_limit)


source_id = os.getenv("EEGDASH_SOURCE_DATASET", "ds005863")
target_id = os.getenv("EEGDASH_TARGET_DATASET", "ds003061")

source_records = _fetch_records(source_id)
target_records = _fetch_records(target_id)
if not source_records or not target_records:
    records = eegdash.find({"task": {"$regex": "oddball", "$options": "i"}}, limit=200)
    dataset_ids = []
    for rec in records:
        ds_id = rec.get("dataset")
        if ds_id and ds_id not in dataset_ids:
            dataset_ids.append(ds_id)
    if dataset_ids:
        source_id = dataset_ids[0]
        target_id = dataset_ids[1] if len(dataset_ids) > 1 else dataset_ids[0]
        source_records = _fetch_records(source_id)
        target_records = _fetch_records(target_id)

if not source_records or not target_records:
    raise RuntimeError("Unable to find two oddball datasets from the API.")

ds_p3 = EEGDashDataset(cache_dir=cache_folder, records=source_records)
ds_avo = EEGDashDataset(cache_dir=cache_folder, records=target_records)

print(f"Source ({source_id}): {len(ds_p3)} recordings")
print(f"Target ({target_id}): {len(ds_avo)} recordings")

# %%
# Data Preprocessing Pipeline
# ----------------------------
#
# Before training, we apply standard EEG preprocessing:
#
# - **Event labeling**: Identify oddball vs. standard stimuli
# - **Filtering**: 0.5-30 Hz bandpass to focus on relevant oscillations
# - **Resampling**: Downsample to 128 Hz to reduce computation
# - **Channel selection**: Keep Fz, Pz, P3, P4, Oz (standard P3 locations)
# - **Windowing**: Extract 1.2 sec epochs (-0.1s before to 1.1s after stimulus)
# - **Normalization**: Z-score normalization per trial

import numpy as np
import torch
import mne
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)

mne.set_log_level("ERROR")

# Preprocessing parameters
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128
TRIAL_START_OFFSET = -0.1  # 100 ms before stimulus
TRIAL_DURATION = 1.1  # Total window 1.1 seconds
COMMON_CHANNELS = ["Fz", "Pz", "P3", "P4", "Oz"]


def preprocess_dataset(dataset, channels, dataset_type="P3"):
    """Apply preprocessing pipeline to an EEG dataset.

    Returns numpy arrays: (n_trials, n_channels, n_times)
    """
    print(f"\nPreprocessing {dataset_type} dataset...")

    # Define preprocessing steps
    preprocessors = [
        Preprocessor("set_eeg_reference", ref_channels="average", projection=True),
        Preprocessor("resample", sfreq=RESAMPLE_FREQ),
        Preprocessor("filter", l_freq=LOW_FREQ, h_freq=HIGH_FREQ),
        Preprocessor(
            "pick_channels", ch_names=[ch.lower() for ch in channels], ordered=False
        ),
    ]

    # Apply preprocessing
    preprocess(dataset, preprocessors)

    # Extract windowed trials around stimulus onset
    trial_start = int(TRIAL_START_OFFSET * RESAMPLE_FREQ)
    trial_stop = int((TRIAL_START_OFFSET + TRIAL_DURATION) * RESAMPLE_FREQ)

    # Define event mapping to handle both datasets
    # ErpCore uses "Target"/"NonTarget"
    # ds005863 (AVO) uses "S 11", "S 21", etc. for targets and "S 12", "S 22" for nontargets
    mapping = {
        "Target": 1,
        "NonTarget": 0,
        "standard": 0,
        "target": 1,
    }
    # Add AVO specific codes if they are missing
    for i in range(1, 6):
        mapping[f"S {i}1"] = 1  # Often S 11, S 21, ... are targets
        mapping[f"S {i}2"] = 0  # Often S 12, S 22, ... are nontargets
        mapping[f"S{i}1"] = 1
        mapping[f"S{i}2"] = 0

    windows_ds = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start,
        trial_stop_offset_samples=trial_stop,
        preload=True,
        drop_bad_windows=True,
        mapping=mapping,
    )

    X, y = [], []
    for i in range(len(windows_ds)):
        data, label, *_ = windows_ds[i]
        X.append(data)
        y.append(label)

    print(f"Extracted {len(X)} trials from {dataset_type}")
    return np.array(X), np.array(y)


# Preprocess both datasets
X_p3, y_p3 = preprocess_dataset(ds_p3, COMMON_CHANNELS, f"Source ({source_id})")
X_avo, y_avo = preprocess_dataset(ds_avo, COMMON_CHANNELS, f"Target ({target_id})")

# Ensure both have the same number of samples (cropping to the minimum)
min_samples = min(X_p3.shape[2], X_avo.shape[2])
if X_p3.shape[2] != X_avo.shape[2]:
    print(f"\nCropping trials to {min_samples} samples for consistency...")
    X_p3 = X_p3[:, :, :min_samples]
    X_avo = X_avo[:, :, :min_samples]

# Combine datasets for training
X_all = np.vstack([X_p3, X_avo])
y_all = np.hstack([y_p3, y_avo])
src_all = np.array([source_id] * len(X_p3) + [target_id] * len(X_avo))

print(f"\nCombined dataset: {len(X_all)} trials ({X_all.shape})")
print(f"  Source: {np.sum(src_all == source_id)} trials")
print(f"  Target: {np.sum(src_all == target_id)} trials")


# %%
# Part 2: Model Architecture and Training
# ========================================
#
# Building the Domain-Adaptive Model
# -----------------------------------
#
# We use **EEGConformer**, a transformer-based architecture designed for EEG signals.
# The key idea in AS-MMD is to combine:
#
# 1. **Classification loss**: Standard cross-entropy on both domains
# 2. **Domain alignment**: MMD loss to match feature distributions
# 3. **Prototype alignment**: Align class centers across domains
# 4. **Data augmentation**: Mixup + Gaussian noise for regularization

from braindecode.models import EEGConformer
import torch.nn.functional as F


def normalize_data(x, eps=1e-7):
    """Normalize each trial independently."""
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std


# %%
# Domain Adaptation Techniques
# ----------------------------
#
# **Mixup**: Interpolates between sample pairs
def mixup_data(x, y, alpha=0.4):
    """Mix samples from the same batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# **Focal Loss**: Down-weights easy examples
def compute_focal_loss(scores, targets, gamma=2.0, alpha=0.25):
    """Focal loss for class imbalance."""
    ce_loss = F.cross_entropy(scores, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


# **Maximum Mean Discrepancy**: Measures domain distribution mismatch
def compute_mmd_rbf(x, y, eps=1e-8):
    """RBF-kernel MMD for distribution alignment."""
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)

    z = torch.cat([x, y], dim=0)
    if z.size(0) > 1:
        dists = torch.cdist(z, z, p=2.0)
        sigma = torch.median(dists)
        sigma = torch.clamp(sigma, min=eps)
    else:
        sigma = torch.tensor(1.0, device=z.device)

    gamma = 1.0 / (2.0 * (sigma**2) + eps)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2.0) ** 2)
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2.0) ** 2)
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2.0) ** 2)

    m, n = x.size(0), y.size(0)
    if m <= 1 or n <= 1:
        return torch.tensor(0.0, device=x.device)

    mmd = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    mmd += (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    mmd -= 2.0 * k_xy.mean()
    return mmd


# **Prototype Alignment**: Align class centers across domains
def compute_prototypes(features, labels, n_classes=2):
    """Compute mean feature vector per class."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    prototypes = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            proto = features[mask].mean(dim=0)
        else:
            proto = torch.zeros(features.size(1), device=features.device)
        prototypes.append(proto)
    return torch.stack(prototypes)


def compute_prototype_loss(features, labels, prototypes):
    """Align features to their class prototypes."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    loss = 0.0
    for i, label in enumerate(labels):
        proto = prototypes[label]
        loss += F.mse_loss(features[i], proto)
    return loss / max(1, len(labels))


# %%
# Training Configuration
# ----------------------
#
# Define hyperparameters for stable cross-domain training

BATCH_SIZE = 22
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2.5e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Part 3: Training and Evaluation
# ===============================
#
# The Training Loop
# -----------------
#
# For each batch, we compute four loss components:
#
# 1. **Classification loss** (source + target): Standard cross-entropy
# 2. **Mixup loss** (target domain): Interpolated samples for regularization
# 3. **MMD loss**: Aligns logit-space feature distributions
# 4. **Prototype loss**: Pulls small-domain features to large-domain class centers
#
# All losses are combined with domain-adaptive weights that increase during training.

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in data_loader:
            x = normalize_data(x).to(device)
            y = y.to(device)
            scores = model(x)
            all_preds.append(scores.argmax(1).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_probs.append(torch.softmax(scores, dim=1)[:, 1].cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    accuracy = (preds == targets).mean()
    auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5

    return {"accuracy": float(accuracy), "auc": float(auc)}


def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_asmmd_model(
    Xtr_p3,
    ytr_p3,
    Xva_p3,
    yva_p3,
    Xtr_avo,
    ytr_avo,
    Xva_avo,
    yva_avo,
    n_channels,
    n_times,
    seed=42,
):
    """Train a single AS-MMD model.

    Parameters
    ----------
    Xtr_*, ytr_* : numpy arrays
        Training data and labels for each domain
    Xva_*, yva_* : numpy arrays
        Validation data and labels for each domain

    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create data loaders

    train_p3 = make_loader(Xtr_p3, ytr_p3, shuffle=True)
    val_p3 = make_loader(Xva_p3, yva_p3, shuffle=False)
    train_avo = make_loader(Xtr_avo, ytr_avo, shuffle=True)
    val_avo = make_loader(Xva_avo, yva_avo, shuffle=False)

    # Initialize model
    model = EEGConformer(
        n_chans=n_channels,
        n_outputs=2,  # Binary: oddball vs. standard
        n_times=n_times,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        num_layers=3,  # Corrected from att_depth
        num_heads=4,  # Corrected from att_heads
        att_drop_prob=0.5,
    ).to(DEVICE)

    optimizer = torch.optim.Adamax(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Compute domain-specific weights
    n_p3, n_avo = len(Xtr_p3), len(Xtr_avo)
    small_domain = "P3" if n_p3 < n_avo else "AVO"
    large_domain = "AVO" if small_domain == "P3" else "P3"

    # Training loop
    best_score = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        # Warmup: gradually increase domain adaptation strength
        warmup_epoch = min(1.0, epoch / 20)

        loaders = {"P3": train_p3, "AVO": train_avo}
        itr_small = iter(loaders[small_domain])

        for xb_large, yb_large in loaders[large_domain]:
            # Large domain batch
            x_large = normalize_data(xb_large).to(DEVICE)
            y_large = yb_large.to(DEVICE)
            scores_large = model(x_large)
            loss_cls = F.cross_entropy(scores_large, y_large)

            # Small domain batch
            try:
                xb_small, yb_small = next(itr_small)
            except StopIteration:
                itr_small = iter(loaders[small_domain])
                xb_small, yb_small = next(itr_small)

            x_small = normalize_data(xb_small).to(DEVICE)
            y_small = yb_small.to(DEVICE)

            # Mixup on small domain
            x_mixed, y_a, y_b, lam = mixup_data(x_small, y_small)
            scores_mixed = model(x_mixed)
            loss_mixup = lam * compute_focal_loss(scores_mixed, y_a) + (
                1 - lam
            ) * compute_focal_loss(scores_mixed, y_b)

            # MMD alignment
            scores_orig = model(x_small)
            loss_mmd = warmup_epoch * compute_mmd_rbf(
                scores_large.detach(), scores_orig.detach()
            )

            # Prototype alignment
            with torch.no_grad():
                proto_large = compute_prototypes(
                    scores_large.detach(), y_large, n_classes=2
                )
            loss_proto = warmup_epoch * compute_prototype_loss(
                scores_orig, y_small, proto_large
            )

            # Combined loss
            loss = loss_cls + loss_mixup + 0.3 * loss_mmd + 0.5 * loss_proto

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        # Validation
        val_p3_metrics = evaluate_model(model, val_p3, DEVICE)
        val_avo_metrics = evaluate_model(model, val_avo, DEVICE)

        # Track best model on small domain
        small_val = (
            val_p3_metrics["accuracy"]
            if small_domain == "P3"
            else val_avo_metrics["accuracy"]
        )

        if small_val > best_score:
            best_score = small_val
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        if (epoch % 10 == 0) or (epoch == 1):
            print(
                f"Epoch {epoch:3d} | P3 val: {val_p3_metrics['accuracy']:.3f} | "
                f"AVO val: {val_avo_metrics['accuracy']:.3f} | Score: {small_val:.3f}"
            )

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# %%
# Nested Cross-Validation
# -----------------------
#
# We use nested CV to robustly estimate model performance:
#
# - **Outer folds (5)**: For test set evaluation
# - **Inner split**: Train/val split for hyperparameter tuning
# - **Repeats (5)**: Multiple random seeds for stability

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def run_nested_cv(X_all, y_all, src_all, channels):
    """Run nested cross-validation with AS-MMD."""
    n_channels = X_all.shape[1]
    n_times = X_all.shape[2]

    results = []
    SEEDS = [42, 123, 456, 789, 321]

    for repeat in range(2):  # 2 repeats for quick demo (use 5 for final results)
        print(f"\n{'=' * 60}")
        print(f"Repeat {repeat + 1}/2")
        print("=" * 60)

        cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=SEEDS[repeat]
        )  # 3 folds for demo

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            print(f"\nFold {fold_idx + 1}/3")

            X_tr, y_tr, src_tr = X_all[train_idx], y_all[train_idx], src_all[train_idx]
            X_te, y_te, src_te = X_all[test_idx], y_all[test_idx], src_all[test_idx]

            # Split train into train/val
            tr_idx, va_idx = train_test_split(
                np.arange(len(X_tr)), test_size=0.15, stratify=y_tr, random_state=42
            )

            # Extract per-domain data
            def get_domain(X, y, src, idx, domain):
                mask = src == domain
                indices = np.intersect1d(np.where(mask)[0], idx)
                return X[indices], y[indices]

            Xtr_p3, ytr_p3 = get_domain(X_tr, y_tr, src_tr, tr_idx, "P3")
            Xtr_avo, ytr_avo = get_domain(X_tr, y_tr, src_tr, tr_idx, "AVO")
            Xva_p3, yva_p3 = get_domain(X_tr, y_tr, src_tr, va_idx, "P3")
            Xva_avo, yva_avo = get_domain(X_tr, y_tr, src_tr, va_idx, "AVO")

            if len(Xtr_p3) == 0 or len(Xtr_avo) == 0:
                print("  Skipping: insufficient training samples")
                continue

            print(f"  Train: P3={len(Xtr_p3)}, AVO={len(Xtr_avo)}")
            print(f"  Val:   P3={len(Xva_p3)}, AVO={len(Xva_avo)}")

            # Train model
            model = train_asmmd_model(
                Xtr_p3,
                ytr_p3,
                Xva_p3,
                yva_p3,
                Xtr_avo,
                ytr_avo,
                Xva_avo,
                yva_avo,
                n_channels,
                n_times,
                seed=SEEDS[repeat],
            )

            # Evaluate on test set
            def test_domain(domain_label):
                mask = src_te == domain_label
                if not np.any(mask):
                    return {"accuracy": 0.0, "auc": 0.5}, 0
                loader = make_loader(X_te[mask], y_te[mask])
                metrics = evaluate_model(model, loader, DEVICE)
                return metrics, np.sum(mask)

            def make_loader(X, y):
                return DataLoader(
                    TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

            m_p3, n_p3 = test_domain("P3")
            m_avo, n_avo = test_domain("AVO")

            overall_acc = (m_p3["accuracy"] * n_p3 + m_avo["accuracy"] * n_avo) / (
                n_p3 + n_avo + 1e-8
            )

            print(
                f"  Test: P3={m_p3['accuracy']:.3f} (n={n_p3}), AVO={m_avo['accuracy']:.3f} (n={n_avo})"
            )

            results.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold_idx + 1,
                    "p3_acc": m_p3["accuracy"],
                    "p3_auc": m_p3["auc"],
                    "avo_acc": m_avo["accuracy"],
                    "avo_auc": m_avo["auc"],
                    "overall_acc": overall_acc,
                }
            )

    return pd.DataFrame(results)


# %%
# Execute Training
# ----------------

print("\nStarting AS-MMD Training with Nested Cross-Validation...")
print("=" * 60)

results_df = run_nested_cv(X_all, y_all, src_all, COMMON_CHANNELS)

# Print summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(
    f"\nOverall Accuracy: {results_df['overall_acc'].mean():.4f} ± {results_df['overall_acc'].std():.4f}"
)
print("\nP3 Dataset:")
print(
    f"  Accuracy: {results_df['p3_acc'].mean():.4f} ± {results_df['p3_acc'].std():.4f}"
)
print(f"  AUC: {results_df['p3_auc'].mean():.4f} ± {results_df['p3_auc'].std():.4f}")
print("\nAVO Dataset:")
print(
    f"  Accuracy: {results_df['avo_acc'].mean():.4f} ± {results_df['avo_acc'].std():.4f}"
)
print(f"  AUC: {results_df['avo_auc'].mean():.4f} ± {results_df['avo_auc'].std():.4f}")
print("=" * 60)

# Save results
results_df.to_csv("asmmd_results.csv", index=False)
print("\nResults saved to: asmmd_results.csv")

# %%
# Key Takeaways
# =============
#
# **Main Components of AS-MMD:**
#
# 1. **Classification Loss**: Standard cross-entropy on both datasets
# 2. **Mixup Regularization**: Interpolate between samples for better generalization
# 3. **MMD Alignment**: Match feature distributions across domains
# 4. **Prototype Alignment**: Pull small-domain features toward large-domain class centers
# 5. **Warmup Schedule**: Gradually introduce domain adaptation during training
#
# **When to Use This Method:**
#
# - You have limited data from your target domain
# - You have access to a related source domain (different equipment/site)
# - You want a single model that performs well on both domains
# - You need robust cross-dataset performance
#
# **Tips for Your Own Data:**
#
# - Verify channel names match between datasets (case-insensitive lowercasing helps)
# - Adjust BATCH_SIZE if memory is limited (try 16 or 32)
# - Increase MAX_EPOCHS if curves haven't plateaued
# - Tune MMD weight (0.2-0.5) and prototype weight (0.5-0.8) based on domain similarity
# - Use more CV folds (5-10) for final results
#
# **References:**
#
# - Chen, W., Delorme, A. (2025). Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 Classification.
# - Song et al. (2019). "EEGConformer: Convolutional Transformer for EEG Decoding"
# - Long et al. (2015). "Learning Transferable Features with Deep Adaptation Networks"

# %%
# Next Steps
# ==========
#
# - Try different EEG components (e.g., N1, P2, N2 instead of P3)
# - Extend to multi-class classification (e.g., oddball paradigm variants)
# - Apply to other tasks (motor imagery, sleep staging, seizure detection)
# - Experiment with other backbones (ResNet, LSTM) instead of EEGConformer
# - Implement subject-independent vs. subject-specific models
