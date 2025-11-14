""".. _tutorial-p3-transfer-learning:

EEG P3 Transfer Learning with AS-MMD
====================================


This tutorial's corresponding method/paper: Chen, W., Delorme, A. (2025). Adaptive Split-MMD Training for Small-Sample Cross-Dataset P300 EEG Classification. arXiv: [2510.21969](https://arxiv.org/abs/2510.21969).

Overview
--------
This tutorial demonstrates transfer learning for EEG P3 component analysis using 
Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD) to adapt between two oddball 
paradigm datasets: ERP CORE P3 and Active Visual Oddball (AVO).

The goal is to train a deep learning classifier (EEGConformer) that performs well 
on both datasets despite differences in participant populations, recording equipment, 
and experimental protocols. We use domain adaptation techniques including:

1. **Prototype-based Discriminative Transfer**: Align class representations across domains
2. **Mixup Data Augmentation**: Enhance learning from smaller sample sizes
3. **MMD Alignment**: Minimize distribution mismatch in feature space
4. **Nested Cross-Validation**: Robust performance estimation
"""
# %%
# Dataset Loading with EEGDashDataset
# ====================================
#
# This tutorial uses EEGDashDataset to load data from BIDS-compatible datasets.
# First, we initialize the datasets using EEGDashDataset.
#
# Download Instructions:
#
# 1. AVO Dataset (ds005863) - Cognitive Electrophysiology in Socioeconomic Context
#    Download: https://nemar.org/dataexplorer/detail?dataset_id=ds005863
#    - 127 participants, ~10.6 GB
#    - Contains Active Visual Oddball task from ERP CORE
#    - Alternative: https://openneuro.org/datasets/ds005863
#
# 2. ERP CORE P3 Dataset
#    Download: https://osf.io/etdkz/files
#    - choose file: P3 Raw Data BIDS-Compatible (extracts to P3_Raw_Data_BIDS-Compatible)
#    - 40 neurotypical adults
#    - P3 active visual oddball task
#
# After downloading and extracting, update the cache_dir path below:

from eegdash import EEGDashDataset

CACHE_DIR = '/home/vivian/eeg' # update to your local cache directory

ds_p3 = EEGDashDataset(
    dataset="P3_Raw_Data_BIDS-Compatible",
    cache_dir=CACHE_DIR,
    task="P3",
    download=False,
)

ds_avo = EEGDashDataset(
    dataset="ds005863",
    cache_dir=CACHE_DIR,
    task="visualoddball",  
    download=False,
)

print(f"P3 Dataset: {len(ds_p3)} recordings loaded")
print(f"AVO Dataset: {len(ds_avo)} recordings loaded")

# %%
# Event Renaming Function
# -----------------------
#
# Simple function to rename oddball events for preprocessing

import mne
import numpy as np
mne.set_log_level('ERROR')

def rename_oddball_events(raw, dataset_type='P3'):
    events, event_id = mne.events_from_annotations(raw)
    
    if len(events) == 0:
        return raw
    
    if dataset_type == 'AVO':
        oddball_codes = [11, 22, 33, 44, 55]
        response_codes = [201, 202]
        
        def get_codes_from_descriptions(event_id, descriptions):
            codes = []
            for desc, code in event_id.items():
                if any(pattern in desc for pattern in descriptions):
                    codes.append(code)
            return codes
        
        oddball_codes = get_codes_from_descriptions(event_id, ['11', '22', '33', '44', '55'])
        response_codes = get_codes_from_descriptions(event_id, ['201', '202'])
    else:  # P3
        oddball_codes = [1, 9, 15, 21, 27]
        response_codes = [6, 7, 201, 202]
    
    response_mask = np.isin(events[:, 2], response_codes)
    events = events[~response_mask]
    
    if len(events) == 0:
        return raw
    
    unique_samples, unique_indices = np.unique(events[:, 0], return_index=True)
    events = events[np.sort(unique_indices)]
    
    trial_start_offset_samples = int(TRIAL_START_OFFSET * raw.info["sfreq"])
    trial_stop_offset_samples = int((TRIAL_START_OFFSET + TRIAL_DURATION) * raw.info["sfreq"])
    recording_length = raw.n_times
    
    min_gap_samples = int(abs(TRIAL_START_OFFSET) * raw.info["sfreq"]) + int(TRIAL_DURATION * raw.info["sfreq"])
    filtered_events = []
    last_kept_sample = None
    
    for event in events:
        event_sample = event[0]
        window_start = event_sample + trial_start_offset_samples
        window_stop = event_sample + trial_stop_offset_samples
        
        if window_start < 0 or window_stop > recording_length:
            continue
        
        if last_kept_sample is None or (event_sample - last_kept_sample) >= min_gap_samples:
            filtered_events.append(event)
            last_kept_sample = event_sample
    
    if len(filtered_events) == 0:
        return raw
    
    events = np.array(filtered_events)
    
    oddball_mask = np.isin(events[:, 2], oddball_codes)
    new_events = events.copy()
    new_events[oddball_mask, 2] = 1
    new_events[~oddball_mask, 2] = 0
    
    annot_from_events = mne.annotations_from_events(
        events=new_events,
        event_desc={0: "standard", 1: "oddball"},
        sfreq=raw.info["sfreq"],
    )
    raw.set_annotations(annot_from_events)
    return raw


# %%
# Data Normalization and Augmentation
# -----------------------------------
#
# Functions for normalizing EEG data and applying data augmentation techniques
# (Gaussian noise and time shifting) to improve model generalization.

import numpy as np
import torch

# Data augmentation parameters
NOISE_STD = 0.006
TIME_SHIFT_RANGE = 6

def normalize_data(x, eps=1e-7):
    """Normalize EEG data along time dimension.
    
    Applies z-score normalization: (x - mean) / std
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std


def augment_data(x, noise_std=NOISE_STD, time_shift=TIME_SHIFT_RANGE):
    """Apply data augmentation to EEG signals.
    
    Two augmentation techniques:
    1. Gaussian noise: Adds random noise to simulate natural variability
    2. Time shifting: Shifts signals along time axis to improve temporal robustness
    """
    augmented = x.clone()
    
    if noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
    
    if time_shift > 0:
        for i in range(x.size(0)):
            shift = np.random.randint(-time_shift, time_shift + 1)
            if shift != 0:
                augmented[i] = torch.roll(augmented[i], shift, dims=1)
    
    return augmented



# %%
# Braindecode Preprocessing Pipeline
# -----------------------------------
#
# Create preprocessors for both datasets using braindecode

from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events

# Preprocessing parameters
LOW_FREQ = 0.5
HIGH_FREQ = 30
RESAMPLE_FREQ = 128
TRIAL_START_OFFSET = -0.1  # seconds before stimulus
TRIAL_DURATION = 1.1  # seconds total window

def create_preprocessors(eeg_channels, dataset_type='P3'):
    """Create preprocessing pipeline for oddball datasets.
    
    Parameters
    ----------
    eeg_channels : list
        List of EEG channel names to use
    dataset_type : str
        Either 'P3' or 'AVO'
        
    Returns
    -------
    list
        List of Preprocessor objects
    """
    def rename_events_fn(raw):
        return rename_oddball_events(raw, dataset_type=dataset_type)
    
    def rename_channels_fn(raw):
        raw.rename_channels({ch: ch.lower() for ch in raw.ch_names})
        return raw
    
    preprocessors = [
        Preprocessor(rename_events_fn, apply_on_array=False),
        Preprocessor(rename_channels_fn, apply_on_array=False),
        Preprocessor("set_eeg_reference", ref_channels='average', projection=True),
        Preprocessor("resample", sfreq=RESAMPLE_FREQ),
        Preprocessor("filter", l_freq=LOW_FREQ, h_freq=HIGH_FREQ),
        Preprocessor("pick_channels", ch_names=[ch.lower() for ch in eeg_channels], ordered=False),
    ]
    
    return preprocessors

# Trials per subject for sampling
TRIALS_PER_SUBJECT_P3 = 80 
TRIALS_PER_SUBJECT_AVO = 10


# %%
# Plot Sample Trials
# ==================
#
# Visualize sample oddball trials from both datasets.
#
# .. image:: /_static/p3_avo_sample_trials.png
#    :alt: Sample trials from P3 and AVO datasets
#    :align: center
#    :width: 800px
#
# The figure above shows example oddball trials from both P3 and AVO datasets,
# displaying the ERP responses across common channels (Fz, Pz, P3, P4, Oz).
# The stimulus onset occurs at t=0s.

# %%
# Deep Learning Model: EEGConformer from Braindecode
# ---------------------------------------------------
#
# Import the EEGConformer model from braindecode library

import torch
import torch.nn as nn
import torch.nn.functional as F

COMMON_CHANNELS = ['Fz', 'Pz', 'P3', 'P4', 'Oz']

from braindecode.models import EEGConformer

def create_model(n_channels, n_times):
    """Create EEGConformer model from braindecode."""
    return EEGConformer(
        n_chans=n_channels,
        n_outputs=2,
        n_times=n_times,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=3,
        att_heads=4,
        att_drop_prob=0.5,
    )


# %%
# AS-MMD Domain Adaptation
# -------------------------
#
# Adaptive Symmetric Maximum Mean Discrepancy (AS-MMD) aligns feature distributions
# between source (P3) and target (AVO) domains using three key components:
#
# 1. **Mixup augmentation**: Interpolates between samples to enhance generalization
# 2. **Prototype alignment**: Aligns class-wise feature representations
# 3. **MMD loss**: Minimizes distribution discrepancy in feature space
#

# %%
# Mixup and Augmentation Utilities
# --------------------------------
#
# Functions for mixup data augmentation and focal loss used in AS-MMD training.

def mixup_data(x, y, alpha=0.4):
    """Perform mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def compute_focal_loss(scores, targets, gamma=2.0, alpha=0.25):
    """Compute focal loss for handling class imbalance."""
    ce_loss = F.cross_entropy(scores, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def mixup_criterion(pred, y_a, y_b, lam, gamma=2.0, alpha=0.5):
    """Compute mixup focal loss."""
    loss_a = compute_focal_loss(pred, y_a, gamma=gamma, alpha=alpha)
    loss_b = compute_focal_loss(pred, y_b, gamma=gamma, alpha=alpha)
    return lam * loss_a + (1 - lam) * loss_b
# %%
# Prototype-based Alignment Utilities
# -----------------------------------
#
# Functions to compute class prototypes and prototype alignment loss.

def compute_prototypes(features, labels, n_classes=2):
    """Compute class prototypes (mean features per class)."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    prototypes = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            proto = features[mask].mean(dim=0)
        else:
            proto = torch.zeros(features.size(1), device=features.device)
        prototypes.append(proto)

    return torch.stack(prototypes)


def compute_prototype_loss(features, labels, prototypes):
    """Compute prototype alignment loss."""
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    loss = 0.0
    n_samples = 0
    for i, label in enumerate(labels):
        proto = prototypes[label]
        dist = F.mse_loss(features[i], proto)
        loss += dist
        n_samples += 1

    return loss / max(1, n_samples)
# %%
# MMD Alignment Utility
# ---------------------
#
# Compute the RBF-kernel Maximum Mean Discrepancy for distribution alignment.

def compute_mmd_rbf(x, y, eps=1e-8):
    """Compute RBF-kernel Maximum Mean Discrepancy."""
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)
    
    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        if z.size(0) > 1:
            dists = torch.cdist(z, z, p=2.0)
            sigma = torch.median(dists)
            sigma = torch.clamp(sigma, min=eps)
        else:
            sigma = torch.tensor(1.0, device=z.device)
    
    gamma = 1.0 / (2.0 * (sigma ** 2) + eps)
    k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2.0) ** 2)
    k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2.0) ** 2)
    k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2.0) ** 2)
    
    m = x.size(0)
    n = y.size(0)
    if m <= 1 or n <= 1:
        return torch.tensor(0.0, device=x.device)
    
    mmd = (k_xx.sum() - torch.trace(k_xx)) / (m * (m - 1) + eps)
    mmd += (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1) + eps)
    mmd -= 2.0 * k_xy.mean()
    
    return mmd
# %%
# Symmetric Weighting Schedule
# ----------------------------
#
# Compute domain weights and warmup schedule based on relative dataset sizes.

# Training parameters
MAX_EPOCHS = 100

def get_symmetric_adjustments(n_train_a, n_train_b):
    """Compute symmetric domain weights based on dataset sizes."""
    n_train_a = max(1, n_train_a)
    n_train_b = max(1, n_train_b)
    ratio_ab = n_train_a / float(n_train_b)
    
    w_small = float(np.clip(np.sqrt(max(ratio_ab, 1.0/ratio_ab)) * 3.0, 1.0, 12.0))
    
    overall_ratio = max(ratio_ab, 1.0 / ratio_ab)
    lambda_mmd = 0.2 if overall_ratio < 2.0 else (0.3 if overall_ratio < 4.0 else 0.45)
    lambda_proto = 0.5 if overall_ratio < 4.0 else 0.8
    warmup = max(20, min(40, int(0.4 * MAX_EPOCHS)))
    
    return w_small, lambda_mmd, lambda_proto, warmup
# %%
# Stratified Sampling
# -------------------
#
# Helper for balanced per-subject sampling while preserving class ratios.

SEEDS = [42, 123, 456, 789, 321]

def stratified_sample_trials(data, labels, n_trials, random_seed):
    """Stratified sampling of trials maintaining class balance."""
    np.random.seed(random_seed)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2 or n_trials >= len(data):
        return data, labels
    
    sampled_indices = []
    for label in unique_labels:
        label_mask = labels == label
        available = np.sum(label_mask)
        
        proportion = available / len(data)
        n_for_label = max(1, int(n_trials * proportion))
        
        if len(sampled_indices) + n_for_label > n_trials:
            n_for_label = n_trials - len(sampled_indices)
        
        if n_for_label > 0 and available >= n_for_label:
            label_indices = np.where(label_mask)[0]
            selected = np.random.choice(label_indices, size=n_for_label, replace=False)
            sampled_indices.extend(selected)
    
    sampled_indices = np.array(sampled_indices)
    return data[sampled_indices], labels[sampled_indices]


# %%
# Combined Loaders (P3 + AVO)
# ---------------------------
#
# Load both datasets using braindecode preprocessing on each raw recording.

def load_combined_arrays(channels):
    """Load and combine P3 and AVO datasets using braindecode preprocessing."""
    X_list = []
    y_list = []
    src_list = []

    for dataset_name in ['P3', 'AVO']:
        print(f"\nProcessing {dataset_name} dataset...")
        dataset = ds_p3 if dataset_name == 'P3' else ds_avo
        
        preprocessors = create_preprocessors(channels, dataset_type=dataset_name)
        
        print(f"Applying preprocessing to {len(dataset.datasets)} recordings...")
        preprocess(dataset, preprocessors)
        
        trial_start_offset_samples = int(TRIAL_START_OFFSET * RESAMPLE_FREQ)
        trial_stop_offset_samples = int((TRIAL_START_OFFSET + TRIAL_DURATION) * RESAMPLE_FREQ)
        
        windows_ds = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            window_size_samples=None,
            window_stride_samples=None,
            preload=True,
            drop_bad_windows=True,
        )
        
        for i in range(len(windows_ds)):
            window_data, window_label, *_ = windows_ds[i]
            X_list.append(window_data)
            y_list.append(window_label)
            src_list.append(dataset_name)
        
        print(f"Extracted {len([s for s in src_list if s == dataset_name])} windows from {dataset_name}")

    if not X_list:
        raise RuntimeError("No valid data loaded")

    X_all = np.array(X_list)
    y_all = np.array(y_list)
    src_all = np.array(src_list)

    print(f"\nFinal dataset: {len(X_all)} total trials")
    print(f"  P3: {np.sum(src_all=='P3')} trials")
    print(f"  AVO: {np.sum(src_all=='AVO')} trials")
    if len(X_all) > 0:
        print(f"  Data shape: {X_all.shape} (n_trials, n_chans, n_times)")

    return X_all, y_all, src_all


# %%
# AS-MMD Training Loop
# --------------------
#
# Joint training combining classification, mixup, prototype alignment, and MMD.

from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle

BATCH_SIZE = 22
LEARNING_RATE = 0.001
WEIGHT_DECAY = 2.5e-4
EARLY_STOPPING_PATIENCE = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def asmmd_train_fold(Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                     Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                     n_channels, n_times, seed=42):
    """Train a single fold with AS-MMD method."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_loader(X, y, shuffle): 
        return DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), 
                         batch_size=BATCH_SIZE, shuffle=shuffle)
    train_loader_p3 = make_loader(Xtr_p3, ytr_p3, True)
    val_loader_p3 = make_loader(Xva_p3, yva_p3, False)
    train_loader_avo = make_loader(Xtr_avo, ytr_avo, True)
    val_loader_avo = make_loader(Xva_avo, yva_avo, False)

    model = create_model(n_channels, n_times).to(DEVICE)
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    n_train_avo, n_train_p3 = len(Xtr_avo), len(Xtr_p3)
    w_small, lambda_mmd_target, lambda_proto_target, warmup_epochs = get_symmetric_adjustments(n_train_avo, n_train_p3)
    small_domain = 'P3' if n_train_p3 <= n_train_avo else 'AVO'
    large_domain = 'AVO' if small_domain == 'P3' else 'P3'

    best_val, best_state, patience_cnt = 0.0, None, 0
    large_prototypes = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        alpha = min(1.0, epoch / max(1, warmup_epochs))
        w_small_cur, w_large_cur = 1.0 + alpha * (w_small - 1.0), 1.0
        lambda_mmd, lambda_proto = alpha * lambda_mmd_target, alpha * lambda_proto_target

        train_loaders = {'P3': train_loader_p3, 'AVO': train_loader_avo}
        itr_large = iter(train_loaders[large_domain])
        itr_small = cycle(train_loaders[small_domain]) if len(train_loaders[small_domain]) > 0 else None

        for _ in train_loaders[large_domain]:
            xb_large, yb_large = next(itr_large)

            xb_small, yb_small = None, None
            if itr_small is not None:
                xb_small, yb_small = next(itr_small)

            optimizer.zero_grad()

            x_large, y_large = augment_data(normalize_data(xb_large)).to(DEVICE), yb_large.to(DEVICE)
            scores_large = model(x_large)
            loss_large = F.cross_entropy(scores_large, y_large)

            with torch.no_grad():
                current_prototypes = compute_prototypes(scores_large.detach(), y_large, n_classes=2)
                large_prototypes = current_prototypes if large_prototypes is None else 0.9 * large_prototypes + 0.1 * current_prototypes

            loss_small, loss_proto, scores_small = torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE), None
            if xb_small is not None:
                x_small, y_small = normalize_data(xb_small).to(DEVICE), yb_small.to(DEVICE)
                x_mixed, y_a, y_b, lam = mixup_data(x_small, y_small, alpha=0.4)
                scores_small = model(augment_data(x_mixed))
                loss_small = mixup_criterion(scores_small, y_a, y_b, lam, gamma=2.0, alpha=0.5)
                
                if large_prototypes is not None and lambda_proto > 0:
                    loss_proto = compute_prototype_loss(model(augment_data(x_small)), y_small, large_prototypes)

            loss_align = torch.tensor(0.0, device=DEVICE)
            if scores_small is not None and lambda_mmd > 0.0:
                scores_orig_small = model(augment_data(normalize_data(xb_small).to(DEVICE)))
                b = min(scores_large.size(0), scores_orig_small.size(0))
                loss_align = compute_mmd_rbf(scores_large[:b].detach(), scores_orig_small[:b].detach())

            total_loss = w_large_cur * loss_large + w_small_cur * loss_small + lambda_mmd * loss_align + lambda_proto * loss_proto
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        scheduler.step()

        p3_val, avo_val = evaluate_domain(model, val_loader_p3, DEVICE), evaluate_domain(model, val_loader_avo, DEVICE)
        print(f"Epoch {epoch}: Val(P3)={p3_val:.3f} | Val(AVO)={avo_val:.3f}")

        small_val = p3_val if small_domain == 'P3' else avo_val
        if small_val > best_val + 1e-4:
            best_val, best_state, patience_cnt = small_val, model.state_dict(), 0
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# %%
# Model Evaluation Utilities
# --------------------------
#
# Functions for model evaluation: simple accuracy and detailed metrics.

from sklearn.metrics import confusion_matrix, roc_auc_score

def evaluate_domain(model, loader, device):
    """Evaluate model accuracy on a specific domain."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = normalize_data(x).to(device), y.to(device)
            _, pred = model(x).max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_with_metrics(model, loader, device):
    """Evaluate with detailed metrics: accuracy, precision, recall, F1, AUC."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = normalize_data(x).to(device), y.to(device)
            scores = model(x)
            all_preds.extend(scores.max(1)[1].cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(scores, dim=1)[:, 1].cpu().numpy())
    
    all_preds, all_targets, all_probs = map(np.array, [all_preds, all_targets, all_probs])
    
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy, precision, recall, f1 = np.mean(all_preds == all_targets), 0.0, 0.0, 0.0
    
    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }


# %%
# Nested Cross-Validation and Execution
# -------------------------------------
#
# Nested CV to robustly evaluate AS-MMD across multiple splits.

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

NESTED_CV_OUTER_FOLDS = 5
NESTED_CV_REPEATS = 5

TRAIN_SIZE = 0.7
VAL_SIZE = 0.1

def run_nested_cv_asmmd(channels):
    """Run nested cross-validation with AS-MMD."""
    print("Loading datasets...")
    X_all, y_all, src_all = load_combined_arrays(channels)
    print(f"Total: {len(X_all)} (P3: {np.sum(src_all=='P3')}, AVO: {np.sum(src_all=='AVO')})")
    
    n_channels, n_times = X_all.shape[1], X_all.shape[2]
    fold_results = []
    
    for repeat in range(NESTED_CV_REPEATS):
        print(f"\n{'='*60}\nRepeat {repeat + 1}/{NESTED_CV_REPEATS}\n{'='*60}")
        cv = StratifiedKFold(n_splits=NESTED_CV_OUTER_FOLDS, shuffle=True, 
                            random_state=SEEDS[repeat % len(SEEDS)])
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
            print(f"\nFold {fold_idx + 1}/{NESTED_CV_OUTER_FOLDS}")
            X_tr_fold, y_tr_fold, src_tr_fold = X_all[train_idx], y_all[train_idx], src_all[train_idx]
            X_te_fold, y_te_fold, src_te_fold = X_all[test_idx], y_all[test_idx], src_all[test_idx]
            
            # Split train/val
            train_ratio = TRAIN_SIZE / (TRAIN_SIZE + VAL_SIZE) if (TRAIN_SIZE + VAL_SIZE) > 0 else 0.875
            tr_idx, va_idx = train_test_split(np.arange(len(X_tr_fold)), train_size=train_ratio, 
                                             stratify=y_tr_fold, random_state=42)
            
            def get_domain_data(X, y, src, indices, domain):
                mask = (src == domain)
                idx = np.intersect1d(np.where(mask)[0], indices)
                return X[idx], y[idx]
            
            Xtr_p3, ytr_p3 = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, tr_idx, 'P3')
            Xtr_avo, ytr_avo = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, tr_idx, 'AVO')
            Xva_p3, yva_p3 = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, va_idx, 'P3')
            Xva_avo, yva_avo = get_domain_data(X_tr_fold, y_tr_fold, src_tr_fold, va_idx, 'AVO')
            
            print(f"  Train sizes - P3: {len(Xtr_p3)}, AVO: {len(Xtr_avo)}")
            print(f"  Val sizes - P3: {len(Xva_p3)}, AVO: {len(Xva_avo)}")
            
            if len(Xtr_p3) == 0 or len(Xtr_avo) == 0:
                print(f"  Skipping fold {fold_idx + 1}: empty training set (P3={len(Xtr_p3)}, AVO={len(Xtr_avo)})")
                continue
            if len(Xva_p3) == 0 or len(Xva_avo) == 0:
                print(f"  Skipping fold {fold_idx + 1}: empty validation set (P3={len(Xva_p3)}, AVO={len(Xva_avo)})")
                continue
            
            model = asmmd_train_fold(Xtr_p3, ytr_p3, Xva_p3, yva_p3,
                                    Xtr_avo, ytr_avo, Xva_avo, yva_avo,
                                    n_channels, n_times, seed=SEEDS[0])
            
            def eval_domain(mask):
                if not np.any(mask):
                    return {'accuracy': 0.0, 'auc': 0.5}, 0
                loader = DataLoader(TensorDataset(torch.FloatTensor(X_te_fold[mask]), 
                                                  torch.LongTensor(y_te_fold[mask])),
                                   batch_size=BATCH_SIZE, shuffle=False)
                return evaluate_with_metrics(model, loader, DEVICE), int(np.sum(mask))
            
            m_p3, n_p3 = eval_domain(src_te_fold == 'P3')
            m_avo, n_avo = eval_domain(src_te_fold == 'AVO')
            acc_overall = (m_p3['accuracy'] * n_p3 + m_avo['accuracy'] * n_avo) / max(1, n_p3 + n_avo)
            
            print(f"  P3 Test: Acc={m_p3['accuracy']:.3f}, AUC={m_p3['auc']:.3f} (n={n_p3})")
            print(f"  AVO Test: Acc={m_avo['accuracy']:.3f}, AUC={m_avo['auc']:.3f} (n={n_avo})")
            print(f"  Overall: Acc={acc_overall:.3f}")
            
            fold_results.append({
                'repeat': repeat + 1, 'fold': fold_idx + 1, 'overall_accuracy': acc_overall,
                'p3_accuracy': m_p3['accuracy'], 'p3_auc': m_p3['auc'],
                'avo_accuracy': m_avo['accuracy'],                 'avo_auc': m_avo['auc'],
            })
    
    df = pd.DataFrame(fold_results)
    return {
        'mean_accuracy': float(df['overall_accuracy'].mean()),
        'std_accuracy': float(df['overall_accuracy'].std()),
        'p3_mean_accuracy': float(df['p3_accuracy'].mean()),
        'p3_std_accuracy': float(df['p3_accuracy'].std()),
        'p3_mean_auc': float(df['p3_auc'].mean()),
        'p3_std_auc': float(df['p3_auc'].std()),
        'avo_mean_accuracy': float(df['avo_accuracy'].mean()),
        'avo_std_accuracy': float(df['avo_accuracy'].std()),
        'avo_mean_auc': float(df['avo_auc'].mean()),
        'avo_std_auc': float(df['avo_auc'].std()),
        'detailed_fold_results': fold_results
    }

# Execute nested cross-validation
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("Starting Nested Cross-Validation with AS-MMD")

results = run_nested_cv_asmmd(COMMON_CHANNELS)

print("\n" + "="*70 + "\nFINAL RESULTS\n" + "="*70)
print(f"Overall Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
print(f"\nP3 Dataset:")
print(f"  Accuracy: {results['p3_mean_accuracy']:.4f} ± {results['p3_std_accuracy']:.4f}")
print(f"  AUC: {results['p3_mean_auc']:.4f} ± {results['p3_std_auc']:.4f}")
print(f"\nAVO Dataset:")
print(f"  Accuracy: {results['avo_mean_accuracy']:.4f} ± {results['avo_std_accuracy']:.4f}")
print(f"  AUC: {results['avo_mean_auc']:.4f} ± {results['avo_std_auc']:.4f}")
print("="*70)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pd.DataFrame(results['detailed_fold_results']).to_csv(f'asmmd_detailed_results_{timestamp}.csv', index=False)
print(f"\nDetailed results saved to: asmmd_detailed_results_{timestamp}.csv")

summary_stats = {k: v for k, v in results.items() if k != 'detailed_fold_results'}
pd.DataFrame([summary_stats]).to_csv(f'asmmd_summary_{timestamp}.csv', index=False)
print(f"Summary statistics saved to: asmmd_summary_{timestamp}.csv")

# %%
# Discussion and Hyperparameters
# -------------------------------
#
# **Key Hyperparameters:**
#
# - **Learning Rate (0.001)**: Moderate learning rate with Adamax optimizer
# - **Batch Size (22)**: Small batches help with domain adaptation
# - **Dropout (0.18)**: Prevents overfitting in the transformer layers
# - **Early Stopping Patience (10)**: Allows sufficient training time
# - **Domain Weights**: Automatically adjusted based on dataset size ratio
# - **MMD Weight (0.2-0.4)**: Balances classification and domain alignment
# - **Prototype Weight (0.5-0.8)**: Encourages discriminative class alignment
#
# **References:**
#
# - Song et al. (2020). "EEGConformer: Convolutional Transformer for EEG Decoding"
# - Long et al. (2015). "Learning Transferable Features with Deep Adaptation Networks"
# - Zhang et al. (2018). "Mixup: Beyond Empirical Risk Minimization"

# %%
# Conclusion
# ----------
# In this tutorial, we demonstrated a complete workflow for EEG P3 transfer learning
# using AS-MMD on two public oddball datasets (ERP CORE P3 and AVO). By combining
# prototype-based alignment, mixup augmentation, and MMD alignment on the logit space
# within an EEGConformer backbone, we achieved robust cross-dataset performance while
# keeping the inference-time model unchanged. The nested cross-validation protocol
# provides reliable estimates across subjects and splits. For practical use, consider
# tuning the domain weights, model capacity, and data budgets per subject, and extend
# this pipeline to other ERP components or datasets. For methodological details and
# ablations, see the accompanying paper: https://arxiv.org/abs/2510.21969
