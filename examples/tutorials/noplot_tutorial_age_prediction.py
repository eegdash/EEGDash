"""Age Prediction from EEG
===========================

**Objective**: Learn how to predict a continuous variable (Subject Age) from raw EEG data using a Convolutional Neural Network (Conformer).

**What you will learn**:

1.  **Data Retrieval**: How to fetch specific datasets (e.g., Healthy Brain Network) using `EEGDash`.
2.  **Preprocessing**: Applying standard EEG cleaning techniques (filtering, resampling) with `BrainDecode`.
3.  **Windowing**: cutting continuous EEG into fixed-length training windows.
4.  **Modeling**: Training a Conformer model (Transformer-based) using PyTorch.
5.  **Interpretation**: Visualizing training progress (MAE/RMSE).

.. tip::
    This tutorial assumes basic familiarity with PyTorch. If you are new to EEG, check out the `Minimal Tutorial` first.
"""

from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from braindecode.datautil import load_concat_dataset
from braindecode.models import EEGConformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from eegdash import EEGDash, EEGDashDataset

# %%
# Configuration & Setup
# ---------------------
# We start by defining our hyperparameters and caching paths.
# Using a centralized ``CACHE_DIR`` ensures we don't re-download data unnecessarily.


# ============================================================================
# Configuration
# ============================================================================
CACHE_DIR_BASE = Path(
    os.getenv("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache")
).resolve()
CACHE_DIR_BASE.mkdir(parents=True, exist_ok=True)
DATASET_NAME = "ds005505"
TARGET_NAME = "age"
CACHE_DIR = CACHE_DIR_BASE / f"reg_{DATASET_NAME}_all_{TARGET_NAME}"
SFREQ = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.00002
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 5
RANDOM_SEED = 41
RECORD_LIMIT = 60

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %%
# Data Preparation
# ----------------
# We need to fetch metadata from the EEGDash API and then download the corresponding raw files.
# The ``EEGDash`` client handles the metadata query, allowing us to find subjects with valid "age" labels.
#
# .. note::
#    For this tutorial, we limit the dataset to just 10 subjects to ensure quick execution.
#    In a real scenario, you would use the full dataset.
#
# .. note::
#    The ``PREPARE_DATA`` flag is a safety switch. In a real workflow, you usually process raw data once
#    and then load the processed windows from disk for all subsequent experiments.

PREPARE_DATA = True  # Set to True to prepare data from scratch

if PREPARE_DATA or not CACHE_DIR.exists():
    eegdash = EEGDash()
    from braindecode.preprocessing import (
        Preprocessor,
        create_fixed_length_windows,
        preprocess,
    )
    from braindecode.datasets.base import BaseConcatDataset

    print(f"Preparing data for {DATASET_NAME} - {TARGET_NAME}...")

    # Load raw dataset from API records, requesting age
    ds_data = EEGDashDataset(
        dataset=DATASET_NAME,
        cache_dir=CACHE_DIR_BASE,
        description_fields=["subject", "session", "run", "task", "age", "sex"],
    )

    # Filter subjects: remove problematic subjects
    sub_rm = {
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
        "041",
    }

    filtered_datasets = []

    # Reconstruct datasets with valid description
    for ds in ds_data.datasets:
        subj = ds.description.get("subject", "")
        if subj is None:
            continue
        subj = str(subj).replace("sub-", "")

        # Check exclusion list
        if subj in sub_rm:
            continue

        # Check age validity
        age_val = ds.description.get("age")
        if age_val is None:
            continue
        try:
            age = float(age_val)
        except (ValueError, TypeError):
            continue

        if np.isnan(age):
            continue

        # Update description with clean values
        ds.description["age"] = age
        ds.description["subject"] = subj

        # Check data is not empty
        if len(ds) == 0:
            continue

        # Data quality checks (moved inside loop to ensure we get 10 VALID subjects)
        # Note: accessing ds.raw triggers download if not cached
        try:
            if ds.raw.n_times < 4 * SFREQ:
                print(f"Skipping {subj}: duration {ds.raw.n_times / SFREQ:.2f}s < 4s")
                continue
            if len(ds.raw.ch_names) != 64:
                print(f"Skipping {subj}: channel count {len(ds.raw.ch_names)} != 64")
                continue
        except Exception as e:
            print(f"Skipping {subj}: failed to load raw data ({e})")
            continue

        filtered_datasets.append(ds)

        # LIMIT FOR TUTORIAL: Stop after 10 valid subjects
        if len(filtered_datasets) >= 10:
            print("Reached limit of 10 valid subjects for tutorial demonstration.")
            break

    if len(filtered_datasets) == 0:
        raise RuntimeError(
            "No valid datasets found (checked for metadata and data quality)."
        )

    all_datasets = BaseConcatDataset(filtered_datasets)

    if len(all_datasets.datasets) == 0:
        raise RuntimeError("No datasets remaining after quality checks.")

    # Define preprocessing pipeline - select a subset of standard 10-20 channels
    # We downsample to 128Hz to reduce computational load while keeping relevant brain frequencies.
    # We filter between 1-55Hz to remove DC drift (<1Hz) and line noise/high-freq artifacts (>55Hz).
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
        "Oz",
        "FC1",
        "FC2",
        "CP1",
        "CP2",
    ]
    preprocessors = [
        Preprocessor("pick_channels", ch_names=ch_names),
        Preprocessor("resample", sfreq=128),
        Preprocessor("filter", l_freq=1, h_freq=55, picks=ch_names),
    ]

    # Apply preprocessing
    print(f"Preprocessing {len(all_datasets.datasets)} datasets...")
    preprocess(all_datasets, preprocessors, n_jobs=16)
    print("Preprocessing completed!")

    # Create fixed-length windows
    windows_ds = create_fixed_length_windows(
        all_datasets,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=256,
        window_stride_samples=256,
        drop_last_window=True,
        preload=False,
    )
    for ds in windows_ds.datasets:
        ds.target_name = "age"

    # Save processed data
    os.makedirs(CACHE_DIR, exist_ok=True)
    windows_ds.save(str(CACHE_DIR), overwrite=True)
    print(f"Data saved to {CACHE_DIR}")

# ============================================================================
# Load Dataset
# ============================================================================
print(f"Loading data from {CACHE_DIR}...")
windows_ds = load_concat_dataset(path=str(CACHE_DIR), preload=False)
print(f"Loaded {len(windows_ds.datasets)} subjects, {len(windows_ds)} windows total")


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


windows_ds.description["age"] = windows_ds.description["age"].apply(_to_float)
for ds in windows_ds.datasets:
    ds.target_name = "age"

# ============================================================================
# Train/Validation Split (80/20)
# ============================================================================
unique_subjects = np.unique(windows_ds.description["subject"])
train_subj, val_subj = train_test_split(
    unique_subjects, train_size=0.8, random_state=RANDOM_SEED
)

# Filter valid age values and create train/val datasets
train_ds = [
    ds
    for ds in windows_ds.datasets
    if ds.description.subject in train_subj
    and ds.description.age is not None
    and float(ds.description.age) > 0.5
]
val_ds = [
    ds
    for ds in windows_ds.datasets
    if ds.description.subject in val_subj
    and ds.description.age is not None
    and float(ds.description.age) > 0.5
]

from braindecode.datasets.base import BaseConcatDataset

train_ds = BaseConcatDataset(train_ds)
val_ds = BaseConcatDataset(val_ds)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

# %%
# Model Architecture: EEGConformer
# --------------------------------
# We use the **Conformer** architecture, which combines Convolutional Neural Networks (CNNs)
# for local feature extraction with Transformers for capturing long-range global dependencies.
#
# * ``n_times=256``: Matches our window length (2 seconds @ 128Hz)
# * ``n_outputs=1``: We are doing regression (predicting a single float value: age).

# ============================================================================
# Initialize Model (EEGConformerSimplified)
# ============================================================================
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = EEGConformer(
    n_chans=24,
    n_outputs=1,
    n_times=256,
    sfreq=128,
    drop_prob=0.7,
    n_filters_time=32,
    filter_time_length=20,
    num_layers=4,
    num_heads=8,
    pool_time_stride=12,
    pool_time_length=64,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)


# ============================================================================
# Helper Function: Normalize EEG Data
# ============================================================================
def normalize_data(x):
    x = x.reshape(x.shape[0], 24, 256).float()
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-7
    return (x - mean) / std


# ============================================================================
# Calculate Baseline Metrics
# ============================================================================
# Collect all training and validation ages for baseline calculation
train_ages = np.array([ds.description.age for ds in train_ds.datasets])
val_ages = np.array([ds.description.age for ds in val_ds.datasets])

# Baseline MAE: predict median age for all samples
train_median = np.median(train_ages)
baseline_train_mae = np.mean(np.abs(train_ages - train_median))
baseline_val_mae = np.mean(np.abs(val_ages - train_median))

# Baseline RMSE: predict mean age for all samples
train_mean = np.mean(train_ages)
baseline_train_rmse = np.sqrt(np.mean((train_ages - train_mean) ** 2))
baseline_val_rmse = np.sqrt(np.mean((val_ages - train_mean) ** 2))

print(
    f"Baseline (predict median={train_median:.1f}): Train MAE={baseline_train_mae:.4f}, Val MAE={baseline_val_mae:.4f}"
)
print(
    f"Baseline (predict mean={train_mean:.1f}): Train RMSE={baseline_train_rmse:.4f}, Val RMSE={baseline_val_rmse:.4f}"
)

# ============================================================================
# Training Loop
# ============================================================================
history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    train_mae_sum, train_mse_sum, train_count = 0.0, 0.0, 0

    for x, y, _ in train_loader:
        x = normalize_data(x).to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = F.l1_loss(preds, y)
        loss.backward()
        optimizer.step()

        train_mae_sum += loss.item() * len(y)
        train_mse_sum += F.mse_loss(preds, y).item() * len(y)
        train_count += len(y)

    train_mae = train_mae_sum / train_count
    train_rmse = np.sqrt(train_mse_sum / train_count)

    # --- Validation Phase ---
    model.eval()
    val_mae_sum, val_mse_sum, val_count = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = normalize_data(x).to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            preds = model(x).squeeze()
            val_mae_sum += F.l1_loss(preds, y).item() * len(y)
            val_mse_sum += F.mse_loss(preds, y).item() * len(y)
            val_count += len(y)

    val_mae = val_mae_sum / val_count
    val_rmse = np.sqrt(val_mse_sum / val_count)

    # Store history
    history["train_loss"].append(train_mae)
    history["val_loss"].append(val_mae)
    history["train_rmse"].append(train_rmse)
    history["val_rmse"].append(val_rmse)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train MAE: {train_mae:.4f} RMSE: {train_rmse:.4f} | Val MAE: {val_mae:.4f} RMSE: {val_rmse:.4f}"
    )

# ============================================================================
# Plot Results
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Train vs Val Loss (MAE)
axes[0].plot(history["train_loss"], label="Train MAE", marker="o")
axes[0].plot(history["val_loss"], label="Val MAE", marker="s")
axes[0].axhline(
    y=baseline_train_mae,
    color="blue",
    linestyle="--",
    alpha=0.5,
    label=f"Train Baseline (median) = {baseline_train_mae:.2f}",
)
axes[0].axhline(
    y=baseline_val_mae,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"Val Baseline (median) = {baseline_val_mae:.2f}",
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MAE")
axes[0].set_title("Train vs Validation Loss (MAE)")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Train vs Val RMSE
axes[1].plot(history["train_rmse"], label="Train RMSE", marker="o")
axes[1].plot(history["val_rmse"], label="Val RMSE", marker="s")
axes[1].axhline(
    y=baseline_train_rmse,
    color="blue",
    linestyle="--",
    alpha=0.5,
    label=f"Train Baseline (mean) = {baseline_train_rmse:.2f}",
)
axes[1].axhline(
    y=baseline_val_rmse,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"Val Baseline (mean) = {baseline_val_rmse:.2f}",
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("RMSE")
axes[1].set_title("Train vs Validation RMSE")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\nTraining complete! Results saved to 'training_results.png'")
plt.show()
