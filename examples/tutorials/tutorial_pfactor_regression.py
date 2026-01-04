"""========================================================================================
P-Factor Regression Tutorial
========================================================================================
A tutorial for training an EEG Conformer model to predict the "p-factor" (a psychometric score) from EEG data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from braindecode.models import EEGConformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from eegdash import EEGDash, EEGDashDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from braindecode.datasets.base import BaseConcatDataset, BaseDataset

from eegdash.paths import get_default_cache_dir

# ============================================================================
# Configuration
# ============================================================================
CACHE_DIR_BASE = Path(get_default_cache_dir()).resolve()
CACHE_DIR_BASE.mkdir(parents=True, exist_ok=True)
DATASET_NAME = "ds005505"
TARGET_NAME = "p_factor"
CACHE_DIR = CACHE_DIR_BASE / f"reg_{DATASET_NAME}_all_{TARGET_NAME}"
SFREQ = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 5
RANDOM_SEED = 42
RECORD_LIMIT = 200

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# Data Preparation
# ============================================================================
if not CACHE_DIR.exists():
    eegdash = EEGDash()
    print(f"Preparing data for {DATASET_NAME} - {TARGET_NAME}...")

    # Fetch records (ignoring API metadata filter effectively, relying on local participants.tsv)
    query = {"dataset": DATASET_NAME}

    records = eegdash.find(query, limit=RECORD_LIMIT)

    if not records:
        # Fallback search if specific dataset returns nothing
        records = eegdash.find({}, limit=RECORD_LIMIT)

    if not records:
        raise RuntimeError("No records found from the API.")

    ds_data = EEGDashDataset(
        cache_dir=CACHE_DIR_BASE,
        records=records,
        description_fields=["subject", "session", "run", "task"],
    )

    # Load real metadata from participants.tsv
    # Try multiple standard locations
    possible_paths = [
        ds_data.data_dir / "participants.tsv",
        CACHE_DIR_BASE / DATASET_NAME / "participants.tsv",
        Path.home() / ".eegdash_cache" / DATASET_NAME / "participants.tsv",
        Path.cwd() / ".eegdash_cache" / DATASET_NAME / "participants.tsv",
    ]
    participants_path = None
    for p in possible_paths:
        if p.exists():
            participants_path = p
            print(f"Using participants.tsv found at: {p}")
            break

    meta_lookup = {}
    if participants_path:
        df_participants = pd.read_csv(participants_path, sep="\t")
        df_participants["subject"] = df_participants["participant_id"].apply(
            lambda s: s.replace("sub-", "")
        )
        # p_factor might be a string, convert to numeric
        df_participants[TARGET_NAME] = pd.to_numeric(
            df_participants[TARGET_NAME], errors="coerce"
        )
        # Create lookup: subject -> p_factor
        meta_lookup = df_participants.set_index("subject")[TARGET_NAME].to_dict()
        print(f"Loaded metadata for {len(meta_lookup)} subjects.")
    else:
        print(f"Warning: participants.tsv not found in {possible_paths}")

    filtered_datasets = []
    for i, ds in enumerate(ds_data.datasets):
        if i < len(records):
            subj_rec = records[i].get("subject")
        else:
            continue  # excessive robustness

        if not subj_rec:
            if ds.description is not None:
                subj_rec = str(ds.description.get("subject"))
            else:
                continue

        subj = subj_rec.replace("sub-", "")

        target_val = meta_lookup.get(subj)

        # Check validity of target
        if target_val is None or np.isnan(target_val):
            continue

        if len(ds) == 0:
            continue

        # Create new description
        new_desc = pd.Series(
            {
                "subject": subj,
                TARGET_NAME: float(target_val),
                "session": records[i].get("session"),
                "run": records[i].get("run"),
            }
        )

        # New BaseDataset
        new_ds = BaseDataset(ds.raw, new_desc)
        filtered_datasets.append(new_ds)

    print(f"Retained {len(filtered_datasets)} datasets with valid {TARGET_NAME}.")

    if not filtered_datasets:
        raise RuntimeError(f"No datasets remained after filtering for {TARGET_NAME}.")

    all_datasets = BaseConcatDataset(filtered_datasets)

    # Preprocessing
    ch_names = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]  # reduced set for speed
    preprocessors = [
        Preprocessor("pick_channels", ch_names=ch_names, ordered=True),
        Preprocessor("resample", sfreq=SFREQ),
        Preprocessor("filter", l_freq=1, h_freq=30),
    ]

    print("Preprocessing...")
    preprocess(all_datasets, preprocessors, n_jobs=2)

    # Windowing
    windows_ds = create_fixed_length_windows(
        all_datasets,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=SFREQ * 2,  # 2 seconds
        window_stride_samples=SFREQ * 2,
        drop_last_window=True,
        preload=False,
    )

    for ds in windows_ds.datasets:
        ds.target_name = TARGET_NAME

    # Save
    windows_ds.save(str(CACHE_DIR), overwrite=True)
    print(f"Data saved to {CACHE_DIR}")

else:
    print(f"Loading data from {CACHE_DIR}...")
    from braindecode.datautil import load_concat_dataset

    windows_ds = load_concat_dataset(path=str(CACHE_DIR), preload=False)

# ============================================================================
# Splitting and Loading
# ============================================================================
# Basic split by subject
subjects = np.array([ds.description["subject"] for ds in windows_ds.datasets])
unique_subs = np.unique(subjects)
train_subs, val_subs = train_test_split(
    unique_subs, test_size=0.2, random_state=RANDOM_SEED
)

train_ds = [ds for ds in windows_ds.datasets if ds.description["subject"] in train_subs]
val_ds = [ds for ds in windows_ds.datasets if ds.description["subject"] in val_subs]

train_ds = BaseConcatDataset(train_ds)
val_ds = BaseConcatDataset(val_ds)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

# ============================================================================
# Model
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
# simple check for mps
if (
    not torch.cuda.is_available()
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = "mps"

print(f"Using device: {device}")

# Assuming 8 channels from our preprocessing
n_chans = 8
n_times = SFREQ * 2

model = EEGConformer(
    n_chans=n_chans,
    n_outputs=1,  # Regression
    n_times=n_times,
    sfreq=SFREQ,
    num_layers=3,  # Simplified for tutorial
    num_heads=4,
    final_fc_length="auto",
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)


def normalize_batch(x):
    # (B, C, T)
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-6
    return (x - mean) / std


# ============================================================================
# Training
# ============================================================================
history = {"train_loss": [], "val_loss": []}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss_accum = 0
    count = 0

    for x, y, _ in train_loader:
        x = x.to(device).float()
        y = y.to(device).float()
        x = normalize_batch(x)

        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item() * len(x)
        count += len(x)

    avg_train_loss = train_loss_accum / count

    model.eval()
    val_loss_accum = 0
    val_count = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            x = normalize_batch(x)

            preds = model(x).squeeze()
            loss = F.mse_loss(preds, y)
            val_loss_accum += loss.item() * len(x)
            val_count += len(x)

    avg_val_loss = val_loss_accum / val_count

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}"
    )

# ============================================================================
# Visualization
# ============================================================================
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train MSE")
plt.plot(history["val_loss"], label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("P-Factor Regression Training")
plt.legend()
plt.tight_layout()
plt.show()  # In a real script we might save it, but here we show
