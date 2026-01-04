"""Minimal Tutorial
================

This is a minimal tutorial demonstrating how to use EEGDash with BrainDecode.
"""

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from braindecode.models import EEGConformer
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.paths import get_default_cache_dir

# Load data
dataset = EEGChallengeDataset(
    release="R1",
    task="contrastChangeDetection",
    description_fields=["p_factor"],
    cache_dir=get_default_cache_dir(),
)

# Filter out any non-EEG files (e.g. .tsv/.json sidecars)
valid_extensions = (".vhdr", ".edf", ".bdf", ".set")
dataset.datasets = [
    ds for ds in dataset.datasets if str(ds.bidspath).endswith(valid_extensions)
]

# Preprocess
preprocess(
    dataset,
    [
        Preprocessor("resample", sfreq=100),
        Preprocessor("filter", l_freq=1, h_freq=35),
    ],
)

# Segment into windows
windows_ds = create_fixed_length_windows(
    dataset,
    window_size_samples=200,
    window_stride_samples=200,
    drop_last_window=True,
)

# Split and create loaders (not splitting by subjects, so there is obvious leakage)
train_ds, test_ds = train_test_split(windows_ds, test_size=0.2, random_state=42)
train_loader = DataLoader(train_ds, batch_size=100)
test_loader = DataLoader(test_ds, batch_size=100)

# Define model and optimizer
model = EEGConformer(
    n_chans=129,
    n_outputs=1,
    n_times=200,
    num_layers=4,
    num_heads=8,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-2)

# Train
for epoch in range(1):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch[0]), batch[1].float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Train Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            loss = F.mse_loss(model(batch[0]), batch[1].float().unsqueeze(1))
            print(f"Epoch {epoch}, Test Loss: {loss.item()}")
