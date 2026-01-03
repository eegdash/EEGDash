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

# Load data
dataset = EEGChallengeDataset(
    release="R1",
    task="contrastChangeDetection",
    description_fields=["p_factor"],
    cache_dir="/Users/arno/eegdash_data/eeg2025_competition",
)

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
    att_depth=4,
    att_heads=8,
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
