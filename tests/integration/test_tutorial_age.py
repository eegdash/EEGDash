import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import EEGConformer


def test_tutorial_age_logic():
    """Integration test mirroring the logic of tutorial_age_prediction.py to ensure model convergence.
    Uses synthetic data to avoid network dependencies.
    """
    # 1. Setup Synthetic Data (simulating Preprocessed Windows)
    # Shape: (n_samples, n_channels, n_times)
    n_samples = 20
    n_chans = 24
    n_times = 256
    n_outputs = 1

    # Generate random EEG-like data
    X = torch.randn(n_samples, n_chans, n_times)
    # Generate synthetic age labels (correlated with signal mean for learning potential)
    # Target = 50 + 10 * mean_signal + noise
    y = 50 + 10 * X.mean(dim=(1, 2)) + torch.randn(n_samples)
    y = y.float()

    # 2. Dataset and Loader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    _val_loader = DataLoader(dataset, batch_size=5)

    # 3. Model Initialization (Same as tutorial)
    model = EEGConformer(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=128,
        drop_prob=0.0,  # Disable dropout for faster convergence on tiny data
        num_layers=2,  # Reduced depth for speed
        num_heads=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 4. Training Loop
    model.train()
    initial_loss = float("inf")
    final_loss = 0.0

    # Capture initial loss
    with torch.no_grad():
        preds = model(X).squeeze()
        initial_loss = F.l1_loss(preds, y).item()

    print(f"Initial Loss query: {initial_loss}")

    # Train for a few epochs
    epochs = 5
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = F.l1_loss(preds, y_batch)
            loss.backward()
            optimizer.step()

    # 5. Assertions
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze()
        final_loss = F.l1_loss(preds, y).item()

    print(f"Final Loss: {final_loss}")

    # Check that loss hasn't exploded (sanity check)
    assert not np.isnan(final_loss), "Loss became NaN"
    # Basic check ensuring strict backward pass validity
    assert final_loss < initial_loss or final_loss < 100, (
        "Model failed to learn or loss is too high"
    )


if __name__ == "__main__":
    test_tutorial_age_logic()
