""".. _tutorial-surround-suppression:
    Surround Suppression Contrastive Learning Tutorial
    =================================================

    This script demonstrates how to use the EEGDash and Braindecode libraries to perform a simple contrastive learning on EEG data from a surround suppression visual task. The workflow includes data retrieval, preprocessing, contrastive pair sampling, model definition, and training.

    Key Steps:
    1. **Data Retrieval**: Uses EEGDashDataset to query and retrieve metadata for a surround suppression EEG dataset.
    2. **Preprocessing**: Applies channel selection, resampling, filtering, and epoch extraction using Braindecode to prepare the data for analysis.
    3. **Contrastive Dataset and Sampler**: Implements custom classes to generate pairs of EEG samples for contrastive learning, ensuring pairs are from the same or different stimulus conditions.
    4. **Model Architecture**: Defines a shallow convolutional neural network (ShallowFBCSPNet) as an encoder to produce embeddings, and a classification head that predicts whether paired samples are from the same class.
    5. **Contrastive Training**: Trains the encoder and classification head jointly using binary cross-entropy loss on paired samples.

    ---------------
    - **Pretrained Encoder Usage**: After training, only the encoder model (ShallowFBCSPNet) is of interest. The classification head is discarded.
    - **Downstream Applications**: For future downstream tasks, the pretrained encoder will be used to transform EEG data into an embedding space. Additional fine-tuning or task-specific heads can then be applied to these embeddings for specific analyses or predictions.
"""

# %%
# Data Retrieval Using EEGDash
# ----------------------------
#
# We retrieve a surround suppression EEG dataset from the EEGDash repository.
# This dataset contains visual stimuli presentations with different stimulus conditions
# that will be used for contrastive learning.
from pathlib import Path

cache_folder = Path.home() / "eegdash"
# %%
from eegdash import EEGDashDataset
from braindecode.datasets.base import BaseConcatDataset

ds_sus = EEGDashDataset(
    query={"dataset": "ds005514", "task": "surroundSupp", "subject": "NDARDB033FW5"},
    cache_dir=cache_folder, download=False
)

sfreq = ds_sus.datasets[0].raw.info["sfreq"]  # Get original sampling frequency

# %%
# Data Preprocessing Using Braindecode
# ------------------------------------
#
# [BrainDecode](https://braindecode.org/stable/install/install.html) is a
# specialized library for preprocessing EEG and MEG data. In this surround suppression dataset,
# the key events are **stim_ON** (visual stimulus presentation) and **fixpoint_ON** (fixation point).
# We extract non-overlapping 2-second epochs after the event onset.
#
# Next, we apply three preprocessing steps in Braindecode:
# 1. **Channel Selection** of 24 specific EEG channels from the original 128.
# 2. **Resampling** the EEG data to a frequency of 128 Hz.
# 3. **Filtering** the EEG signals to retain frequencies between 1 Hz and 55 Hz.
#
# When calling the `preprocess` function, the data is retrieved from the remote
# repository.
# %%
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
import numpy as np
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# BrainDecode preprocessors
preprocessors = [
    Preprocessor(
        "pick_channels",
        ch_names=[
            "E22", "E9", "E33", "E24", "E11", "E124", "E122", "E29", "E6", "E111",
            "E45", "E36", "E104", "E108", "E42", "E55", "E93", "E58", "E52", "E62",
            "E92", "E96", "E70", "Cz",
        ],  # 24 channels selected for optimal visual processing coverage including occipital and parietal regions
    ),
    Preprocessor("resample", sfreq=128),  # Downsample to 128 Hz for computational efficiency
    Preprocessor("filter", l_freq=1, h_freq=55),  # Bandpass filter to remove slow drifts and high-frequency noise
]
preprocess(ds_sus, preprocessors)

# %%
# Finally, we use `create_windows_from_events` to extract 2-second epochs from
# the data. These epochs serve as the dataset samples. At this stage, each
# sample is automatically labeled with the corresponding event type (stimulus
# presentation or fixation point). `windows_ds` is a PyTorch dataset, and when queried, it
# returns labels for the different stimulus conditions (assigned as labels 0 and 1,
# corresponding to their respective event markers).

# Extract 2-second segments
windows_ds = create_windows_from_events(
    ds_sus,
    mapping={"stim_ON": 1,
             "fixpoint_ON": 0},  # Map event markers to labels: stimulus=1, fixation=0
    window_size_samples=int(128 * 2),  # 2 seconds at 128 Hz
    window_stride_samples=int(128 * 2),  
    trial_start_offset_samples=0,
    trial_stop_offset_samples=int(128 * 2),
    preload=True,
)
print("Number of samples:", len(windows_ds))
print("Classes:", np.unique(windows_ds.get_metadata().target, return_counts=True))

# %%
# Plotting a Single Channel for One Sample
# ----------------------------------------
#
# It’s always a good practice to verify that the data has been properly loaded
# and processed. Here, we plot a single channel from one sample to ensure the
# signal is present and looks as expected.

# %%
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(windows_ds[2][0][0, :].transpose())  # first channel of first epoch
# plt.show()


# %%
# Creating the Contrastive DataLoader
# -----------------------------------
#
# We use a custom **ContrastiveSampler** and **ContrastiveDataset** 
# to generate pairs of EEG samples for contrastive learning. 
# The sampler presamples pairs of indices (with labels indicating
# whether they are from the same class), and the dataset returns the
# corresponding EEG data for each pair.
#
# 1. **Set Random Seed** – The random seed is fixed using
#    `torch.manual_seed(random_state)` and `np.random.seed(random_state)` to
#    ensure reproducibility in sampling and model training.
# 2. **Create ContrastiveDataset** – Wrap the windowed EEG datasets in a
#    `ContrastiveDataset` to enable paired sample retrieval.
# 3. **Create ContrastiveSampler** – Use `ContrastiveSampler` to generate
#    presampled pairs of indices and labels for contrastive learning.
# 4. **Create DataLoader** – The DataLoader uses the custom dataset and sampler,
#    providing batches of paired EEG samples and their labels for training.
#
# This setup is tailored for contrastive learning, where each batch contains
# pairs of EEG samples and a label indicating if they belong to the same class.
from braindecode.samplers import RecordingSampler

class ContrastiveSampler(RecordingSampler):
    """Sample examples for the contrastive task.

    This sampler creates balanced pairs of EEG epochs for contrastive learning.
    It ensures 50% of pairs are from the same stimulus condition (positive pairs)
    and 50% are from different conditions (negative pairs). This balanced sampling
    is crucial for effective contrastive learning.

    Sample examples as tuples of two window indices, with a label indicating
    whether the windows are from the same class (1) or not (0).

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_examples : int
        Number of pairs to extract.
    random_state : None | np.RandomState | int
        Random state.
    """
    def __init__(
        self,
        metadata,
        n_examples,
        random_state=None,
    ):
        super().__init__(metadata, random_state=random_state)
        self.n_examples = n_examples

    def _sample_pair(self):
        """Sample a pair of two windows for contrastive learning.
        
        This method implements the core logic for creating balanced positive and negative pairs:
        1. Randomly decides whether to create a positive pair (same class) or negative pair (different classes)
        2. Samples two different epochs that satisfy the chosen pair type
        3. Ensures no self-pairing (same epoch used twice)
        
        Returns
        -------
        tuple
            (win_ind1, win_ind2, label) where label=1 for same class, label=0 for different classes
        """
        # Sample first window
        win_ind1, rec_ind1 = self.sample_window()
        ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
        ts1_target = self.metadata.iloc[win_ind1]["target"]

        # Sample second window
        win_ind2, rec_ind2 = self.sample_window()
        ts2 = self.metadata.iloc[win_ind2]["i_start_in_trial"]
        ts2_target = self.metadata.iloc[win_ind2]["target"]

        current_pair_type = 0 if ts1_target != ts2_target else 1
        selected_pair_type = self.rng.binomial(1, 0.5)  # 50/50 chance for same/different type pairs

        # Keep sampling until we have two different windows and the right pair type
        while ts1 == ts2 or current_pair_type != selected_pair_type:
            win_ind2, rec_ind2 = self.sample_window()
            ts2 = self.metadata.iloc[win_ind2]["i_start_in_trial"]
            ts2_target = self.metadata.iloc[win_ind2]["target"]
            current_pair_type = 0 if ts1_target != ts2_target else 1

        return win_ind1, win_ind2, float(selected_pair_type)

    def presample(self):
        """Presample examples.

        Once presampled, the examples are the same from one epoch to another.
        """
        self.examples = [self._sample_pair() for _ in range(self.n_examples)]
        return self

    def __iter__(self):
        for i in range(self.n_examples):
            if hasattr(self, "examples"):
                yield self.examples[i]
            else:
                yield self._sample_pair()

    def __len__(self):
        return self.n_examples

class ContrastiveDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
    
    This dataset wrapper enables contrastive learning by modifying the __getitem__
    method to return pairs of EEG epochs instead of single epochs. When return_pair
    is True, it expects a tuple (index1, index2, label) and returns the corresponding
    pair of EEG data along with the contrastive label.
    
    The contrastive label indicates whether the two epochs are from the same stimulus
    condition (1) or different conditions (0).
    """

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0], super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value

contrastive_windows_ds = ContrastiveDataset(windows_ds.datasets)

multiply_factor = 10  # How many times to oversample the dataset
n_examples = multiply_factor * len(contrastive_windows_ds.datasets)
random_state = 42

contrastive_sampler = ContrastiveSampler(
    contrastive_windows_ds.get_metadata(),
    n_examples=n_examples,
    random_state=random_state,
)

# %%
import torch
from torch.utils.data import DataLoader

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

train_dataloader = DataLoader(
    contrastive_windows_ds, batch_size=10, sampler=contrastive_sampler
)

# %%
# Create model
# ------------
#
# The model consists of a shallow convolutional neural network encoder (ShallowFBCSPNet)
# that outputs a 64-dimensional embedding for each EEG sample (24 channels, 256 time points).
# For contrastive learning, two samples are passed through the encoder, and the absolute
# difference of their embeddings is computed. This difference is then passed through a
# linear classification head to predict whether the two samples belong to the same class.

# %%
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

torch.manual_seed(random_state)
embedding_dim = 64
encoder_model = ShallowFBCSPNet(24, embedding_dim, n_times=256, final_conv_length="auto")
classification_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    nn.Dropout(0.5),
    torch.nn.Linear(embedding_dim, 1)  
)
summary(encoder_model, input_size=(1, 24, 256))

# %%
# Model Training Loop for Contrastive Learning
# --------------------------------------------
#
# This section trains the encoder and classification head using the Adamax optimizer.
# The contrastive learning approach works as follows:
# 
# 1. **Pair Processing**: Each batch contains pairs of EEG samples (x1, x2) and binary labels y
# 2. **Normalization**: Both samples are z-score normalized across time dimension
# 3. **Encoding**: Both samples pass through the ShallowFBCSPNet encoder → 64D embeddings
# 4. **Similarity Computation**: Absolute difference |z1 - z2| captures embedding similarity
# 5. **Binary Classification**: Linear head predicts if pairs are from same condition
# 6. **Loss**: Binary cross-entropy loss encourages similar embeddings for same-condition pairs
#
# Training details:
# - Optimizer: Adamax (lr=0.002, weight_decay=0.001)
# - Scheduler: ExponentialLR (gamma=1, no decay)
# - Epochs: 2 (for demonstration)
# - Device: Auto-detects CUDA > MPS > CPU

# %%
params = list(encoder_model.parameters()) + list(classification_model.parameters())
optimizer = torch.optim.Adamax(params, lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
encoder_model = encoder_model.to(device=device)  # move the model parameters to CPU/GPU
classification_model = classification_model.to(device=device)
epochs = 2

def normalize_data(x):
    """Normalize EEG data using z-score normalization across time dimension.
    
    This normalization is applied per channel and per sample, ensuring that each
    EEG channel has zero mean and unit variance across the time dimension. This
    standardization helps the model focus on temporal patterns rather than absolute
    amplitude differences between channels or trials.
    
    Parameters
    ----------
    x : torch.Tensor
        Input EEG data of shape (batch_size, n_channels, n_timepoints)
        
    Returns
    -------
    torch.Tensor
        Normalized EEG data moved to the specified device
    """
    mean = x.mean(dim=2, keepdim=True)  # Mean across time dimension
    std = x.std(dim=2, keepdim=True) + 1e-7  # Std across time + epsilon for numerical stability
    x = (x - mean) / std  # Z-score normalization
    x = x.to(device=device, dtype=torch.float32)  # Move to device (GPU/MPS/CPU)
    return x

encoder_model.train()  # put model to training mode
classification_model.train()  # put model to training mode
for e in range(epochs):
    # training
    epoch_loss = 0
    for t, batch in enumerate(train_dataloader):
        (x1, x2), y = batch
        y = y.to(device=device, dtype=torch.long)

        x1, x2 = normalize_data(x1), normalize_data(x2)

        z1, z2 = encoder_model(x1), encoder_model(x2)

        y_pred = classification_model(torch.abs(z1 - z2).squeeze()).squeeze()

        loss = F.binary_cross_entropy(y_pred, y.float())
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {e} average loss: {epoch_loss / len(train_dataloader):.4f}")

# %%
# Save encoder model weights
# --------------------------
#
# Save the trained encoder weights for future use. This encoder has learned
# meaningful representations from the surround suppression EEG data through
# contrastive learning and can be used for downstream tasks.
# %%
torch.save(encoder_model.state_dict(), "encoder_model_weights.pth")