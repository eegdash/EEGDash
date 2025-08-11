"""EEGDash example for eyes open vs. closed classification

The code below provides an example of using the *EEGDash* library in combination with PyTorch to develop a deep learning model for analyzing EEG data, specifically for eyes open vs. closed classification in a single subject.

1. **Data Retrieval Using EEGDash**: An instance of *EEGDashDataset* is created to search and retrieve an EEG dataset. At this step, only the metadata is transferred.

2. **Data Preprocessing Using BrainDecode**: This process preprocesses EEG data using Braindecode by reannotating events, selecting specific channels, resampling, filtering, and extracting 2-second epochs, ensuring balanced eyes-open and eyes-closed data for analysis.

3. **Extracting EEG Features Using EEGDash.features**: Building a feature extraction tree using existing and new features.

4. **Creating train and testing sets**: The dataset is split into training (80%) and testing (20%) sets with balanced labels, converted into PyTorch tensors, and wrapped in DataLoader objects for efficient mini-batch training.

5. **Model Definition**: The model is a MLP with `n_features` input channels, 2 output classes (eyes-open and eyes-closed).

6. **Model Training and Evaluation Process**: This section trains the neural network, computes cross-entropy loss, updates model parameters, and evaluates classification accuracy over six epochs.

"""

# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset. This dataset contains both eyes open and eyes closed data.

# %%
from eegdash import EEGDashDataset

ds_eoec = EEGDashDataset(
    {"dataset": "ds005514", "task": "RestingState", "subject": "NDARDB033FW5"}
)

# %% [markdown]
# ## Data Preprocessing Using Braindecode
#
# [BrainDecode](https://braindecode.org/stable/install/install.html) is a specialized library for preprocessing EEG and MEG data. In this dataset, there are two key events in the continuous data: **instructed_toCloseEyes**, marking the start of a 40-second eyes-closed period, and **instructed_toOpenEyes**, indicating the start of a 20-second eyes-open period.
#
# For the eyes-closed event, we extract 14 seconds of data from 15 to 29 seconds after the event onset. Similarly, for the eyes-open event, we extract data from 5 to 19 seconds after the event onset. This ensures an equal amount of data for both conditions. The event extraction is handled by the custom function **hbn_ec_ec_reannotation**.
#
# Next, we apply four preprocessing steps in Braindecode:
# 1.	**Reannotation** of event markers using hbn_ec_ec_reannotation().
# 2.	**Selection** of 24 specific EEG channels from the original 128.
# 3.	**Resampling** the EEG data to a frequency of 128 Hz.
# 4.	**Filtering** the EEG signals to retain frequencies between 1 Hz and 55 Hz.
#
# When calling the **preprocess** function, the data is retrieved from the remote repository.
#
# Finally, we use **create_windows_from_events** to extract 5-second epochs from the data. These epochs serve as the dataset samples. At this stage, each sample is automatically labeled with the corresponding event type (eyes-open or eyes-closed). windows_ds is a PyTorch dataset, and when queried, it returns labels for eyes-open and eyes-closed (assigned as labels 0 and 1, corresponding to their respective event markers).

# %%
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
import numpy as np
import mne
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)


class hbn_ec_ec_reannotation(Preprocessor):
    def __init__(self):
        super().__init__(
            fn=self.transform, apply_on_array=False
        )  # Pass the transform method as the function

    def transform(self, raw):  # Changed from 'apply' to 'transform'
        # Create events array from annotations
        events, event_id = mne.events_from_annotations(raw)

        print(event_id)

        # Create new events array for 2-second segments
        new_events = []
        sfreq = raw.info["sfreq"]
        for event in events[events[:, 2] == event_id["instructed_toCloseEyes"]]:
            # For each original event, create events every 2 seconds from 15s to 29s after
            start_times = event[0] + np.arange(15, 29, 2) * sfreq
            new_events.extend([[int(t), 0, 1] for t in start_times])

        for event in events[events[:, 2] == event_id["instructed_toOpenEyes"]]:
            # For each original event, create events every 2 seconds from 5s to 19s after
            start_times = event[0] + np.arange(5, 19, 2) * sfreq
            new_events.extend([[int(t), 0, 2] for t in start_times])

        # replace events in raw
        new_events = np.array(new_events)
        annot_from_events = mne.annotations_from_events(
            events=new_events,
            event_desc={1: "eyes_closed", 2: "eyes_open"},
            sfreq=raw.info["sfreq"],
        )
        raw.set_annotations(annot_from_events)
        return raw


# BrainDecode preprocessors
preprocessors = [
    hbn_ec_ec_reannotation(),
    Preprocessor(
        "pick_channels",
        ch_names=[
            "E22",
            "E9",
            "E33",
            "E24",
            "E11",
            "E124",
            "E122",
            "E29",
            "E6",
            "E111",
            "E45",
            "E36",
            "E104",
            "E108",
            "E42",
            "E55",
            "E93",
            "E58",
            "E52",
            "E62",
            "E92",
            "E96",
            "E70",
            "Cz",
        ],
    ),
    Preprocessor("resample", sfreq=128),
    Preprocessor("filter", l_freq=1, h_freq=55),
]
preprocess(ds_eoec, preprocessors)

# Extract 2-second segments
windows_ds = create_windows_from_events(
    ds_eoec,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=int(5 * ds_eoec.datasets[0].raw.info["sfreq"]),
    preload=True,
)

# %% [markdown]
# ## Plotting a Single Channel for One Sample
#
# It’s always a good practice to verify that the data has been properly loaded and processed. Here, we plot a single channel from one sample to ensure the signal is present and looks as expected.

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(windows_ds[2][0][0, :].transpose())  # first channel of first epoch
plt.show()

# %% [markdown]
# ## Features

# %% [markdown]
# * We start by extracting the signal variance from each channel (EEG electrode) as a feature, so we get 24 features (one per channel).
#
# * The function `signal_variance_feature` gets a *batch* of samples, represented by a numpy array of size (`batch_size`$\times$`num_channels`$\times$`time_points_per_window`).
#
# * The function returns a numpy array of size (`batch_size`$\times$`num_channels`).
#
# * To automatically match the channel name to each feature, we use the `univariate_feature` decorator.
#
# * The features extraction is performed by the `extract_features` function, getting a `braindecode` windows dataset and a features dictionary mapping feature names to feature extraction functions.

# %%
from eegdash import features
from eegdash.features import extract_features


@features.univariate_feature
def signal_variaince_feature(x):
    return x.var(axis=-1)


features_dict = {"sig_var": signal_variaince_feature}

features_ds = extract_features(windows_ds, features_dict, batch_size=512)

# %% [markdown]
# Let us have a look at the feature values.
#
# In this example, the first three columns represent the window crop indices, and are optional.

# %%
features_ds.to_dataframe(include_crop_inds=True)

# %% [markdown]
# * Now we add two spectral features: the root of the total power, and the power in different power bands.
#
# * Keyword parameters can be passed to each feature using the `functools.partial` function.
#
# * Multiple similar features can be returned from a feature extraction function by passing a dictionary of numpy arrays.

# %%
from functools import partial
from scipy.signal import welch

sfreq = windows_ds.datasets[0].raw.info["sfreq"]


@features.univariate_feature
def spectral_root_total_power_feature(x, **kwargs):
    f, p = welch(x, **kwargs)
    return p.sum(axis=-1)


DEFAULT_FREQ_BANDS = {
    "delta": (1, 4.5),
    "theta": (4.5, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}


@features.univariate_feature
def spectral_power_bands_feature(x, bands=DEFAULT_FREQ_BANDS, **kwargs):
    f, p = welch(x, **kwargs)
    power_bands = dict()
    for band_name, band_lims in bands.items():
        ind = np.logical_and(f > band_lims[0], f < band_lims[1])
        power_bands[band_name] = p[..., ind].sum(axis=-1)
    return power_bands


features_dict = {
    "sig_var": signal_variaince_feature,
    "spec_rtotpow": partial(spectral_root_total_power_feature, fs=sfreq),
    "sig_pband": partial(spectral_power_bands_feature, fs=sfreq),
}

features_ds = extract_features(windows_ds, features_dict, batch_size=512)

# %% [markdown]
# Again, let us have a look at the feature values (this time without the window crop indices).

# %%
features_ds.to_dataframe()

# %% [markdown]
# You might have noticed that both of the spectral feature extraction functions call the `welch` function with exact same parameters, so the computation will happen twice. As we may add more spectral features, this repeating computation will slow down the feature extraction computations. This can be solved by creating a mid-step computation of the power spectrum, then reusing its result to compute different spectral features.
#
# * Mid-step computations is implemented by inheriting the `FeatureExtractor` class and overriding its `preprocess` method.
#
# * The output of the `preprocess` method will pass as-is to downstream feature extraction functions.
#
# * The `FeaturePredecessor` decorator is used to make sure each feature extraction function receives a properly preprocessed input.
#
# * The new processing step is included as a new feature, getting its own descendants in a new features dictionary. The feature names will be a concatenation of the processing steps.

# %%
sfreq = windows_ds.datasets[0].raw.info["sfreq"]


class WelchFeatureExtractor(features.FeatureExtractor):
    def preprocess(self, x, **kwargs):
        f, p = welch(x, **kwargs)
        return f, p


@features.FeaturePredecessor(WelchFeatureExtractor)
@features.univariate_feature
def spectral_root_total_power_feature(f, p, **kwargs):
    return p.sum(axis=-1)


@features.FeaturePredecessor(WelchFeatureExtractor)
@features.univariate_feature
def spectral_power_bands_feature(f, p, bands=DEFAULT_FREQ_BANDS, **kwargs):
    power_bands = dict()
    for band_name, band_lims in bands.items():
        ind = np.logical_and(f > band_lims[0], f < band_lims[1])
        power_bands[band_name] = p[..., ind].sum(axis=-1)
    return power_bands


features_dict = {
    "sig_var": signal_variaince_feature,
    "spec": WelchFeatureExtractor(
        {
            "rtotpow": spectral_root_total_power_feature,
            "pband": spectral_power_bands_feature,
        },
        fs=sfreq,
    ),
}

features_ds = extract_features(windows_ds, features_dict, batch_size=512)

# %% [markdown]
# Again, let us have a look at the feature values.

# %%
features_ds.to_dataframe()

# %% [markdown]
# Finally, let us extract the same features using features already implemented in the `EEGDash.features` package.

# %%
sfreq = windows_ds.datasets[0].raw.info["sfreq"]


features_dict = {
    "sig_var": features.signal_variance,
    "spec": features.SpectralFeatureExtractor(
        {
            "rtotpow": features.spectral_root_total_power,
            "pband": features.spectral_bands_power,
        },
        fs=sfreq,
    ),
}

features_ds = extract_features(windows_ds, features_dict, batch_size=512)

# %%
features_ds.to_dataframe()

# %% [markdown]
# The function `get_all_features` returns a list of all currently implemented features:

# %%
features.get_all_features()

# %% [markdown]
# The function `get_all_feature_extractors` returns a list of all currently implemented feature extractors:

# %%
features.get_all_feature_extractors()

# %% [markdown]
# Now we can add some new features.

# %%
sfreq = windows_ds.datasets[0].raw.info["sfreq"]
filter_freqs = dict(windows_ds.datasets[0].raw_preproc_kwargs)["filter"]


features_dict = {
    "sig_var": features.signal_variance,
    "spec": features.SpectralFeatureExtractor(
        {
            "rtotpow": features.spectral_root_total_power,
            "pband": features.spectral_bands_power,
            0: features.NormalizedSpectralFeatureExtractor(
                {
                    "entropy": features.spectral_entropy,
                    "moment": features.spectral_moment,
                    "edge": partial(features.spectral_edge, edge=0.9),
                }
            ),
            1: features.DBSpectralFeatureExtractor(
                {
                    "slope": features.spectral_slope,
                }
            ),
        },
        fs=sfreq,
        nperseg=2 * sfreq,
        noverlap=int(1.5 * sfreq),
        f_min=filter_freqs["l_freq"],
        f_max=filter_freqs["h_freq"],
    ),
}

features_ds = extract_features(windows_ds, features_dict, batch_size=512)

# %%
features_ds.to_dataframe()

# %% [markdown]
# Note that the signal of Cz electrode is always zero, so some of its features(e.g., 'spec_moment_Cz') are *NaN*. To avoid future problems, let us replace them with zeros.

# %%
features_ds.fillna(0)

# %%
features_ds.to_dataframe()

# %% [markdown]
# #### Advanced usage
#
# * The feature extraction process can be controlled via the `batch_size` and `n_jobs` parameters, allowing for efficient parallel and batched processing.
#
# * The resulting `FeaturesConcatDataset` (in this example, `features_ds`) can be saved to disk using the `save` method, then loaded using the `load_features_concat_dataset` function.
#
# * A `FeaturesConcatDataset` object also supports a subset of pandas-dataframe-like operations, such as `mean`, `var`, `zscore`, `fillna`, `join` and more.
#
# * A feature extraction function may be any callable object. If necessary, the relevant decorators can be applied directly to the class definition.
#
#    - Feature extraction functions decorated by a `numba.jit` decorator are explicitly supported.
#
# * By default, any new feature assumes its predecessor preprocessing step is a simple `FeatureExtractor`; otherwise, the `FeaturePredecessor` decorator is used to enforce a specific type of a preprocessing step. If relevant, multiple possible preprocessing steps can be passed to the decorator (for example, spectral power bands may be computed for different types of power normalizations, each performed by a different preprocessing step).
#
#    - Each object inheriting from `FeatureExtractor` may be decorated with a `FeaturePredecessor` to create a tree of processing steps.
#
#    - The function `get_feature_predecessors` returns a list of all possible predecessors for a given feature.
#
#    - The feature name is derived by concatenating the names of its processing steps. To ignore a certain step (such as a simple normalization), replace its key in the dictionary by an empty string or a non-string value.
#
# * Just like the `univariate_feature` decorator one may use the `bivariate_feature` and `multivariate_feature` decorators. In each case, the second dimension returned by the feature extraction function should match the feature kind (i.e., for a `bivariate_feature`, the second dimension should be equal to `num_channels`$\times$(`num_channels` - 1)/2, and their order should match the one computed via `BivariateFeature.get_pair_iterators`). For a `multivariate_feature` this dimension should be omitted completely.
#
# * If necessary, one may create new feature kinds (e.g., triplet features) by inheriting from `MultivariateFeature` and overriding its `feature_channel_names` method. The new feature kind can be enforces using the `FeatureKind` decorator (e.g., `univariate_feature` is just a shorthand for `FeatureKind(UnivariateFeature())`).
#
#    - The function `get_feature_kind` returns the `FeatureKind` of a given feature.
#
#    - The function `get_all_feature_kinds` returns a list of all currently implemented `FeatureKind`s.
#
# * Trainable features (e.g., Common Spatial Pattern features) can be implemented by inheriting the `TrainableFeature` class and overriding its `partial_fit`, `fit` and `__call__` methods, then call the `fit_feature_extractors` function before `extract_features`. For an example, see the built-in CSP implementation.

# %% [markdown]
# ## Creating training and test sets
#
# The code below creates a training and test set. We first split the data into training and test sets using the **train_test_split** function from the **sklearn** library. We then create a **TensorDataset** for the training and test sets.
#
# 1.	**Set Random Seed** – The random seed is fixed using torch.manual_seed(random_state) to ensure reproducibility in dataset splitting and model training.
# 2.	**Extract Labels from the Dataset** – Labels (eye-open or eye-closed events) are extracted from windows_ds, stored as a NumPy array, and printed for verification.
# 3.	**Split Dataset into Train and Test Sets** – The dataset is split into training (80%) and testing (20%) subsets using train_test_split(), ensuring balanced stratification based on the extracted labels. Stratification means that we have as many eyes-open and eyes-closed samples in the training and testing sets.
# 4.	**Convert Data to PyTorch Tensors** – The selected training and testing samples are converted into FloatTensor for input features and LongTensor for labels, making them compatible with PyTorch models.
# 5.	**Create DataLoaders** – The datasets are wrapped in PyTorch DataLoader objects with a batch size of 10, enabling efficient mini-batch training and shuffling.
#

# %%
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Set random seed for reproducibility
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)

# Extract labels from the dataset
eo_ec = np.array([ds[1] for ds in features_ds]).ravel()  # check labels
print("labels: ", eo_ec)

# Get balanced indices for male and female subjects
train_indices, test_indices = train_test_split(
    range(len(features_ds)), test_size=0.2, stratify=eo_ec, random_state=random_state
)

# Convert the data to tensors
X_train = torch.FloatTensor(
    np.array([features_ds[i][0] for i in train_indices])
)  # Convert list of arrays to single tensor
X_test = torch.FloatTensor(
    np.array([features_ds[i][0] for i in test_indices])
)  # Convert list of arrays to single tensor
y_train = torch.LongTensor(eo_ec[train_indices])  # Convert targets to tensor
y_test = torch.LongTensor(eo_ec[test_indices])  # Convert targets to tensor
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)

# Create data loaders for training and testing (batch size 10)
train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)

# Print shapes and sizes to verify split
print(
    f"Shape of data {X_train.shape} number of samples - Train: {len(train_loader)}, Test: {len(test_loader)}"
)
print(
    f"Eyes-Open/Eyes-Closed balance, train: {np.mean(eo_ec[train_indices]):.2f}, test: {np.mean(eo_ec[test_indices]):.2f}"
)

# %% [markdown]
# ## Check labels
#
# It is good practice to verify the labels and ensure the random seed is functioning correctly. If all labels are 0s (eyes closed) or 1s (eyes open), it could indicate an issue with data loading or stratification, requiring further investigation.

# %%
# Visualize a batch of target labels
dataiter = iter(train_loader)
first_item, label = dataiter.__next__()
label

# %% [markdown]
# ## Create model
#
# The model is a MLP with `n_features` input channels, and 2 output classes (eyes-open and eyes-closed).

# %%
import torch
from torch import nn
from torchinfo import summary

torch.manual_seed(random_state)
# MLP
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(features_ds.datasets[0].n_features, 100),
    nn.Linear(100, 100),
    nn.Linear(100, 100),
    nn.Linear(100, 2),
)

summary(model, input_size=first_item.shape)

# %% [markdown]
# ## Model Training and Evaluation Process
#
# This section trains the neural network using the Adamax optimizer, normalizes input data, computes cross-entropy loss, updates model parameters, and tracks accuracy across six epochs.
#
# 1. **Set Up Optimizer and Learning Rate Scheduler** – The `Adamax` optimizer initializes with a learning rate of 0.002 and weight decay of 0.001 for regularization. An `ExponentialLR` scheduler with a decay factor of 1 keeps the learning rate constant.
#
# 2. **Allocate Model to Device** – The model moves to the specified device (CPU, GPU, or MPS for Mac silicon) to optimize computation efficiency.
#
# 3. **Normalize Input Data** – The `normalize_data` function standardizes input data by subtracting the mean and dividing by the standard deviation along the time dimension before transferring it to the appropriate device.
#
# 4. **Evaluates Classification Accuracy Over Six Epochs** – The training loop iterates through data batches with the model in training mode. It normalizes inputs, computes predictions, calculates cross-entropy loss, performs backpropagation, updates model parameters, and steps the learning rate scheduler. It tracks correct predictions to compute accuracy.
#
# 5. **Evaluate on Test Data** – After each epoch, the model runs in evaluation mode on the test set. It computes predictions on normalized data and calculates test accuracy by comparing outputs with actual labels.

# %%
from torch.nn import functional as F

optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = model.to(device=device)  # move the model parameters to CPU/GPU
epochs = 6

x_mean = X_train.mean(dim=0, keepdim=True)
x_std = X_train.std(dim=0, keepdim=True) + 1e-7


def normalize_data(x):
    x = (x - x_mean) / x_std
    x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
    return x


for e in range(epochs):
    # training
    correct_train = 0
    for t, (x, y) in enumerate(train_loader):
        model.train()  # put model to training mode
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)
        _, preds = scores.max(1)
        correct_train += (preds == y).sum() / len(dataset_train)

        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    correct_test = 0
    for t, (x, y) in enumerate(test_loader):
        model.eval()  # put model to testing mode
        scores = model(normalize_data(x))
        y = y.to(device=device, dtype=torch.long)
        _, preds = scores.max(1)
        correct_test += (preds == y).sum() / len(dataset_test)

    # Reporting
    print(
        f"Epoch {e}, Train accuracy: {correct_train:.2f}, Test accuracy: {correct_test:.2f}"
    )
