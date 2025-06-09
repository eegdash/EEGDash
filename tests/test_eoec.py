import numpy as np
import torch
import warnings
import unittest
from eegdash import EEGDashDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
import mne
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from braindecode.models import ShallowFBCSPNet
from torchinfo import summary

# %% [markdown]
# # EEGDash example for eyes open vs. closed classification
# 
# The code below provides an example of using the *EEGDash* library in combination with PyTorch to develop a deep learning model for analyzing EEG data, specifically for eyes open vs. closed classification in a single subject.
# 
# 1. **Data Retrieval Using EEGDash**: An instance of *EEGDashDataset* is created to search and retrieve an EEG dataset. At this step, only the metadata is transferred.
# 
# 2. **Data Preprocessing Using BrainDecode**: This process preprocesses EEG data using Braindecode by reannotating events, selecting specific channels, resampling, filtering, and extracting 2-second epochs, ensuring balanced eyes-open and eyes-closed data for analysis.
# 
# 3. **Creating train and testing sets**: The dataset is split into training (80%) and testing (20%) sets with balanced labels, converted into PyTorch tensors, and wrapped in DataLoader objects for efficient mini-batch training.
# 
# 4. **Model Definition**: The model is a shallow convolutional neural network (ShallowFBCSPNet) with 24 input channels (EEG channels), 2 output classes (eyes-open and eyes-closed).
# 
# 5. **Model Training and Evaluation Process**: This section trains the neural network, normalizes input data, computes cross-entropy loss, updates model parameters, and evaluates classication accuracy over six epochs.
# 
# 

# %% [markdown]
# ## Data Retrieval Using EEGDash
# 
# First we find one resting state dataset. This dataset contains both eyes open and eyes closed data.

# %%
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


class TestEEGDashEOEC(unittest.TestCase):

    def test_eoec_dataset(self):
        ds_eoec = EEGDashDataset(
            {"dataset": "ds005514", "task": "RestingState", "subject": "NDARDB033FW5"}
        )
        assert isinstance(ds_eoec, EEGDashDataset)

        warnings.simplefilter("ignore", category=RuntimeWarning)


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
            trial_stop_offset_samples=256,
            preload=True,
        )

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
        # Set random seed for reproducibility
        random_state = 42
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Extract labels from the dataset
        eo_ec = np.array([ds[1] for ds in windows_ds]).transpose()  # check labels
        print("labels: ", eo_ec)

        # Get balanced indices for male and female subjects
        train_indices, test_indices = train_test_split(
            range(len(windows_ds)), test_size=0.2, stratify=eo_ec, random_state=random_state
        )

        # Convert the data to tensors
        X_train = torch.FloatTensor(
            np.array([windows_ds[i][0] for i in train_indices])
        )  # Convert list of arrays to single tensor
        X_test = torch.FloatTensor(
            np.array([windows_ds[i][0] for i in test_indices])
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
        # # Check labels
        # 
        # It is good practice to verify the labels and ensure the random seed is functioning correctly. If all labels are 0s (eyes closed) or 1s (eyes open), it could indicate an issue with data loading or stratification, requiring further investigation.

        # %%
        # Visualize a batch of target labels
        dataiter = iter(train_loader)
        first_item, label = dataiter.__next__()
        label

        # %% [markdown]
        # # Create model
        # 
        # The model is a shallow convolutional neural network (ShallowFBCSPNet) with 24 input channels (EEG channels), 2 output classes (eyes-open and eyes-closed), and an input window size of 256 samples (2 seconds of EEG data). 

        # %%
        torch.manual_seed(random_state)
        model = ShallowFBCSPNet(24, 2, n_times=256, final_conv_length="auto")
        summary(model, input_size=(1, 24, 256))
        # %%
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


        def normalize_data(x):
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
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


if __name__ == "__main__":
    unittest.main()
