# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset for a collection of subject. The dataset ds005505 contains 136 subjects with both male and female participants.
# %%
from eegdash import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset
from pathlib import Path
import numpy as np
import pandas as pd
import os
cache_dir = Path("/mnt/v1/arno/eeg2025")

def run_task(release, task, target_name):

    ds_sexdata = EEGChallengeDataset(
        release=release,
        cache_dir=cache_dir,
        task=task,
        mini=False,
        download=False,
        target_name="sex"
    )
    # %%
    num_ignore = 0
    if target_name != "sex":
        for ds in ds_sexdata.datasets:
            # randomly assign gender to "M" or "F"
            # ds.description['sex'] = np.random.choice(['M', 'F'])
            
            if ds.description[target_name] is not None and not pd.isna(ds.description[target_name]):
                if ds.description[target_name] > 0:
                    ds.description['sex'] = 'M'
                else:
                    ds.description['sex'] = 'F'
            else:
                num_ignore += 1
                ds.description['sex'] = 'M'

    print(f"Number of subjects ignored: {num_ignore}")

    # %% [markdown]
    # ## Data Preprocessing Using Braindecode
    from braindecode.preprocessing import (
        Preprocessor,
        create_fixed_length_windows,
        preprocess,
    )

    # Alternatively, if you want to include this as a preprocessing step in a Braindecode pipeline:
    preprocessors = [
        Preprocessor(
            "pick_channels",
            ch_names=[ "E22","E9","E33","E24","E11","E124","E122","E29","E6","E111","E45","E36","E104","E108","E42","E55","E93","E58","E52","E62","E92","E96","E70","Cz"],
        ),
        Preprocessor("resample", sfreq=128),
        Preprocessor("filter", l_freq=1, h_freq=55),
    ]
    preprocess(
        ds_sexdata, preprocessors, n_jobs=-1
    )  # , save_dir='xxxx'' will save and set preload to false

    # extract windows and save to disk
    windows_ds = create_fixed_length_windows(
        ds_sexdata,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=256,
        window_stride_samples=256,
        drop_last_window=True,
        preload=False
    )
    # os.makedirs("data/hbn_preprocessed_restingstate", exist_ok=True)
    # windows_ds.save("data/hbn_preprocessed_restingstate", overwrite=True)
    # # %%
    # from braindecode.datautil import load_concat_dataset

    # print("Loading data from disk")
    # windows_ds = load_concat_dataset(
    #     path="data/hbn_preprocessed_restingstate", preload=False
    # )

    # %% [markdown]
    # ## Creating a Training and Test Set

    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    # %%
    from braindecode.datasets import BaseConcatDataset
    correct_train_list = []
    correct_test_list = []
    for random_state in range(10):
        # random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Get balanced indices for male and female subjects and create a balanced dataset
        male_subjects = windows_ds.description["subject"][windows_ds.description["sex"] == "M"]
        female_subjects = windows_ds.description["subject"][
            windows_ds.description["sex"] == "F"
        ]
        n_samples = min(len(male_subjects), len(female_subjects))
        balanced_subjects = np.concatenate(
            [male_subjects[:n_samples], female_subjects[:n_samples]]
        )
        balanced_gender = ["M"] * n_samples + ["F"] * n_samples
        train_subj, val_subj, train_gender, val_gender = train_test_split(
            balanced_subjects,
            balanced_gender,
            train_size=0.8,
            stratify=balanced_gender,
            random_state=random_state,
        )

        # Create datasets
        train_ds = BaseConcatDataset(
            [ds for ds in windows_ds.datasets if ds.description.subject in train_subj]
        )
        val_ds = BaseConcatDataset(
            [ds for ds in windows_ds.datasets if ds.description.subject in val_subj]
        )

        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=100, shuffle=True)

        # Check the balance of the dataset
        assert len(balanced_subjects) == len(balanced_gender)
        print(f"Number of subjects in balanced dataset: {len(balanced_subjects)}")
        print(
            f"Gender distribution in balanced dataset: {np.unique(balanced_gender, return_counts=True)}"
        )

        # create model
        from torch import nn
        from torchinfo import summary
        from braindecode.models import EEGNeX

        model = EEGNeX(
            n_chans=24,      # 129 channels
            n_outputs=2,      # 1 output for regression
            n_times=256,      # 2 seconds
            sfreq=128,        # sample frequency 100 Hz
        )

        print(model)
        # %% [markdown]
        # # Model Training and Evaluation Process

        # %%
        from torch.nn import functional as F

        optimizer = torch.optim.Adamax(model.parameters(), lr=0.00002, weight_decay=0.001)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model.to(device=device)


        def normalize_data(x):
            x = x.reshape(x.shape[0], 24, 256)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            return x


        # dictionary of genders for converting sample labels to numerical values
        gender_dict = {"M": 0, "F": 1}

        epochs = 20
        for e in range(epochs):
            # training
            correct_train = 0
            for t, (x, y, sz) in enumerate(train_loader):
                model.train()  # put model to training mode
                scores = model(normalize_data(x))
                _, preds = scores.max(1)
                y = torch.tensor(
                    [gender_dict[gender] for gender in y], device=device, dtype=torch.long
                )
                correct_train += (preds == y).sum() / len(train_ds)

                # Calculates the cross-entropy loss and performs backpropagation
                loss = F.cross_entropy(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % 50 == 0:
                    print("Epoch %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))

            # validation
            correct_test = 0
            for t, (x, y, sz) in enumerate(val_loader):
                model.eval()  # put model to testing mode
                scores = model(normalize_data(x))
                _, preds = scores.max(1)
                y = torch.tensor(
                    [gender_dict[gender] for gender in y], device=device, dtype=torch.long
                )
                correct_test += (preds == y).sum() / len(val_ds)

            print(
                f"Epoch {e}, Train accuracy: {correct_train:.2f}, Test accuracy: {correct_test:.2f}\n"
            )
        # convert values to numpy arrays
        correct_train = float(correct_train.item())
        correct_test = float(correct_test.item())
        correct_train_list.append(correct_train)
        correct_test_list.append(correct_test)
    return correct_train_list, correct_test_list

# %%

from pathlib import Path
import json

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

tasks = [  'DespicableMe',
  'DiaryOfAWimpyKid',
  'FunwithFractals',
  'RestingState',
  'ThePresent',
  'contrastChangeDetection',
  'seqLearning6target',
  'seqLearning8target',
  'surroundSupp',
  'symbolSearch']

factors = ["sex", "p_factor", "attention", "internalizing", "externalizing"]

# run_task("R5", tasks[0], 'p_factor')

for task in tasks:
    for factor in factors:
        json_file = f"results/{task}_{factor}.json"
        if os.path.exists(json_file):
            print(f"Skipping {task}_{factor} because it already exists")
            continue
        try:
            res_train, res_test = run_task("R5", task, factor)
            with open(json_file, "w") as f:
                json.dump({"train": res_train, "test": res_test}, f)
            print(f"Task: {task}, Factor: {factor}, Train: {res_train}, Test: {res_test}")
        except Exception as e:
            # append to log file
            with open("error_log.txt", "a") as f:
                f.write(f"{task}_{factor}: {e}\n")
            print(f"Error running {task}_{factor}: {e}")