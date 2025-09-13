# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset for a collection of subject. The dataset ds005505 contains 136 subjects with both male and female participants.
# %%
from eegdash import EEGDashDataset
from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess
from braindecode.datasets.base import BaseConcatDataset
from pathlib import Path
import numpy as np
import pandas as pd
import os
cache_dir = Path("/mnt/v1/arno/eeg2025")
SFREQ = 100  # sampling frequency

def run_task(release, task, target_name, repeat=10, weights=None, model_freeze=False, random_add=42, train_epochs=20, save_weights=""):
    
    # Memory monitoring function
    def print_memory_usage():
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
        return memory_mb


    cached_data_folder_name = "data/hbn_" + release + "_" + task + "_" + target_name

    # check if data already exists
    if os.path.exists(cached_data_folder_name):
        print(f"Data already exists in {cached_data_folder_name}")
        
    else:
        if release != "R12":
            ds_sexdata = EEGChallengeDataset(
                release=release,
                cache_dir=cache_dir,
                task=task,
                mini=False,
                download=False,
                target_name="sex"
            )
        else:
            ds_sexdata = EEGDashDataset(
                dataset="HBN-R12_L100",
                cache_dir=cache_dir,
                task=task,
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
                    ds.description['sex'] = 'B'
                    
        print(f"Number of subjects ignored: {num_ignore}")
        print_memory_usage()

        sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1", "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
        all_datasets = BaseConcatDataset(
            [
                ds
                for ds in ds_sexdata.datasets
                if not ds.description.subject in sub_rm
                and ds.raw.n_times >= 4 * SFREQ
                and len(ds.raw.ch_names) == 129
                and ds.description['sex'] != 'B'
            ]
        )

        # %% [markdown]
        # ## Data Preprocessing Using Braindecode

        # Preprocess all datasets together to avoid threading conflicts
        import copy
        ch_names = ["E22","E9","E33","E24","E11","E124","E122","E29","E6","E111","E45","E36","E104","E108","E42","E55","E93","E58","E52","E62","E92","E96","E70","Cz"]
        preprocessors = [
            Preprocessor(
                "pick_channels",
                ch_names=ch_names,
            ),
            Preprocessor("resample", sfreq=128),
            Preprocessor("filter", l_freq=1, h_freq=55, picks=ch_names),
        ]
        
        print(f"Preprocessing {len(all_datasets.datasets)} datasets...")
        print_memory_usage()
        
        # Add a small delay to ensure any previous database connections are properly closed
        import time
        time.sleep(0.1)
        
        try:
            # Preprocess all datasets at once with limited parallelism to save memory
            preprocess(all_datasets, preprocessors, n_jobs=-1)  # Reduced from -1 to 2
            print("Preprocessing completed successfully!")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            print("Trying alternative approach with individual dataset preprocessing...")
            # Fallback: preprocess datasets individually with error handling
            for i, ds in enumerate(all_datasets.datasets):
                try:
                    ds2 = BaseConcatDataset([ds])
                    print(f"Preprocessing dataset {i+1}/{len(all_datasets.datasets)}: {ds.description['subject']}")
                    preprocess(ds2, preprocessors, n_jobs=1)
                    # Small delay between individual preprocessing to avoid connection conflicts
                    time.sleep(0.05)
                except Exception as individual_error:
                    print(f"Failed to preprocess dataset {ds.description['subject']}: {individual_error}")
                    continue

        # extract windows and save to disk
        windows_ds = create_fixed_length_windows(
            all_datasets,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=256,
            window_stride_samples=256,
            drop_last_window=True,
            preload=False  # Keep preload=False to save memory
        )

        # save to disk
        os.makedirs(cached_data_folder_name, exist_ok=True)
        windows_ds.save(cached_data_folder_name, overwrite=True)

        # Clear original datasets from memory after windowing
        del all_datasets
        import gc
        gc.collect()
        print("After windowing:")
        print_memory_usage()
        # os.makedirs("data/hbn_preprocessed_restingstate", exist_ok=True)
        # windows_ds.save("data/hbn_preprocessed_restingstate", overwrite=True)
        # # %%
        # from braindecode.datautil import load_concat_dataset

    print("Loading data from disk")
    from braindecode.datautil import load_concat_dataset
    windows_ds = load_concat_dataset(path=cached_data_folder_name, preload=False)

    # %% [markdown]
    # ## Creating a Training and Test Set

    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    # %%
    correct_train_list = []
    correct_test_list = []
    # Don't store all model weights in memory - save them individually
    # model_weights = []
    for random_state in range(repeat):
        # random seed for reproducibility
        np.random.seed(random_state + random_add)
        torch.manual_seed(random_state + random_add)

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
            random_state=random_state + random_add,
        )

        # Create datasets
        train_ds = BaseConcatDataset(
            [ds for ds in windows_ds.datasets if ds.description.subject in train_subj]
        )
        val_ds = BaseConcatDataset(
            [ds for ds in windows_ds.datasets if ds.description.subject in val_subj]
        )

        # Create dataloaders with smaller batch size to save memory
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

        if weights is not None:
            # Load individual model weights instead of loading all at once
            if os.path.exists(weights):
                model.load_state_dict(torch.load(weights))
            else:
                print(f"Warning: Model weights not found at {weights}")

        # print(model)
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

        for e in range(train_epochs):
            # training
            correct_train = 0
            if not model_freeze:
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
                f"Iteration {random_state}, Epoch {e}, Train accuracy: {correct_train:.2f}, Test accuracy: {correct_test:.2f}\n"
            )
        # convert values to numpy arrays
        try:
            correct_train = float(correct_train.item())
        except:
            pass
        correct_test = float(correct_test.item())
        correct_train_list.append(correct_train)
        correct_test_list.append(correct_test)
        
        # Save model weights immediately to avoid storing in memory
        if save_weights is not None and save_weights != "":
            torch.save(model.state_dict(), save_weights)
            print(f"Saved model {random_state} weights to {save_weights}")
        
        # Clear model and datasets from memory
        del model, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Final memory cleanup
    print("Final memory usage:")
    print_memory_usage()
    
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
releases = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
# releases = ["R12"]

# run_task("R1", 'contrastChangeDetection', 'sex', save_weights="weights.pth", repeat=1, train_epochs=20)
res_test = {}
for release1 in releases:
    run_task(release1, 'contrastChangeDetection', 'p_factor', save_weights="weights.pth", repeat=1, train_epochs=20)
    for release2 in releases:
        filename = f"results/{release1}train_{release2}test_contrastChangeDetection_p_factor.json"
        if os.path.exists(filename):
            print(f"Skipping {filename} because it already exists")
            continue
        _, res_test_release = run_task(release2, 'contrastChangeDetection', 'p_factor', weights="weights.pth", repeat=10, train_epochs=1, model_freeze=True)
        with open(filename, "w") as f:
            json.dump({"test": res_test_release}, f)

if 0:
    count = 0
    for release in releases[::-1]:
        res_train, res_test = run_task(release, 'contrastChangeDetection', 'p_factor', save_weights="weights" + release + ".pth", repeat=10, train_epochs=20)
        _, res_test_r12     = run_task("R12",   'contrastChangeDetection', 'p_factor', weights="weights" + release + ".pth", repeat=10, train_epochs=1, random_add=count, model_freeze=True)
        count += 1
        with open(f"results/{release}_contrastChangeDetection_p_factor.json", "w") as f:
            json.dump({"train": res_train, "test": res_test, "test_r12": res_test_r12}, f)

# for release in releases:
#     try:
#         run_task(release, 'contrastChangeDetection', 'sex', save_weights="weights.pth", repeat=1, train_epochs=1)
#         with open("error_log.txt", "a") as f:
#             f.write(f"{release}: success\n")
#     except Exception as e:
#         with open("error_log.txt", "a") as f:
#             f.write(f"{release}: {e}\n")

if 0:
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