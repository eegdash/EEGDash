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
from braindecode.datautil import load_concat_dataset
from pathlib import Path
from torch.utils.data import Subset

import numpy as np
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import os
from scipy.stats import bootstrap
import gc

cache_dir = Path("/mnt/v1/arno/eeg2025")
SFREQ = 100  # sampling frequency

def balance_windows_by_class(ds, random_seed=42):
    """
    Returns a torch.utils.data.Subset of ds with equal number of windows for M and F classes.
    """
    import numpy as np
    # Extract all labels at once for better performance
    metadata_df = ds.get_metadata()
    labels = np.array([metadata_df.iloc[i]['sex'] for i in range(len(ds))])
    m_indices = np.where(labels == 'M')[0]
    f_indices = np.where(labels == 'F')[0]
    n_min = min(len(m_indices), len(f_indices))
    rng = np.random.default_rng(random_seed)
    m_indices = rng.permutation(m_indices)[:n_min]
    f_indices = rng.permutation(f_indices)[:n_min]
    selected_indices = np.concatenate([m_indices, f_indices])
    selected_indices = rng.permutation(selected_indices)
    return Subset(ds, selected_indices)

def balance_windows_by_class(ds, random_seed=42):
    """
    Returns a torch.utils.data.Subset of ds with equal number of windows for M and F classes.
    """
    import numpy as np
    desc = ds.description
    m_indices = np.where(desc['sex'] == 'M')[0]
    f_indices = np.where(desc['sex'] == 'F')[0]
    n_min = min(len(m_indices), len(f_indices))
    rng = np.random.default_rng(random_seed)
    m_indices = rng.permutation(m_indices)[:n_min]
    f_indices = rng.permutation(f_indices)[:n_min]
    selected_indices = np.concatenate([m_indices, f_indices])
    selected_indices = rng.permutation(selected_indices)
    return Subset(ds, selected_indices)

def process_data(releases, tasks, target_names):
    
    for release in releases:
        missing = []
        for task in tasks:
            for target_name in target_names:
                cached_data_folder_name = "data/hbn_" + release + "_" + task + "_" + target_name
                if not os.path.exists(cached_data_folder_name):
                    if target_name == target_names[0]:
                        missing.append(task)
                    
        if len(missing) < 3:
            print(f"All data exists in {release} missing [{', '.join(missing)}]")
            continue
        else:
            print(f"Incomplete data in {release}: Missing [{', '.join(missing)}]")
        
        if release != "R12":
            ds_sexdata = EEGChallengeDataset(
                release=release,
                cache_dir=cache_dir,
                task=tasks,
                mini=False,
                # run="1",
                download=False,
                target_name="sex"
            )
        else:
            ds_sexdata = EEGDashDataset(
                dataset="HBN-R12_L100",
                cache_dir=cache_dir,
                task=tasks,
                # run="1",
                download=False,
                target_name="sex"
            )        
        
        sub_rm = ["NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1", "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV", "NDARBA381JGH"]
        all_datasets = BaseConcatDataset(
            [
                ds
                for ds in ds_sexdata.datasets
                if not ds.description.subject in sub_rm
                and ds.raw.n_times >= 4 * SFREQ
                and len(ds.raw.ch_names) == 129
            ]
        )

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
        
        preprocess(all_datasets, preprocessors, n_jobs=16)  # Reduced from -1 to 2
        print("Preprocessing completed successfully!")

        for task in tasks:
            for target_name in target_names:
                # %%
                num_ignore = 0
                if target_name != "sex":
                    for ds in all_datasets.datasets:
                        # randomly assign gender to "M" or "F"
                        # ds.description['sex'] = np.random.choice(['M', 'F'])
                        
                        if ds.description[target_name] is not None and not pd.isna(ds.description[target_name]):
                            if target_name == "age":
                                if ds.description[target_name] > 12:
                                    ds.description['sex'] = 'M'
                                elif ds.description[target_name] < 8:
                                    ds.description['sex'] = 'F'
                                else:
                                    ds.description['sex'] = 'B'
                            else:
                                if ds.description[target_name] > 0.5:
                                    ds.description['sex'] = 'M'
                                elif ds.description[target_name] < 0.5:
                                    ds.description['sex'] = 'F'
                                else:
                                    ds.description['sex'] = 'B'
                        else:
                            num_ignore += 1
                            ds.description['sex'] = 'B'
                
                print(f"Number of subjects ignored: {num_ignore}")
                ds_list = [ds for ds in all_datasets.datasets if ds.description['task'] == task and ds.description['sex'] != 'B' ]

                if len(ds_list) > 0:                 
                    all_datasets2 = BaseConcatDataset(ds_list)
                    
                    print(f"Preprocessing {len(all_datasets2.datasets)} datasets...")
                    
                    # extract windows and save to disk
                    windows_ds = create_fixed_length_windows(
                        all_datasets2,
                        start_offset_samples=0,
                        stop_offset_samples=None,
                        window_size_samples=256,
                        window_stride_samples=256,
                        drop_last_window=True,
                        preload=False  # Keep preload=False to save memory
                    )

                    # save to disk
                    cached_data_folder_name = "data/hbn_" + release + "_" + task + "_" + target_name
                    os.makedirs(cached_data_folder_name, exist_ok=True)
                    windows_ds.save(cached_data_folder_name, overwrite=True)
                    
                    # reload to create metadata_df.pkl
                    windows_ds = load_concat_dataset(cached_data_folder_name, preload=False)
                    print(f"Number of datasets in {cached_data_folder_name}: {len(windows_ds.datasets)}")
                    print(f"number of samples in {cached_data_folder_name} : {len(windows_ds)}")

def run_task(releases, tasks, target_name, folds=10, weights=None, model_freeze=False, random_add=42, train_epochs=20, save_weights="", batch_size=100, lrate=0.00002, model_name = 'EEGNeX'):
    global deep_copy_dataset
    
    if not isinstance(releases, list):
        releases = [releases]
    if not isinstance(tasks, list):
        tasks = [tasks]
        
    cached_data_folder_names = []
    for release in releases:
        for task in tasks:
            cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_" + release + "_" + task + "_" + target_name
            if os.path.exists(cached_data_folder_name):
                cached_data_folder_names.append(cached_data_folder_name)
            else:
                print(f"Missing DataError({cached_data_folder_name}): You first run process_data to run the task for each release")
    
    print("Loading data from disk")
    windows_ds = []
    for cached_data_folder_name in cached_data_folder_names:
        windows_ds_tmp = load_concat_dataset(path=cached_data_folder_name, preload=False)
        windows_ds.extend([ds for ds in windows_ds_tmp.datasets])
        print(f"Number of datasets in {cached_data_folder_name}: {len(windows_ds_tmp.datasets)}")
        
    windows_ds = BaseConcatDataset(windows_ds)
    print(f"Number of datasets in all releases: {len(windows_ds.datasets)}")
    print(f"number of samples in windows_ds: {len(windows_ds)}")

    # random seed for reproducibility
    np.random.seed(random_add)
    torch.manual_seed(random_add)

    # %% [markdown]
    # ## Creating a Training and Test Set
    correct_train_list        = []
    correct_test_list         = []
    correct_test_list_subject = []
    unique_subjects, unique_indices = np.unique(windows_ds.description["subject"], return_index=True)
    unique_gender = windows_ds.description["sex"][unique_indices].values
    
    # Ensure we don't have more folds than the minimum class count
    min_class_count = min((unique_gender == "M").sum(), (unique_gender == "F").sum())
    actual_folds = min(folds, min_class_count)
    
    if actual_folds > 1:
        splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_add)
        splits = splitter.split(unique_subjects, unique_gender)
    else:
        train_idx, val_idx = train_test_split(np.arange(len(unique_subjects)),train_size=0.8,stratify=unique_gender,random_state=random_add)
        splits = [(train_idx, val_idx)]
        
    for it_fold, (train_idx, val_idx) in enumerate(splits):
        
        # balance the dataset so we have 50% of each class
        train_gender = unique_gender[train_idx]
        val_gender   = unique_gender[  val_idx]
        train_subj   = unique_subjects[train_idx]
        val_subj     = unique_subjects[  val_idx]
        train_n = min((train_gender == "M").sum(), (train_gender == "F").sum()) # take the minimum number of subjects for each class to balance the dataset
        val_n   = min((  val_gender == "M").sum(), (  val_gender == "F").sum()) # take the minimum number of subjects for each class to balance the dataset
        train_subj = np.concatenate([np.random.choice(train_subj[train_gender == "M"], train_n, replace=False),np.random.choice(train_subj[train_gender == "F"], train_n, replace=False)])        
        val_subj   = np.concatenate([np.random.choice(  val_subj[  val_gender == "M"],   val_n, replace=False),np.random.choice(  val_subj[  val_gender == "F"],   val_n, replace=False)])        
        
        # Create datasets
        if model_freeze: # When model_freeze=True, evaluate on all (balanced) data
            val_ds   = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in train_subj or ds.description.subject in val_subj]) 
            train_ds = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in train_subj])
        else:
            train_ds = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in train_subj])
            val_ds   = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in val_subj])
            
        # Balance windows per class for train and val sets
        #train_ds, train_male_count, train_female_count = balance_windows_by_class(train_ds, random_seed=random_add)
        #val_ds, val_male_count, val_female_count = balance_windows_by_class(val_ds, random_seed=random_add)
        train_ds = balance_windows_by_class(train_ds, random_seed=random_add)
        val_ds   = balance_windows_by_class(val_ds, random_seed=random_add)

        # show the number of sample in each class for each gender
        # print(f"Number of samples in training set for male: {train_male_count}")
        # print(f"Number of samples in training set for female: {train_female_count}")
        # print(f"Number of samples in validation set for male: {val_male_count}")
        # print(f"Number of samples in validation set for female: {val_female_count}")
        
        # Create dataloaders with smaller batch size to save memory
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4)
        val_loader   = DataLoader(  val_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4)

        # Check the balance of the dataset
        print(f"Number of datasets in balanced set: {len(train_subj)}")
        print(f"Class distribution in training set {train_n} of each class; validation set {val_n} of each class")

        # create model
        from torch import nn
        from torchinfo import summary
        from braindecode.models import EEGNeX, TSception, EEGConformer	

        if model_name == 'EEGNeX':
            model = EEGNeX(
                n_chans=24,      # 129 channels
                n_outputs=2,      # 1 output for regression
                n_times=256,      # 2 seconds
                sfreq=128,        # sample frequency 100 Hz
            )
        elif model_name == 'TSception':
            model = TSception(
                n_chans=24,      # 129 channels
                n_outputs=2,      # 1 output for regression
                n_times=256,      # 2 seconds
                sfreq=128,        # sample frequency 100 Hz
            )
        elif model_name == 'EEGConformer':
            model = EEGConformer(
                    n_chans=24,
                    n_outputs=2,
                    n_times=256,
                    sfreq=128,
                    drop_prob=0.7,
                    n_filters_time=32,        # Try 32 instead of default 40
                    filter_time_length=20,    # Try 20
                    att_depth=4,             # Reduce attention layers
                    att_heads=8,             # Reduce attention heads
                    pool_time_stride=12,     # Adjust pooling
                    pool_time_length=64,     # Adjust pooling window
                )

        if weights is not None:
            model.load_state_dict(torch.load(weights))

        # print(summary(model, input_size=(batch_size, 24, 256)))

        # print(model)
        # %% [markdown]
        # # Model Training and Evaluation Process
        optimizer = torch.optim.Adamax(model.parameters(), lr=lrate, weight_decay=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
            all_preds, all_ground_truth, all_subjects = [], [], []
            model.eval()
            for x, y, ss in val_loader:
                scores = model(normalize_data(x))
                _, preds = scores.max(1)
                y = torch.tensor([gender_dict[gender] for gender in y], device=device, dtype=torch.long)
                correct_test += (preds == y).sum() / len(val_ds)
                all_preds.extend(preds.cpu().numpy())
                all_ground_truth.extend(y.cpu().numpy())
                all_subjects.extend(ss)
            
            # Subject-level voting: average predictions per subject and vote
            subject_uniq, subject_indices = np.unique(np.array(all_subjects), return_inverse=True)
            subject_gt    = np.array([np.array(all_ground_truth)[subject_indices == i][0] for i in range(len(subject_uniq))])
            subject_preds = np.array([np.bincount(np.array(all_preds)[subject_indices == i]).argmax() for i in range(len(subject_uniq))])
            subject_level_accuracy = (subject_preds == subject_gt).mean()
            
            print(f"Iteration {it_fold}, Epoch {e}, Train accuracy: {correct_train:.3f}, Test accuracy: {correct_test:.3f}, Subject-level accuracy: {subject_level_accuracy:.3f}\n")
            
        # convert values to numpy arrays
        correct_train = float(correct_train.item()) if hasattr(correct_train, "item") else correct_train
        correct_test  = float(correct_test.item())
        correct_train_list.extend([correct_train])
        correct_test_list.extend([correct_test])  # Use subject-level accuracy instead of sample-level
        correct_test_list_subject.extend([subject_level_accuracy])
        
        # do garbage collection
        gc.collect()
        
        # Save model weights immediately to avoid storing in memory
        if save_weights is not None and save_weights != "":
            torch.save(model.state_dict(), save_weights.replace(".pth", "") + f"_{it_fold}.pth")
            print(f"Saved model {it_fold} weights to {save_weights.replace('.pth', '')}_{it_fold}.pth")
    
    return correct_train_list, correct_test_list, correct_test_list_subject

# %%
from pathlib import Path
import json

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# releases = ["R12"]

import sys
import numpy as np
factor = "attention"

# load data to avoid errors
tasks = [  'DespicableMe',
  'DiaryOfAWimpyKid',
  'FunwithFractals',
  'ThePresent',
  'RestingState',
  'contrastChangeDetection',
  'seqLearning6target',
  'seqLearning8target',
  'surroundSupp',
  'symbolSearch']
factors = ["sex", "age", "p_factor", "attention", "internalizing", "externalizing"]
releases = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"]

# process_data(releases, tasks, factors) # just import the data to avoid errors
# sys.exit()


if 0:
    # train large model
    _, res_test_all = run_task(releases[:-1], 'contrastChangeDetection', factor, save_weights="weights_" + factor + ".pth", repeat=1, train_epochs=20, batch_size=2000, lrate=0.00002*10*4)
    #_, res_test_r12 = run_task(releases[:-1], 'contrastChangeDetection', 'p_factor', weights="weights" + "_all" + ".pth", repeat=20, train_epochs=1, model_freeze=True)
    print("Performance on all releases: ", res_test_all)
    #print("Performance on R12: "         , res_test_r12)

if 0:
    # test large model
    # _, res_test_r12 = run_task("R12", 'contrastChangeDetection', factor, weights="weights_" + factor + ".pth", repeat=20, train_epochs=1, model_freeze=True)
    _, res_test_r12 = run_task("R12", 'contrastChangeDetection', factor, weights="weights_" + factor + ".pth", repeat=20, train_epochs=1, model_freeze=True)
    print("Performance on R12: ", res_test_r12)
    print("Mean performance on R12: ", np.mean(np.array(res_test_r12)))
    print("95% confidence interval on R12: ", np.std(np.array(res_test_r12), ddof=1) / np.sqrt(len(res_test_r12)) * 1.96)
    sys.exit()

if 0:
    # run_task("R1", 'contrastChangeDetection', 'sex', save_weights="weights.pth", repeat=1, train_epochs=20)
    res_test = {}
    factor = "sex"
    for release1 in releases:
        run_task(release1, 'contrastChangeDetection', factor, save_weights="weights" + release1 + "_" + factor + ".pth", repeat=1, train_epochs=20, batch_size=500, lrate=0.00002*10)
        for release2 in releases:
            filename = f"results/{release1}train_{release2}test_contrastChangeDetection_{factor}.json"
            if os.path.exists(filename):
                print(f"Skipping {filename} because it already exists")
                continue
            _, res_test_release = run_task(release2, 'contrastChangeDetection', factor, weights="weights" + release1 + "_" + factor + ".pth", repeat=10, train_epochs=1, model_freeze=True)
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

new_tasks = { 'movies': ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent'],
  'restingstate': ['RestingState'],
  'contrastChangeDetection': ['contrastChangeDetection'],
  'seqLearning': ['seqLearning6target', 'seqLearning8target'],
  'surroundSupp': ['surroundSupp'],
  'symbolSearch': ['symbolSearch'],
  'all_tasks': ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'RestingState', 'ThePresent', 'contrastChangeDetection', 'seqLearning6target', 'seqLearning8target', 'surroundSupp', 'symbolSearch']
  }
# model_name = 'EEGNeX'
# model_name = 'TSception'
model_name = 'EEGConformer'
releases_train = releases[:-1]
new_tasks = {'contrastChangeDetection': ['contrastChangeDetection']}
factors = ['attention']
folds = 1
if True:
    for task in list(new_tasks.keys()):
        for factor in factors:
            # train set on all releases
            json_file = f"results/{task}_{factor}_{model_name}.json"
            if os.path.exists(json_file):
                print(f"Skipping {task}_{factor} because it already exists")
                continue
            weights_file_base = "weights_" + task+ "_" + factor + "_" + model_name + ".pth"
            res_train, res_test, res_test_s = run_task("R12", new_tasks[task], factor, folds=folds, train_epochs=20, batch_size=64, random_add=10, save_weights=weights_file_base, lrate=0.0005, model_name=model_name)

            # test set on R12
            res_test_r12 = []
            res_test_r12_subject = []
            for fold in range(folds):
                weights_file = weights_file_base.replace(".pth", "") + f"_{fold}.pth"
                _, res_test_r12_tmp, res_test_r12_subject_tmp = run_task("R12", new_tasks[task], factor, folds=1,  train_epochs=1,  batch_size=1000, weights=weights_file, model_freeze=True, model_name=model_name, random_add=fold)
                res_test_r12.extend(res_test_r12_tmp)
                res_test_r12_subject.extend(res_test_r12_subject_tmp)
                
                
            with open(json_file, "w") as f:
                # make the dump pretty
                dump_dict = {"train": res_train, "validation": res_test, "validation_subject": res_test_s, "test_r12": res_test_r12, "test_r12_subject": res_test_r12_subject}
                print(json.dumps(dump_dict, indent=4))
                json.dump(dump_dict, f, indent=4)
