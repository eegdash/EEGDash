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
import numpy as np
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import os
from scipy.stats import bootstrap
import lightning as L
from braindecode.models import EEGNeX, TSception	
from lightning.pytorch.tuner import Tuner
from torchmetrics import Recall, Precision, F1Score, Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
cache_dir = Path("/mnt/v1/arno/eeg2025")
SFREQ = 100  # sampling frequency

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

def create_model(config):
    class EEGModel(L.LightningModule):
        def __init__(self, config):
            super(EEGModel, self).__init__()
            drop_prob = config.get('dropout', 0.7)
            if config['model_name'] == 'EEGNeX':
                self.model = EEGNeX(
                    n_chans=24,
                    n_outputs=2,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            elif config['model_name'] == 'TSception':
                self.model = TSception(
                    n_chans=24,
                    n_outputs=2,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            self.lr = config['lr']
            self.precision = Precision(task="binary")
            self.recall = Recall(task="binary")
            self.f1 = F1Score(task="binary")
            self.accuracy = Accuracy(task="binary")
            self.val_precision = Precision(task="binary")
            self.val_recall = Recall(task="binary")
            self.val_f1 = F1Score(task="binary")
            self.val_accuracy = Accuracy(task="binary")
            self.save_hyperparameters(config)

        def normalize_data(self, x):
            x = x.reshape(x.shape[0], 24, 256)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
            return x

        def _forward(self, batch):
            x, y, subjects = batch
            gender_mapping = {"M": 0, "F": 1}
            # y is a tuple of (n_samples,) with values "M" or "F"
            y = torch.tensor([gender_mapping[gender] for gender in y], dtype=torch.long, device=self.device)
            scores = self.model(self.normalize_data(x))
            _, preds = scores.max(1)
            loss = F.cross_entropy(scores, y)
            return loss, preds, y

        def training_step(self, batch, batch_idx):
            loss, preds, y = self._forward(batch)
            self.accuracy.update(preds, y)
            self.precision.update(preds, y)
            self.recall.update(preds, y)
            self.f1.update(preds, y)
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def on_train_epoch_end(self):
            self.log("train/recall_epoch", self.recall, on_step=False, on_epoch=True)
            self.log("train/precision_epoch", self.precision, on_step=False, on_epoch=True)
            self.log("train/f1_epoch", self.f1, on_step=False, on_epoch=True)
            self.log("train/accuracy_epoch", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            loss, preds, y = self._forward(batch)
            self.val_accuracy.update(preds, y)
            self.val_precision.update(preds, y)
            self.val_recall.update(preds, y)
            self.val_f1.update(preds, y)
            self.log('val/loss', loss, on_step=False, on_epoch=True)
        
        def on_validation_epoch_end(self):
            self.log("val/recall_epoch", self.recall, on_step=False, on_epoch=True)
            self.log("val/precision_epoch", self.val_precision, on_step=False, on_epoch=True)
            self.log("val/f1_epoch", self.val_f1, on_step=False, on_epoch=True)
            val_acc = self.val_accuracy.compute()
            self.log("val/accuracy_epoch", val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("hp_metric", val_acc, on_epoch=True)

        def configure_optimizers(self):
            print('Learning rate:', self.lr)
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    return EEGModel(config)

def run_task(releases, tasks, target_name, folds=10, weights=None, model_freeze=False, random_add=42, train_epochs=20, save_weights="", batch_size=100, lrate=0.00002, model_name = 'EEGNeX', dropout=0.5):
    # random seed for reproducibility
    L.seed_everything(random_add)

    global deep_copy_dataset

    if not isinstance(releases, list):
        releases = [releases]
    if not isinstance(tasks, list):
        tasks = [tasks]



    # # Save hparams to results dir
    # hparams_file = f"results/hparams_{'_'.join(tasks)}_{target_name}_{model_name}.json"
    # with open(hparams_file, "w") as f:
    #     json.dump(hparams, f, indent=2)

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

    # ## Creating a Training and Test Set
    correct_train_list = []
    correct_test_list  = []
    correct_test_ci    = []
    unique_subjects, unique_indices = np.unique(windows_ds.description["subject"], return_index=True)
    unique_gender = windows_ds.description["sex"][unique_indices].values
    print(f"Class distribution in full set: {(unique_gender == 'M').sum()} male, {(unique_gender == 'F').sum()} female")
    if folds > 1:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_add)
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

        # Check the balance of the dataset
        print(f"Number of datasets in balanced set: {len(train_ds.datasets)} train, {len(val_ds.datasets)} val")
        print(f"Number of samples in balanced set: {len(train_ds)} train, {len(val_ds)} val")
        # print(f"Class distribution in training set {train_n} of each class; validation set {val_n} of each class")

        # Create dataloaders with smaller batch size to save memory
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4)
        val_loader   = DataLoader(  val_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)


        # create model and hparams dict
        hparams = {
            "batch_size": batch_size,
            "dropout": dropout,
            "lr": lrate,
            "random_seed": random_add,
            "model_name": model_name,
        }
        model = create_model(hparams)

        if weights is not None:
            model.load_state_dict(torch.load(weights))
        # from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        # early_stop = EarlyStopping(
        #     monitor="val/accuracy_epoch",
        #     mode="max",
        #     patience=30,
        #     verbose=True,
        #     min_delta=0.0
        # )
        # Find learning rate before logger/trainer creation
        # trainer_tmp = L.Trainer(
        #     max_epochs=1,
        #     accelerator="auto",
        #     devices=1 if torch.cuda.is_available() else None,
        #     enable_checkpointing=False,
        #     logger=False,
        #     callbacks=[]
        # )
        # tuner = Tuner(trainer_tmp)
        # lr_finder = tuner.lr_find(model, train_loader, val_loader)
        # new_lr = lr_finder.suggestion()
        # model.hparams.lr = new_lr
        # hparams["lr_final"] = float(new_lr)


        # Now create logger with correct hparams
        tb_logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name=f"experiment_{'_'.join(tasks)}_{target_name}_{model_name}_k{folds}",
        )
        # Log hparams 
        tb_logger.log_hyperparams(hparams)
        # Now create trainer with logger and early stopping
        trainer = L.Trainer(
            max_epochs=train_epochs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            enable_checkpointing=False,
            logger=tb_logger,
            # callbacks=[early_stop]
        )

        # Fit model
        trainer.fit(model, train_loader, val_loader)

        # Use PyTorch Lightning metrics for kfold reporting
        train_acc = float(trainer.callback_metrics.get('train/accuracy_epoch', 0.0))
        test_acc = float(trainer.callback_metrics.get('val/accuracy_epoch', 0.0))
        correct_train_list.append(train_acc)
        correct_test_list.append(test_acc)


        # Optionally, compute bootstrap CI for test set using Lightning predictions
        # try:
        #     from scipy.stats import bootstrap
        #     preds = trainer.callback_metrics.get('val/accuracy_epoch', None)
        #     # If you want to use predictions, you can use a Lightning callback to store them for more advanced CI
        #     # Here, just use test_acc as both bounds if not available
        #     if preds is not None:
        #         correct_test_ci.append([test_acc, test_acc])
        #     else:
        #         correct_test_ci.append([test_acc, test_acc])
        # except Exception as e:
        #     correct_test_ci.append([test_acc, test_acc])

        # # Save model weights immediately to avoid storing in memory
        # if save_weights is not None and save_weights != "":
        #     torch.save(model.state_dict(), save_weights.replace(".pth", "") + f"_{it_fold}.pth")
        #     print(f"Saved model {it_fold} weights to {save_weights.replace('.pth', '')}_{it_fold}.pth")

    # Print summary statistics for kfolds
    def ci95(arr):
        arr = np.array(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        ci = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return mean, std, ci

    if len(correct_train_list) > 0:
        train_mean, train_std, train_ci = ci95(correct_train_list)
        print(f"Train acc: mean={train_mean:.4f}, std={train_std:.4f}, 95% CI=±{train_ci:.4f}")
    if len(correct_test_list) > 0:
        test_mean, test_std, test_ci = ci95(correct_test_list)
        print(f"Test acc: mean={test_mean:.4f}, std={test_std:.4f}, 95% CI=±{test_ci:.4f}")
    if len(correct_test_ci) > 0:
        print(f"Test bootstrap CIs: {correct_test_ci}")

    return correct_train_list, correct_test_list, correct_test_ci

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
model_name = 'EEGNeX'
# model_name = 'TSception'
releases_train = ['R12'] #releases[:-1]
new_tasks = {'contrastChangeDetection': ['contrastChangeDetection']}
factors = ['attention']
folds = 10

import random
if True:
    for task in list(new_tasks.keys()):
        for factor in factors:
            # train set on all releases
            # if os.path.exists(json_file):
            #     print(f"Skipping {task}_{factor} because it already exists")
            #     continue
            weights_file_base = "weights_" + task+ "_" + factor + "_" + model_name + ".pth"
            batch_sizes = [64, 128, 256, 512, 1024, 2048]
            lrates = [0.002, 0.0002, 0.00006, 0.00002]
            dropouts = [0.5, 0.6, 0.7, 0.8, 0.9]
            seeds = [0, 42, 9]
            models = ['EEGNeX', 'TSception']
            for model_name in models: 
                combinations = [ (batch_size, lrate, random_add, dropout) for batch_size in batch_sizes for lrate in lrates for random_add in seeds for dropout in dropouts]
                # shuffle combinations
                random.shuffle(combinations)
                for batch_size, lrate, random_add, dropout in combinations:
                    print(f"Running task {task}, factor {factor}, model {model_name}, batch_size {batch_size}, lrate {lrate}, random_add {random_add}, dropout {dropout}")
                    res_train, res_test, res_test_ci = run_task(releases_train, task, factor, folds=folds, random_add=random_add, train_epochs=150, batch_size=batch_size, lrate=lrate, model_name=model_name, dropout=dropout, save_weights=weights_file_base)
                    json_file = f"results/{task}_{factor}_{model_name}_bs{batch_size}_lr{lrate}_seed{random_add}_dropout{dropout}_k{folds}.json"
                    with open(json_file, "w") as f:
                        json.dump({"train": res_train, "test": res_test, "test_ci": res_test_ci}, f)
            # test set on R12
            # res_test_r12 = []
            # for fold in range(folds):
            #     weights_file = weights_file_base.replace(".pth", "") + f"_{fold}.pth"
            #     _, res_test_r12_tmp, _ = run_task("R12", new_tasks[task], factor, folds=1,  train_epochs=1,  batch_size=2000, weights=weights_file, model_freeze=True, model_name=model_name, random_add=fold)
            #     res_test_r12.append(res_test_r12_tmp)
            # with open(json_file, "w") as f:
            #     json.dump({"train": res_train, "test": res_test, "test_ci": res_test_ci, "test_r12": res_test_r12}, f)
            # print(f"Task: {task}, Factor: {factor}, Train: {res_train}, Test: {res_test_ci}")
