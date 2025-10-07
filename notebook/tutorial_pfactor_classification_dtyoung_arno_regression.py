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
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import os
import lightning as L
from braindecode.models import EEGNeX, TSception, EEGConformer
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import Subset
from pathlib import Path
import json
from torch.utils.data import Subset
import numpy as np


cache_dir = Path("/mnt/v1/arno/eeg2025")
SFREQ = 100  # sampling frequency

def process_data(releases, tasks, target_names):
    for release in releases:
        missing = []
        for task in tasks:
            for target_name in target_names:
                cached_data_folder_name = "data/hbn_reg_" + release + "_" + task + "_" + target_name
                if not os.path.exists(cached_data_folder_name):
                    if target_name == target_names[0]:
                        missing.append(task)
                    
        # if len(missing) < 3:
        #     print(f"All data exists in {release} missing [{', '.join(missing)}]")
        #     continue
        # else:
        #     print(f"Incomplete data in {release}: Missing [{', '.join(missing)}]")
        
        if release != "R12":
            ds_sexdata = EEGChallengeDataset(
                release=release,
                cache_dir=cache_dir,
                task=tasks,
                mini=False,
                # run="1",
                download=False,
                target_name=target_names[0]
            )
        else:
            ds_sexdata = EEGDashDataset(
                dataset="HBN-R12_L100",
                cache_dir=cache_dir,
                task=tasks,
                # run="1",
                download=False,
                target_name=target_names[0]
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
            if len(all_datasets) > 0:                                 
                print(f"Preprocessing {len(all_datasets.datasets)} datasets...")
                
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
                cached_data_folder_name = "data/hbn_reg_" + release + "_" + task + "_" + target_name
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
                    n_outputs=1,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            elif config['model_name'] == 'TSception':
                self.model = TSception(
                    n_chans=24,
                    n_outputs=1,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            elif config['model_name'] == 'EEGConformer':
                self.model = EEGConformer(
                    n_chans=24,
                    n_outputs=1,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            elif config['model_name'] == 'EEGConformerSimplified':
                self.model = EEGConformer(
                    n_chans=24,
                    n_outputs=1,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                    n_filters_time=32,        # Try 32 instead of default 40
                    filter_time_length=20,    # Try 20
                    att_depth=4,             # Reduce attention layers
                    att_heads=8,             # Reduce attention heads
                    pool_time_stride=12,     # Adjust pooling
                    pool_time_length=64,     # Adjust pooling window
                )
            self.lr = config['lr']
            self.weight_decay = config['weight_decay']
            self.mae = MeanAbsoluteError()
            self.mse = MeanSquaredError()
            self.r2 = R2Score()
            self.val_mae = MeanAbsoluteError()
            self.val_mse = MeanSquaredError()
            self.val_r2 = R2Score()
            self.target_std = config.get('target_std', None)
            # Baseline MAE (median-based) passed from run_task for S metric
            self.train_baseline_mae = float(config.get('train_baseline_mae', 1.0))
            self.model_freeze = bool(config.get('model_freeze', False))
            if self.model_freeze:
                self.automatic_optimization = False
            self.save_hyperparameters(config)

        def normalize_data(self, x):
            x = x.reshape(x.shape[0], 24, 256)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
            return x

        def _forward(self, batch):
            x, y, subjects = batch
            # Debug: print information about target values
            if hasattr(self, '_debug_printed') == False:
                print(f"Target values type: {type(y[0]) if len(y) > 0 else 'empty'}")
                print(f"Target values sample: {y[:5] if len(y) >= 5 else y}")
                print(f"Target values unique types: {set(type(val) for val in y)}")
                print(f"Target values unique count in batch: {len(set([float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v) for v in y]))}")
                print(f"Batch size: {len(y)}")
                self._debug_printed = True
            
            # Ensure float32 types for model inputs and targets
            x = x.float()
            # Accept list/array/tuple of floats from dataset; convert to float32 tensor on device
            if not torch.is_tensor(y):
                y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
            else:
                y = y.to(self.device, dtype=torch.float32)
            scores = self.model(self.normalize_data(x))
            preds = scores.squeeze().to(dtype=torch.float32)
            # loss = F.mse_loss(preds, y)
            loss = F.l1_loss(preds, y)
            return loss, preds, y

        def training_step(self, batch, batch_idx):
            if self.model_freeze:
                with torch.no_grad():
                    loss, preds, y = self._forward(batch)
            else:
                loss, preds, y = self._forward(batch)

            self.mae.update(preds, y)
            self.mse.update(preds, y)
            self.r2.update(preds, y)

            denom = (self.target_std if self.target_std is not None else 1.0) + 1e-8
            normalized_rmse = torch.sqrt(F.mse_loss(preds, y)) / denom
            self.log('train/normalized_rmse', normalized_rmse, on_step=True, on_epoch=True)
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        
        def on_train_epoch_end(self):
            # 1) read scalar values BEFORE any logging of metric objects
            train_mse_value = self.mse.compute()
            train_mae_value = self.mae.compute()
            train_r2_value  = self.r2.compute()
            denom = (self.target_std if self.target_std is not None else 1.0) + 1e-8
            train_normalized_rmse = torch.sqrt(train_mse_value) / denom

            # 2) log scalars (not metric objects) to avoid implicit resets mid-function
            self.log("train/mse_epoch", train_mse_value, on_step=False, on_epoch=True)
            self.log("train/mae_epoch", train_mae_value, on_step=False, on_epoch=True)
            self.log("train/r2_epoch",  train_r2_value,  on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/normalized_rmse_epoch", train_normalized_rmse, on_step=False, on_epoch=True)
            # S = 1 - MAE(model)/MAE(baseline median)
            s_train = float(train_mae_value) / (self.train_baseline_mae + 1e-8)
            self.log("train/S_epoch", s_train, on_step=False, on_epoch=True)

            # 3) now reset explicitly
            self.mse.reset(); self.mae.reset(); self.r2.reset()

        def validation_step(self, batch, batch_idx):
            loss, preds, y = self._forward(batch)
            self.val_mae.update(preds, y)
            self.val_mse.update(preds, y)
            self.val_r2.update(preds, y)
            
            # Compute normalized RMSE (RMSE / std)
            # Guard against None target_std
            denom = (self.target_std if self.target_std is not None else 1.0) + 1e-8
            val_normalized_rmse = torch.sqrt(F.mse_loss(preds, y)) / denom
            self.log('val/normalized_rmse', val_normalized_rmse, on_step=False, on_epoch=True)
            self.log('val/loss', loss, on_step=False, on_epoch=True)
        
        def on_train_start(self):
            # Initialize hp_metric to 0.5 to avoid -1 in TensorBoard
            self.log("hp_metric", 0.5, on_epoch=True)

        def on_validation_epoch_end(self):
            val_mse_value = self.val_mse.compute()
            val_mae_value = self.val_mae.compute()
            val_r2_value  = self.val_r2.compute()
            denom = (self.target_std if self.target_std is not None else 1.0) + 1e-8
            val_normalized_rmse = torch.sqrt(val_mse_value) / denom

            self.log("val/mse_epoch", val_mse_value, on_step=False, on_epoch=True)
            self.log("val/mae_epoch", val_mae_value, on_step=False, on_epoch=True)
            self.log("val/r2_epoch",  val_r2_value,  on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/normalized_rmse_epoch", val_normalized_rmse, on_step=False, on_epoch=True)
            self.log("hp_metric", val_r2_value, on_epoch=True)
            # S = 1 - MAE(model)/MAE(baseline median)
            s_val = float(val_mae_value) / (self.train_baseline_mae + 1e-8)
            self.log("val/S_epoch", s_val, on_step=False, on_epoch=True)

            self.val_mse.reset(); self.val_mae.reset(); self.val_r2.reset()

        def configure_optimizers(self):
            if self.model_freeze:
                return None

            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/r2_epoch",
                    "frequency": 1
                }
            }

    return EEGModel(config)

def run_task(releases, tasks, target_name, folds=10, weights=None, model_freeze=False, experiment_name=None, random_add=42, train_epochs=20, save_weights="", batch_size=100, weight_decay=1e-4, lrate=0.00002, model_name = 'EEGNeX', dropout=0.5):
    from torch.utils.data import Subset

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
            cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_reg_" + release + "_" + task + "_" + target_name
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
    correct_val_list  = []
    train_norm_rmse_list = []
    val_norm_rmse_list = []
    train_norm_mae_list = []
    val_norm_mae_list = []
    target_std_fold = None
    unique_subjects, unique_indices = np.unique(windows_ds.description["subject"], return_index=True)
    
    if folds > 1:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_add)
        splits = splitter.split(unique_subjects)
    else:
        train_idx, val_idx = train_test_split(np.arange(len(unique_subjects)),train_size=0.8,random_state=random_add)
        splits = [(train_idx, val_idx)]
        
    for it_fold, (train_idx, val_idx) in enumerate(splits):
        train_ds = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in unique_subjects[train_idx] \
            and ds.description[target_name] != np.nan and ds.description[target_name] != None and not isinstance(ds.description[target_name], str) \
            and (ds.description[target_name] < -0.5 or ds.description[target_name] > 0.5)])
        val_ds   = BaseConcatDataset([ds for ds in windows_ds.datasets if ds.description.subject in unique_subjects[val_idx]   \
            and ds.description[target_name] != np.nan and ds.description[target_name] != None and not isinstance(ds.description[target_name], str) \
            and (ds.description[target_name] < -0.5 or ds.description[target_name] > 0.5)])

        # Create dataloaders with smaller batch size to save memory
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4)
        val_loader   = DataLoader(  val_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)

        # compute dataset-wide target std on training set (from metadata/description)
        # We take per-recording targets from description[target_name]
        raw_targets = []
        for ds in train_ds.datasets:
            v = ds.description.get(target_name, None)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                raw_targets.append(v)
        targets_np = np.array(raw_targets, dtype=np.float32)
        if targets_np.size == 0:
            target_std = 1.0
            train_baseline_mae = 1.0
        else:
            dd = 1 if targets_np.size > 1 else 0
            target_std = float(np.std(targets_np, ddof=dd))
            med = float(np.median(targets_np))
            train_baseline_mae = float(np.mean(np.abs(targets_np - med)))
            if not np.isfinite(target_std) or target_std <= 0:
                target_std = 1.0
            if not np.isfinite(train_baseline_mae) or train_baseline_mae <= 0:
                train_baseline_mae = 1.0

        # Print target statistics
        print(f"Training targets - count: {targets_np.size}, mean: {targets_np.mean() if targets_np.size else 0:.4f}, std: {target_std:.4f}, mae: {train_baseline_mae:.4f}, min: {targets_np.min() if targets_np.size else 0:.4f}, max: {targets_np.max() if targets_np.size else 0:.4f}")

        # create model and hparams dict
        hparams = {
            "batch_size": batch_size,
            "dropout": dropout,
            "lr": lrate,
            "weight_decay": weight_decay,
            "random_seed": random_add,
            "model_name": model_name,
            "fold": it_fold,
            "target_std": target_std,
            "model_freeze": bool(model_freeze),
            "train_baseline_mae": train_baseline_mae,
        }
        model = create_model(hparams)

        if weights is not None:
            model.load_state_dict(torch.load(weights))
        # Optionally freeze all model parameters (no training)
        if model_freeze:
            for p in model.parameters():
                p.requires_grad = False
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val/r2_epoch', # normalized mae
            patience=15,
            mode='max', # min because we are monitoring a loss
            min_delta=0.001
        )

        # Now create logger with correct hparams
        if experiment_name is None or experiment_name == "":
            experiment_name = f"balanced_experiment_{'_'.join(tasks)}_{target_name}_{model_name}_k{folds}"
        tb_logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name=experiment_name,
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
            callbacks=[early_stopping]
        )

        # Fit model
        trainer.fit(model, train_loader, val_loader)

        # Use PyTorch Lightning metrics for kfold reporting
        train_norm_rmse = float(trainer.callback_metrics.get('train/normalized_rmse_epoch', 0.0))
        val_norm_rmse = float(trainer.callback_metrics.get('val/normalized_rmse_epoch', 0.0))
        # keep R2 lists for console summary only if needed; but do not save
        train_r2 = float(trainer.callback_metrics.get('train/r2_epoch', 0.0))
        val_r2 = float(trainer.callback_metrics.get('val/r2_epoch', 0.0))
        train_norm_mae = float(trainer.callback_metrics.get('train/S_epoch', 0.0))
        val_norm_mae = float(trainer.callback_metrics.get('val/S_epoch', 0.0))
        correct_train_list.append(train_r2)
        correct_val_list.append(val_r2)
        train_norm_rmse_list.append(train_norm_rmse)
        val_norm_rmse_list.append(val_norm_rmse)
        train_norm_mae_list.append(train_norm_mae)
        val_norm_mae_list.append(val_norm_mae)

        # Save model weights 
        if save_weights is not None and save_weights != "":
            torch.save(model.state_dict(), save_weights.replace(".pth", f"_{it_fold}.pth"))
            print(f"Saved model {it_fold} weights to {save_weights.replace('.pth', '')}_{it_fold}.pth")

    # Print summary statistics for kfolds
    def ci95(arr):
        arr = np.array(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        ci = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return mean, std, ci

    if len(correct_train_list) > 0:
        train_mean, train_std, train_ci = ci95(correct_train_list)
        print(f"Train R2: mean={train_mean:.4f}, std={train_std:.4f}, 95% CI=±{train_ci:.4f}")
    if len(correct_val_list) > 0:
        val_mean, val_std, val_ci = ci95(correct_val_list)
        print(f"Val R2: mean={val_mean:.4f}, std={val_std:.4f}, 95% CI=±{val_ci:.4f}")
    if len(train_norm_rmse_list) > 0:
        train_norm_rmse_mean, train_norm_rmse_std, train_norm_rmse_ci = ci95(train_norm_rmse_list)
        print(f"Train Normalized RMSE: mean={train_norm_rmse_mean:.4f}, std={train_norm_rmse_std:.4f}, 95% CI=±{train_norm_rmse_ci:.4f}")
    if len(val_norm_rmse_list) > 0:
        val_norm_rmse_mean, val_norm_rmse_std, val_norm_rmse_ci = ci95(val_norm_rmse_list)
        print(f"Val Normalized RMSE: mean={val_norm_rmse_mean:.4f}, std={val_norm_rmse_std:.4f}, 95% CI=±{val_norm_rmse_ci:.4f}")

    return correct_train_list, correct_val_list, train_norm_rmse_list, val_norm_rmse_list, train_norm_mae_list, val_norm_mae_list, target_std_fold, train_baseline_mae, unique_subjects[val_idx]

def check_experiment_exists(task, factor, model_name, folds, batch_size, lrate, random_add, dropout):
    log_folder = Path("lightning_logs") / f"experiment_{task}_{factor}_{model_name}_k{folds}"
    if not log_folder.exists():
        return False
    # check if any subfolder exists
    subfolders = [f for f in log_folder.iterdir() if f.is_dir()]
    if len(subfolders) == 0:
        return False
    # scan all hparams.yaml files in subfolders for matching hyperparameters
    for subfolder in subfolders:
        hparams_file = subfolder / "hparams.yaml"
        if not hparams_file.exists():
            continue
        with open(hparams_file, "r") as f:
            lines = f.readlines()
            hparams = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    hparams[key.strip()] = value.strip()
            # check if all hyperparameters match
            if (hparams.get("batch_size") == str(batch_size) and
                hparams.get("dropout") == str(dropout) and
                hparams.get("lr") == str(lrate) and
                hparams.get("random_seed") == str(random_add) and
                hparams.get("model_name") == model_name):
                print(f"Experiment already exists in {subfolder}")
                return True
    return False

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

new_tasks = { 'movies': ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent'],
  'restingstate': ['RestingState'],
  'contrastChangeDetection': ['contrastChangeDetection'],
  'seqLearning': ['seqLearning6target', 'seqLearning8target'],
  'surroundSupp': ['surroundSupp'],
  'symbolSearch': ['symbolSearch'],
  'all_tasks': ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'RestingState', 'ThePresent', 'contrastChangeDetection', 'seqLearning6target', 'seqLearning8target', 'surroundSupp', 'symbolSearch']
  }
releases_train = releases[:-1]
# releases_train = ["R12"] #"R11"
tasks = [  
# # 'DespicableMe',
# #   'DiaryOfAWimpyKid',
# #   'FunwithFractals',
# #   'ThePresent',
# #   'RestingState',
  'contrastChangeDetection',
# #   'seqLearning6target',
# #   'seqLearning8target',
# #   'surroundSupp',
# #   'symbolSearch'
]
# process_data(releases, tasks, ['attention'])
# sys.exit()

factors = ['attention']

models = ['EEGConformerSimplified']#, 'TSception']#, 'EEGNeX']
folds = 1
bypass_run = False

import random
from torch.utils.data import Subset
import numpy as np
# create empty pandas dataframe with columns seeds, val, subj
df = pd.DataFrame(columns=['seeds', 'val', 'subj'])
for factor in factors:
    # train set on all releases
    # if os.path.exists(json_file):
    #     print(f"Skipping {task}_{factor} because it already exists")
    #     continue
    batch_sizes = [128] #[64, 128]#, 256, 512, 1024, 2048]
    lrates = [0.00002]#[0.002, 0.0002, 0.00006, 0.00002]
    dropouts = [0.7] #[0.6, 0.7, 0.8]
    seeds = [77, 15, 20, 31, 64, 88, 99, 0, 42, 9]
    seeds_R11 = np.array(range(30)) + 77
    #seeds = seeds_R11
    #seeds = [77]
    # seeds = [0]
    for model_name in models:
        combinations = [(batch_size, lrate, random_add, dropout) for batch_size in batch_sizes for lrate in lrates for random_add in seeds for dropout in dropouts]
        # combinations = random.sample(combinations, min(20, len(combinations)))
        # random.shuffle(combinations)
        # combinations = [(64, 0.002, seed, 0.6) for seed in seeds]
        print(f"Total combinations to run: {len(combinations)}")
        for iter_comb, (batch_size, lrate, random_add, dropout) in enumerate(combinations):
            # if check_experiment_exists(task, factor, model_name, folds, batch_size, lrate, random_add, dropout):
            #     print(f"Skipping existing experiment for {task}, {factor}, {model_name}, {folds}, {batch_size}, {lrate}, {random_add}, {dropout}")
            #     continue
            task_str = '_'.join(tasks) if len(tasks) < 3 else 'alltasks'
            release_str = '_'.join(releases_train) if len(releases_train) < 3 else 'allreleases'
            experiment_name = f"__regression_{release_str}_{task_str}_{factor}_{model_name}_k{folds}"
            print(f"Running experiment {experiment_name} with hyperparams: bs={batch_size}, lr={lrate}, seed={random_add}, dropout={dropout}")
            weights_file_base = f"checkpoints/{experiment_name}_bs{batch_size}_lr{lrate}_seed{random_add}_dropout{dropout}.pth"
            json_file = f"results/{experiment_name}_bs{batch_size}_lr{lrate}_seed{random_add}_dropout{dropout}.json"
            if bypass_run == False:
                res_train, res_val, train_norm_rmse, val_norm_rmse, train_norm_mae, val_norm_mae, target_std, train_baseline_mae, res_val_subject = run_task(
                    releases_train, tasks, factor, folds=folds, random_add=random_add, experiment_name=experiment_name,
                    train_epochs=100, batch_size=batch_size, lrate=lrate, model_freeze=False, weight_decay=1e-2,
                    model_name=model_name, dropout=dropout, save_weights=weights_file_base
                )
            else:   
                res_train, res_val, train_norm_rmse, val_norm_rmse, target_std, res_val_subject = [], [], [], [], 1.0, []
                print(f"Skipping run for {experiment_name} because bypass_run is True")
                
            # test set on R12
            cached_data_folder_names = []
            for task in tasks:
                cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_reg_R12_" + task + "_" + factor
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

            test_ds = BaseConcatDataset(windows_ds)
            
            # filter None, string, and NaN
            test_ds = BaseConcatDataset([ds for ds in test_ds.datasets if ds.description[factor] != np.nan \
                and ds.description[factor] != None and not isinstance(ds.description[factor], str) \
                and ds.description[factor] != np.nan and (ds.description[factor] < -0.5 or ds.description[factor] > 0.5)])
            
            # analyze_fold_distribution(test_ds, test_ds, 'R12')
            test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

            res_test_r12 = []
            test_norm_rmse_r12 = []
            test_norm_mae_r12 = []
            for fold in range(folds):
                weights_file = weights_file_base.replace(".pth", "") + f"_{fold}.pth"
                # load model weights and run inference on R12
                if not os.path.exists(weights_file):
                    print(f"Missing weights file {weights_file}, skipping")
                    continue
                model = create_model({
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "lr": lrate,
                    "random_seed": random_add,
                    "model_name": model_name,
                    "fold": fold,
                    "target_std": target_std,
                    "train_baseline_mae": train_baseline_mae,
                    "weight_decay": 1e-2,
                })
                model.load_state_dict(torch.load(weights_file, map_location="cpu"))
                model.eval()
                trainer_test = L.Trainer(accelerator="auto", devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False)
                test_result = trainer_test.validate(model, test_loader, verbose=False)
                test_r2 = float(test_result[0].get('val/r2_epoch', 0.0))
                test_norm_rmse = float(test_result[0].get('val/normalized_rmse_epoch', 0.0))
                test_norm_mae  = float(test_result[0].get('val/S_epoch', 0.0))
                res_test_r12.append(test_r2)
                test_norm_rmse_r12.append(test_norm_rmse)
                test_norm_mae_r12.append(test_norm_mae)
            print(f"Test on R12 R2: {res_test_r12}")
            print(f"Test on R12 Normalized RMSE: {test_norm_rmse_r12}")
            
            with open(json_file, "w") as f:
                json.dump({
                    "version": iter_comb,
                    "seed": float(random_add), 
                    "train_norm_rmse": train_norm_rmse,
                    "val_norm_rmse": val_norm_rmse,
                    "test_norm_rmse": test_norm_rmse_r12,
                    "train_norm_mae": train_norm_mae,
                    "val_norm_mae": val_norm_mae,
                    "test_norm_mae": test_norm_mae_r12
                }, f)
                
            # radon_add an res_val are single values but res_val_subject is a list, so we need to duplicate the values for each subject
            random_add_dup = [random_add] * len(res_val_subject)
            res_val_dup = [res_val] * len(res_val_subject)
            # Create a new row as a DataFrame and concatenate
            new_row = pd.DataFrame({'seeds': random_add_dup, 'val': res_val_dup, 'subj': res_val_subject})
            df = pd.concat([df, new_row], ignore_index=True)

# save pandas dataframe to csv with 5 decimal places
df.to_csv(f"results/R12_regression_val_{models[0]}.csv", index=False, float_format='%.5f')