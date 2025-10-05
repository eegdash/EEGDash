# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset for a collection of subject. The dataset ds005505 contains 136 subjects with both male and female participants.
# %%
from eegdash import EEGDashDataset
from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess
from braindecode.datasets.base import BaseConcatDataset, EEGWindowsDataset, BaseDataset
from braindecode.datautil import load_concat_dataset
from pathlib import Path
import numpy as np
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
import os
from scipy.stats import bootstrap
import lightning as L
from braindecode.models import EEGNeX, TSception, EEGConformer
from lightning.pytorch.tuner import Tuner
from torchmetrics import Recall, Precision, F1Score, Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Subset
from typing import Dict
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

class RelabelWindowsDatasetWrapper(BaseDataset):
    def __init__(self, dataset: EEGWindowsDataset, factor="attention", positive_threshold=0.5, negative_threshold=-0.5):
        """
        Wrapper for EEGWindowsDataset to relabel windows based on a specific factor.
        """
        self.dataset = dataset
        self._description = dataset.description
        self.factor = factor
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    @property
    def description(self):
        return self._description

    def __len__(self):
        return len(self.dataset)

    def _get_recording_metadata(self) -> Dict:
        return self._description

    def __getitem__(self, index):
        X, _, subject = self.dataset[index]

        subject_metadata = self._get_recording_metadata()
        factor_score = subject_metadata[self.factor]

        if factor_score >= self.positive_threshold:
            label = 'Positive'
        elif factor_score <= self.negative_threshold:
            label = 'Negative'
        else:
            label = 'Neutral'

        return X, label, subject

def is_dataset_within_threshold_class_by_factor(ds, factor="attention", positive_threshold=0.5, negative_threshold=-0.5):
    '''Check if a dataset has a factor score within the positive or negative threshold'''
    factor_score = ds.description[factor]
    return factor_score < negative_threshold or factor_score > positive_threshold

def create_model(config):
    class EEGModel(L.LightningModule):
        def __init__(self, config):
            super(EEGModel, self).__init__()
            drop_prob = config.get('dropout', 0.7)
            print(f"Using model {config['model_name']} with dropout {drop_prob}")
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
            elif config['model_name'] == 'FBCNet':
                from braindecode.models import FBCNet
                self.model = FBCNet(
                    n_chans=24,
                    n_outputs=2,
                    n_times=256,
                    sfreq=128,
                )
            elif config['model_name'] == 'EEGConformer':
                self.model = EEGConformer(
                    n_chans=24,
                    n_outputs=2,
                    n_times=256,
                    sfreq=128,
                    drop_prob=drop_prob,
                )
            elif config['model_name'] == 'EEGConformerSimplified':
                self.model = EEGConformer(
                    n_chans=24,
                    n_outputs=2,
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
            self.precision = Precision(task="binary")
            self.recall = Recall(task="binary")
            self.f1 = F1Score(task="binary")
            self.accuracy = Accuracy(task="binary")
            self.val_precision = Precision(task="binary")
            self.val_recall = Recall(task="binary")
            self.val_f1 = F1Score(task="binary")
            self.val_accuracy = Accuracy(task="binary")
            self.test_precision = Precision(task="binary")
            self.test_recall = Recall(task="binary")
            self.test_f1 = F1Score(task="binary")
            self.test_accuracy = Accuracy(task="binary")
            self.save_hyperparameters(config)

        def normalize_data(self, x):
            x = x.reshape(x.shape[0], 24, 256)
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
            return x

        def _forward(self, batch):
            x, y, subjects = batch
            label_mapping = {"Negative": 0, "Positive": 1}
            # y is a tuple of (n_samples,) with values "Negative" or "Positive"
            y = torch.tensor([label_mapping[label] for label in y], dtype=torch.long, device=self.device)
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
            self.log("val/recall_epoch", self.val_recall, on_step=False, on_epoch=True) # AI says it should be self.val_recall
            self.log("val/precision_epoch", self.val_precision, on_step=False, on_epoch=True)
            self.log("val/f1_epoch", self.val_f1, on_step=False, on_epoch=True)
            val_acc = self.val_accuracy.compute()
            self.log("val/accuracy_epoch", val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("hp_metric", val_acc, on_epoch=True)

        def test_step(self, batch, batch_idx):
            loss, preds, y = self._forward(batch)
            self.test_accuracy.update(preds, y)
            self.test_precision.update(preds, y)
            self.test_recall.update(preds, y)
            self.test_f1.update(preds, y)
            self.log('test/loss', loss, on_step=False, on_epoch=True)

        def on_test_epoch_end(self):
            self.log("test/recall_epoch", self.test_recall, on_step=False, on_epoch=True)
            self.log("test/precision_epoch", self.test_precision, on_step=False, on_epoch=True)
            self.log("test/f1_epoch", self.test_f1, on_step=False, on_epoch=True)
            test_acc = self.test_accuracy.compute()
            self.log("test/accuracy_epoch", test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("hp_metric", test_acc, on_epoch=True)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr,
                weight_decay=1e-4
            )
            
            # Reduce LR when validation accuracy plateaus
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
                    "monitor": "val/accuracy_epoch",
                    "frequency": 1
                }
            }

    return EEGModel(config)

# Add this analysis before training
def analyze_fold_distribution(train_ds, val_ds):
    """Analyze gender and subject distribution per fold"""
    # For Subset, get description DataFrame
    # Use list comprehensions for efficiency, but if datasets are large, consider vectorized access if possible.
    train_labels, train_subjects = zip(*[(label, subject) for _, label, subject in train_ds])
    train_label_dist = pd.Series(train_labels).value_counts()
    print(f"Statistics of balanced dataset")
    print(f"Train subjects: {len(set(train_subjects))}, Train labels: {train_label_dist}")
    if val_ds is not None:
        val_labels, val_subjects = zip(*[(label, subject) for _, label, subject in val_ds])
        val_label_dist = pd.Series(val_labels).value_counts()
        subject_overlap = set(train_subjects) & set(val_subjects)
        if len(subject_overlap) > 0:
            raise ValueError(f"Subject overlap between train and val sets: {subject_overlap}")
        print(f"Val subjects: {len(set(val_subjects))}, Val labels: {val_label_dist}")
        print(f"Subject overlap: {subject_overlap}")

def balance_windows_by_class(ds, labels, random_seed=42):
    """
    Returns a torch.utils.data.Subset of ds with equal number of windows for M and F classes.
    """
    import numpy as np
    labels = np.array(labels)
    positive_indices = np.where(labels == 'Positive')[0]
    negative_indices = np.where(labels == 'Negative')[0]
    n_min = min(len(positive_indices), len(negative_indices))
    rng = np.random.default_rng(random_seed)
    positive_indices = rng.permutation(positive_indices)[:n_min]
    negative_indices = rng.permutation(negative_indices)[:n_min]
    selected_indices = np.concatenate([positive_indices, negative_indices])
    selected_indices = rng.permutation(selected_indices)
    return Subset(ds, selected_indices)

def run_task(releases, tasks, target_name, folds=10, weights=None, model_freeze=False, random_add=42, train_epochs=20, save_weights="", batch_size=100, lrate=0.00002, model_name = 'EEGNeX', dropout=0.5, data_split_seed=42, test_release=None, experiment_name=""):

    # random seed for reproducibility
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
    # Filter out subjects with neutral labels based on factor
    windows_ds = BaseConcatDataset([ds for ds in windows_ds.datasets 
                                    if is_dataset_within_threshold_class_by_factor(
                                        ds, 
                                        factor=target_name, 
                                        positive_threshold=0.5, 
                                        negative_threshold=-0.5
                                    )
    ])
    # Relabel windows based on factor
    relabled_windows_ds = BaseConcatDataset([RelabelWindowsDatasetWrapper(ds, factor=target_name, positive_threshold=0.5, negative_threshold=-0.5) for ds in windows_ds.datasets])
    print(f"Number of samples in relabeled_windows_ds: {len(relabled_windows_ds)}")

    # Efficiently extract subjects and labels using list comprehensions over datasets, not windows
    subjects = []
    labels = []
    for _, label, subj in relabled_windows_ds:
        labels.append(label)
        subjects.append(subj)
    subjects = np.array(subjects)
    labels = np.array(labels)

    # ## Creating a Training and Test Set
    # unique_subjects, unique_indices = np.unique(windows_ds.description["subject"], return_index=True)
    # unique_gender = windows_ds.description["sex"][unique_indices].values
    if folds > 1:
        # TODO double check
        # splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=data_split_seed)
        # splits = splitter.split(subjects, labels)
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=data_split_seed)
        dummy_X = np.zeros_like(subjects) # Dummy features
        splits = sgkf.split(X=dummy_X, y=labels, groups=subjects)
    else:
        # train_idx, val_idx = train_test_split(np.arange(len(unique_subjects)),train_size=0.8,stratify=unique_subjects_labels,random_state=data_split_seed)
        # splits = [(train_idx, val_idx)]
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=data_split_seed)
        splits = gss.split(X=np.zeros_like(subjects), y=labels, groups=subjects)

    correct_train_list = []
    correct_val_list  = []
    for it_fold, (train_idx, val_idx) in enumerate(splits):
        # train_ds = BaseConcatDataset([ds for ds in relabled_windows_ds.datasets if ds.description.subject in unique_subjects[train_idx]])
        # val_ds   = BaseConcatDataset([ds for ds in relabled_windows_ds.datasets if ds.description.subject in unique_subjects[val_idx]])
        train_ds = Subset(relabled_windows_ds, train_idx)
        val_ds   = Subset(relabled_windows_ds, val_idx)

        # Balance windows per class for train and val sets
        train_labels = [label for _, label, _ in train_ds]
        val_labels = [label for _, label, _ in val_ds]
        train_ds = balance_windows_by_class(train_ds, train_labels, random_seed=data_split_seed)
        val_ds = balance_windows_by_class(val_ds, val_labels, random_seed=data_split_seed)

        analyze_fold_distribution(train_ds, val_ds)

        L.seed_everything(random_add)

        # Create dataloaders with smaller batch size to save memory
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, prefetch_factor=4, num_workers=4)
        val_loader   = DataLoader(  val_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)


        # Load test data
        # test_cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_" + test_release + "_" + tasks[0] + "_" + target_name
        # test_windows_ds = load_concat_dataset(path=test_cached_data_folder_name, preload=False)
        # test_windows_ds = BaseConcatDataset([ds for ds in test_windows_ds.datasets
        #                                 if is_dataset_within_threshold_class_by_factor(
        #                                     ds, 
        #                                     factor=target_name, 
        #                                     positive_threshold=0.5, 
        #                                     negative_threshold=-0.5
        #                                 )
        # ])
        # # Relabel windows based on factor
        # test_relabled_windows_ds = BaseConcatDataset([RelabelWindowsDatasetWrapper(ds, factor=target_name, positive_threshold=0.5, negative_threshold=-0.5) for ds in test_windows_ds.datasets])
        # test_loader = DataLoader(test_relabled_windows_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)

        # create model and hparams dict
        hparams = {
            "batch_size": batch_size,
            "dropout": dropout,
            "lr": lrate,
            "random_seed": random_add,
            "model_name": model_name,
            "fold": it_fold,
        }
        model = create_model(hparams)

        if weights is not None:
            model.load_state_dict(torch.load(weights))
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val/accuracy_epoch',
            patience=10,
            mode='max',
            min_delta=0.001
        )
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
        # tb_logger = TensorBoardLogger(
        #     save_dir="lightning_logs",
        #     name=f"balanced_experiment_{'_'.join(tasks)}_{target_name}_{model_name}_k{folds}",
        # )
        from lightning.pytorch.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project="eegchallenge-test",
            name=experiment_name,
            config=hparams
        )
        # Log hparams
        # tb_logger.log_hyperparams(hparams)
        # Now create trainer with logger and early stopping
        trainer = L.Trainer(
            max_epochs=train_epochs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            enable_checkpointing=False,
            logger=None,# wandb_logger, #tb_logger,
            callbacks=[early_stopping]
        )

        # Fit model
        trainer.fit(model, train_loader, val_loader)

        # Use PyTorch Lightning metrics for kfold reporting
        train_acc = float(trainer.callback_metrics.get('train/accuracy_epoch', 0.0))
        val_acc = float(trainer.callback_metrics.get('val/accuracy_epoch', 0.0))
        correct_train_list.append(train_acc)
        correct_val_list.append(val_acc)
        print('correct_train_list:', correct_train_list)
        print('correct_val_list:', correct_val_list)

        # Test model
        # test_results = trainer.test(model, test_loader)
        # print(f"Test results: {test_results}")


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
        print(f"Train acc: mean={train_mean:.4f}, std={train_std:.4f}, 95% CI=±{train_ci:.4f}")
    if len(correct_val_list) > 0:
        val_mean, val_std, val_ci = ci95(correct_val_list)
        print(f"Val acc: mean={val_mean:.4f}, std={val_std:.4f}, 95% CI=±{val_ci:.4f}")

    return correct_train_list, correct_val_list

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
tasks = [  
# 'DespicableMe',
#   'DiaryOfAWimpyKid',
#   'FunwithFractals',
#   'ThePresent',
#   'RestingState',
  'contrastChangeDetection',
#   'seqLearning6target',
#   'seqLearning8target',
#   'surroundSupp',
#   'symbolSearch'
]
factors = ['attention']
folds = 1

import random
from torch.utils.data import Subset
import numpy as np
for release in releases[:-1]:
    releases_train = [release]
    for factor in factors:
        # train set on all releases
        # if os.path.exists(json_file):
        #     print(f"Skipping {task}_{factor} because it already exists")
        #     continue
        batch_sizes = [256] #[64, 128]#, 256, 512, 1024, 2048]
        lrates = [0.0005]#[0.002, 0.0002, 0.00006, 0.00002]
        dropouts = [0.5] #[0.6, 0.7, 0.8]
        seeds = [1390] #[15, 20, 31, 64, 77, 88, 99, 0, 42, 9]
        data_split_seed = 1925
        epochs = 10
        # seeds = [0]
        models = ['FBCNet'] #['EEGConformer']#, 'TSception']#, 'EEGNeX']
        for model_name in models:
            combinations = [(batch_size, lrate, random_add, dropout) for batch_size in batch_sizes for lrate in lrates for random_add in seeds for dropout in dropouts]
            # combinations = random.sample(combinations, min(20, len(combinations)))
            random.shuffle(combinations)
            # combinations = [(64, 0.002, seed, 0.6) for seed in seeds]
            print(f"Total combinations to run: {len(combinations)}")
            for batch_size, lrate, random_add, dropout in combinations:
                # if check_experiment_exists(task, factor, model_name, folds, batch_size, lrate, random_add, dropout):
                #     print(f"Skipping existing experiment for {task}, {factor}, {model_name}, {folds}, {batch_size}, {lrate}, {random_add}, {dropout}")
                #     continue
                releases_str = f"R{releases_train[0].replace('R','')}-{releases_train[-1].replace('R','')}" if len(releases_train) > 1 else releases_train[0]
                experiment_name = f"{releases_str}_task_{'_'.join(tasks)}_{factor}_{model_name}_bs{batch_size}_lr{lrate}_seed{random_add}_dropout{dropout}_k{folds}_datasplit{data_split_seed}_epochs{epochs}"
                print(f"Running experiment {experiment_name}")
                weights_file_base = f"checkpoints/{experiment_name}.pth"
                res_train, res_val = run_task(
                    releases_train, tasks, factor, folds=folds, random_add=random_add,
                    train_epochs=epochs, batch_size=batch_size, lrate=lrate,
                    model_name=model_name, dropout=dropout, save_weights=weights_file_base,
                    data_split_seed=data_split_seed, test_release="R12", experiment_name=experiment_name
                )
                json_file = f"results/{experiment_name}.json"
                with open(json_file, "w") as f:
                    json.dump({"train": res_train, "val": res_val}, f)

                # test set on R12
                cached_data_folder_names = []
                for task in tasks:
                    cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_R12_" + task + "_" + factor
                    if os.path.exists(cached_data_folder_name):
                        cached_data_folder_names.append(cached_data_folder_name)
                    else:
                        print(f"Missing DataError({cached_data_folder_name}): You first run process_data to run the task for each release")

                test_cached_data_folder_name = "/home/arno/v1/eegdash/notebook/data/hbn_R12_" + tasks[0] + "_" + factor
                test_windows_ds = load_concat_dataset(path=test_cached_data_folder_name, preload=False)
                test_windows_ds = BaseConcatDataset([ds for ds in test_windows_ds.datasets
                                                if is_dataset_within_threshold_class_by_factor(
                                                    ds, 
                                                    factor=factor, 
                                                    positive_threshold=0.5, 
                                                    negative_threshold=-0.5
                                                )
                ])
                # Relabel windows based on factor
                test_relabled_windows_ds = BaseConcatDataset([RelabelWindowsDatasetWrapper(ds, factor=factor, positive_threshold=0.5, negative_threshold=-0.5) for ds in test_windows_ds.datasets])
                test_loader = DataLoader(test_relabled_windows_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)

                test_labels = [label for _, label, _ in test_relabled_windows_ds]
                test_balanced_ds = balance_windows_by_class(test_relabled_windows_ds, test_labels, random_seed=data_split_seed)
                analyze_fold_distribution(test_balanced_ds, None)
                test_balanced_loader = DataLoader(test_balanced_ds, batch_size=batch_size, prefetch_factor=4, num_workers=4)

                res_test_r12 = []
                res_test_balanced_r12 = []
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
                    })
                    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
                    model.eval()
                    trainer_test = L.Trainer(accelerator="auto", devices=1 if torch.cuda.is_available() else None, logger=False, enable_checkpointing=False)

                    test_result = trainer_test.test(model, test_loader)
                    res_test_r12.append(float(trainer_test.callback_metrics.get('test/accuracy_epoch', 0.0)))
                    test_balanced_result = trainer_test.test(model, test_balanced_loader)
                    res_test_balanced_r12.append(float(trainer_test.callback_metrics.get('test/accuracy_epoch', 0.0)))
                print(f"Test on R12 acc: {res_test_r12}, balanced acc: {res_test_balanced_r12}")
                
                with open(json_file, "w") as f:
                    json.dump({"train": res_train, "val": res_val, "test": res_test_r12, "test_balanced": res_test_balanced_r12}, f)
