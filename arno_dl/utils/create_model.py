# %% [markdown]
# ## Data Retrieval Using EEGDash
#
# First we find one resting state dataset for a collection of subject. The dataset ds005505 contains 136 subjects with both male and female participants.
# %%
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.models import EEGConformer, EEGNeX
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from eegdash import EEGDashDataset
from eegdash.api import EEGDashDataset
from eegdash.dataset.dataset import EEGChallengeDataset


def create_model(config):
    class EEGModel(L.LightningModule):
        def __init__(self, config):
            super(EEGModel, self).__init__()
            drop_prob = config.get("dropout", 0.7)
            n_chans = config.get("n_chans", 24)
            n_times = config.get("n_times", 256)
            sfreq = config.get("sfreq", 128)
            self.n_outputs = config.get("n_outputs", 1)
            if config["model_name"] == "EEGNeX":
                self.model = EEGNeX(
                    n_chans=n_chans,
                    n_outputs=self.n_outputs,
                    n_times=n_times,
                    sfreq=sfreq,
                    drop_prob=drop_prob,
                )
            elif config["model_name"] == "EEGConformer":
                self.model = EEGConformer(
                    n_chans=n_chans,
                    n_outputs=self.n_outputs,
                    n_times=n_times,
                    sfreq=sfreq,
                    drop_prob=drop_prob,
                )
            elif config["model_name"] == "EEGConformerSimplified":
                self.model = EEGConformer(
                    n_chans=n_chans,
                    n_outputs=self.n_outputs,
                    n_times=n_times,
                    sfreq=sfreq,
                    drop_prob=drop_prob,
                    n_filters_time=32,  # Try 32 instead of default 40
                    filter_time_length=20,  # Try 20
                    att_depth=4,  # Reduce attention layers
                    att_heads=8,  # Reduce attention heads
                    pool_time_stride=12,  # Adjust pooling
                    pool_time_length=64,  # Adjust pooling window
                )
            if self.n_outputs > 1:
                self.precision = Precision(task="binary")
                self.recall = Recall(task="binary")
                self.f1 = F1Score(task="binary")
                self.accuracy = Accuracy(task="binary")
                self.val_precision = Precision(task="binary")
                self.val_recall = Recall(task="binary")
                self.val_f1 = F1Score(task="binary")
                self.val_accuracy = Accuracy(task="binary")
            else:
                self.mae = MeanAbsoluteError()
                self.mse = MeanSquaredError()
                self.r2 = R2Score()
                self.val_mae = MeanAbsoluteError()
                self.val_mse = MeanSquaredError()
                self.val_r2 = R2Score()
                self.target_std = config.get("target_std", None)
                self.train_baseline_mae = float(config.get("train_baseline_mae", 1.0))
                self.loss = config.get("loss", "l1_loss")

            self.lr = config["lr"]
            self.weight_decay = config.get("weight_decay", False)
            self.model_freeze = bool(config.get("model_freeze", False))
            if self.model_freeze:
                self.automatic_optimization = False
            self.save_hyperparameters(config)

        def normalize_data(self, x):
            x = x.reshape(x.shape[0], 24, 256)
            mean = x.mean(dim=2, keepdim=True)
            std = (
                x.std(dim=2, keepdim=True) + 1e-7
            )  # add small epsilon for numerical stability
            x = (x - mean) / std
            return x

        def _forward(self, batch):
            x, y, subjects = batch
            # Debug: print information about target values
            if hasattr(self, "_debug_printed") == False:
                print(f"Target values type: {type(y[0]) if len(y) > 0 else 'empty'}")
                print(f"Target values sample: {y[:5] if len(y) >= 5 else y}")
                print(f"Target values unique types: {set(type(val) for val in y)}")
                print(
                    f"Target values unique count in batch: {len(set([float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v) for v in y]))}"
                )
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

            # CHECK IF WE CAN REMOVE THAT
            # if y.shape != preds.shape and y.numel() == preds.numel():
            #     y = y.reshape(preds.shape)

            if self.n_outputs > 1:
                _, preds = scores.max(1)
                loss = F.cross_entropy(scores, y)
            else:
                preds = scores.squeeze().to(dtype=torch.float32)
                if self.loss == "l1_loss":
                    loss = F.l1_loss(preds, y)
                else:
                    loss = F.mse_loss(preds, y)
            return loss, preds, y

        def training_step(self, batch, batch_idx):
            if self.model_freeze:
                with torch.no_grad():
                    loss, preds, y = self._forward(batch)
            else:
                loss, preds, y = self._forward(batch)

            if self.n_outputs > 1:
                self.precision.update(preds, y)
                self.recall.update(preds, y)
                # self.f1.update(preds, y)
                self.accuracy.update(preds, y)
            else:
                self.mae.update(preds, y)
                self.mse.update(preds, y)
                self.r2.update(preds, y)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def on_train_epoch_end(self):
            if self.n_outputs > 1:
                self.log(
                    "train/precision_epoch",
                    self.precision.compute(),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "train/recall_epoch",
                    self.recall.compute(),
                    on_step=False,
                    on_epoch=True,
                )
                # self.log('train/f1_epoch', self.f1.compute(), on_step=False, on_epoch=True)
                self.log(
                    "train/accuracy_epoch",
                    self.accuracy.compute(),
                    on_step=False,
                    on_epoch=True,
                )
            else:
                # 1) read scalar values BEFORE any logging of metric objects
                train_mse_value = self.mse.compute()
                train_mae_value = self.mae.compute()
                train_r2_value = self.r2.compute()
                train_normalized_rmse = torch.sqrt(F.mse_loss(preds, y)) / (
                    self.target_std + 1e-8
                )

                # 2) log scalars (not metric objects) to avoid implicit resets mid-function
                self.log(
                    "train/mse_epoch", train_mse_value, on_step=False, on_epoch=True
                )
                self.log(
                    "train/mae_epoch", train_mae_value, on_step=False, on_epoch=True
                )
                self.log(
                    "train/r2_epoch",
                    train_r2_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.log(
                    "train/normalized_rmse_epoch",
                    train_normalized_rmse,
                    on_step=False,
                    on_epoch=True,
                )
                s_train = 1.0 - float(train_mae_value) / (
                    self.train_baseline_mae + 1e-8
                )
                self.log("train/S_epoch", s_train, on_step=False, on_epoch=True)

                # 3) now reset explicitly
                self.mse.reset()
                self.mae.reset()
                self.r2.reset()

        def validation_step(self, batch, batch_idx):
            loss, preds, y = self._forward(batch)
            if self.n_outputs > 1:
                self.val_precision.update(preds, y)
                self.val_recall.update(preds, y)
                # self.val_f1.update(preds, y)
                self.val_accuracy.update(preds, y)
            else:
                self.val_mae.update(preds, y)
                self.val_mse.update(preds, y)
                self.val_r2.update(preds, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True)

        def on_train_start(self):
            # Initialize hp_metric to 0.5 to avoid -1 in TensorBoard
            if self.n_outputs > 1:
                self.log("hp_metric", 1 / self.n_outputs, on_epoch=True)
            else:
                self.log("hp_metric", 0.5, on_epoch=True)

        def on_validation_epoch_end(self):
            if self.n_outputs > 1:
                self.log(
                    "val/recall_epoch", self.val_recall, on_step=False, on_epoch=True
                )  # AI says it should be self.val_recall
                self.log(
                    "val/precision_epoch",
                    self.val_precision,
                    on_step=False,
                    on_epoch=True,
                )
                self.log("val/f1_epoch", self.val_f1, on_step=False, on_epoch=True)
                val_acc = self.val_accuracy.compute()
                self.log(
                    "val/accuracy_epoch",
                    val_acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.log("hp_metric", val_acc, on_epoch=True)

                self.val_acc.reset()
            else:
                val_mse_value = self.val_mse.compute()
                val_mae_value = self.val_mae.compute()
                val_r2_value = self.val_r2.compute()
                denom = (self.target_std if self.target_std is not None else 1.0) + 1e-8
                val_normalized_rmse = torch.sqrt(val_mse_value) / denom

                self.log(
                    "val/r2_epoch",
                    val_r2_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.log(
                    "val/norm_rmse_epoch",
                    val_normalized_rmse,
                    on_step=False,
                    on_epoch=True,
                )
                self.log("hp_metric", val_r2_value, on_epoch=True)
                s_val = 1.0 - float(val_mae_value) / (self.train_baseline_mae + 1e-8)
                self.log("val/norm_mae_epoch", s_val, on_step=False, on_epoch=True)

                self.val_mse.reset()
                self.val_mae.reset()
                self.val_r2.reset()

        def configure_optimizers(self):
            if self.model_freeze:
                return None

            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
            )

            if self.n_outputs > 1:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/accuracy_epoch",
                        "frequency": 1,
                    },
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/norm_mae_epoch",
                        "frequency": 1,
                    },
                }

    return EEGModel(config)


if __name__ == "__main__":
    config = {
        "model_name": "EEGNeX",
        "lr": 0.0001,
        "weight_decay": 0.0001,
    }
    model = create_model(config)
    print(model)

    # create model and hparams dict
    hparams = {
        "batch_size": 128,
        "dropout": 0.5,
        "lr": 0.0002,
        "weight_decay": 0.01,
        "random_seed": 13,
        "model_name": "EEGConformerSimplified",
        "fold": 1,
        "target_std": 1,
        "model_freeze": bool(True),
        "train_baseline_mae": 1,
    }
    model = create_model(hparams)

    # test forward pass
    x = torch.randn(1, 24, 256)
    y = torch.randn(1)
    loss, preds, y = model._forward((x, y, None))
    print(loss, preds, y)
