from __future__ import annotations
import os
import json
import shutil
import warnings
from typing import Dict, no_type_check
from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from braindecode.datasets.base import (
    EEGWindowsDataset,
    BaseConcatDataset,
    _create_description,
)


class FeaturesDataset(EEGWindowsDataset):
    """Returns samples from a pandas DataFrame object along with a target.

    Dataset which serves samples from a pandas DataFrame object along with a
    target. The target is unique for the dataset, and is obtained through the
    `description` attribute.

    Parameters
    ----------
    features : a pandas DataFrame
        Tabular data.
    description : dict | pandas.Series | None
        Holds additional description about the continuous signal / subject.
    target_name : str | tuple | None
        Name(s) of the column that should be used to provide the target (e.g.,
        to be used in a prediction task later on).
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        description: dict | pd.Series | None = None,
        target_name: str | tuple[str, ...] | None = None,
        transform: Callable | None = None,
        raw_info: Dict | None = None,
        raw_preproc_kwargs: Dict | None = None,
        window_kwargs: Dict | None = None,
        window_preproc_kwargs: Dict | None = None,
        features_kwargs: Dict | None = None,
    ):
        self.features = features
        self.n_features = features.columns.size
        self.metadata = metadata
        self._description = _create_description(description)
        self.transform = transform
        self.raw_info = raw_info
        self.raw_preproc_kwargs = raw_preproc_kwargs
        self.window_kwargs = window_kwargs
        self.window_preproc_kwargs = window_preproc_kwargs
        self.features_kwargs = features_kwargs

        # save target name for load/save later
        if target_name is not None:
            # hush "target_name not in description" warning
            self.set_description({target_name: target_name}, overwrite=True)
            self.n_features -= 1
        self.target_name = self._target_name(target_name)

    def __getitem__(self, index):
        X = self.features.iloc[index]
        y = None
        if self.target_name is not None:
            # y = self.description[self.target_name]
            y = X[self.target_name]
            X = X.drop(self.target_name, inplace=False)
        else:
            X = X.copy()
        if isinstance(y, pd.Series):
            y = y.copy().to_list()
        elif not isinstance(y, Iterable):
            y = [y]
        if isinstance(X, pd.Series | pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.features.index)


def _compute_stats(
    ds: FeaturesDataset,
    return_count=False,
    return_mean=False,
    return_var=False,
    ddof=1,
    numeric_only=False,
):
    df = ds.features.drop(columns=ds.target_name)
    res = []
    if return_count:
        res.append(df.count(numeric_only=numeric_only))
    if return_mean:
        res.append(df.mean(numeric_only=numeric_only))
    if return_var:
        res.append(df.var(ddof=ddof, numeric_only=numeric_only))
    return tuple(res)


def _pooled_var(counts, means, variances, ddof):
    count = counts.sum(axis=0)
    mean = np.sum((counts / count) * means, axis=0)
    var = np.sum(((counts - ddof) / (count - ddof)) * variances, axis=0)
    var += np.sum((counts / (count - ddof)) * (means**2), axis=0)
    var -= (count / (count - ddof)) * (mean**2)
    var = var.clip(min=0)
    return count, mean, var


class FeaturesConcatDataset(BaseConcatDataset):
    """A base class for concatenated datasets.

    Holds either mne.Raw or mne.Epoch in self.datasets and has
    a pandas DataFrame with additional description.

    Parameters
    ----------
    list_of_ds : list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    target_transform : callable | None
        Optional function to call on targets before returning them.

    """

    def __init__(
        self,
        list_of_ds: list[FeaturesDataset] | None = None,
        target_transform: Callable | None = None,
    ):
        # if we get a list of FeaturesConcatDataset, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], FeaturesConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)

        self.target_transform = target_transform

    def split(
        self,
        by: str | list[int] | list[list[int]] | dict[str, list[int]],
    ) -> dict[str, FeaturesConcatDataset]:
        """Split the dataset based on information listed in its description.

        The format could be based on a DataFrame or based on indices.

        Parameters
        ----------
        by : str | list | dict
            If ``by`` is a string, splitting is performed based on the
            description DataFrame column with this name.
            If ``by`` is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
            If a dict then each key will be used in the returned
            splits dict and each value should be a list of int.

        Returns
        -------
        splits : dict
            A dictionary with the name of the split (a string) as key and the
            dataset as value.
        """
        if isinstance(by, str):
            split_ids = {
                k: list(v) for k, v in self.description.groupby(by).groups.items()
            }
        elif isinstance(by, dict):
            split_ids = by
        else:
            # assume list(int)
            if not isinstance(by[0], list):
                by = [by]
            # assume list(list(int))
            split_ids = {split_i: split for split_i, split in enumerate(by)}

        return {
            str(split_name): FeaturesConcatDataset(
                [self.datasets[ds_ind] for ds_ind in ds_inds],
                target_transform=self.target_transform,
            )
            for split_name, ds_inds in split_ids.items()
        }

    def get_metadata(self) -> pd.DataFrame:
        """Concatenate the metadata and description of the wrapped Epochs.

        Returns
        -------
        metadata : pd.DataFrame
            DataFrame containing as many rows as there are windows in the
            BaseConcatDataset, with the metadata and description information
            for each window.
        """
        if not all([isinstance(ds, FeaturesDataset) for ds in self.datasets]):
            raise TypeError(
                "Metadata dataframe can only be computed when all "
                "datasets are FeaturesDataset."
            )

        all_dfs = list()
        for ds in self.datasets:
            df = ds.metadata
            for k, v in ds.description.items():
                df[k] = v
            all_dfs.append(df)

        return pd.concat(all_dfs)

    def save(self, path: str, overwrite: bool = False, offset: int = 0):
        """Save datasets to files by creating one subdirectory for each dataset:
        path/
            0/
                0-feat.parquet
                description.json
                raw-info.fif (if raw info was saved)
                raw_preproc_kwargs.json (if raws were preprocessed)
                window_kwargs.json (if this is a windowed dataset)
                window_preproc_kwargs.json  (if windows were preprocessed)
                features_kwargs.json
                target_name.json (if target_name is not None)
            1/
                1-feat.parquet
                description.json
                raw-info.fif (if raw info was saved)
                raw_preproc_kwargs.json (if raws were preprocessed)
                window_kwargs.json (if this is a windowed dataset)
                window_preproc_kwargs.json  (if windows were preprocessed)
                features_kwargs.json
                target_name.json (if target_name is not None)

        Parameters
        ----------
        path : str
            Directory in which subdirectories are created to store
             -feat.parquet and .json files to.
        overwrite : bool
            Whether to delete old subdirectories that will be saved to in this
            call.
        offset : int
            If provided, the integer is added to the id of the dataset in the
            concat. This is useful in the setting of very large datasets, where
            one dataset has to be processed and saved at a time to account for
            its original position.
        """
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        path_contents = os.listdir(path)
        n_sub_dirs = len([os.path.isdir(e) for e in path_contents])
        for i_ds, ds in enumerate(self.datasets):
            # remove subdirectory from list of untouched files / subdirectories
            if str(i_ds + offset) in path_contents:
                path_contents.remove(str(i_ds + offset))
            # save_dir/i_ds/
            sub_dir = os.path.join(path, str(i_ds + offset))
            if os.path.exists(sub_dir):
                if overwrite:
                    shutil.rmtree(sub_dir)
                else:
                    raise FileExistsError(
                        f"Subdirectory {sub_dir} already exists. Please select"
                        f" a different directory, set overwrite=True, or "
                        f"resolve manually."
                    )
            # save_dir/{i_ds+offset}/
            os.makedirs(sub_dir)
            # save_dir/{i_ds+offset}/{i_ds+offset}-feat.parquet
            self._save_features(sub_dir, ds, i_ds, offset)
            # save_dir/{i_ds+offset}/metadata_df.pkl
            self._save_metadata(sub_dir, ds)
            # save_dir/{i_ds+offset}/description.json
            self._save_description(sub_dir, ds.description)
            # save_dir/{i_ds+offset}/raw-info.fif
            self._save_raw_info(sub_dir, ds)
            # save_dir/{i_ds+offset}/raw_preproc_kwargs.json
            # save_dir/{i_ds+offset}/window_kwargs.json
            # save_dir/{i_ds+offset}/window_preproc_kwargs.json
            # save_dir/{i_ds+offset}/features_kwargs.json
            self._save_kwargs(sub_dir, ds)
            # save_dir/{i_ds+offset}/target_name.json
            self._save_target_name(sub_dir, ds)
        if overwrite:
            # the following will be True for all datasets preprocessed and
            # stored in parallel with braindecode.preprocessing.preprocess
            if i_ds + 1 + offset < n_sub_dirs:
                warnings.warn(
                    f"The number of saved datasets ({i_ds + 1 + offset}) "
                    f"does not match the number of existing "
                    f"subdirectories ({n_sub_dirs}). You may now "
                    f"encounter a mix of differently preprocessed "
                    f"datasets!",
                    UserWarning,
                )
        # if path contains files or directories that were not touched, raise
        # warning
        if path_contents:
            warnings.warn(
                f"Chosen directory {path} contains other "
                f"subdirectories or files {path_contents}."
            )

    @staticmethod
    def _save_features(sub_dir, ds, i_ds, offset):
        parquet_file_name = f"{i_ds + offset}-feat.parquet"
        parquet_file_path = os.path.join(sub_dir, parquet_file_name)
        ds.features.to_parquet(parquet_file_path)

    @staticmethod
    def _save_raw_info(sub_dir, ds):
        if hasattr(ds, "raw_info"):
            fif_file_name = "raw-info.fif"
            fif_file_path = os.path.join(sub_dir, fif_file_name)
            ds.raw_info.save(fif_file_path)

    @staticmethod
    def _save_kwargs(sub_dir, ds):
        for kwargs_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
            "features_kwargs",
        ]:
            if hasattr(ds, kwargs_name):
                kwargs_file_name = ".".join([kwargs_name, "json"])
                kwargs_file_path = os.path.join(sub_dir, kwargs_file_name)
                kwargs = getattr(ds, kwargs_name)
                if kwargs is not None:
                    with open(kwargs_file_path, "w") as f:
                        json.dump(kwargs, f)

    def to_dataframe(self, include_metadata=False):
        if include_metadata:
            dataframes = [
                ds.metadata.join(ds.features, lsuffix="_metadata")
                for ds in self.datasets
            ]
        else:
            dataframes = [ds.features for ds in self.datasets]
        return pd.concat(dataframes, axis=0, ignore_index=True)

    def _numeric_columns(self):
        return self.datasets[0].features.select_dtypes(include=np.number).columns

    def count(self, numeric_only=False, n_jobs=1):
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(ds, return_count=True, numeric_only=numeric_only)
            for ds in self.datasets
        )
        counts = np.array([s[0] for s in stats])
        count = counts.sum(axis=0)
        return pd.Series(count, index=self._numeric_columns())

    def mean(self, numeric_only=False, n_jobs=1):
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(
                ds, return_count=True, return_mean=True, numeric_only=numeric_only
            )
            for ds in self.datasets
        )
        counts, means = np.array([s[0] for s in stats]), np.array([s[1] for s in stats])
        count = counts.sum(axis=0, keepdims=True)
        mean = np.sum((counts / count) * means, axis=0)
        return pd.Series(mean, index=self._numeric_columns())

    def var(self, ddof=1, numeric_only=False, n_jobs=1):
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(
                ds,
                return_count=True,
                return_mean=True,
                return_var=True,
                ddof=ddof,
                numeric_only=numeric_only,
            )
            for ds in self.datasets
        )
        counts, means, variances = (
            np.array([s[0] for s in stats]),
            np.array([s[1] for s in stats]),
            np.array([s[2] for s in stats]),
        )
        _, _, var = _pooled_var(counts, means, variances, ddof)
        return pd.Series(var, index=self._numeric_columns())

    def std(self, ddof=1, numeric_only=False, n_jobs=1):
        return np.sqrt(self.var(ddof=ddof, numeric_only=numeric_only, n_jobs=n_jobs))

    def zscore(self, ddof=1, numeric_only=False, eps=0, n_jobs=1):
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(
                ds,
                return_count=True,
                return_mean=True,
                return_var=True,
                ddof=ddof,
                numeric_only=numeric_only,
            )
            for ds in self.datasets
        )
        counts, means, variances = (
            np.array([s[0] for s in stats]),
            np.array([s[1] for s in stats]),
            np.array([s[2] for s in stats]),
        )
        _, mean, var = _pooled_var(counts, means, variances, ddof)
        std = np.sqrt(var) + eps
        for ds in self.datasets:
            cols_without_target = ds.features.columns[
                ds.features.columns != ds.target_name
            ]
            ds.features[cols_without_target] = (
                ds.features[cols_without_target] - mean
            ) / std

    @staticmethod
    def _enforce_inplace_operations(func_name, kwargs):
        if "inplace" in kwargs and kwargs["inplace"] is False:
            raise ValueError(
                f"{func_name} only works inplace, please change "
                + "to inplace=True (default)."
            )
        kwargs["inplace"] = True

    def fillna(self, *args, **kwargs):
        FeaturesConcatDataset._enforce_inplace_operations("fillna", kwargs)
        for ds in self.datasets:
            ds.features.fillna(*args, **kwargs)

    def interpolate(self, *args, **kwargs):
        FeaturesConcatDataset._enforce_inplace_operations("interpolate", kwargs)
        for ds in self.datasets:
            ds.features.interpolate(*args, **kwargs)

    def dropna(self, *args, **kwargs):
        FeaturesConcatDataset._enforce_inplace_operations("dropna", kwargs)
        for ds in self.datasets:
            ds.features.dropna(*args, **kwargs)

    def drop(self, *args, **kwargs):
        FeaturesConcatDataset._enforce_inplace_operations("drop", kwargs)
        for ds in self.datasets:
            ds.features.drop(*args, **kwargs)

    def join(self, concat_dataset: FeaturesConcatDataset, **kwargs):
        assert len(self.datasets) == len(concat_dataset.datasets)
        for ds1, ds2 in zip(self.datasets, concat_dataset.datasets):
            assert len(ds1) == len(ds2)
            ds1.features.join(ds2, **kwargs)

    def transform(self, *args, **kwargs):
        for ds in self.datasets:
            ds.features.transform(*args, **kwargs)
