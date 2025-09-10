"""
This module provides classes for handling feature datasets, extending the
functionality of braindecode's dataset classes to work with tabular feature data.
"""
from __future__ import annotations

import json
import os
import shutil
import warnings
from collections.abc import Callable
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from braindecode.datasets.base import (
    BaseConcatDataset,
    EEGWindowsDataset,
    _create_description,
)


class FeaturesDataset(EEGWindowsDataset):
    """A dataset class for handling tabular feature data.

    This dataset serves samples from a pandas DataFrame of features, along with
    a target variable obtained from the dataset's description.

    Parameters
    ----------
    features : pd.DataFrame
        A DataFrame containing the feature data.
    metadata : pd.DataFrame | None
        A DataFrame containing metadata for the features.
    description : dict | pd.Series | None
        Additional information about the continuous signal or subject.
    transform : callable | None
        A function to apply to the features on-the-fly.
    raw_info : dict | None
        Information about the raw data.
    raw_preproc_kwargs : dict | None
        Keyword arguments for raw data preprocessing.
    window_kwargs : dict | None
        Keyword arguments for windowing.
    window_preproc_kwargs : dict | None
        Keyword arguments for window preprocessing.
    features_kwargs : dict | None
        Keyword arguments for feature extraction.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        description: dict | pd.Series | None = None,
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

        self.crop_inds = metadata.loc[
            :, ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"]
        ].to_numpy()
        self.y = metadata.loc[:, "target"].to_list()

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the features, target, and crop indices.
        """
        crop_inds = self.crop_inds[index].tolist()
        X = self.features.iloc[index].to_numpy()
        X = X.copy()
        X.astype("float32")
        if self.transform is not None:
            X = self.transform(X)
        y = self.y[index]
        return X, y, crop_inds

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.features.index)


def _compute_stats(
    ds: FeaturesDataset,
    return_count=False,
    return_mean=False,
    return_var=False,
    ddof=1,
    numeric_only=False,
):
    """Compute statistics for a FeaturesDataset.

    Parameters
    ----------
    ds : FeaturesDataset
        The dataset to compute statistics for.
    return_count : bool
        Whether to return the count of non-NA cells.
    return_mean : bool
        Whether to return the mean of the values.
    return_var : bool
        Whether to return the variance of the values.
    ddof : int
        Delta Degrees of Freedom for variance calculation.
    numeric_only : bool
        Whether to include only numeric columns.

    Returns
    -------
    tuple
        A tuple of computed statistics.
    """
    res = []
    if return_count:
        res.append(ds.features.count(numeric_only=numeric_only))
    if return_mean:
        res.append(ds.features.mean(numeric_only=numeric_only))
    if return_var:
        res.append(ds.features.var(ddof=ddof, numeric_only=numeric_only))
    return tuple(res)


def _pooled_var(counts, means, variances, ddof, ddof_in=None):
    """Compute the pooled variance.

    Parameters
    ----------
    counts : array-like
        The counts of each group.
    means : array-like
        The means of each group.
    variances : array-like
        The variances of each group.
    ddof : int
        Delta Degrees of Freedom for the pooled variance.
    ddof_in : int, optional
        Delta Degrees of Freedom for the input variances.

    Returns
    -------
    tuple
        A tuple of (count, mean, variance).
    """
    if ddof_in is None:
        ddof_in = ddof
    count = counts.sum(axis=0)
    mean = np.sum((counts / count) * means, axis=0)
    var = np.sum(((counts - ddof_in) / (count - ddof)) * variances, axis=0)
    var[:] += np.sum((counts / (count - ddof)) * (means**2), axis=0)
    var[:] -= (count / (count - ddof)) * (mean**2)
    var[:] = var.clip(min=0)
    return count, mean, var


class FeaturesConcatDataset(BaseConcatDataset):
    """A concatenated dataset of FeaturesDataset objects.

    This class holds a list of `FeaturesDataset` objects and provides methods
    for splitting, saving, and manipulating the concatenated data.

    Parameters
    ----------
    list_of_ds : list[FeaturesDataset]
        A list of `FeaturesDataset` objects to concatenate.
    target_transform : callable | None
        An optional function to apply to targets before returning them.
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
        """Save datasets to files by creating one subdirectory for each dataset.

        Parameters
        ----------
        path : str
            Directory in which to save the datasets.
        overwrite : bool
            Whether to overwrite existing subdirectories.
        offset : int
            An offset to add to the subdirectory names.

        Raises
        ------
        ValueError
            If the dataset is empty.
        FileExistsError
            If a subdirectory already exists and `overwrite` is False.
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
        """Save the features of a dataset to a parquet file.

        Parameters
        ----------
        sub_dir : str
            The directory to save the file in.
        ds : FeaturesDataset
            The dataset to save.
        i_ds : int
            The index of the dataset.
        offset : int
            The offset for the filename.
        """
        parquet_file_name = f"{i_ds + offset}-feat.parquet"
        parquet_file_path = os.path.join(sub_dir, parquet_file_name)
        ds.features.to_parquet(parquet_file_path)

    @staticmethod
    def _save_raw_info(sub_dir, ds):
        """Save the raw info of a dataset to a fif file.

        Parameters
        ----------
        sub_dir : str
            The directory to save the file in.
        ds : FeaturesDataset
            The dataset to save.
        """
        if hasattr(ds, "raw_info"):
            fif_file_name = "raw-info.fif"
            fif_file_path = os.path.join(sub_dir, fif_file_name)
            ds.raw_info.save(fif_file_path)

    @staticmethod
    def _save_kwargs(sub_dir, ds):
        """Save the kwargs of a dataset to json files.

        Parameters
        ----------
        sub_dir : str
            The directory to save the files in.
        ds : FeaturesDataset
            The dataset to save.
        """
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

    def to_dataframe(
        self,
        include_metadata: bool | str | List[str] = False,
        include_target: bool = False,
        include_crop_inds: bool = False,
    ) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame.

        Parameters
        ----------
        include_metadata : bool | str | list[str]
            Whether to include metadata in the DataFrame.
        include_target : bool
            Whether to include the target in the DataFrame.
        include_crop_inds : bool
            Whether to include crop indices in the DataFrame.

        Returns
        -------
        pd.DataFrame
            The dataset as a DataFrame.
        """
        if (
            not isinstance(include_metadata, bool)
            or include_metadata
            or include_crop_inds
        ):
            include_dataset = False
            if isinstance(include_metadata, bool) and include_metadata:
                include_dataset = True
                cols = self.datasets[0].metadata.columns
            else:
                cols = include_metadata
                if isinstance(cols, bool) and not cols:
                    cols = []
                elif isinstance(cols, str):
                    cols = [cols]
                cols = set(cols)
                if include_crop_inds:
                    cols = {
                        "i_dataset",
                        "i_window_in_trial",
                        "i_start_in_trial",
                        "i_stop_in_trial",
                        *cols,
                    }
                if include_target:
                    cols.add("target")
                cols = list(cols)
                include_dataset = "i_dataset" in cols
                if include_dataset:
                    cols.remove("i_dataset")
            dataframes = [
                ds.metadata[cols].join(ds.features, how="right", lsuffix="_metadata")
                for ds in self.datasets
            ]
            if include_dataset:
                for i, df in enumerate(dataframes):
                    df.insert(loc=0, column="i_dataset", value=i)
        elif include_target:
            dataframes = [
                ds.features.join(ds.metadata["target"], how="left", rsuffix="_metadata")
                for ds in self.datasets
            ]
        else:
            dataframes = [ds.features for ds in self.datasets]
        return pd.concat(dataframes, axis=0, ignore_index=True)

    def _numeric_columns(self) -> pd.Index:
        """Get the numeric columns of the features.

        Returns
        -------
        pd.Index
            The numeric columns.
        """
        return self.datasets[0].features.select_dtypes(include=np.number).columns

    def count(self, numeric_only=False, n_jobs=1) -> pd.Series:
        """Compute the count of non-NA cells for each column.

        Parameters
        ----------
        numeric_only : bool
            Whether to include only numeric columns.
        n_jobs : int
            The number of parallel jobs to run.

        Returns
        -------
        pd.Series
            The counts for each column.
        """
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(ds, return_count=True, numeric_only=numeric_only)
            for ds in self.datasets
        )
        counts = np.array([s[0] for s in stats])
        count = counts.sum(axis=0)
        return pd.Series(count, index=self._numeric_columns())

    def mean(self, numeric_only=False, n_jobs=1) -> pd.Series:
        """Compute the mean of the values for each column.

        Parameters
        ----------
        numeric_only : bool
            Whether to include only numeric columns.
        n_jobs : int
            The number of parallel jobs to run.

        Returns
        -------
        pd.Series
            The means for each column.
        """
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

    def var(self, ddof=1, numeric_only=False, n_jobs=1) -> pd.Series:
        """Compute the variance of the values for each column.

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom.
        numeric_only : bool
            Whether to include only numeric columns.
        n_jobs : int
            The number of parallel jobs to run.

        Returns
        -------
        pd.Series
            The variances for each column.
        """
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(
                ds,
                return_count=True,
                return_mean=True,
                return_var=True,
                ddof=0,
                numeric_only=numeric_only,
            )
            for ds in self.datasets
        )
        counts, means, variances = (
            np.array([s[0] for s in stats]),
            np.array([s[1] for s in stats]),
            np.array([s[2] for s in stats]),
        )
        _, _, var = _pooled_var(counts, means, variances, ddof, ddof_in=0)
        return pd.Series(var, index=self._numeric_columns())

    def std(self, ddof=1, numeric_only=False, eps=0, n_jobs=1) -> pd.Series:
        """Compute the standard deviation of the values for each column.

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom.
        numeric_only : bool
            Whether to include only numeric columns.
        eps : float
            A small value to add to the variance to avoid taking the square root of zero.
        n_jobs : int
            The number of parallel jobs to run.

        Returns
        -------
        pd.Series
            The standard deviations for each column.
        """
        return np.sqrt(
            self.var(ddof=ddof, numeric_only=numeric_only, n_jobs=n_jobs) + eps
        )

    def zscore(self, ddof=1, numeric_only=False, eps=0, n_jobs=1):
        """Compute the z-score of the values for each column.

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom.
        numeric_only : bool
            Whether to include only numeric columns.
        eps : float
            A small value to add to the variance to avoid taking the square root of zero.
        n_jobs : int
            The number of parallel jobs to run.
        """
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(
                ds,
                return_count=True,
                return_mean=True,
                return_var=True,
                ddof=0,
                numeric_only=numeric_only,
            )
            for ds in self.datasets
        )
        counts, means, variances = (
            np.array([s[0] for s in stats]),
            np.array([s[1] for s in stats]),
            np.array([s[2] for s in stats]),
        )
        _, mean, var = _pooled_var(counts, means, variances, ddof, ddof_in=0)
        std = np.sqrt(var + eps)
        for ds in self.datasets:
            ds.features = (ds.features - mean) / std

    @staticmethod
    def _enforce_inplace_operations(func_name, kwargs):
        """Enforce that a method is called with `inplace=True`.

        Parameters
        ----------
        func_name : str
            The name of the method.
        kwargs : dict
            The keyword arguments passed to the method.

        Raises
        ------
        ValueError
            If `inplace` is False.
        """
        if "inplace" in kwargs and kwargs["inplace"] is False:
            raise ValueError(
                f"{func_name} only works inplace, please change "
                + "to inplace=True (default)."
            )
        kwargs["inplace"] = True

    def fillna(self, *args, **kwargs):
        """Fill NA/NaN values using the specified method.

        Parameters
        ----------
        *args
            Positional arguments passed to `pandas.DataFrame.fillna`.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.fillna`.
        """
        FeaturesConcatDataset._enforce_inplace_operations("fillna", kwargs)
        for ds in self.datasets:
            ds.features.fillna(*args, **kwargs)

    def replace(self, *args, **kwargs):
        """Replace values given in `to_replace` with `value`.

        Parameters
        ----------
        *args
            Positional arguments passed to `pandas.DataFrame.replace`.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.replace`.
        """
        FeaturesConcatDataset._enforce_inplace_operations("replace", kwargs)
        for ds in self.datasets:
            ds.features.replace(*args, **kwargs)

    def interpolate(self, *args, **kwargs):
        """Interpolate values according to different methods.

        Parameters
        ----------
        *args
            Positional arguments passed to `pandas.DataFrame.interpolate`.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.interpolate`.
        """
        FeaturesConcatDataset._enforce_inplace_operations("interpolate", kwargs)
        for ds in self.datasets:
            ds.features.interpolate(*args, **kwargs)

    def dropna(self, *args, **kwargs):
        """Remove missing values.

        Parameters
        ----------
        *args
            Positional arguments passed to `pandas.DataFrame.dropna`.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.dropna`.
        """
        FeaturesConcatDataset._enforce_inplace_operations("dropna", kwargs)
        for ds in self.datasets:
            ds.features.dropna(*args, **kwargs)

    def drop(self, *args, **kwargs):
        """Drop specified labels from rows or columns.

        Parameters
        ----------
        *args
            Positional arguments passed to `pandas.DataFrame.drop`.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.drop`.
        """
        FeaturesConcatDataset._enforce_inplace_operations("drop", kwargs)
        for ds in self.datasets:
            ds.features.drop(*args, **kwargs)

    def join(self, concat_dataset: FeaturesConcatDataset, **kwargs):
        """Join columns of another FeaturesConcatDataset.

        Parameters
        ----------
        concat_dataset : FeaturesConcatDataset
            The dataset to join with.
        **kwargs
            Keyword arguments passed to `pandas.DataFrame.join`.
        """
        assert len(self.datasets) == len(concat_dataset.datasets)
        for ds1, ds2 in zip(self.datasets, concat_dataset.datasets):
            assert len(ds1) == len(ds2)
            ds1.features.join(ds2, **kwargs)
