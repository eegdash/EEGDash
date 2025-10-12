from __future__ import annotations

import json
import os
import shutil
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

from ..logging import logger


__all__ = [
    "FeaturesDataset",
    "FeaturesConcatDataset",
]


class FeaturesDataset(EEGWindowsDataset):
    """A dataset of features extracted from EEG windows.

    This class holds features in a pandas DataFrame and provides an interface
    compatible with braindecode's dataset structure. Each row in the feature
    DataFrame corresponds to a single sample (e.g., an EEG window).

    Parameters
    ----------
    features : pandas.DataFrame
        A DataFrame where each row is a sample and each column is a feature.
    metadata : pandas.DataFrame, optional
        A DataFrame containing metadata for each sample, indexed consistently
        with `features`. Must include columns 'i_window_in_trial',
        'i_start_in_trial', 'i_stop_in_trial', and 'target'.
    description : dict or pandas.Series, optional
        Additional high-level information about the dataset (e.g., subject ID).
    transform : callable, optional
        A function or transform to apply to the feature data on-the-fly.
    raw_info : dict, optional
        Information about the original raw recording, for provenance.
    raw_preproc_kwargs : dict, optional
        Keyword arguments used for preprocessing the raw data.
    window_kwargs : dict, optional
        Keyword arguments used for windowing the data.
    window_preproc_kwargs : dict, optional
        Keyword arguments used for preprocessing the windowed data.
    features_kwargs : dict, optional
        Keyword arguments used for feature extraction.

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

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, list]:
        """Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the feature vector (X), the target (y), and the
            cropping indices.

        """
        crop_inds = self.crop_inds[index].tolist()
        X = self.features.iloc[index].to_numpy()
        X = X.copy()
        X.astype("float32")
        if self.transform is not None:
            X = self.transform(X)
        y = self.y[index]
        return X, y, crop_inds

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            The total number of feature samples.

        """
        return len(self.features.index)


def _compute_stats(
    ds: FeaturesDataset,
    return_count: bool = False,
    return_mean: bool = False,
    return_var: bool = False,
    ddof: int = 1,
    numeric_only: bool = False,
) -> tuple:
    """Compute statistics for a single FeaturesDataset."""
    res = []
    if return_count:
        res.append(ds.features.count(numeric_only=numeric_only))
    if return_mean:
        res.append(ds.features.mean(numeric_only=numeric_only))
    if return_var:
        res.append(ds.features.var(ddof=ddof, numeric_only=numeric_only))
    return tuple(res)


def _pooled_var(
    counts: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    ddof: int,
    ddof_in: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pooled variance across multiple datasets."""
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
    """A concatenated dataset of `FeaturesDataset` objects.

    This class holds a list of :class:`FeaturesDataset` instances and allows
    them to be treated as a single, larger dataset. It provides methods for

    splitting, saving, and performing DataFrame-like operations (e.g., `mean`,
    `var`, `fillna`) across all contained datasets.

    Parameters
    ----------
    list_of_ds : list of FeaturesDataset
        A list of :class:`FeaturesDataset` objects to concatenate.
    target_transform : callable, optional
        A function to apply to the target values before they are returned.

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
        """Split the dataset into subsets.

        The splitting can be done based on a column in the description
        DataFrame or by providing explicit indices for each split.

        Parameters
        ----------
        by : str or list or dict
            - If a string, splits are created for each unique value in the
              description column `by`.
            - If a list of integers, a single split is created containing the
              datasets at the specified indices.
            - If a list of lists of integers, multiple splits are created, one
              for each sublist of indices.
            - If a dictionary, keys are used as split names and values are
              lists of dataset indices.

        Returns
        -------
        dict[str, FeaturesConcatDataset]
            A dictionary where keys are split names and values are the new
            :class:`FeaturesConcatDataset` subsets.

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
        """Get the metadata of all datasets as a single DataFrame.

        Concatenates the metadata from all contained datasets and adds columns
        from their `description` attributes.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the metadata for every sample in the
            concatenated dataset.

        Raises
        ------
        TypeError
            If any of the contained datasets is not a :class:`FeaturesDataset`.

        """
        if not all([isinstance(ds, FeaturesDataset) for ds in self.datasets]):
            raise TypeError(
                "Metadata dataframe can only be computed when all "
                "datasets are FeaturesDataset."
            )

        all_dfs = list()
        for ds in self.datasets:
            df = ds.metadata.copy()
            for k, v in ds.description.items():
                df[k] = v
            all_dfs.append(df)

        return pd.concat(all_dfs)

    def save(self, path: str, overwrite: bool = False, offset: int = 0) -> None:
        """Save the concatenated dataset to a directory.

        Creates a directory structure where each contained dataset is saved in
        its own numbered subdirectory.

        .. code-block::

            path/
                0/
                    0-feat.parquet
                    metadata_df.pkl
                    description.json
                    ...
                1/
                    1-feat.parquet
                    ...

        Parameters
        ----------
        path : str
            The directory where the dataset will be saved.
        overwrite : bool, default False
            If True, any existing subdirectories that conflict with the new
            ones will be removed.
        offset : int, default 0
            An integer to add to the subdirectory names. Useful for saving
            datasets in chunks.

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
        n_sub_dirs = len([os.path.isdir(os.path.join(path, e)) for e in path_contents])
        for i_ds, ds in enumerate(self.datasets):
            sub_dir_name = str(i_ds + offset)
            if sub_dir_name in path_contents:
                path_contents.remove(sub_dir_name)
            sub_dir = os.path.join(path, sub_dir_name)
            if os.path.exists(sub_dir):
                if overwrite:
                    shutil.rmtree(sub_dir)
                else:
                    raise FileExistsError(
                        f"Subdirectory {sub_dir} already exists. Please select"
                        f" a different directory, set overwrite=True, or "
                        f"resolve manually."
                    )
            os.makedirs(sub_dir)
            self._save_features(sub_dir, ds, i_ds, offset)
            self._save_metadata(sub_dir, ds)
            self._save_description(sub_dir, ds.description)
            self._save_raw_info(sub_dir, ds)
            self._save_kwargs(sub_dir, ds)
        if overwrite and i_ds + 1 + offset < n_sub_dirs:
            logger.warning(
                f"The number of saved datasets ({i_ds + 1 + offset}) "
                f"does not match the number of existing "
                f"subdirectories ({n_sub_dirs}). You may now "
                f"encounter a mix of differently preprocessed "
                f"datasets!",
                UserWarning,
            )
        if path_contents:
            logger.warning(
                f"Chosen directory {path} contains other "
                f"subdirectories or files {path_contents}."
            )

    @staticmethod
    def _save_features(sub_dir: str, ds: FeaturesDataset, i_ds: int, offset: int):
        """Save the feature DataFrame to a Parquet file."""
        parquet_file_name = f"{i_ds + offset}-feat.parquet"
        parquet_file_path = os.path.join(sub_dir, parquet_file_name)
        ds.features.to_parquet(parquet_file_path)

    @staticmethod
    def _save_metadata(sub_dir: str, ds: FeaturesDataset):
        """Save the metadata DataFrame to a pickle file."""
        metadata_file_name = "metadata_df.pkl"
        metadata_file_path = os.path.join(sub_dir, metadata_file_name)
        ds.metadata.to_pickle(metadata_file_path)

    @staticmethod
    def _save_description(sub_dir: str, description: pd.Series):
        """Save the description Series to a JSON file."""
        desc_file_name = "description.json"
        desc_file_path = os.path.join(sub_dir, desc_file_name)
        description.to_json(desc_file_path)

    @staticmethod
    def _save_raw_info(sub_dir: str, ds: FeaturesDataset):
        """Save the raw info dictionary to a FIF file if it exists."""
        if hasattr(ds, "raw_info") and ds.raw_info is not None:
            fif_file_name = "raw-info.fif"
            fif_file_path = os.path.join(sub_dir, fif_file_name)
            ds.raw_info.save(fif_file_path, overwrite=True)

    @staticmethod
    def _save_kwargs(sub_dir: str, ds: FeaturesDataset):
        """Save various keyword argument dictionaries to JSON files."""
        for kwargs_name in [
            "raw_preproc_kwargs",
            "window_kwargs",
            "window_preproc_kwargs",
            "features_kwargs",
        ]:
            if hasattr(ds, kwargs_name):
                kwargs = getattr(ds, kwargs_name)
                if kwargs is not None:
                    kwargs_file_name = ".".join([kwargs_name, "json"])
                    kwargs_file_path = os.path.join(sub_dir, kwargs_file_name)
                    with open(kwargs_file_path, "w") as f:
                        json.dump(kwargs, f)

    def to_dataframe(
        self,
        include_metadata: bool | str | List[str] = False,
        include_target: bool = False,
        include_crop_inds: bool = False,
    ) -> pd.DataFrame:
        """Convert the dataset to a single pandas DataFrame.

        Parameters
        ----------
        include_metadata : bool or str or list of str, default False
            If True, include all metadata columns. If a string or list of
            strings, include only the specified metadata columns.
        include_target : bool, default False
            If True, include the 'target' column.
        include_crop_inds : bool, default False
            If True, include window cropping index columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the features and requested metadata.

        """
        if (
            not isinstance(include_metadata, bool)
            or include_metadata
            or include_crop_inds
        ):
            include_dataset = False
            if isinstance(include_metadata, bool) and include_metadata:
                include_dataset = True
                cols = self.datasets[0].metadata.columns.tolist()
            else:
                cols = include_metadata
                if isinstance(cols, bool) and not cols:
                    cols = []
                elif isinstance(cols, str):
                    cols = [cols]
                cols = set(cols)
                if include_crop_inds:
                    cols.update(
                        {
                            "i_dataset",
                            "i_window_in_trial",
                            "i_start_in_trial",
                            "i_stop_in_trial",
                        }
                    )
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
        """Get the names of numeric columns from the feature DataFrames."""
        return self.datasets[0].features.select_dtypes(include=np.number).columns

    def count(self, numeric_only: bool = False, n_jobs: int = 1) -> pd.Series:
        """Count non-NA cells for each feature column.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        n_jobs : int, default 1
            Number of jobs to run in parallel.

        Returns
        -------
        pandas.Series
            The count of non-NA cells for each column.

        """
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(ds, return_count=True, numeric_only=numeric_only)
            for ds in self.datasets
        )
        counts = np.array([s[0] for s in stats])
        count = counts.sum(axis=0)
        return pd.Series(count, index=self._numeric_columns())

    def mean(self, numeric_only: bool = False, n_jobs: int = 1) -> pd.Series:
        """Compute the mean for each feature column.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        n_jobs : int, default 1
            Number of jobs to run in parallel.

        Returns
        -------
        pandas.Series
            The mean of each column.

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

    def var(
        self, ddof: int = 1, numeric_only: bool = False, n_jobs: int = 1
    ) -> pd.Series:
        """Compute the variance for each feature column.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        n_jobs : int, default 1
            Number of jobs to run in parallel.

        Returns
        -------
        pandas.Series
            The variance of each column.

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

    def std(
        self, ddof: int = 1, numeric_only: bool = False, eps: float = 0, n_jobs: int = 1
    ) -> pd.Series:
        """Compute the standard deviation for each feature column.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        eps : float, default 0
            A small epsilon value to add to the variance before taking the
            square root to avoid numerical instability.
        n_jobs : int, default 1
            Number of jobs to run in parallel.

        Returns
        -------
        pandas.Series
            The standard deviation of each column.

        """
        return np.sqrt(
            self.var(ddof=ddof, numeric_only=numeric_only, n_jobs=n_jobs) + eps
        )

    def zscore(
        self, ddof: int = 1, numeric_only: bool = False, eps: float = 0, n_jobs: int = 1
    ) -> None:
        """Apply z-score normalization to numeric columns in-place.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom for variance calculation.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        eps : float, default 0
            Epsilon for numerical stability.
        n_jobs : int, default 1
            Number of jobs to run in parallel for statistics computation.

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
            ds.features.loc[:, self._numeric_columns()] = (
                ds.features.loc[:, self._numeric_columns()] - mean
            ) / std

    @staticmethod
    def _enforce_inplace_operations(func_name: str, kwargs: dict):
        """Raise an error if 'inplace=False' is passed to a method."""
        if "inplace" in kwargs and kwargs["inplace"] is False:
            raise ValueError(
                f"{func_name} only works inplace, please change "
                + "to inplace=True (default)."
            )
        kwargs["inplace"] = True

    def fillna(self, *args, **kwargs) -> None:
        """Fill NA/NaN values in-place. See :meth:`pandas.DataFrame.fillna`."""
        FeaturesConcatDataset._enforce_inplace_operations("fillna", kwargs)
        for ds in self.datasets:
            ds.features.fillna(*args, **kwargs)

    def replace(self, *args, **kwargs) -> None:
        """Replace values in-place. See :meth:`pandas.DataFrame.replace`."""
        FeaturesConcatDataset._enforce_inplace_operations("replace", kwargs)
        for ds in self.datasets:
            ds.features.replace(*args, **kwargs)

    def interpolate(self, *args, **kwargs) -> None:
        """Interpolate values in-place. See :meth:`pandas.DataFrame.interpolate`."""
        FeaturesConcatDataset._enforce_inplace_operations("interpolate", kwargs)
        for ds in self.datasets:
            ds.features.interpolate(*args, **kwargs)

    def dropna(self, *args, **kwargs) -> None:
        """Remove missing values in-place. See :meth:`pandas.DataFrame.dropna`."""
        FeaturesConcatDataset._enforce_inplace_operations("dropna", kwargs)
        for ds in self.datasets:
            ds.features.dropna(*args, **kwargs)

    def drop(self, *args, **kwargs) -> None:
        """Drop specified labels from rows or columns in-place. See :meth:`pandas.DataFrame.drop`."""
        FeaturesConcatDataset._enforce_inplace_operations("drop", kwargs)
        for ds in self.datasets:
            ds.features.drop(*args, **kwargs)

    def join(self, concat_dataset: FeaturesConcatDataset, **kwargs) -> None:
        """Join columns with other FeaturesConcatDataset in-place.

        Parameters
        ----------
        concat_dataset : FeaturesConcatDataset
            The dataset to join with. Must have the same number of datasets,
            and each corresponding dataset must have the same length.
        **kwargs
            Keyword arguments to pass to :meth:`pandas.DataFrame.join`.

        """
        assert len(self.datasets) == len(concat_dataset.datasets)
        for ds1, ds2 in zip(self.datasets, concat_dataset.datasets):
            assert len(ds1) == len(ds2)
            ds1.features = ds1.features.join(ds2.features, **kwargs)
