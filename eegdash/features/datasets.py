r"""
Datasets for Feature Management.

This module defines the core data structures for storing, manipulating, and 
serializing extracted features.

Provides the base classes:
- :class:`FeaturesDataset` — Represents features from a single recording.
- :class:`FeaturesConcatDataset` — Manages multiple :class:`FeaturesDataset`
  objects as a unified dataset.

"""

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
    r"""A dataset of features extracted from a single recording.

    This class holds features in a :class:`pandas.DataFrame` and provides an interface
    compatible with braindecode's dataset structure. A single object corresponds 
    to one recording.

    Parameters
    ----------
    features : pandas.DataFrame
        A DataFrame where each row is a sample (e.g, EEG window) 
        and each column is a feature.
    metadata : pandas.DataFrame, optional
        A DataFrame containing metadata for each sample, indexed consistently
        with `features`. Must include columns 'i_window_in_trial',
        'i_start_in_trial', 'i_stop_in_trial', and 'target'.
    description : dict or pandas.Series, optional
        Additional high-level information about the dataset.
    transform : callable, optional
        A function or transform to apply to the feature data.
    raw_info : dict, optional
        Information about the original raw recording (e.g., sampling rate,
        montage, channel names).
    raw_preproc_kwargs : dict, optional
        Keyword arguments used for preprocessing the raw data.
    window_kwargs : dict, optional
        Keyword arguments used for windowing the data.
    window_preproc_kwargs : dict, optional
        Keyword arguments used for preprocessing the windowed data.
    features_kwargs : dict, optional
        Keyword arguments used for feature extraction.

    Attributes
    ----------
    features : pandas.DataFrame
        Table of extracted features.
    n_features : int
        Number of feature columns in the dataset.
    metadata : pandas.DataFrame
        Metadata describing each window.
    transform : callable or None
        The transform applied to each sample.
    raw_info : dict or None
        Information about the raw recording.
    raw_preproc_kwargs : dict or None
        Parameters used during raw data preprocessing.
    window_kwargs : dict or None
        Parameters used during window segmentation.
    window_preproc_kwargs : dict or None
        Parameters used during window-level preprocessing.
    features_kwargs : dict or None
        Parameters used during feature extraction.
    crop_inds : numpy.ndarray of shape (n_samples, 3)
        Indices specifying window position within each trial:
        (i_window_in_trial, i_start_in_trial, i_stop_in_trial).
    y : list of int
        Target labels corresponding to each window.
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
        r"""Returns a single sample and its corresponding label and metadata.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        X : numpy.ndarray of shape (n_features,)
            Feature vector corresponding to the requested index.
        y : int
            Target label or class associated with the sample.
        crop_inds : list of int
            Trial-related indices for the window.
            
        Notes
        -----
        - If a transformation function was provided at initialization, it is
        applied to ``X`` before returning.
        - This method enables indexing of the dataset object using square brackets.

        Examples
        --------
        >>> X, y, crop_inds = dataset[42]
        
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
        r"""Returns the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples.
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
    r"""Compute basic feature statistics for a single FeaturesDataset.

    Depending on the specified flags, this function computes and returns
    the count, mean, and/or variance of all numeric features in the dataset.

    Parameters
    ----------
    ds : FeaturesDataset
        The dataset containing feature values in a pandas DataFrame.
    return_count : bool, default=False
        If True, include the feature counts.
    return_mean : bool, default=False
        If True, include the feature means.
    return_var : bool, default=False
        If True, include the feature variances.
    ddof : int, default=1
        Delta degrees of freedom for variance computation.
    numeric_only : bool, default=False
        Whether to include only numeric columns in the computation.

    Returns
    -------
    tuple of pandas.Series
        A tuple containing one or more pandas Series, in the order of the
        requested statistics (count, mean, var). Each Series has feature
        names as its index.

    Examples
    --------
    >>> stats = _compute_stats(dataset, return_mean=True, return_var=True)
    >>> len(stats)
    2
    >>> stats[0].head()
    feature_1    0.12
    feature_2    0.34
    dtype: float64
                  
    """
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
    r"""Compute the pooled variance across multiple datasets.

    This function combines per-dataset statistics (sample counts, means,
    and variances) into a single set of pooled statistics.

    Parameters
    ---------- 
    counts : ndarray of shape (n_datasets, n_features)
        Number of samples per feature in each dataset.
    means : ndarray of shape (n_datasets, n_features)
        Mean value of each feature per dataset.
    variances : ndarray of shape (n_datasets, n_features)
        Variance of each feature per dataset.
    ddof : int
        Delta degrees of freedom for the pooled variance (typically 1).
    ddof_in : int or None, optional
        Delta degrees of freedom used in the input variances. If None,
        it defaults to the same value as ``ddof``.

    Returns
    -------
    count : ndarray of shape (n_features,)
        Total number of samples across datasets, per feature.
    mean : ndarray of shape (n_features,)
        Pooled mean across all datasets.
    var : ndarray of shape (n_features,)
        Pooled variance across all datasets.

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
    r"""A concatenated dataset composed of multiple :class:`FeaturesDataset` objects.

    This class manages a collection of :class:`FeaturesDataset` instances and
    provides an interface for treating them as a single, unified dataset.
    Supports concatenation, splitting, saving, and performing DataFrame-like
    operations across all contained datasets.

    Parameters
    ----------
    list_of_ds : list of FeaturesDataset or None, optional
        A list of :class:`FeaturesDataset` objects to concatenate.
        If a list of :class:`FeaturesConcatDataset` objects is provided,
        all contained datasets are automatically flattened into a single list.
    target_transform : callable or None, optional
        A function to apply to target values before they are returned.

    Attributes
    ----------
    datasets : list of FeaturesDataset
        The list of individual datasets contained in this object.
    target_transform : callable or None
        Optional transform applied to target labels.

    """
    def __init__(
        self,
        list_of_ds: list[FeaturesDataset] | None = None,
        target_transform: Callable | None = None,
    ):
        # if a list of FeaturesConcatDataset is provided, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], FeaturesConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)

        self.target_transform = target_transform

    def split(
        self,
        by: str | list[int] | list[list[int]] | dict[str, list[int]],
    ) -> dict[str, FeaturesConcatDataset]:
        r"""Split the concatenated dataset into multiple subsets.

        This method allows flexible splitting of the concatenated dataset into
        several :class:`FeaturesConcatDataset` objects based on a metadata field,
        explicit indices, or custom grouping definitions.

        Parameters
        ----------
        by : str or list of int or list of list of int or dict of {str: list of int}
            Defines how the dataset is split:
            
            * **str** — Name of a column in the dataset description.
              Each unique value in that column defines a separate split.
            * **list of int** — Indices of datasets to include in one split.
            * **list of list of int** — A list of groups of indices, where each sub-list
              defines one split.
            * **dict of {str: list of int}** — Explicit mapping of split names to
              lists of dataset indices.

        Returns
        -------
        dict[str, FeaturesConcatDataset]
            A dictionary where each key is the split name (or index)
            and each value is a :class:`FeaturesConcatDataset` containing
            the corresponding subset of datasets.

        Examples
        --------
        >>> # Split by a metadata column (str)
        >>> splits = concat_ds.split(by='subject_id')
        >>> list(splits.keys())
        ['subj_01', 'subj_02', 'subj_03']
        >>> splits['subj_01']
        <FeaturesConcatDataset>

        >>> # Split by explicit indices (list of int)
        >>> splits = concat_ds.split(by=[0, 2, 4])
        >>> splits["0"]
        <FeaturesConcatDataset>

        >>> # Split by groups of indices (list of list of int)
        >>> splits = concat_ds.split(by=[[0, 1], [2, 3], [4, 5]])      
        >>> list(splits.keys())
        ['0', '1', '2']

        >>> # Split by custom mapping (dict)
        >>> splits = concat_ds.split(by={'train': [0, 1, 2], 'test': [3, 4]})
        >>> splits["train"], splits["test"]
        (<FeaturesConcatDataset>, <FeaturesConcatDataset>)

        Notes
        -----
        The resulting splits inherit the same ``target_transform`` as the original
        dataset. Splitting by a string requires that ``self.description`` contains
        the specified column.

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
        r"""Return a concatenated metadata DataFrame from all contained datasets.

        Collects the metadata of each :class:`FeaturesDataset` contained in
        the :class:`FeaturesConcatDataset` and concatenates them into a single
        pandas DataFrame, adding each dataset's description entries as 
        additional columns in the resulting DataFrame.

        Returns
        -------
        pandas.DataFrame
            Combined metadata from all contained datasets.
            Each row corresponds to a single sample from one of the underlying
            :class:`FeaturesDataset` objects.
            Columns include both window-level metadata (e.g., ``target``,
            ``i_window_in_trial``, ``i_start_in_trial``, ``i_stop_in_trial``)
            and dataset-level description fields (e.g., ``subject_id``,
            ``session``, etc.).

        Raises
        ------
        TypeError
            If one or more contained datasets are not instances
            of :class:`FeaturesDataset`.

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
        r"""Save the concatenated dataset to a directory.

        Each contained :class:`FeaturesDataset` is saved in its own
        numbered subdirectory within the specified ``path``. The resulting
        structure is compatible with later reloading using
        :func:`serialization.load_features_concat_dataset`.

        Directory structure example
        ----------------------------
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
            Path to the parent directory where the dataset should be saved.
            The directory will be created if it does not exist.
        overwrite : bool, default=False
            If True, existing subdirectories that conflict with the new ones
            are removed before saving.
        offset : int, default=0
            Integer offset added to subdirectory names. Useful when saving
            datasets in chunks or continuing a previous save session.

        Raises
        ------
        ValueError
            If the concatenated dataset is empty.
        FileExistsError
            If a subdirectory already exists and ``overwrite`` is False.

        Warns
        -----
        UserWarning
            If the number of saved subdirectories does not match the number
            of existing ones, or if unrelated files remain in the directory.

        Notes
        -----
        Each subdirectory contains:
        
        - ``*-feat.parquet`` — feature DataFrame for that dataset.
        - ``metadata_df.pkl`` — corresponding metadata.
        - ``description.json`` — dataset-level metadata.
        - ``raw_info.pkl`` — recording information (optional).
        - ``*_kwargs.json`` — preprocessing parameters.

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
        r"""Save the feature DataFrame to a Parquet file.

        Parameters
        ----------
        sub_dir : str
            Path to the directory where the file will be saved.
        ds : FeaturesDataset
            The dataset instance containing the features.
        i_ds : int
            The index of the dataset within the collection.
        offset : int
            An integer offset used for file naming.
        """
        parquet_file_name = f"{i_ds + offset}-feat.parquet"
        parquet_file_path = os.path.join(sub_dir, parquet_file_name)
        ds.features.to_parquet(parquet_file_path)

    @staticmethod
    def _save_metadata(sub_dir: str, ds: FeaturesDataset):
        r"""Save the metadata DataFrame to a pickle file.

        Parameters
        ----------
        sub_dir : str
            Path to the directory where the file will be saved.
        ds : FeaturesDataset
            The dataset instance containing the metadata.
        """
        metadata_file_name = "metadata_df.pkl"
        metadata_file_path = os.path.join(sub_dir, metadata_file_name)
        ds.metadata.to_pickle(metadata_file_path)

    @staticmethod
    def _save_description(sub_dir: str, description: pd.Series):
        r"""Save the description Series to a JSON file.

        Parameters
        ----------
        sub_dir : str
            Path to the directory where the file will be saved.
        description : pandas.Series
            Series containing dataset-level description/metadata.
        """
        desc_file_name = "description.json"
        desc_file_path = os.path.join(sub_dir, desc_file_name)
        description.to_json(desc_file_path)

    @staticmethod
    def _save_raw_info(sub_dir: str, ds: FeaturesDataset):
        r"""Save the raw info dictionary to a FIF file if it exists.

        Parameters
        ----------
        sub_dir : str
            Path to the directory where the file will be saved.
        ds : FeaturesDataset
            The dataset instance containing the raw information.
        """
        if hasattr(ds, "raw_info") and ds.raw_info is not None:
            fif_file_name = "raw-info.fif"
            fif_file_path = os.path.join(sub_dir, fif_file_name)
            ds.raw_info.save(fif_file_path, overwrite=True)

    @staticmethod
    def _save_kwargs(sub_dir: str, ds: FeaturesDataset):
        r"""Save various keyword argument dictionaries to JSON files.

        Iterates through known preprocessing and feature extraction 
        keyword argument attributes and saves them if they are not None.

        Parameters
        ----------
        sub_dir : str
            Path to the directory where the files will be saved.
        ds : FeaturesDataset
            The dataset instance containing the keyword arguments.
        """
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
        r"""Convert the concatenated dataset into a single unified pandas DataFrame.

        This method flattens the collection of individual recording datasets into 
        one table, allowing for the selective inclusion of metadata, target 
        labels, and window-cropping indices alongside features.

        Parameters
        ----------
        include_metadata : bool, str, or list of str, default=False
            Controls the inclusion of window-level metadata: 
            * If **True** — includes all metadata columns available in the 
                underlying datasets.
            * If **str** or **list of str** — includes only the specified 
                metadata column(s).
            * If **False** — excludes metadata (unless overridden by other 
                flags).
        include_target : bool, default=False
            If True, ensures the 'target' column is included in the resulting 
            DataFrame.
        include_crop_inds : bool, default=False
            If True, includes the internal windowing indices: 'i_dataset', 
            'i_window_in_trial', 'i_start_in_trial', and 'i_stop_in_trial'.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame where each row represents a sample (window) 
            and columns contain features and requested metadata.

        Notes
        -----
        When metadata columns and feature columns share the same name, the 
        metadata columns are suffixed with ``_metadata`` to avoid name 
        collisions.

        Examples
        --------
        >>> # Get only features
        >>> df = concat_ds.to_dataframe()
        
        >>> # Get features with target labels and specific metadata
        >>> df = concat_ds.to_dataframe(
        ...     include_metadata=['subject_id'], 
        ...     include_target=True
        ... )

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
        r"""Get the names of numeric columns from the feature DataFrames.

        Returns
        -------
        pandas.Index
            The names of the columns containing numeric data.

        Notes
        -----
        This method assumes that all :class:`FeaturesDataset` objects in the 
        concatenated collection share the same feature column schema.
        
        """
        return self.datasets[0].features.select_dtypes(include=np.number).columns

    def count(self, numeric_only: bool = False, n_jobs: int = 1) -> pd.Series:
        r"""Count non-NA cells for each feature column across all datasets.

        Parameters
        ----------
        numeric_only : bool, default=False
            If True, only includes columns with float, int, or boolean data 
            types.
        n_jobs : int, default=1
            The number of CPU cores to use for parallel processing of 
            individual datasets.

        Returns
        -------
        pd.Series
            A Series containing the total count of non-missing values for 
            each feature column, indexed by feature names.

        """
        stats = Parallel(n_jobs)(
            delayed(_compute_stats)(ds, return_count=True, numeric_only=numeric_only)
            for ds in self.datasets
        )
        counts = np.array([s[0] for s in stats])
        count = counts.sum(axis=0)
        return pd.Series(count, index=self._numeric_columns())

    def mean(self, numeric_only: bool = False, n_jobs: int = 1) -> pd.Series:
        r"""Compute the mean for each feature column across all datasets.

        This method calculates the mean of each feature by aggregating the 
        individual means of each dataset, weighted by their respective 
        sample counts.

        Parameters
        ----------
        numeric_only : bool, default=False
            If True, only includes columns with float, int, or boolean data 
            types.
        n_jobs : int, default=1
            The number of CPU cores to use for parallel processing of 
            individual datasets.

        Returns
        -------
        pd.Series
            A Series containing the weighted mean of each feature column, 
            indexed by feature names.
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
        r"""Compute the variance for each feature column across all datasets.

        This method calculates the total variance by combining within-dataset 
        variability and between-dataset mean differences.

        Parameters
        ----------
        ddof : int, default=1
            Delta Degrees of Freedom.
        numeric_only : bool, default=False
            If True, only includes columns with float, int, or boolean data 
            types.
        n_jobs : int, default=1
            The number of CPU cores to use for parallel processing of 
            individual datasets.

        Returns
        -------
        pd.Series
            A Series containing the pooled variance of each feature column, 
            indexed by feature names.
           
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
        r"""Compute the standard deviation for each feature column across all datasets.

        Parameters
        ----------
        ddof : int, default=1
            Delta Degrees of Freedom for the variance calculation.
        numeric_only : bool, default=False
            If True, only includes numeric data types.
        eps : float, default=0
            Small constant added to variance for numerical stability.
        n_jobs : int, default=1
            Number of CPU cores for parallel processing.

        Returns
        -------
        pd.Series
            Standard deviation of each feature column. Indexed by feature names.

        """
        return np.sqrt(
            self.var(ddof=ddof, numeric_only=numeric_only, n_jobs=n_jobs) + eps
        )

    def zscore(
        self, ddof: int = 1, numeric_only: bool = False, eps: float = 0, n_jobs: int = 1
    ) -> None:
        r"""Apply z-score normalization to numeric columns in-place.

        This method scales features to a mean of 0 and a standard deviation 
        of 1 based on statistics pooled across all contained datasets.

        Parameters
        ----------
        ddof : int, default=1
            Delta Degrees of Freedom for the pooled variance.
        numeric_only : bool, default=False
            If True, only includes numeric data types.
        eps : float, default=0
            Small constant added to variance for numerical stability.
        n_jobs : int, default=1
            Number of CPU cores for parallel statistics computation.
        
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
        r"""Ensure that the operation is performed in-place.

        Validates that 'inplace=False' is not passed and explicitly
        sets the 'inplace' key to True in the provided arguments.

        Parameters
        ----------
        func_name : str
            The name of the calling method to be used in the error message.
        kwargs : dict
            Dictionary of keyword arguments to be modified.

        Raises
        ------
        ValueError
            If 'inplace' is present in `kwargs` and set to False.
        
        """
        if "inplace" in kwargs and kwargs["inplace"] is False:
            raise ValueError(
                f"{func_name} only works inplace, please change "
                + "to inplace=True (default)."
            )
        kwargs["inplace"] = True

    def fillna(self, *args, **kwargs) -> None:
        r"""Fill NA/NaN values in-place across all datasets.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`pandas.DataFrame.fillna`.
        
        Notes
        -----
        ``inplace`` is enforced as True.

        See Also
        --------
        :meth:pandas.DataFrame.fillna : The underlying pandas method.
        """
        FeaturesConcatDataset._enforce_inplace_operations("fillna", kwargs)
        for ds in self.datasets:
            ds.features.fillna(*args, **kwargs)

    def replace(self, *args, **kwargs) -> None:
        r"""Replace values in-place across all datasets.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`pandas.DataFrame.replace`.
        
        Notes
        -----
        ``inplace`` is enforced as True.

        See Also
        --------
        :meth:pandas.DataFrame.replace : The underlying pandas method.
        """
        FeaturesConcatDataset._enforce_inplace_operations("replace", kwargs)
        for ds in self.datasets:
            ds.features.replace(*args, **kwargs)

    def interpolate(self, *args, **kwargs) -> None:
        r"""Interpolate values in-place across all datasets.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`pandas.DataFrame.interpolate`.
        
        Notes
        -----
        ``inplace`` is enforced as True.

        See Also
        --------
        :meth:pandas.DataFrame.interpolate : The underlying pandas method.
        """
        FeaturesConcatDataset._enforce_inplace_operations("interpolate", kwargs)
        for ds in self.datasets:
            ds.features.interpolate(*args, **kwargs)

    def dropna(self, *args, **kwargs) -> None:
        r"""Remove missing values in-place across all datasets.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`pandas.DataFrame.dropna`.
        
        Notes
        -----
        ``inplace`` is enforced as True.

        See Also
        --------
        :meth:pandas.DataFrame.dropna : The underlying pandas method.
        """
        FeaturesConcatDataset._enforce_inplace_operations("dropna", kwargs)
        for ds in self.datasets:
            ds.features.dropna(*args, **kwargs)

    def drop(self, *args, **kwargs) -> None:
        r"""Drop specified labels from rows or columns in-place across all datasets.

        This method removes features (columns) or samples (rows) from every 
        underlying dataset in the collection.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :meth:`pandas.DataFrame.drop`.
        
        Notes
        -----
        ``inplace`` is enforced as True.

        See Also
        --------
        :meth:pandas.DataFrame.drop : The underlying pandas method.

        Examples
        --------
        >>> # Remove specific feature columns by name from all datasets
        >>> concat_ds.drop(columns=['Alpha_Power', 'Beta_Power'])

        >>> # Remove the first and third window (rows) from every dataset
        >>> concat_ds.drop(index=[0, 2])

        """
        FeaturesConcatDataset._enforce_inplace_operations("drop", kwargs)
        for ds in self.datasets:
            ds.features.drop(*args, **kwargs)

    def join(self, concat_dataset: FeaturesConcatDataset, **kwargs) -> None:
        r"""Join columns with another FeaturesConcatDataset in-place.

        This method merges the feature columns of another dataset into the 
        current one. Both collections must contain the same number of 
        individual datasets, and corresponding datasets must have matching 
        lengths.

        Parameters
        ----------
        concat_dataset : FeaturesConcatDataset
            The dataset containing the new columns to be joined.
        **kwargs
            Keyword arguments passed to :meth:`pandas.DataFrame.join`.

        Raises
        ------
        AssertionError
            If the number of datasets or the lengths of corresponding 
            datasets do not match.

        Notes
        -----
        This operation is performed in-place. The ``ds.features`` attribute 
        of each underlying dataset is updated with the new columns.

        """
        assert len(self.datasets) == len(concat_dataset.datasets)
        for ds1, ds2 in zip(self.datasets, concat_dataset.datasets):
            assert len(ds1) == len(ds2)
            ds1.features = ds1.features.join(ds2.features, **kwargs)
