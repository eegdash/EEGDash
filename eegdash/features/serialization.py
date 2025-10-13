"""Convenience functions for storing and loading features datasets.

See Also
--------
https://github.com/braindecode/braindecode/blob/master/braindecode/datautil/serialization.py#L165-L229

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from mne.io import read_info

from braindecode.datautil.serialization import _load_kwargs_json

from .datasets import FeaturesConcatDataset, FeaturesDataset

__all__ = [
    "load_features_concat_dataset",
]


def load_features_concat_dataset(
    path: str | Path, ids_to_load: list[int] | None = None, n_jobs: int = 1
) -> FeaturesConcatDataset:
    """Load a stored `FeaturesConcatDataset` from a directory.

    This function reconstructs a :class:`FeaturesConcatDataset` by loading
    individual :class:`FeaturesDataset` instances from subdirectories within
    the given path. It uses joblib for parallel loading.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the directory where the dataset was saved. This directory
        should contain subdirectories (e.g., "0", "1", "2", ...) for each
        individual dataset.
    ids_to_load : list of int, optional
        A list of specific dataset IDs (subdirectory names) to load. If None,
        all subdirectories in the path will be loaded.
    n_jobs : int, default 1
        The number of jobs to use for parallel loading. -1 means using all
        processors.

    Returns
    -------
    eegdash.features.datasets.FeaturesConcatDataset
        A concatenated dataset containing the loaded `FeaturesDataset` instances.

    """
    # Make sure we always work with a pathlib.Path
    path = Path(path)

    if ids_to_load is None:
        # Get all subdirectories and sort them numerically
        ids_to_load = [p.name for p in path.iterdir() if p.is_dir()]
        ids_to_load = sorted(ids_to_load, key=lambda i: int(i))
    ids_to_load = [str(i) for i in ids_to_load]

    datasets = Parallel(n_jobs)(delayed(_load_parallel)(path, i) for i in ids_to_load)
    return FeaturesConcatDataset(datasets)


def _load_parallel(path: Path, i: str) -> FeaturesDataset:
    """Load a single `FeaturesDataset` from its subdirectory.

    This is a helper function for `load_features_concat_dataset` that handles
    the loading of one dataset's files (features, metadata, descriptions, etc.).

    Parameters
    ----------
    path : pathlib.Path
        The root directory of the saved `FeaturesConcatDataset`.
    i : str
        The identifier of the dataset to load, corresponding to its
        subdirectory name.

    Returns
    -------
    eegdash.features.datasets.FeaturesDataset
        The loaded dataset instance.

    """
    sub_dir = path / i

    parquet_name_pattern = "{}-feat.parquet"
    parquet_file_name = parquet_name_pattern.format(i)
    parquet_file_path = sub_dir / parquet_file_name

    features = pd.read_parquet(parquet_file_path)

    description_file_path = sub_dir / "description.json"
    description = pd.read_json(description_file_path, typ="series")

    raw_info_file_path = sub_dir / "raw-info.fif"
    raw_info = None
    if raw_info_file_path.exists():
        raw_info = read_info(raw_info_file_path)

    raw_preproc_kwargs = _load_kwargs_json("raw_preproc_kwargs", sub_dir)
    window_kwargs = _load_kwargs_json("window_kwargs", sub_dir)
    window_preproc_kwargs = _load_kwargs_json("window_preproc_kwargs", sub_dir)
    features_kwargs = _load_kwargs_json("features_kwargs", sub_dir)
    metadata = pd.read_pickle(path / i / "metadata_df.pkl")

    dataset = FeaturesDataset(
        features,
        metadata=metadata,
        description=description,
        raw_info=raw_info,
        raw_preproc_kwargs=raw_preproc_kwargs,
        window_kwargs=window_kwargs,
        window_preproc_kwargs=window_preproc_kwargs,
        features_kwargs=features_kwargs,
    )
    return dataset
