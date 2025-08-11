"""Convenience functions for storing and loading of features datasets.

see also: https://github.com/braindecode/braindecode//blob/master/braindecode/datautil/serialization.py#L165-L229
"""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from mne.io import read_info

from braindecode.datautil.serialization import _load_kwargs_json

from .datasets import FeaturesConcatDataset, FeaturesDataset


def load_features_concat_dataset(path, ids_to_load=None, n_jobs=1):
    """Load a stored FeaturesConcatDataset of FeaturesDatasets from files.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to the directory of the .fif / -epo.fif and .json files.
    ids_to_load: list of int | None
        Ids of specific files to load.
    n_jobs: int
        Number of jobs to be used to read files in parallel.

    Returns
    -------
    concat_dataset: FeaturesConcatDataset of FeaturesDatasets

    """
    # Make sure we always work with a pathlib.Path
    path = Path(path)

    # else we have a dataset saved in the new way with subdirectories in path
    # for every dataset with description.json and -feat.parquet,
    # target_name.json, raw_preproc_kwargs.json, window_kwargs.json,
    # window_preproc_kwargs.json, features_kwargs.json
    if ids_to_load is None:
        ids_to_load = [p.name for p in path.iterdir()]
        ids_to_load = sorted(ids_to_load, key=lambda i: int(i))
    ids_to_load = [str(i) for i in ids_to_load]

    datasets = Parallel(n_jobs)(delayed(_load_parallel)(path, i) for i in ids_to_load)
    return FeaturesConcatDataset(datasets)


def _load_parallel(path, i):
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
