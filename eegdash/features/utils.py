import copy
from collections.abc import Callable
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

from braindecode.datasets.base import (
    BaseConcatDataset,
    EEGWindowsDataset,
    WindowsDataset,
)

from .datasets import FeaturesConcatDataset, FeaturesDataset
from .extractors import FeatureExtractor

__all__ = [
    "extract_features",
    "fit_feature_extractors",
]


def _extract_features_from_windowsdataset(
    win_ds: EEGWindowsDataset | WindowsDataset,
    feature_extractor: FeatureExtractor,
    batch_size: int = 512,
) -> FeaturesDataset:
    """Extract features from a single `WindowsDataset`.

    This is a helper function that iterates through a `WindowsDataset` in
    batches, applies a `FeatureExtractor`, and returns the results as a
    `FeaturesDataset`.

    Parameters
    ----------
    win_ds : EEGWindowsDataset or WindowsDataset
        The windowed dataset to extract features from.
    feature_extractor : FeatureExtractor
        The feature extractor instance to apply.
    batch_size : int, default 512
        The number of windows to process in each batch.

    Returns
    -------
    FeaturesDataset
        A new dataset containing the extracted features and associated metadata.

    """
    metadata = win_ds.metadata
    if not win_ds.targets_from == "metadata":
        metadata = copy.deepcopy(metadata)
        metadata["orig_index"] = metadata.index
        metadata.set_index(
            ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"],
            drop=False,
            inplace=True,
        )
    win_dl = DataLoader(win_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    features_dict = dict()
    ch_names = win_ds.raw.ch_names
    for X, y, crop_inds in win_dl:
        X = X.numpy()
        if hasattr(y, "tolist"):
            y = y.tolist()
        win_dict = dict()
        win_dict.update(
            feature_extractor(X, _batch_size=X.shape[0], _ch_names=ch_names)
        )
        if not win_ds.targets_from == "metadata":
            metadata.loc[crop_inds, "target"] = y
        for k, v in win_dict.items():
            if k not in features_dict:
                features_dict[k] = []
            features_dict[k].extend(v)
    features_df = pd.DataFrame(features_dict)
    if not win_ds.targets_from == "metadata":
        metadata.reset_index(drop=True, inplace=True)
        metadata.drop("orig_index", axis=1, inplace=True, errors="ignore")

    return FeaturesDataset(
        features_df,
        metadata=metadata,
        description=win_ds.description,
        raw_info=win_ds.raw.info,
        raw_preproc_kwargs=getattr(win_ds, "raw_preproc_kwargs", None),
        window_kwargs=getattr(win_ds, "window_kwargs", None),
        features_kwargs=feature_extractor.features_kwargs,
    )


def extract_features(
    concat_dataset: BaseConcatDataset,
    features: FeatureExtractor | Dict[str, Callable] | List[Callable],
    *,
    batch_size: int = 512,
    n_jobs: int = 1,
) -> FeaturesConcatDataset:
    """Extract features from a concatenated dataset of windows.

    This function applies a feature extractor to each `WindowsDataset` within a
    `BaseConcatDataset` in parallel and returns a `FeaturesConcatDataset`
    with the results.

    Parameters
    ----------
    concat_dataset : BaseConcatDataset
        A concatenated dataset of `WindowsDataset` or `EEGWindowsDataset`
        instances.
    features : FeatureExtractor or dict or list
        The feature extractor(s) to apply. Can be a `FeatureExtractor`
        instance, a dictionary of named feature functions, or a list of
        feature functions.
    batch_size : int, default 512
        The size of batches to use for feature extraction.
    n_jobs : int, default 1
        The number of parallel jobs to use for extracting features from the
        datasets.

    Returns
    -------
    FeaturesConcatDataset
        A new concatenated dataset containing the extracted features.

    """
    if isinstance(features, list):
        features = dict(enumerate(features))
    if not isinstance(features, FeatureExtractor):
        features = FeatureExtractor(features)
    feature_ds_list = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_extract_features_from_windowsdataset)(
                    win_ds, features, batch_size
                )
                for win_ds in concat_dataset.datasets
            ),
            total=len(concat_dataset.datasets),
            desc="Extracting features",
        )
    )
    return FeaturesConcatDataset(feature_ds_list)


def fit_feature_extractors(
    concat_dataset: BaseConcatDataset,
    features: FeatureExtractor | Dict[str, Callable] | List[Callable],
    batch_size: int = 8192,
) -> FeatureExtractor:
    """Fit trainable feature extractors on a dataset.

    If the provided feature extractor (or any of its sub-extractors) is
    trainable (i.e., subclasses `TrainableFeature`), this function iterates
    through the dataset to fit it.

    Parameters
    ----------
    concat_dataset : BaseConcatDataset
        The dataset to use for fitting the feature extractors.
    features : FeatureExtractor or dict or list
        The feature extractor(s) to fit.
    batch_size : int, default 8192
        The batch size to use when iterating through the dataset for fitting.

    Returns
    -------
    FeatureExtractor
        The fitted feature extractor.

    """
    if isinstance(features, list):
        features = dict(enumerate(features))
    if not isinstance(features, FeatureExtractor):
        features = FeatureExtractor(features)
    if not features._is_trainable:
        return features
    features.clear()
    concat_dl = DataLoader(
        concat_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    for X, y, _ in tqdm(
        concat_dl, total=len(concat_dl), desc="Fitting feature extractors"
    ):
        features.partial_fit(X.numpy(), y=np.array(y))
    features.fit()
    return features
