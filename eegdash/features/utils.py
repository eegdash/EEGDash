r"""Feature Extraction Utilities.

This module provides the primary entry points for applying feature extraction
pipelines to windowed datasets.

The module provides the following functions:

- :func:`extract_features` — The main interface for computing features
  across an entire concatenated dataset.
- :func:`fit_feature_extractors` — Fits trainable features using a
  representative dataset.
- :func:`_extract_features_from_windowsdataset` — Internal helper for
  processing individual recording datasets.

"""

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

from . import extractors
from .datasets import FeaturesConcatDataset, FeaturesDataset

__all__ = [
    "extract_features",
    "fit_feature_extractors",
    "channel_names_to_indices",
]


def _get_record_metadata(win_ds):
    """Get record metadata.

    Parameters
    ----------
    win_ds : EEGWindowsDataset
        A braindecode wimdowed EEG dataset.

    Returns
    -------
    dict
        Record metadata, including:

        - info : MNE's record info.
        - description : braindesode's dataset description.

    """
    return {
        "info": win_ds.raw.info,
        "description": win_ds.description,
    }


def _get_batch_metadata(win_ds, X, crop_inds):
    """Get batch metadata.

    Parameters
    ----------
    win_ds : EEGWindowsDataset
        A braindecode wimdowed EEG dataset.
    X : ndarray
        A batch of EEG windows.
    crop_inds : list of tuples
        a tuple of `(i_window_in_trial, i_start_in_trial, i_stop_in_trial)` for each
        sample in the batch.

    Returns
    -------
    dict
        Batch metadata, including:

        - batch_size : the number of samples in the batch.
        - crop_inds : a tuple of `(i_window_in_trial, i_start_in_trial, i_stop_in_trial)`
           for each sample in the batch.

    """
    return {
        "batch_size": X.shape[0],
        "crop_inds": crop_inds,
    }


def _extract_features_from_windowsdataset(
    win_ds: EEGWindowsDataset | WindowsDataset,
    feature_extractor: extractors.FeatureExtractor,
    batch_size: int = 512,
) -> FeaturesDataset:
    r"""Extract features from a single recording windowed dataset.

    This helper function iterates through a :class:`WindowsDataset` in
    batches, applies a :class:`FeatureExtractor`, and packages the
    resulting feature vectors into a :class:`FeaturesDataset` instance.

    Parameters
    ----------
    win_ds : EEGWindowsDataset or WindowsDataset
        The windowed dataset containing raw EEG data to extract features from.
    feature_extractor : ~eegdash.features.extractors.FeatureExtractor
        The configured feature extractor pipeline to apply to the windows.
    batch_size : int, default 512
        The number of windows to process in each batch via the DataLoader.

    Returns
    -------
    ~eegdash.features.datasets.FeaturesDataset
        A recording-level dataset containing the extracted feature table
        and associated recording metadata.

    Notes
    -----
    If the input dataset does not have targets pre-loaded in metadata,
    this function will automatically extract them during the iteration
    and update the returned metadata accordingly.

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
    batch_metadata = _get_record_metadata(win_ds)
    for X, y, crop_inds in win_dl:
        X = X.numpy()
        if hasattr(y, "tolist"):
            y = y.tolist()
        batch_metadata.update(_get_batch_metadata(win_ds, X, crop_inds))
        win_dict = dict()
        win_dict.update(feature_extractor(X, _metadata=batch_metadata))
        if not win_ds.targets_from == "metadata":
            # Convert transposed crop_inds from DataLoader to list of tuples for MultiIndex
            crop_inds_tuples = list(zip(*[idx.tolist() for idx in crop_inds]))
            metadata.loc[crop_inds_tuples, "target"] = y
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
    features: extractors.FeatureExtractor | Dict[str, Callable] | List[Callable],
    *,
    batch_size: int = 512,
    n_jobs: int = 1,
) -> FeaturesConcatDataset:
    r"""Extract features from a collection of windowed recordings.

    This function applies a feature extraction pipeline to every
    individual recording in a :class:`BaseConcatDataset`.

    Parameters
    ----------
    concat_dataset : BaseConcatDataset
        A concatenated dataset of :class:`WindowsDataset` or
        :class:`EEGWindowsDataset` instances.
    features : ~eegdash.features.extractors.FeatureExtractor or dict or list
        The feature extractor(s) to apply. Can be a
        :class:`~eegdash.features.extractors.FeatureExtractor` instance,
        a dictionary of named feature functions, or a list of feature
        functions.
    batch_size : int, default 512
        The size of batches used for feature extraction within each recording.
    n_jobs : int, default 1
        The number of parallel jobs to use for processing different
        recordings simultaneously.

    Returns
    -------
    ~eegdash.features.datasets.FeaturesConcatDataset
        A unified collection of feature datasets corresponding to the
        input recordings.

    """
    if isinstance(features, list):
        features = dict(enumerate(features))
    if not isinstance(features, extractors.FeatureExtractor):
        features = extractors.FeatureExtractor(features)
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
    features: extractors.FeatureExtractor | Dict[str, Callable] | List[Callable],
    batch_size: int = 8192,
) -> extractors.FeatureExtractor:
    r"""Fit trainable feature extractors on a concatenated dataset.

    Scans the provided feature pipeline for components that require training
    (subclasses of :class:`~eegdash.features.extractors.TrainableFeature`).
    If found, the function iterates through the dataset in batches to
    perform partial fitting before finalization.

    Parameters
    ----------
    concat_dataset : BaseConcatDataset
        The dataset used to train the feature extractors.
    features : ~eegdash.features.extractors.FeatureExtractor or dict or list
        The feature extractor pipeline(s) to fit.
    batch_size : int, default 8192
        The batch size to use when streaming data through the
        :meth:`partial_fit` phase.

    Returns
    -------
    ~eegdash.features.extractors.FeatureExtractor
        The fitted feature extractor instance, ready for feature extraction.

    Notes
    -----
    If the provided extractors are not trainable, the function returns
    the original input without modification.

    """
    if isinstance(features, list):
        features = dict(enumerate(features))
    if not isinstance(features, extractors.FeatureExtractor):
        features = extractors.FeatureExtractor(features)
    if not features._is_trainable:
        return features
    features.clear()
    for win_ds in tqdm(
        concat_dataset.datasets,
        total=len(concat_dataset.datasets),
        desc="Fitting feature extractors",
    ):
        win_dl = DataLoader(
            win_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )
        batch_metadata = _get_record_metadata(win_ds)
        for X, y, crop_inds in win_dl:
            batch_metadata.update(_get_batch_metadata(win_ds, X, crop_inds))
            features.partial_fit(X.numpy(), y=np.array(y), _metadata=batch_metadata)
    features.fit()
    return features


def channel_names_to_indices(channels: List[str], ch_names: List[str]) -> List[int]:
    r"""Converts a list of channel names to channel indices in another list.

    Parameters
    ----------
    channels : List[str]
        A list of channel names.
    ch_names : List[str]
        A list of existing channel names to take indices from.

    Returns
    -------
    List[int]
        A list of channel indices.

    Raises
    ------
    ValueError
        If the channel name was not found in the existing channels list.

    """
    channel_idx = []
    for channel in channels:
        if channel in ch_names:
            channel_idx.append(ch_names.index(channel))
        else:
            raise ValueError(
                f"Channel {channel} not found in metadata channels: {ch_names}."
            )
    return channel_idx
