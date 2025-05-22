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
from .extractors import FeatureExtractor, _get_underlying_func


def _extract_features_from_windowsdataset(
    win_ds: EEGWindowsDataset | WindowsDataset,
    feature_extractor: FeatureExtractor,
    batch_size: int = 512,
):
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
        metadata.set_index("orig_index", drop=False, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        metadata.drop("orig_index", axis=1, inplace=True)

    # FUTURE: truly support WindowsDataset objects
    return FeaturesDataset(
        features_df,
        metadata=metadata,
        description=win_ds.description,
        raw_info=win_ds.raw.info,
        raw_preproc_kwargs=win_ds.raw_preproc_kwargs,
        window_kwargs=win_ds.window_kwargs,
        features_kwargs=feature_extractor.features_kwargs,
    )


def extract_features(
    concat_dataset: BaseConcatDataset,
    features: FeatureExtractor | Dict[str, Callable] | List[Callable],
    *,
    batch_size: int = 512,
    n_jobs: int = 1,
):
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
):
    if isinstance(features, list):
        features = dict(enumerate(features))
    if not isinstance(features, FeatureExtractor):
        features = FeatureExtractor(features)
    if not features._is_fitable:
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


def get_feature_predecessors(feature_or_extractor: Callable):
    current = _get_underlying_func(feature_or_extractor)
    if current is FeatureExtractor:
        return [current]
    predecessor = getattr(current, "parent_extractor_type", [FeatureExtractor])
    if len(predecessor) == 1:
        return [current, *get_predecessors(predecessor[0])]
    else:
        return [current, [get_predecessors(pred) for pred in predecessor]]
