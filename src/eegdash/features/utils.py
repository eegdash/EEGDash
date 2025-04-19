from typing import Dict, List
from collections.abc import Callable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.utils.data import DataLoader
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset
from .datasets import FeaturesDataset, FeaturesConcatDataset
from .extractors import FeatureExtractor


def _extract_features_from_eegwindowsdataset(
    win_ds: EEGWindowsDataset,
    feature_extractor: FeatureExtractor,
    target_name: str = "target",
    batch_size: int = 512,
):
    win_dl = DataLoader(win_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    features_dict = dict()
    ch_names = win_ds.raw.ch_names
    for X, y, _ in win_dl:
        X = X.numpy()
        if hasattr(y, "tolist"):
            y = y.tolist()
        win_dict = dict()
        win_dict.update(feature_extractor(X.shape[0], ch_names, X))
        win_dict[target_name] = y
        for k, v in win_dict.items():
            if k not in features_dict:
                features_dict[k] = []
            features_dict[k].extend(v)
    features_df = pd.DataFrame(features_dict)
    return FeaturesDataset(
        features_df,
        metadata=win_ds.metadata,
        description=win_ds.description,
        raw_info=win_ds.raw.info,
        raw_preproc_kwargs=win_ds.raw_preproc_kwargs,
        window_kwargs=win_ds.window_kwargs,
        features_kwargs=feature_extractor.features_kwargs,
        target_name=target_name,
    )


def extract_features(
    concat_dataset: BaseConcatDataset,
    features: FeatureExtractor | Dict[str, Callable] | List[Callable],
    *,
    target_name: str = "target",
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
                delayed(_extract_features_from_eegwindowsdataset)(
                    win_ds, features, target_name, batch_size
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
