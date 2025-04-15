from typing import Dict
from collections.abc import Callable
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset
from .datasets import FeaturesDataset, FeaturesConcatDataset
from .extractors import FeatureExtractor


def _extract_features_from_eegwindowsdataset(
    win_ds: EEGWindowsDataset,
    feature_extractor: FeatureExtractor,
    target_name: str = "target",
):
    features_dict = dict()
    ch_names = win_ds.raw.ch_names
    for X, y, _ in win_ds:
        win_dict = dict()
        win_dict.update(feature_extractor(ch_names, X))
        win_dict[target_name] = y
        for k, v in win_dict.items():
            if k not in features_dict:
                features_dict[k] = []
            features_dict[k].append(v)
    features_df = pd.DataFrame(features_dict)
    return FeaturesDataset(
        features_df,
        metadata=win_ds.metadata,
        description=win_ds.description,
        raw_preproc_kwargs=win_ds.raw_preproc_kwargs,
        window_kwargs=win_ds.window_kwargs,
        features_kwargs=feature_extractor.features_kwargs,
        target_name=target_name,
    )


def extract_features(
    concat_dataset: BaseConcatDataset,
    feature_extractor: FeatureExtractor | Dict[str, Callable],
    target_name: str = "target",
    n_jobs=1,
):
    if not isinstance(feature_extractor, FeatureExtractor):
        feature_extractor = FeatureExtractor(feature_extractor)
    feature_ds_list = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_extract_features_from_eegwindowsdataset)(
                    win_ds, feature_extractor, target_name
                )
                for win_ds in concat_dataset.datasets
            ),
            total=len(concat_dataset.datasets),
            desc="Extracting features",
        )
    )
    return FeaturesConcatDataset(feature_ds_list)
