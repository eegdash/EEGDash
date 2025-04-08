import pandas as pd
from joblib import Parallel, delayed
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset
from .datasets import FeaturesDataset, FeaturesConcatDataset
from .extractors import FeatureExtractor


def _extract_features_from_eegwindowsdataset(
    win_ds: EEGWindowsDataset,
    feature_extractor: FeatureExtractor,
    target_name: str = "target",
    ):

    features_dict = dict()
    for X, y, _ in win_ds:
        win_dict = dict()
        win_dict.update(feature_extractor(X))
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
        target_name=target_name
        )


def extract_features(concat_dataset: BaseConcatDataset,
                     feature_extractor: FeatureExtractor,
                     target_name: str = "target",
                     n_jobs=1,
                     ):

    feature_ds_list = Parallel(n_jobs)(
        delayed(_extract_features_from_eegwindowsdataset)(
            win_ds, feature_extractor, target_name)
        for win_ds in concat_dataset.datasets
    )
    return FeaturesConcatDataset(feature_ds_list)
