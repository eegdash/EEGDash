import numpy as np
from typing import Dict, List, Type
from collections.abc import Callable
from functools import partial


class FeatureExtractor:
    def __init__(
        self,
        feature_extractors: Dict[str, Callable],
        channel_names: List[str] | None = None,
        **preprocess_kwargs: Dict
    ):
        self.feature_extractors_dict = self._validate_execution_tree(feature_extractors)
        self.channel_names = channel_names
        self.preprocess_kwargs = preprocess_kwargs
        if self.preprocess_kwargs is None:
            self.preprocess_kwargs = dict()
        self.features_kwargs = {
            "channel_names": channel_names,
            "preprocess_kwargs": preprocess_kwargs,
        }
        for fn, fe in feature_extractors.items():
            if isinstance(fe, FeatureExtractor):
                self.features_kwargs[fn] = fe.features_kwargs
            if isinstance(fe, partial):
                self.features_kwargs[fn] = fe.keywords

    def preprocess(self, *x, **kwargs):
        return (*x,)

    def feature_channel_names(self, n_channels):
        return [""]

    def __call__(self, *x):
        results_dict = dict()
        z = self.preprocess(*x, **self.preprocess_kwargs)
        for fname, f in self.feature_extractors_dict.items():
            r = f(*z)
            if isinstance(r, dict):
                if fname:
                    fname += "_"
                for k, v in r.items():
                    self._add_feature_to_dict(results_dict, fname + k, v)
            else:
                self._add_feature_to_dict(results_dict, fname, r)
        return results_dict

    def _validate_execution_tree(self, feature_extractors):
        for fname, f in feature_extractors.items():
            if isinstance(f, partial):
                f.parent_extractor_type = f.func.parent_extractor_type
            assert type(self) is f.parent_extractor_type
        # TODO: automatic generation of missing links?
        return feature_extractors

    def _add_feature_to_dict(self, results_dict, name, value):
        if isinstance(value, np.ndarray):
            f_channels = self.feature_channel_names(value.shape[0])
            for cname, v in zip(f_channels, value):
                if cname:
                    cname = "_" + cname
                results_dict[name + cname] = v
        else:
            results_dict[name] = value


class ByChannelFeatureExtractor(FeatureExtractor):
    def feature_channel_names(self, n_channels):
        if self.channel_names:
            assert n_channels == len(self.channel_names)
            ch_names = self.channel_names
        else:
            ch_names = [f"ch{i}" for i in range(n_channels)]
        return ch_names


class Feature:
    def __init__(self, parent_extractor_type: Type = FeatureExtractor):
        parent_cls = parent_extractor_type
        if isinstance(parent_cls, Feature):
            parent_cls = parent_cls.parent_extractor_type
        assert issubclass(parent_cls, FeatureExtractor)
        self.parent_extractor_type = parent_cls

    def __call__(self, func: Callable):
        func.parent_extractor_type = self.parent_extractor_type
        return func
