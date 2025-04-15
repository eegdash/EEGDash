from typing import Dict, List, Type
from collections.abc import Callable
from functools import partial
import numpy as np


class FeatureExtractor:
    def __init__(
        self, feature_extractors: Dict[str, Callable], **preprocess_kwargs: Dict
    ):
        self.feature_extractors_dict = self._validate_execution_tree(feature_extractors)
        self.preprocess_kwargs = preprocess_kwargs
        if self.preprocess_kwargs is None:
            self.preprocess_kwargs = dict()
        self.features_kwargs = {
            "preprocess_kwargs": preprocess_kwargs,
        }
        for fn, fe in feature_extractors.items():
            if isinstance(fe, FeatureExtractor):
                self.features_kwargs[fn] = fe.features_kwargs
            if isinstance(fe, partial):
                self.features_kwargs[fn] = fe.keywords

    def preprocess(self, *x, **kwargs):
        return (*x,)

    def feature_channel_names(self, ch_names):
        return [""]

    def __call__(self, ch_names, *x):
        f_channels = self.feature_channel_names(ch_names)
        results_dict = dict()
        z = self.preprocess(*x, **self.preprocess_kwargs)
        for fname, f in self.feature_extractors_dict.items():
            if isinstance(f, FeatureExtractor):
                r = f(ch_names, *z)
            else:
                r = f(*z)
            if not isinstance(fname, str) or not fname:
                fun = f.func if isinstance(f, partial) else f
                if isinstance(fun, FeatureExtractor) or not hasattr(fun, "__name__"):
                    fname = ""
                else:
                    fname = fun.__name__
            if isinstance(r, dict):
                if fname:
                    fname += "_"
                for k, v in r.items():
                    self._add_feature_to_dict(results_dict, fname + k, v, f_channels)
            else:
                self._add_feature_to_dict(results_dict, fname, r, f_channels)
        return results_dict

    def _validate_execution_tree(self, feature_extractors):
        for fname, f in feature_extractors.items():
            if isinstance(f, partial):
                f.parent_extractor_type = f.func.parent_extractor_type
            assert type(self) in f.parent_extractor_type
        return feature_extractors

    def _add_feature_to_dict(self, results_dict, name, value, f_channels):
        if isinstance(value, np.ndarray):
            assert value.shape[0] == len(f_channels)
            for cname, v in zip(f_channels, value):
                if cname:
                    cname = "_" + cname
                results_dict[name + cname] = v
        else:
            results_dict[name] = value


class Feature:
    def __init__(self, *parent_extractor_type: List[Type]):
        parent_cls = parent_extractor_type
        if not parent_cls:
            parent_cls = [FeatureExtractor]
        for i, p_cls in enumerate(parent_cls):
            if isinstance(p_cls, Feature):
                parent_cls[i] = p_cls.parent_extractor_type
            assert issubclass(p_cls, FeatureExtractor)
        self.parent_extractor_type = parent_cls

    def __call__(self, func: Callable):
        func.parent_extractor_type = self.parent_extractor_type
        return func


@Feature(FeatureExtractor)
class ByChannelFeatureExtractor(FeatureExtractor):
    def feature_channel_names(self, ch_names):
        return ch_names


@Feature(FeatureExtractor)
class ByChannelPairFeatureExtractor(ByChannelFeatureExtractor):
    def __init__(
        self, feature_extractors, *, channel_pair_format="{}<>{}", **preprocess_kwargs
    ):
        super().__init__(feature_extractors, **preprocess_kwargs)
        self.channel_pair_format = channel_pair_format
        self.features_kwargs["channel_pair_format"] = channel_pair_format

    @staticmethod
    def get_pair_iterators(n):
        return np.triu_indices(n, 1)

    def feature_channel_names(self, ch_names):
        return [
            self.channel_pair_format.format(ch_names[i], ch_names[j])
            for i, j in zip(
                *ByChannelPairFeatureExtractor.get_pair_iterators(len(ch_names))
            )
        ]
