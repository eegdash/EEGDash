from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Dict

import numpy as np
from numba.core.dispatcher import Dispatcher


def _get_underlying_func(func):
    f = func
    if isinstance(f, partial):
        f = f.func
    if isinstance(f, Dispatcher):
        f = f.py_func
    return f


class TrainableFeature(ABC):
    def __init__(self):
        self._is_trained = False
        self.clear()

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def partial_fit(self, *x, y=None):
        pass

    def fit(self):
        self._is_fitted = True

    def __call__(self, *args, **kwargs):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__} cannot be called, it has to be fitted first."
            )


class FeatureExtractor(TrainableFeature):
    def __init__(
        self, feature_extractors: Dict[str, Callable], **preprocess_kwargs: Dict
    ):
        self.feature_extractors_dict = self._validate_execution_tree(feature_extractors)
        self._is_trainable = self._check_is_trainable(feature_extractors)
        super().__init__()

        # bypassing FeaturePredecessor to avoid circular import
        if not hasattr(self, "parent_extractor_type"):
            self.parent_extractor_type = [FeatureExtractor]

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

    def _validate_execution_tree(self, feature_extractors):
        for fname, f in feature_extractors.items():
            f = _get_underlying_func(f)
            pe_type = getattr(f, "parent_extractor_type", [FeatureExtractor])
            assert type(self) in pe_type
        return feature_extractors

    def _check_is_trainable(self, feature_extractors):
        is_trainable = False
        for fname, f in feature_extractors.items():
            if isinstance(f, FeatureExtractor):
                is_trainable = f._is_trainable
            else:
                f = _get_underlying_func(f)
                if isinstance(f, TrainableFeature):
                    is_trainable = True
            if is_trainable:
                break
        return is_trainable

    def preprocess(self, *x, **kwargs):
        return (*x,)

    def feature_channel_names(self, ch_names):
        return [""]

    def __call__(self, *x, _batch_size=None, _ch_names=None):
        assert _batch_size is not None
        assert _ch_names is not None
        if self._is_trainable:
            super().__call__()
        results_dict = dict()
        z = self.preprocess(*x, **self.preprocess_kwargs)
        for fname, f in self.feature_extractors_dict.items():
            if isinstance(f, FeatureExtractor):
                r = f(*z, _batch_size=_batch_size, _ch_names=_ch_names)
            else:
                r = f(*z)
            f = _get_underlying_func(f)
            if hasattr(f, "feature_kind"):
                r = f.feature_kind(r, _ch_names=_ch_names)
            if not isinstance(fname, str) or not fname:
                if isinstance(f, FeatureExtractor) or not hasattr(f, "__name__"):
                    fname = ""
                else:
                    fname = f.__name__
            if isinstance(r, dict):
                if fname:
                    fname += "_"
                for k, v in r.items():
                    self._add_feature_to_dict(results_dict, fname + k, v, _batch_size)
            else:
                self._add_feature_to_dict(results_dict, fname, r, _batch_size)
        return results_dict

    def _add_feature_to_dict(self, results_dict, name, value, batch_size):
        if not isinstance(value, np.ndarray):
            results_dict[name] = value
        else:
            assert value.shape[0] == batch_size
            results_dict[name] = value

    def clear(self):
        if not self._is_trainable:
            return
        for fname, f in self.feature_extractors_dict.items():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.clear()

    def partial_fit(self, *x, y=None):
        if not self._is_trainable:
            return
        z = self.preprocess(*x, **self.preprocess_kwargs)
        for fname, f in self.feature_extractors_dict.items():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.partial_fit(*z, y=y)

    def fit(self):
        if not self._is_trainable:
            return
        for fname, f in self.feature_extractors_dict.items():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.fit()
        super().fit()


class MultivariateFeature:
    def __call__(self, x, _ch_names=None):
        assert _ch_names is not None
        f_channels = self.feature_channel_names(_ch_names)
        if isinstance(x, dict):
            r = dict()
            for k, v in x.items():
                r.update(self._array_to_dict(v, f_channels, k))
            return r
        return self._array_to_dict(x, f_channels)

    @staticmethod
    def _array_to_dict(x, f_channels, name=""):
        assert isinstance(x, np.ndarray)
        if len(f_channels) == 0:
            assert x.ndim == 1
            if name:
                return {name: x}
            return x
        assert x.shape[1] == len(f_channels)
        x = x.swapaxes(0, 1)
        names = [f"{name}_{ch}" for ch in f_channels] if name else f_channels
        return dict(zip(names, x))

    def feature_channel_names(self, ch_names):
        return []


class UnivariateFeature(MultivariateFeature):
    def feature_channel_names(self, ch_names):
        return ch_names


class BivariateFeature(MultivariateFeature):
    def __init__(self, *args, channel_pair_format="{}<>{}"):
        super().__init__(*args)
        self.channel_pair_format = channel_pair_format

    @staticmethod
    def get_pair_iterators(n):
        return np.triu_indices(n, 1)

    def feature_channel_names(self, ch_names):
        return [
            self.channel_pair_format.format(ch_names[i], ch_names[j])
            for i, j in zip(*self.get_pair_iterators(len(ch_names)))
        ]


class DirectedBivariateFeature(BivariateFeature):
    @staticmethod
    def get_pair_iterators(n):
        return [
            np.append(a, b)
            for a, b in zip(np.tril_indices(n, -1), np.triu_indices(n, 1))
        ]
