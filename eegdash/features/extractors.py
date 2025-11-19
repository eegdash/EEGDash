from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Dict

import numpy as np
from numba.core.dispatcher import Dispatcher

__all__ = [
    "BivariateFeature",
    "DirectedBivariateFeature",
    "FeatureExtractor",
    "MultivariateFeature",
    "TrainableFeature",
    "UnivariateFeature",
]


def _get_underlying_func(func: Callable) -> Callable:
    """Get the underlying function from a potential wrapper.

    This helper unwraps functions that might be wrapped by `functools.partial`
    or `numba.dispatcher.Dispatcher`.

    Parameters
    ----------
    func : callable
        The function to unwrap.

    Returns
    -------
    callable
        The underlying Python function.

    """
    f = func
    if isinstance(f, partial):
        f = f.func
    if isinstance(f, Dispatcher):
        f = f.py_func
    return f


class TrainableFeature(ABC):
    """Abstract base class for features that require training.

    This ABC defines the interface for feature extractors that need to be
    fitted on data before they can be used. It includes methods for fitting
    the feature extractor and for resetting its state.
    """

    def __init__(self):
        self._is_trained = False
        self.clear()

    @abstractmethod
    def clear(self):
        """Reset the internal state of the feature extractor."""
        pass

    @abstractmethod
    def partial_fit(self, *x, y=None):
        """Update the feature extractor's state with a batch of data.

        Parameters
        ----------
        *x : tuple
            The input data for fitting.
        y : any, optional
            The target data, if required for supervised training.

        """
        pass

    def fit(self):
        """Finalize the training of the feature extractor.

        This method should be called after all data has been seen via
        `partial_fit`. It marks the feature as fitted.
        """
        self._is_fitted = True

    def __call__(self, *args, **kwargs):
        """Check if the feature is fitted before execution."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__} cannot be called, it has to be fitted first."
            )


class FeatureExtractor(TrainableFeature):
    """A composite feature extractor that applies multiple feature functions.

    This class orchestrates the application of a dictionary of feature
    extraction functions to input data. It can handle nested extractors,
    pre-processing, and trainable features.

    Parameters
    ----------
    feature_extractors : dict[str, callable]
        A dictionary where keys are feature names and values are the feature
        extraction functions or other `FeatureExtractor` instances.
    preprocessor
        A shared preprocessing function for all child feature extraction functions.

    """

    def __init__(
        self,
        feature_extractors: Dict[str, Callable],
        preprocessor: Callable | None = None,
    ):
        self.preprocessor = preprocessor
        self.feature_extractors_dict = self._validate_execution_tree(feature_extractors)
        self._is_trainable = self._check_is_trainable(feature_extractors)
        super().__init__()

        # bypassing FeaturePredecessor to avoid circular import
        if not hasattr(self, "parent_extractor_type"):
            self.parent_extractor_type = [None]

        self.features_kwargs = dict()
        if preprocessor is not None and isinstance(preprocessor, partial):
            self.features_kwargs["preprocess_kwargs"] = preprocessor.args
        for fn, fe in feature_extractors.items():
            if isinstance(fe, FeatureExtractor):
                self.features_kwargs[fn] = fe.features_kwargs
            if isinstance(fe, partial):
                self.features_kwargs[fn] = fe.keywords

    def _validate_execution_tree(self, feature_extractors: dict) -> dict:
        """Validate the feature dependency graph."""
        preprocessor = (
            None
            if self.preprocessor is None
            else _get_underlying_func(self.preprocessor)
        )
        for fname, f in feature_extractors.items():
            if isinstance(f, FeatureExtractor):
                f = f.preprocessor
            f = _get_underlying_func(f)
            pe_type = getattr(f, "parent_extractor_type", [None])
            if preprocessor not in pe_type:
                parent = getattr(preprocessor, "__name__", preprocessor)
                child = getattr(f, "__name__", f)
                raise TypeError(
                    f"Feature '{fname}: {child}' cannot be a child of {parent}"
                )
        return feature_extractors

    def _check_is_trainable(self, feature_extractors: dict) -> bool:
        """Check if any of the contained features are trainable."""
        for fname, f in feature_extractors.items():
            if isinstance(f, FeatureExtractor):
                if f._is_trainable:
                    return True
            elif isinstance(_get_underlying_func(f), TrainableFeature):
                return True
        return False

    def preprocess(self, *x):
        """Apply pre-processing to the input data.

        Parameters
        ----------
        *x : tuple
            Input data.

        Returns
        -------
        tuple
            The pre-processed data.

        """
        if self.preprocessor is None:
            return (*x,)
        else:
            return self.preprocessor(*x)

    def __call__(self, *x, _batch_size=None, _ch_names=None) -> dict:
        """Apply all feature extractors to the input data.

        Parameters
        ----------
        *x : tuple
            Input data.
        _batch_size : int, optional
            The number of samples in the batch.
        _ch_names : list of str, optional
            The names of the channels in the input data.

        Returns
        -------
        dict
            A dictionary where keys are feature names and values are the
            computed feature values.

        """
        assert _batch_size is not None
        assert _ch_names is not None
        if self._is_trainable:
            super().__call__()
        results_dict = dict()
        z = self.preprocess(*x)
        if not isinstance(z, tuple):
            z = (z,)
        for fname, f in self.feature_extractors_dict.items():
            if isinstance(f, FeatureExtractor):
                r = f(*z, _batch_size=_batch_size, _ch_names=_ch_names)
            else:
                r = f(*z)
            f_und = _get_underlying_func(f)
            if hasattr(f_und, "feature_kind"):
                r = f_und.feature_kind(r, _ch_names=_ch_names)
            if not isinstance(fname, str) or not fname:
                fname = getattr(f_und, "__name__", "")
            if isinstance(r, dict):
                prefix = f"{fname}_" if fname else ""
                for k, v in r.items():
                    self._add_feature_to_dict(results_dict, prefix + k, v, _batch_size)
            else:
                self._add_feature_to_dict(results_dict, fname, r, _batch_size)
        return results_dict

    def _add_feature_to_dict(
        self, results_dict: dict, name: str, value: any, batch_size: int
    ):
        """Add a computed feature to the results dictionary."""
        if isinstance(value, np.ndarray):
            assert value.shape[0] == batch_size
        results_dict[name] = value

    def clear(self):
        """Clear the state of all trainable sub-features."""
        if not self._is_trainable:
            return
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.clear()

    def partial_fit(self, *x, y=None):
        """Partially fit all trainable sub-features."""
        if not self._is_trainable:
            return
        z = self.preprocess(*x)
        if not isinstance(z, tuple):
            z = (z,)
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.partial_fit(*z, y=y)

    def fit(self):
        """Fit all trainable sub-features."""
        if not self._is_trainable:
            return
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.fit()
        super().fit()


class MultivariateFeature:
    """A mixin for features that operate on multiple channels.

    This class provides a `__call__` method that converts a feature array into
    a dictionary with named features, where names are derived from channel
    names.
    """

    def __call__(
        self, x: np.ndarray, _ch_names: list[str] | None = None
    ) -> dict | np.ndarray:
        """Convert a feature array to a named dictionary.

        Parameters
        ----------
        x : numpy.ndarray
            The computed feature array.
        _ch_names : list of str, optional
            The list of channel names.

        Returns
        -------
        dict or numpy.ndarray
            A dictionary of named features, or the original array if feature
            channel names cannot be generated.

        """
        assert _ch_names is not None
        f_channels = self.feature_channel_names(_ch_names)
        if isinstance(x, dict):
            r = dict()
            for k, v in x.items():
                r.update(self._array_to_dict(v, f_channels, k))
            return r
        return self._array_to_dict(x, f_channels)

    @staticmethod
    def _array_to_dict(
        x: np.ndarray, f_channels: list[str], name: str = ""
    ) -> dict | np.ndarray:
        """Convert a numpy array to a dictionary with named keys."""
        assert isinstance(x, np.ndarray)
        if not f_channels:
            return {name: x} if name else x
        assert x.shape[1] == len(f_channels), f"{x.shape[1]} != {len(f_channels)}"
        x = x.swapaxes(0, 1)
        prefix = f"{name}_" if name else ""
        names = [f"{prefix}{ch}" for ch in f_channels]
        return dict(zip(names, x))

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Generate feature names based on channel names.

        Parameters
        ----------
        ch_names : list of str
            The names of the input channels.

        Returns
        -------
        list of str
            The names for the output features.

        """
        return []


class UnivariateFeature(MultivariateFeature):
    """A feature kind for operations applied to each channel independently."""

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Return the channel names themselves as feature names."""
        return ch_names


class BivariateFeature(MultivariateFeature):
    """A feature kind for operations on pairs of channels.

    Parameters
    ----------
    channel_pair_format : str, default="{}<>{}"
        A format string used to create feature names from pairs of
        channel names.

    """

    def __init__(self, *args, channel_pair_format: str = "{}<>{}"):
        super().__init__(*args)
        self.channel_pair_format = channel_pair_format

    @staticmethod
    def get_pair_iterators(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Get indices for unique, unordered pairs of channels."""
        return np.triu_indices(n, 1)

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Generate feature names for each pair of channels."""
        return [
            self.channel_pair_format.format(ch_names[i], ch_names[j])
            for i, j in zip(*self.get_pair_iterators(len(ch_names)))
        ]


class DirectedBivariateFeature(BivariateFeature):
    """A feature kind for directed operations on pairs of channels."""

    @staticmethod
    def get_pair_iterators(n: int) -> list[np.ndarray]:
        """Get indices for all ordered pairs of channels (excluding self-pairs)."""
        return [
            np.append(a, b)
            for a, b in zip(np.tril_indices(n, -1), np.triu_indices(n, 1))
        ]
