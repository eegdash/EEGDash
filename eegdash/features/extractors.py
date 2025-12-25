"""
Core Feature Extraction Orchestration.

This module defines the fundamental building blocks for creating complex EEG 
feature extraction pipelines. It implements a dependency-aware architecture 
that supports functional feature definitions, trainable extractors, and 
automated result formatting based on channel names.

The module provides the base classes:
    * :class:`FeatureExtractor` - The central pipeline for execution trees.
    * :class:`TrainableFeature` - The interface for features requiring a 
      fitting phase.
    * :class:`MultivariateFeature` and its subclasses - Logic for mapping 
      raw arrays to named features (Univariate, Bivariate, etc.).
"""

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
    """Retrieve the original Python function from a potential wrapper.

    This helper is essential for inspecting metadata (like predecessors or 
    feature kinds) attached to functions that have been transformed by 
    optimization or utility wrappers.

    Parameters
    ----------
    func : callable
        The function to unwrap. Typically a raw function, a 
        :class:`functools.partial` object, or a Numba :class:`Dispatcher`.

    Returns
    -------
    callable
        The underlying Python function.

    Notes
    -----
    This utility specifically handles:
    * **functools.partial**: Returns the ``.func`` attribute.
    * **numba.Dispatcher**: Returns the ``.py_func`` attribute.

    """
    f = func
    if isinstance(f, partial):
        f = f.func
    if isinstance(f, Dispatcher):
        f = f.py_func
    return f


class TrainableFeature(ABC):
    """Abstract base class for features requiring a training phase.

    This class provides the interface for features that must be
    fitted on a representative dataset before they can process new samples. 
    It enforces a workflow of partial fitting across batches followed by a 
    finalization step.

    Attributes
    ----------
    _is_trained : bool
        Internal flag indicating whether the feature has completed 
        its training phase.
    
    """

    def __init__(self):
        self._is_trained = False
        self.clear()

    @abstractmethod
    def clear(self):
        """Reset the internal state of the feature.

        This method must be implemented by subclasses to clear any learned 
        parameters, statistics, or buffers.
        """
        pass

    @abstractmethod
    def partial_fit(self, *x, y=None):
        """Update the extractor's state using a single batch of data.

        This method allows for incremental learning, making it possible to 
        train on datasets that are too large to fit into memory at once.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch. Typically contains the raw EEG windows 
            or preprocessed signal representations.
        y : ndarray, optional
            Target labels associated with the batch, required for supervised 
            feature extraction methods (like CSP).
        """
        pass

    def fit(self):
        """Finalize the training of the feature extractor.

        This method should be called after the entire training set has been 
        processed via :meth:`partial_fit`. It transitions the object to a 
        "fitted" state, enabling the :meth:`__call__` method.
        """
        self._is_fitted = True

    def __call__(self, *args, **kwargs):
        """Validate the fitted state before execution.

        Raises
        ------
        RuntimeError
            If the feature is called before :meth:`fit` has been executed.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__} cannot be called, it has to be fitted first."
            )


class FeatureExtractor(TrainableFeature):
    """Pipeline for multi-stage EEG feature extraction.

    This class manages a collection of feature extraction functions or nested 
    extractors. It handles the application of shared preprocessing, validates 
    the dependency graph between components, and aggregates results into a 
    named dictionary compatible with :class:`FeaturesDataset`.

    Parameters
    ----------
    feature_extractors : dict[str, callable]
        A dictionary where keys are the base names for the features and 
        values are the extraction functions or other :class:`FeatureExtractor` 
        instances.
    preprocessor : callable, optional
        A shared preprocessing function applied to the input data
        before it is passed to child extractors.

    Attributes
    ----------
    preprocessor : callable or None
        The shared preprocessing stage for this extractor.
    feature_extractors_dict : dict
        The validated dictionary of child extractors.
    _is_trainable : bool
        'True' if any of the contained features are trainable
    features_kwargs : dict
        A collection of all keyword arguments used by the preprocessor and 
        child functions, preserved for metadata tracking.

    Notes
    -----
    The extractor automatically detects if any child components are 
    trainable and will require a :meth:`fit` phase before 
    extraction can occur.

    Examples
    --------
    >>> # Create a simple extractor
    >>> fe = FeatureExtractor(
    ...     feature_extractors={'mean': signal_mean, 'std': signal_std}
    ... )

    >>> # Extract from a batch (2 windows, 3 channels, 100 samples)
    >>> X = np.random.randn(2, 3, 100)
    >>> results = fe(X, _batch_size=2, _ch_names=['O1', 'Oz', 'O2'])
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
        """Validate the consistency of the feature dependency graph.

        Parameters
        ----------
        feature_extractors : dict
            The dictionary of extractors to validate.

        Returns
        -------
        dict
            The validated dictionary.

        Raises
        ------
        TypeError
            If a child feature's required predecessors do not match the 
            current preprocessor.
        """
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
        """Scan the execution tree for components requiring training.

        Returns
        -------
        bool
            True if any child function or nested extractor is trainable.
        """
        for fname, f in feature_extractors.items():
            if isinstance(f, FeatureExtractor):
                if f._is_trainable:
                    return True
            elif isinstance(_get_underlying_func(f), TrainableFeature):
                return True
        return False

    def preprocess(self, *x):
        """Apply the shared preprocessor to the input data.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.

        Returns
        -------
        tuple
            The preprocessed data passed as a tuple to support multi-output 
            preprocessors.
        """
        if self.preprocessor is None:
            return (*x,)
        else:
            return self.preprocessor(*x)

    def __call__(self, *x, _batch_size=None, _ch_names=None) -> dict:
        """Execute the full extraction pipeline on a batch of data.

        This method applies preprocessing, executes all child extractors, 
        maps results to channel names, and flattens the output.

        Parameters
        ----------
        *x : tuple of ndarray
            The input EEG data batch.
        _batch_size : int
            The number of windows in the current batch.
        _ch_names : list of str
            The names of the EEG channels.

        Returns
        -------
        dict
            A dictionary where keys are formatted as ``{extractor_name}_{channel}`` 
            and values are the extracted feature arrays.

        Raises
        ------
        RuntimeError
            If the extractor contains trainable components and has not 
            been fitted.

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
        """Safely add a feature array to the results collection.

        Parameters
        ----------
        results_dict : dict
            The dictionary to add features to.
        name : str
            The name of the feature.
        value : any
            The feature value to add.
        batch_size : int
            The expected batch size for validation.
        """
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
        """Propagate partial fitting to all trainable children.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        y : ndarray, optional
            Target labels for supervised training.

        """
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
    """"Mixin** for features that operate on one or more EEG channels.

    This class defines the logic for mapping raw numerical results into 
    structured, named dictionaries. It determines the "kind" of a feature 
    (e.g., univariate, bivariate) and handles the association of feature 
    values with specific channels or channel groupings.

    Notes
    -----
    Subclasses should override :meth:`feature_channel_names` to define 
    specific naming conventions for the extracted features.
    """

    def __call__(
        self, x: np.ndarray, _ch_names: list[str] | None = None
    ) -> dict | np.ndarray:
        """Convert a raw feature array into a named dictionary.

        Parameters
        ----------
        x : numpy.ndarray
            The computed feature array from the extraction function.
        _ch_names : list of str, optional
            The list of channel names from the original EEG recording.

        Returns
        -------
        dict or numpy.ndarray
            A dictionary where keys are formatted feature names and values 
            are feature arrays. Returns the original array if channel names 
            cannot be resolved.
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
        """Map a numpy array to a dictionary with named keys.

        Parameters
        ----------
        x : numpy.ndarray
            The feature values to be mapped.
        f_channels : list of str
            The list of generated feature channel names.
        name : str, default=""
            A prefix for the feature name (e.g., the name of the function).

        Returns
        -------
        dict or numpy.ndarray
            A dictionary of named features or the original array if 
            `f_channels` is empty.
        
        """
        assert isinstance(x, np.ndarray)
        if not f_channels:
            return {name: x} if name else x
        assert x.shape[1] == len(f_channels), f"{x.shape[1]} != {len(f_channels)}"
        x = x.swapaxes(0, 1)
        prefix = f"{name}_" if name else ""
        names = [f"{prefix}{ch}" for ch in f_channels]
        return dict(zip(names, x))

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Generate feature-specific names based on input channels.

        Parameters
        ----------
        ch_names : list of str
            The names of the input EEG channels.

        Returns
        -------
        list of str
            A list of strings defining the naming for each output feature. 
            Returns an empty list in the base implementation.
        """
        return []


class UnivariateFeature(MultivariateFeature):
    """Feature kind for operations applied to each channel independently.

    Used when a single feature value is produced per channel.

    """

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Return the channel names themselves as feature names."""
        return ch_names


class BivariateFeature(MultivariateFeature):
    """Feature kind for operations on pairs of channels.

    Designed for undirected relationship measures between two signals.

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
        """Get indices for unique, unordered pairs of channels.

        Computes the upper triangle indices of an (n, n) matrix, 
        excluding the diagonal.

        Parameters
        ----------
        n : int
            The number of channels.

        Returns
        -------
        tuple of ndarray
            The row and column indices for the unique pairs.
        """
        return np.triu_indices(n, 1)

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        """Generate feature names for each unique pair of channels.

        Parameters
        ----------
        ch_names : list of str
            The input EEG channel names.

        Returns
        -------
        list of str
            Formatted strings representing channel pairs (e.g., 'F3<>F4').
        """
        return [
            self.channel_pair_format.format(ch_names[i], ch_names[j])
            for i, j in zip(*self.get_pair_iterators(len(ch_names)))
        ]


class DirectedBivariateFeature(BivariateFeature):
    """Feature kind for directed operations on pairs of channels.

    Used for features where the interaction from channel A to B is 
    distinct from the interaction from B to A.

    """

    @staticmethod
    def get_pair_iterators(n: int) -> list[np.ndarray]:
        """Get indices for all ordered pairs of channels.

        Includes both directions (A to B and B to A) while excluding 
        self-pairs (A to A).

        Parameters
        ----------
        n : int
            The number of channels.

        Returns
        -------
        list of ndarray
            A list containing two arrays: the row indices and column indices.
        """
        return [
            np.append(a, b)
            for a, b in zip(np.tril_indices(n, -1), np.triu_indices(n, 1))
        ]
