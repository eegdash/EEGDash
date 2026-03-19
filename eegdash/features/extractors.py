r"""Core Feature Extraction Orchestration.

This module defines the fundamental building blocks for creating feature
extraction pipelines.

The module provides the base classes:
    * :class:`FeatureExtractor` - The central pipeline for execution trees.
    * :class:`TrainableFeature` - The interface for features requiring a
      fitting phase.
    * :class:`MultivariateFeature` and its subclasses - Logic for mapping
      raw arrays to named features.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import FunctionType
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
    "BasePreprocessorOutputType",
]


def _get_underlying_func(func: Callable) -> Callable:
    r"""Retrieve the original Python function from a potential wrapper.

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


def _func_to_dict(func: FunctionType | partial) -> dict:
    """Dumps a function to a dictionary.

    Parameters
    ----------
    func : FunctionType | functools.partial
        A function

    Returns
    -------
    dict
        A dictionary representing the function, containing its name, as well
        as its arguments and keyword arguments (for partial functions).

    See Also
    --------
    ~eegdash.features.serialization._func_from_dict

    """
    func_dict = {"name": _get_underlying_func(func).__name__}
    if isinstance(func, partial):
        if func.args:
            func_dict["args"] = list(func.args)
        if func.keywords:
            func_dict["kwargs"] = func.keywords
    return func_dict


class BasePreprocessorOutputType(ABC):
    """An abstract class representing a type of preprocessor output.

    Parameters
    ----------
    preprocessor : callable
        The underlying preprocessor callable.

    """

    def __init__(self, preprocessor: Callable):
        super().__init__()
        self.preprocessor = preprocessor
        uf_preprocessor = _get_underlying_func(preprocessor)
        if hasattr(uf_preprocessor, "parent_extractor_type"):
            self.parent_extractor_type = preprocessor.parent_extractor_type
        if hasattr(uf_preprocessor, "feature_kind"):
            self.feature_kind = preprocessor.feature_kind
        if hasattr(uf_preprocessor, "__name__"):
            self.__name__ = preprocessor.__name__

    def __call__(self, *args, **kwargs):
        r"""Call the underlying preprocessor with the provided arguments."""
        return self.preprocessor(*args, **kwargs)


class TrainableFeature(ABC):
    r"""Abstract base class for features requiring a training phase.

    This class provides the interface for features that must be
    fitted on a representative dataset before they can process new samples.

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
        r"""Reset the internal state of the feature.

        This method must be implemented by subclasses to clear any learned
        parameters, statistics, or buffers.
        """
        pass

    @abstractmethod
    def partial_fit(self, *x, y=None):
        r"""Update the extractor's state using a single batch of data.

        This method allows for incremental learning, making it possible to
        train on datasets that are too large to fit into memory at once.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        y : ndarray, optional
            Target labels associated with the batch, required for supervised
            feature extraction methods.

        """
        pass

    def fit(self):
        r"""Finalize the training of the feature extractor.

        This method should be called after the entire training set has been
        processed via :meth:`partial_fit`. It transitions the object to a
        "trained" state, enabling the :meth:`__call__` method.
        """
        self._is_trained = True

    def __call__(self, *args, **kwargs):
        r"""Validate the fitted state before execution.

        Raises
        ------
        RuntimeError
            If the feature is called before :meth:`fit` has been executed.

        """
        if not self._is_trained:
            raise RuntimeError(
                f"{self.__class__} cannot be called, it has to be trained first."
            )


class FeatureExtractor(TrainableFeature):
    r"""Pipeline for multi-stage feature extraction.

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
        r"""Validate the consistency of the feature dependency graph.

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
                is_valid_by_output_type = False
                for pet in pe_type:
                    if (
                        inspect.isclass(pet)
                        and issubclass(pet, BasePreprocessorOutputType)
                        and pet is not BasePreprocessorOutputType
                        and isinstance(preprocessor, pet)
                    ):
                        is_valid_by_output_type = True
                        break
                if not is_valid_by_output_type:
                    parent = getattr(preprocessor, "__name__", preprocessor)
                    child = getattr(f, "__name__", f)
                    raise TypeError(
                        f"Feature '{fname}: {child}' cannot be a child of {parent}"
                    )
        return feature_extractors

    def _check_is_trainable(self, feature_extractors: dict) -> bool:
        r"""Scan the execution tree for components requiring training.

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

    def preprocess(self, *x, _metadata):
        r"""Apply the shared preprocessor to the input data.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        _metadata : dict
            A dictionary of record and batch metadata.

        Returns
        -------
        tuple
            The preprocessed data passed as a tuple to support multi-output
            preprocessors.

        """
        if self.preprocessor is None:
            return (*x,)
        elif "_metadata" in inspect.signature(self.preprocessor).parameters:
            return self.preprocessor(*x, _metadata=_metadata)
        else:
            return self.preprocessor(*x)

    def __call__(self, *x, _metadata) -> dict:
        r"""Execute the full extraction pipeline on a batch of data.

        This method applies preprocessing, executes all child extractors,
        maps results to channel names, and flattens the output.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        _metadata : dict
            A dictionary of record and batch metadata.

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
        assert _metadata is not None
        if self._is_trainable:
            super().__call__()
        results_dict = dict()
        z = self.preprocess(*x, _metadata=_metadata)
        if not isinstance(z, tuple):
            z = (z,)
        for fname, f in self.feature_extractors_dict.items():
            if (
                isinstance(f, FeatureExtractor)
                or "_metadata" in inspect.signature(f).parameters
            ):
                r = f(*z, _metadata=_metadata)
            else:
                r = f(*z)
            f_und = _get_underlying_func(f)
            if hasattr(f_und, "feature_kind"):
                r = f_und.feature_kind(r, _ch_names=_metadata["info"]["ch_names"])
            if not isinstance(fname, str) or not fname:
                fname = getattr(f_und, "__name__", "")
            if isinstance(r, dict):
                prefix = f"{fname}_" if fname else ""
                for k, v in r.items():
                    self._add_feature_to_dict(
                        results_dict, prefix + k, v, _metadata["batch_size"]
                    )
            else:
                self._add_feature_to_dict(
                    results_dict, fname, r, _metadata["batch_size"]
                )
        return results_dict

    def _add_feature_to_dict(
        self, results_dict: dict, name: str, value: any, batch_size: int
    ):
        r"""Safely add a feature array to the results collection.

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
        r"""Clear the state of all trainable sub-features."""
        if not self._is_trainable:
            return
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.clear()

    def partial_fit(self, *x, y=None, _metadata):
        r"""Propagate partial fitting to all trainable children.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        y : ndarray, optional
            Target labels for supervised training.
        _metadata : dict
            A dictionary of record and batch metadata.

        """
        if not self._is_trainable:
            return
        z = self.preprocess(*x, _metadata=_metadata)
        if not isinstance(z, tuple):
            z = (z,)
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                if (
                    isinstance(f, FeatureExtractor)
                    or "_metadata" in inspect.signature(f).parameters
                ):
                    f.partial_fit(*z, y=y, _metadata=_metadata)
                else:
                    f.partial_fit(*z, y=y)

    def fit(self):
        r"""Fit all trainable sub-features."""
        if not self._is_trainable:
            return
        for f in self.feature_extractors_dict.values():
            f = _get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.fit()
        super().fit()

    def to_dict(self) -> dict:
        """Dumps the feature extractor to a dictionary.

        Returns
        -------
        dict
            A dictionary representing the feature extractor, with
            `"feature_extractors"` and `"preprocessor"` fields (if applicable).

        See Also
        --------
        ~eegdash.features.serialization.feature_extractor_from_dict

        Notes
        -----
        Feature extractors including non-function callables are not supported.

        """
        fe_dict = {}
        if self.preprocessor is not None:
            fe_dict["preprocessor"] = _func_to_dict(self.preprocessor)
        fes = {}
        for k, v in self.feature_extractors_dict.items():
            if isinstance(v, FeatureExtractor):
                fes[k] = v.to_dict()
            else:
                fes[k] = _func_to_dict(v)
        fe_dict["feature_extractors"] = fes
        return fe_dict

    def to_json(self, path: str | Path):
        """Dumps the feature extractor to a json file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the json file.

        See Also
        --------
        ~eegdash.features.serialization.load_feature_extractor_from_json

        Notes
        -----
        Feature extractors including non-function callables are not supported.

        """
        import json

        # Verify work with a pathlib.Path
        path = Path(path)

        with open(path, "w") as file:
            json.dump(self.to_dict(), file, sort_keys=False, indent=4)

    def to_yaml(self, path: str | Path):
        """Dumps the feature extractor to a yaml file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the yaml file.

        See Also
        --------
        ~eegdash.features.serialization.load_feature_extractor_from_yaml

        Notes
        -----
        - Feature extractors including non-function callables are not
           supported.
        - Requires the `pyyaml` package.

        """
        import yaml

        # Verify work with a pathlib.Path
        path = Path(path)

        with open(path, "w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)

    def to_hocon(self, path: str | Path):
        """Dumps the feature extractor to a HOCON's conf file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the conf file.

        See Also
        --------
        ~eegdash.features.serialization.load_feature_extractor_from_hocon

        Notes
        -----
        - Feature extractors including non-function callables are not
           supported.
        - Requires the `pyhocon` package.

        """
        from pyhocon import ConfigFactory, HOCONConverter

        # Verify work with a pathlib.Path
        path = Path(path)

        with open(path, "w") as outfile:
            outfile.write(
                HOCONConverter.to_hocon(ConfigFactory.from_dict(self.to_dict()))
            )


class MultivariateFeature:
    r"""Logic wrapper for features that operate on one or more EEG channels.

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
        r"""Convert a raw feature array into a named dictionary.

        Parameters
        ----------
        x : numpy.ndarray
            The computed feature array from the extraction function.
        _ch_names : list of str, optional
            The list of channel names from the original recording.

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
        r"""Map a numpy array to a dictionary with named keys.

        Parameters
        ----------
        x : numpy.ndarray
            The feature values to be mapped.
        f_channels : list of str
            The list of generated feature channel names.
        name : str, default=""
            A prefix for the feature name.

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
        r"""Generate feature-specific names based on input channels.

        Parameters
        ----------
        ch_names : list of str
            The names of the input channels.

        Returns
        -------
        list of str
            A list of strings defining the naming for each output feature.
            Returns an empty list in the base implementation.

        """
        return []


class UnivariateFeature(MultivariateFeature):
    r"""Feature kind for operations applied to each channel independently.

    Used when a single feature value is produced per channel.
    """

    def feature_channel_names(self, ch_names: list[str]) -> list[str]:
        r"""Return the channel names themselves as feature names."""
        return ch_names


class BivariateFeature(MultivariateFeature):
    r"""Feature kind for operations on pairs of channels.

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
        r"""Get indices for unique, unordered pairs of channels.

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
        r"""Generate feature names for each unique pair of channels.

        Parameters
        ----------
        ch_names : list of str
            The input channel names.

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
    r"""Feature kind for directed operations on pairs of channels.

    Used for features where the interaction from channel A to B is
    distinct from the interaction from B to A.
    """

    @staticmethod
    def get_pair_iterators(n: int) -> list[np.ndarray]:
        r"""Get indices for all ordered pairs of channels.

        Includes both directions while excluding self-pairs.

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
