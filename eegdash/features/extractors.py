r"""Core Feature Extraction Orchestration.

This module defines the fundamental building blocks for creating feature
extraction pipelines.

The module provides the base class:

- :class:`FeatureExtractor` - The central pipeline for execution trees.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Dict

import numpy as np

from .base_utils import get_underlying_func
from .output_types import AsInputOutputType, BasePreprocessorOutputType
from .trainable import TrainableFeature

__all__ = [
    "FeatureExtractor",
]


def _func_to_dict(func: FunctionType | partial) -> dict:
    r"""Dumps a function to a dictionary.

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
    _func_from_dict

    """
    func_dict = {"name": get_underlying_func(func).__name__}
    if isinstance(func, partial):
        if func.args:
            func_dict["args"] = list(func.args)
        if func.keywords:
            func_dict["kwargs"] = func.keywords
    return func_dict


def _adjust_dict_types(d: dict) -> dict:
    """Adjust a dictionary keys so they can be saved to config files.

    Parameters
    ----------
    d : dict
        The dictionary to adjust.

    Returns
    -------
    dict
        The adjusted dictionary.

    """
    dd = {}
    for k, v in d.items():
        # values
        if isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, dict):
            v = _adjust_dict_types(v)
        # keys
        if not isinstance(k, str):
            k = str(k)
        elif k == "":
            k = "_"
        dd[k] = v
    return dd


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
        self._validate_execution_tree(feature_extractors)
        self.feature_extractors_dict = feature_extractors
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

    def _validate_execution_tree(
        self, feature_extractors: dict, parent_type=None
    ) -> dict:
        r"""Validate the consistency of the feature dependency graph.

        Parameters
        ----------
        feature_extractors : dict
            The dictionary of extractors to validate.
        parent_type :
            Parent preprocessor type (optional).

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
        if parent_type is None:
            preprocessor = (
                None
                if self.preprocessor is None
                else get_underlying_func(self.preprocessor)
            )
            pp_parent_type = getattr(preprocessor, "parent_extractor_type", [None])
            if preprocessor is None or AsInputOutputType in pp_parent_type:
                assert preprocessor is None or len(pp_parent_type) == 1
                return
            parent_type = preprocessor

        for fname, f in feature_extractors.items():
            fe = None
            if isinstance(f, FeatureExtractor):
                fe = f
                f = f.preprocessor
            f = get_underlying_func(f)
            pe_type = getattr(f, "parent_extractor_type", [None])
            if fe is not None and AsInputOutputType in pe_type:
                fe._validate_execution_tree(fe.feature_extractors_dict, parent_type)
                continue
            if parent_type not in pe_type:
                is_valid_by_output_type = False
                for pet in pe_type:
                    if (
                        inspect.isclass(pet)
                        and issubclass(pet, BasePreprocessorOutputType)
                        and pet is not BasePreprocessorOutputType
                        and isinstance(parent_type, pet)
                    ):
                        is_valid_by_output_type = True
                        break
                if not is_valid_by_output_type:
                    parent = getattr(parent_type, "__name__", parent_type)
                    child = getattr(f, "__name__", f)
                    raise TypeError(
                        f"Feature '{fname}: {child}' cannot be a child of {parent}"
                    )

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
            elif isinstance(get_underlying_func(f), TrainableFeature):
                return True
        return False

    def preprocess(self, *x, _metadata: dict):
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
        _metadata: dict
            The preprocessed metadata. Only relevant for metadata preprocessors.

        """
        if self.preprocessor is None:
            z = (*x,)
        elif "_metadata" in inspect.signature(self.preprocessor).parameters:
            if hasattr(get_underlying_func(self.preprocessor), "metadata_preprocessor"):
                *z, _metadata = self.preprocessor(*x, _metadata=_metadata.copy())
                z = (*z,)
            else:
                z = self.preprocessor(*x, _metadata=_metadata)
        else:
            z = self.preprocessor(*x)
        if not isinstance(z, tuple):
            z = (z,)
        return z, _metadata

    def __call__(self, *x, _metadata: dict) -> dict:
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
        z, _metadata = self.preprocess(*x, _metadata=_metadata)
        for fname, f in self.feature_extractors_dict.items():
            if (
                isinstance(f, FeatureExtractor)
                or "_metadata" in inspect.signature(f).parameters
            ):
                r = f(*z, _metadata=_metadata)
            else:
                r = f(*z)
            f_und = get_underlying_func(f)
            if hasattr(f_und, "feature_kind"):
                r = f_und.feature_kind(r, _metadata=_metadata)
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
            f = get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.clear()

    def partial_fit(self, *x, y=None, _metadata: dict):
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
        z, _metadata = self.preprocess(*x, _metadata=_metadata)
        for f in self.feature_extractors_dict.values():
            f = get_underlying_func(f)
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
            f = get_underlying_func(f)
            if isinstance(f, TrainableFeature):
                f.fit()
        super().fit()

    def to_dict(self) -> dict:
        r"""Dumps the feature extractor to a dictionary.

        Returns
        -------
        dict
            A dictionary representing the feature extractor, with
            ``"feature_extractors"`` and ``"preprocessor"`` fields (if applicable).

        See Also
        --------
        feature_extractor_from_dict

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
        return _adjust_dict_types(fe_dict)

    def to_json(self, path: str | Path):
        r"""Dumps the feature extractor to a json file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the json file.

        See Also
        --------
        load_feature_extractor_from_json, FeatureExtractor.to_dict

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
        r"""Dumps the feature extractor to a yaml file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the yaml file.

        See Also
        --------
        load_feature_extractor_from_yaml, FeatureExtractor.to_dict

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
        r"""Dumps the feature extractor to a HOCON's conf file.

        Parameters
        ----------
        path : str | pathlib.Path
            The path to the conf file.

        See Also
        --------
        load_feature_extractor_from_hocon, FeatureExtractor.to_dict

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
