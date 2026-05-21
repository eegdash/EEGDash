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
from typing import Dict, Tuple, Type

import numpy as np

from .base_utils import get_underlying_func
from .output_types import (
    AsInputOutputType,
    BasePreprocessorOutputType,
    SignalOutputType,
)
from .trainable import TrainableFeature

__all__ = [
    "FeatureExtractor",
]


def _get_func_name(func: Callable) -> str:
    """Get the name of a function or callable object.

    Parameters
    ----------
    func: Callable
        A function or a callable object.

    Returns
    -------
    str
        The name of the function or callable object.

    """
    func = get_underlying_func(func)
    if hasattr(func, "__name__"):
        return func.__name__
    if hasattr(func, "__class__"):
        return func.__class__.__name__
    return str(func)


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
    func_dict = {"name": _get_func_name(func)}
    if isinstance(func, partial):
        if func.args:
            func_dict["args"] = list(func.args)
        if func.keywords:
            func_dict["kwargs"] = func.keywords
    return func_dict


def _adjust_list_types(l: list) -> list:
    """Adjust a dictionary keys so they can be saved to config files.

    Parameters
    ----------
    l : list
        The list to adjust.

    Returns
    -------
    list
        The adjusted list.

    """
    ll = []
    for v in l:
        # values
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, list):
            v = _adjust_list_types(v)
        if isinstance(v, dict):
            v = _adjust_dict_types(v)
        ll.append(v)
    return ll


def _adjust_dict_types(d: dict) -> dict:
    """Adjust a dictionary keys and values so they can be saved to config files.

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
        if isinstance(v, list):
            v = _adjust_list_types(v)
        if isinstance(v, dict):
            v = _adjust_dict_types(v)
        # keys
        if not isinstance(k, str):
            k = str(k)
        elif k == "":
            k = "_"
        dd[k] = v
    return dd


def _tree_to_str(d: dict, s=None, prefix: str = "") -> str:
    """Convert a tree-style dict to directory-style string.

    Parameters
    ----------
    d: dict
        A tree-like dictionary.
    s:
        A stem object (optional).
    prefix: str
        A common prefix for all rows.

    Returns
    -------
    str
        A directory-tree-like string representing the tree-like dictionary.

    """
    out_str = []
    if s is not None:
        out_str.append(f"{s}")
    for i, (k, v) in enumerate(d.items()):
        conn = "╠═ " if i < len(d) - 1 else "╚═ "
        if isinstance(v, tuple):
            v, p = v
            if p is not None:
                out_str.append(f"{prefix}{conn}{k}: {p}")
        else:
            p = None
        if isinstance(v, dict):
            if p is None:
                out_str.append(f"{prefix}{conn}{k}:")
            out_str.append(
                _tree_to_str(v, prefix=prefix + ("║  " if i < len(d) - 1 else "   "))
            )
        else:
            out_str.append(f"{prefix}{conn}{k}: {v}")
    return "\n".join(out_str)


def _call_with_metadata(func: Callable, *x, _metadata: dict) -> Tuple[tuple, dict]:
    """Calls a feature or preprocessor with metadata if required.

    Parameters
    ----------
    func : Callable
        A function to be called.
    *x : tuple[Any]
        Positional arguments to pass to `func`.
    _metadata : dict
        A metadata dictionary to pass into `func` if `func` receives
        a keyword argument called `_metadata`.

    Returns
    -------
    *z : tuple[Any]
        `func`s output.
    _metadata : dict
        A metadata dictionary. May be altered by `func` if it is a
        metadata preprocessor.

    """
    f = get_underlying_func(func)
    if (
        isinstance(func, FeatureExtractor)
        or "_metadata" in inspect.signature(func).parameters
    ):
        if hasattr(f, "metadata_preprocessor"):
            *z, _metadata = func(*x, _metadata=_metadata.copy())
            z = (*z,)
        else:
            z = func(*x, _metadata=_metadata)
    else:
        z = func(*x)
    if not isinstance(z, tuple):
        z = (z,)
    if hasattr(f, "output_type"):
        f.output_type.validate_output(*z, _metadata=_metadata)
    return z, _metadata


def _concat_calls(funcs: list[Callable], *x, _metadata: dict) -> tuple:
    """Call a list of callable in a chain.

    Parameters
    ----------
    funcs : list[Callable]
        A list of callable to chain.
    *x : tuple[Any]
        An input batch to the first callable in `funcs`.
    _metadata : dict
        A metadata dictionary to pass to the first callable in `funcs`.

    Returns
    -------
    *z : tuple
        The chained function output.
    _metadata (optional)
        An altered metadata dictionary. Only returned if the last callable
        in `funcs` is a metadata preprocessor.

    """
    z = (*x,)
    for func in funcs:
        z, _metadata = _call_with_metadata(func, *z, _metadata=_metadata)
    if "_metadata" in inspect.signature(funcs[-1]).parameters and hasattr(
        get_underlying_func(funcs[-1]), "metadata_preprocessor"
    ):
        return *z, _metadata
    if len(z) == 1:
        return z[0]
    return z


def _merge_call_list(funcs: list[Callable]) -> list[Callable]:
    r"""Merge a list of callables and chained callables.

    Parameters
    ----------
    funcs : list[Callable]
        A list of callables.

    Returns
    -------
    list[Callables]
        Same as `funcs`, but any chained callable (a partial of
        :func:`_concat_calls`) is replaced by the chained callables themselves.

    """
    func_list = []
    for f in funcs:
        if isinstance(f, partial) and f.func is _concat_calls:
            func_list.extend(f.args[0])
        else:
            func_list.append(f)
    return func_list


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
        self.feature_extractors_dict = feature_extractors
        self._validate_execution_tree()
        self._is_trainable = self._check_is_trainable()
        super().__init__()

        # bypassing feature_predecessor to avoid circular import
        if self.preprocessor is not None:
            f = get_underlying_func(self.preprocessor)
            if hasattr(f, "parent_extractor_type"):
                self.parent_extractor_type = f.parent_extractor_type
        elif not hasattr(self, "parent_extractor_type"):
            self.parent_extractor_type = [SignalOutputType]

        self.features_kwargs = self.to_dict()

    def _validate_execution_tree(self, parent_type: Type | Callable | None = None):
        r"""Validate the consistency of the feature dependency graph.

        Parameters
        ----------
        feature_extractors : dict
            The dictionary of extractors to validate.
        parent_type : Type | Callable | None
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
            if self.preprocessor is None:
                parent_type = SignalOutputType
                is_parent_type_output_type = True
            else:
                preprocessor = get_underlying_func(self.preprocessor)
                if hasattr(preprocessor, "output_type"):
                    if issubclass(preprocessor.output_type, AsInputOutputType):
                        return
                    parent_type = preprocessor.output_type
                    is_parent_type_output_type = True
                else:
                    parent_type = preprocessor
                    is_parent_type_output_type = False

        for fname, f in self.feature_extractors_dict.items():
            fe = None
            if isinstance(f, FeatureExtractor):
                fe = f
                f = f.preprocessor
            f = get_underlying_func(f)
            if hasattr(f, "output_type") and issubclass(
                f.output_type, AsInputOutputType
            ):
                if fe is not None:
                    fe._validate_execution_tree(parent_type)
                    continue
                elif hasattr(parent_type, "feature_kind"):
                    continue
            pe_type = getattr(f, "parent_extractor_type", [SignalOutputType])
            if parent_type in pe_type:
                continue
            is_valid_by_output_type = False
            if is_parent_type_output_type:
                for pet in pe_type:
                    if (
                        inspect.isclass(pet)
                        and issubclass(pet, BasePreprocessorOutputType)
                        and pet is not BasePreprocessorOutputType
                    ):
                        if issubclass(parent_type, pet):
                            is_valid_by_output_type = True
                            break
            if not is_valid_by_output_type:
                parent = _get_func_name(parent_type)
                child = _get_func_name(f)
                raise TypeError(
                    f"Feature '{fname}: {child}' cannot be a child of {parent}"
                )

    def _check_is_trainable(self) -> bool:
        r"""Scan the execution tree for components requiring training.

        Returns
        -------
        bool
            True if any child function or nested extractor is trainable.

        """
        for fname, f in self.feature_extractors_dict.items():
            if isinstance(f, FeatureExtractor):
                if f._is_trainable:
                    return True
            elif isinstance(get_underlying_func(f), TrainableFeature):
                return True
        return False

    def preprocess(self, *x, _metadata: dict) -> tuple:
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
        else:
            z, _metadata = _call_with_metadata(
                self.preprocessor, *x, _metadata=_metadata
            )
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
        preprocessor_f_und = get_underlying_func(self.preprocessor)
        for fname, f in self.feature_extractors_dict.items():
            r, tmp_metadata = _call_with_metadata(f, *z, _metadata=_metadata)
            assert len(r) == 1
            r = r[0]
            f_und = get_underlying_func(f)
            if hasattr(f_und, "feature_kind"):
                r = f_und.feature_kind(r, _metadata=tmp_metadata)
            elif hasattr(preprocessor_f_und, "feature_kind") and not isinstance(
                f_und, FeatureExtractor
            ):
                r = preprocessor_f_und.feature_kind(r, _metadata=tmp_metadata)
            if (not isinstance(fname, str) or not fname) and not isinstance(
                f, FeatureExtractor
            ):
                fname = _get_func_name(f_und)
            if isinstance(r, dict):
                prefix = f"{fname}_" if isinstance(fname, str) and fname else ""
                for k, v in r.items():
                    self._add_feature_to_dict(
                        results_dict, prefix + k, v, tmp_metadata["batch_size"]
                    )
            else:
                self._add_feature_to_dict(
                    results_dict, fname, r, tmp_metadata["batch_size"]
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

    def __len__(self) -> int:
        """Get the number of children features/extractors."""
        return np.sum(
            [
                len(f) if isinstance(f, FeatureExtractor) else 1
                for f in self.feature_extractors_dict.values()
            ]
        )

    def _concat_preprocessor_to_child_func(self, func: Callable) -> Callable:
        r"""Chain the preprocessor (if there is any) in front of a callable.

        Parameters
        ----------
        func : Callable
            A function to chain to the preprocessor.

        Returns
        -------
        Callable
            A chained callable.

        """
        if self.preprocessor is None:
            return func
        return partial(_concat_calls, _merge_call_list([self.preprocessor, func]))

    def __getitem__(self, key) -> Callable:
        """Get a feature/extractor by its key.

        Parameters
        ----------
        key : str or Any
            Either a key from the feature extractors dict, or a string
            representing a path to a node in the execution tree.

        Returns
        -------
        Callable
            A callable receiving an input batch as positional argument and
            `_metadata` as a mandatory keyword argument. The callable chains
            all preprocessors in the execution path of the tree, with the
            required node (either a feature or a preprocessor) on top.

        """
        if key in self.feature_extractors_dict:
            return self._concat_preprocessor_to_child_func(
                self.feature_extractors_dict[key]
            )
        if not isinstance(key, str):
            raise ValueError(
                "Non-string keys are supported only for direct keys.\n"
                + f"Key {key} is not a direct key of the FeatureExtractor.\n"
                + "Possible direct keys are: "
                + f"{self.feature_extractors_dict.keys()}"
            )
        for k, f in self.feature_extractors_dict.items():
            if not isinstance(f, FeatureExtractor):
                continue
            if not isinstance(k, str) or k == "":
                kk = key
            elif key.startswith(k + "_"):
                kk = key[len(k) + 1 :]
            else:
                continue
            try:
                func = f[kk]
            except ValueError:
                continue
            else:
                return self._concat_preprocessor_to_child_func(func)

        raise ValueError(
            f"Key {key} not found in FeatureExtractor.\n"
            + "Possible direct keys are: "
            + f"{self.feature_extractors_dict.keys()}"
        )

    @property
    def feature_names(self) -> list[str]:
        """A list of full feature names (without the channel names)."""
        fnames = []
        for fname, f in self.feature_extractors_dict.items():
            if (not isinstance(fname, str) or not fname) and not isinstance(
                f, FeatureExtractor
            ):
                fname = _get_func_name(get_underlying_func(f))
            if isinstance(f, FeatureExtractor):
                prefix = f"{fname}_" if isinstance(fname, str) and fname else ""
                fnames.extend([prefix + fn for fn in f.feature_names])
            else:
                fnames.append(fname)
        return fnames

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

    def _repr(self) -> Tuple[dict, Callable | None]:
        d = {}
        for k, v in self.feature_extractors_dict.items():
            if isinstance(v, FeatureExtractor):
                v = v._repr()
            d[k] = v
        if self.preprocessor is not None:
            s = self.preprocessor
        else:
            s = None
        return d, s

    def __repr__(self) -> str:
        return _tree_to_str(*self._repr())

    def _str(self) -> Tuple[dict, str | None]:
        d = {}
        for k, v in self.feature_extractors_dict.items():
            if isinstance(v, FeatureExtractor):
                v = v._str()
            else:
                v = _get_func_name(v)
            d[k] = v
        if self.preprocessor is not None:
            s = _get_func_name(self.preprocessor)
        else:
            s = None
        return d, s

    def __str__(self) -> str:
        return _tree_to_str(*self._str())
