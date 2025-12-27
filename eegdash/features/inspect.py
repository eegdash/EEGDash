r"""
Feature Bank Inspection and Discovery.

This module provides utilities for introspecting the feature extraction 
registry. It allows users and system components to discover available 
features, identify their kinds, and traverse the preprocessing dependency 
graph.

The module provides the following utilities:
- :func:`get_all_features` — Lists all final feature functions.
- :func:`get_all_feature_preprocessors` — Lists all available preprocessing 
  steps.
- :func:`get_feature_kind` — Identifies the dimensionality of a feature.
- :func:`get_feature_predecessors` — Traces the dependency lineage of a 
  feature.
- :func:`get_all_feature_kinds` — Lists all valid feature categories.

"""
from __future__ import annotations

import inspect
from collections.abc import Callable

from . import extractors, feature_bank
from .extractors import _get_underlying_func

__all__ = [
    "get_all_feature_extractors",
    "get_all_feature_preprocessors",
    "get_all_feature_kinds",
    "get_all_features",
    "get_feature_kind",
    "get_feature_predecessors",
]


def get_feature_predecessors(feature_or_extractor: Callable | None) -> list:
    r"""Get the dependency hierarchy for a feature or feature extractor.

    This function recursively traverses the `parent_extractor_type` attribute 
    of a feature or extractor to build a list representing its dependency 
    lineage.

    Parameters
    ----------
    feature_or_extractor : callable
        The feature function or 
        :class:`~eegdash.features.extractors.FeatureExtractor` instance 
        to inspect.

    Returns
    -------
    list
        A nested list representing the dependency tree. For a simple linear 
        chain, this will be a flat list from the specific feature up to the 
        base signal input. For multiple dependencies, it contains tuples 
        of sub-dependencies.

    Notes
    -----
    The traversal stops when it reaches a predecessor of ``None``, which 
    typically represents the raw signal.

    Examples
    --------
    >>> # Example: Linear dependency with a branching dependency
    >>> print(get_feature_predecessors(feature_bank.spectral_entropy))
        [<function spectral_entropy at 0x...>,
        <function spectral_normalized_preprocessor at 0x...>,
        <function spectral_preprocessor at 0x...>,
        (None, [<function signal_hilbert_preprocessor at 0x...>, None])]

    """
    current = feature_or_extractor
    if current is None:
        return [None]
    if isinstance(current, extractors.FeatureExtractor):
        current = current.preprocessor
    current = _get_underlying_func(feature_or_extractor)
    predecessor = getattr(current, "parent_extractor_type", [None])
    if len(predecessor) == 1:
        return [current, *get_feature_predecessors(predecessor[0])]
    else:
        predecessors = [get_feature_predecessors(pred) for pred in predecessor]
        for i in range(len(predecessors)):
            if isinstance(predecessors[i], list) and len(predecessors[i]) == 1:
                predecessors[i] = predecessors[i][0]
        return [current, tuple(predecessors)]


def get_feature_kind(feature: Callable) -> extractors.MultivariateFeature:
    r"""Get the 'kind' of a feature function.

    Identifies whether a feature is univariate, bivariate, or multivariate 
    using decorators.

    Parameters
    ----------
    feature : callable
        The feature function to inspect.

    Returns
    -------
    :class:`~eegdash.features.extractors.MultivariateFeature`
        An instance of the feature kind.

    """
    return _get_underlying_func(feature).feature_kind


def get_all_features() -> list[tuple[str, Callable]]:
    r"""Get a list of all available feature functions.

    Scans the :mod:`~eegdash.features.feature_bank` module for functions 
    that have been decorated with a `feature_kind`.

    Returns
    -------
    list of tuple
        A list of (name, function) tuples for all discovered feature functions.
    """

    def isfeature(x):
        return hasattr(_get_underlying_func(x), "feature_kind")

    return inspect.getmembers(feature_bank, isfeature)


def get_all_feature_extractors() -> list[tuple[str, Callable]]:
    r"""Get a list of all available feature extractor callables.

    A feature extractor is any callable in the feature bank that declares 
    a ``parent_extractor_type``. This includes both intermediate 
    preprocessors and final feature functions.

    Returns
    -------
    list of tuple
        A list of (name, callable) tuples for all discovered feature 
        extractors.
    """

    def isfeatureextractor(x):
        y = _get_underlying_func(x)
        return callable(y) and hasattr(y, "parent_extractor_type")

    return inspect.getmembers(feature_bank, isfeatureextractor)


def get_all_feature_preprocessors() -> list[tuple[str, Callable]]:
    r"""Get a list of all available preprocessor functions.

    Scans the :mod:`~eegdash.features.feature_bank` module for all functions 
    that participate in the dependency graph but do not produce final 
    features (e.g., lack a `feature_kind`).

    Returns
    -------
    list of tuple
        A list of (name, function) tuples for all discovered feature 
        preprocessors.
    """

    def isfeatureextractor(x):
        y = _get_underlying_func(x)
        return (
            callable(y)
            and not hasattr(y, "feature_kind")
            and hasattr(y, "parent_extractor_type")
        )

    return inspect.getmembers(feature_bank, isfeatureextractor)


def get_all_feature_kinds() -> list[tuple[str, type[extractors.MultivariateFeature]]]:
    r"""Get a list of all available feature 'kind' classes.

    Scans the :mod:`~eegdash.features.extractors` module for all classes 
    that subclass :class:`~eegdash.features.extractors.MultivariateFeature`.
    
    Returns
    -------
    list of tuple
        A list of (name, class) tuples for all discovered feature kinds.
    """

    def isfeaturekind(x):
        return inspect.isclass(x) and issubclass(x, extractors.MultivariateFeature)

    return inspect.getmembers(extractors, isfeaturekind)
