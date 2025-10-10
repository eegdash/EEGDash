from __future__ import annotations

import inspect
from collections.abc import Callable

from . import extractors, feature_bank
from .extractors import FeatureExtractor, MultivariateFeature, _get_underlying_func


def get_feature_predecessors(feature_or_extractor: Callable) -> list:
    """Get the dependency hierarchy for a feature or feature extractor.

    This function recursively traverses the `parent_extractor_type` attribute
    of a feature or extractor to build a list representing its dependency
    lineage.

    Parameters
    ----------
    feature_or_extractor : callable
        The feature function or :class:`FeatureExtractor` class to inspect.

    Returns
    -------
    list
        A nested list representing the dependency tree. For a simple linear
        chain, this will be a flat list from the specific feature up to the
        base `FeatureExtractor`. For multiple dependencies, it will contain
        tuples of sub-dependencies.

    """
    current = _get_underlying_func(feature_or_extractor)
    if current is FeatureExtractor:
        return [current]
    predecessor = getattr(current, "parent_extractor_type", [FeatureExtractor])
    if len(predecessor) == 1:
        return [current, *get_feature_predecessors(predecessor[0])]
    else:
        predecessors = [get_feature_predecessors(pred) for pred in predecessor]
        for i in range(len(predecessors)):
            if isinstance(predecessors[i], list) and len(predecessors[i]) == 1:
                predecessors[i] = predecessors[i][0]
        return [current, tuple(predecessors)]


def get_feature_kind(feature: Callable) -> MultivariateFeature:
    """Get the 'kind' of a feature function.

    The feature kind (e.g., univariate, bivariate) is typically attached by a
    decorator.

    Parameters
    ----------
    feature : callable
        The feature function to inspect.

    Returns
    -------
    MultivariateFeature
        An instance of the feature kind (e.g., `UnivariateFeature()`).

    """
    return _get_underlying_func(feature).feature_kind


def get_all_features() -> list[tuple[str, Callable]]:
    """Get a list of all available feature functions.

    Scans the `eegdash.features.feature_bank` module for functions that have
    been decorated to have a `feature_kind` attribute.

    Returns
    -------
    list[tuple[str, callable]]
        A list of (name, function) tuples for all discovered features.

    """

    def isfeature(x):
        return hasattr(_get_underlying_func(x), "feature_kind")

    return inspect.getmembers(feature_bank, isfeature)


def get_all_feature_extractors() -> list[tuple[str, type[FeatureExtractor]]]:
    """Get a list of all available `FeatureExtractor` classes.

    Scans the `eegdash.features.feature_bank` module for all classes that
    subclass :class:`~eegdash.features.extractors.FeatureExtractor`.

    Returns
    -------
    list[tuple[str, type[FeatureExtractor]]]
        A list of (name, class) tuples for all discovered feature extractors,
        including the base `FeatureExtractor` itself.

    """

    def isfeatureextractor(x):
        return inspect.isclass(x) and issubclass(x, FeatureExtractor)

    return [
        ("FeatureExtractor", FeatureExtractor),
        *inspect.getmembers(feature_bank, isfeatureextractor),
    ]


def get_all_feature_kinds() -> list[tuple[str, type[MultivariateFeature]]]:
    """Get a list of all available feature 'kind' classes.

    Scans the `eegdash.features.extractors` module for all classes that
    subclass :class:`~eegdash.features.extractors.MultivariateFeature`.

    Returns
    -------
    list[tuple[str, type[MultivariateFeature]]]
        A list of (name, class) tuples for all discovered feature kinds.

    """

    def isfeaturekind(x):
        return inspect.isclass(x) and issubclass(x, MultivariateFeature)

    return inspect.getmembers(extractors, isfeaturekind)
