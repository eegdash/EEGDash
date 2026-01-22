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


def _is_feature(x) -> bool:
    """Check if x is a feature (has feature_kind attribute)."""
    return hasattr(_get_underlying_func(x), "feature_kind")


def _is_feature_extractor(x) -> bool:
    """Check if x is a feature extractor (has parent_extractor_type)."""
    y = _get_underlying_func(x)
    return callable(y) and hasattr(y, "parent_extractor_type")


def _is_feature_preprocessor(x) -> bool:
    """Check if x is a preprocessor (has parent_extractor_type but no feature_kind)."""
    y = _get_underlying_func(x)
    return (
        callable(y)
        and not hasattr(y, "feature_kind")
        and hasattr(y, "parent_extractor_type")
    )


def _is_feature_kind(x) -> bool:
    """Check if x is a feature kind class (subclass of MultivariateFeature)."""
    return inspect.isclass(x) and issubclass(x, extractors.MultivariateFeature)


def get_feature_predecessors(feature_or_extractor: Callable | None) -> list:
    """Get the dependency hierarchy for a feature or feature extractor.

    This function recursively traverses the `parent_extractor_type` attribute
    of a feature or extractor to build a list representing its dependency
    lineage.

    Parameters
    ----------
    feature_or_extractor : callable
        The feature function or :class:`~eegdash.features.extractors.FeatureExtractor`
        class to inspect.

    Returns
    -------
    list
        A nested list representing the dependency tree. For a simple linear
        chain, this will be a flat list from the specific feature up to the
        base :class:`~eegdash.features.extractors.FeatureExtractor`. For
        multiple dependencies, it will contain tuples of sub-dependencies.

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
    """Get the 'kind' of a feature function.

    The feature kind (e.g., univariate, bivariate) is typically attached by a
    decorator.

    Parameters
    ----------
    feature : callable
        The feature function to inspect.

    Returns
    -------
    :class:`~eegdash.features.extractors.MultivariateFeature`
        An instance of the feature kind (e.g., ``UnivariateFeature()``).

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
    return inspect.getmembers(feature_bank, _is_feature)


def get_all_feature_extractors() -> list[tuple[str, Callable]]:
    """Get a list of all available feature extractor callables.

    A feature extractor is any callable in the feature bank that participates
    in the feature graph, meaning it declares a ``parent_extractor_type``
    via :class:`~eegdash.features.decorators.FeaturePredecessor`. This
    includes both preprocessors and the final feature functions.

    Returns
    -------
    list[tuple[str, callable]]
        A list of (name, callable) tuples for all discovered feature
        extractors.

    """
    return inspect.getmembers(feature_bank, _is_feature_extractor)


def get_all_feature_preprocessors() -> list[tuple[str, Callable]]:
    """Get a list of all available preprocessor functions.

    Scans the `eegdash.features.feature_bank` module for all preprocessor functions.

    Returns
    -------
    list[tuple[str, Callable]]
        A list of (name, function) tuples for all discovered feature preprocessors.

    """
    return inspect.getmembers(feature_bank, _is_feature_preprocessor)


def get_all_feature_kinds() -> list[tuple[str, type[extractors.MultivariateFeature]]]:
    """Get a list of all available feature 'kind' classes.

    Scans the `eegdash.features.extractors` module for all classes that
    subclass :class:`~eegdash.features.extractors.MultivariateFeature`.

    Returns
    -------
    list[tuple[str, type[eegdash.features.extractors.MultivariateFeature]]]
        A list of (name, class) tuples for all discovered feature kinds.

    """
    return inspect.getmembers(extractors, _is_feature_kind)
