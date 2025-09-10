"""
This module provides functions for inspecting feature extractors and their
properties.

It includes functions for retrieving feature predecessors, kinds, and for
listing all available features and feature extractors.
"""
import inspect
from collections.abc import Callable

from . import extractors, feature_bank
from .extractors import FeatureExtractor, MultivariateFeature, _get_underlying_func


def get_feature_predecessors(feature_or_extractor: Callable):
    """Get the predecessors of a feature or feature extractor.

    This function traverses the feature's inheritance tree to find all
    predecessor classes.

    Parameters
    ----------
    feature_or_extractor : callable
        The feature or feature extractor to inspect.

    Returns
    -------
    list
        A list of predecessor classes.
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


def get_feature_kind(feature: Callable):
    """Get the kind of a feature.

    Parameters
    ----------
    feature : callable
        The feature to inspect.

    Returns
    -------
    type
        The kind of the feature (e.g., UnivariateFeature, BivariateFeature).
    """
    return _get_underlying_func(feature).feature_kind


def get_all_features():
    """Get all available features from the feature bank.

    Returns
    -------
    list
        A list of (name, feature) tuples.
    """

    def isfeature(x):
        return hasattr(_get_underlying_func(x), "feature_kind")

    return inspect.getmembers(feature_bank, isfeature)


def get_all_feature_extractors():
    """Get all available feature extractors from the feature bank.

    Returns
    -------
    list
        A list of (name, extractor) tuples.
    """

    def isfeatureextractor(x):
        return inspect.isclass(x) and issubclass(x, FeatureExtractor)

    return [
        ("FeatureExtractor", FeatureExtractor),
        *inspect.getmembers(feature_bank, isfeatureextractor),
    ]


def get_all_feature_kinds():
    """Get all available feature kinds.

    Returns
    -------
    list
        A list of (name, kind) tuples.
    """

    def isfeaturekind(x):
        return inspect.isclass(x) and issubclass(x, MultivariateFeature)

    return inspect.getmembers(extractors, isfeaturekind)
