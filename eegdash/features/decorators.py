from collections.abc import Callable
from typing import List

from .extractors import (
    BivariateFeature,
    DirectedBivariateFeature,
    MultivariateFeature,
    UnivariateFeature,
    _get_underlying_func,
)

__all__ = [
    "bivariate_feature",
    "FeatureKind",
    "FeaturePredecessor",
    "multivariate_feature",
    "univariate_feature",
]


class FeaturePredecessor:
    """A decorator to specify parent extractors for a feature function.

    This decorator attaches a list of immediate parent preprocessing steps to a feature
    extraction function. This information can be used to build a dependency graph of
    features.

    Parameters
    ----------
    *parent_extractor_type : list of Type
        A list of preprocessing functions (subclasses of
        :class:`~collections.abc.Callable` or None) that this feature immediately depends
        on.

    """

    def __init__(self, *parent_extractor_type: List[Callable | None]):
        parent_func = parent_extractor_type
        if not parent_func:
            parent_func = [None]
        for p_func in parent_func:
            assert p_func is None or callable(p_func)
        self.parent_extractor_type = parent_func

    def __call__(self, func: Callable) -> Callable:
        """Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The feature extraction function to decorate.

        Returns
        -------
        callable
            The decorated function with the `parent_extractor_type` attribute
            set.

        """
        f = _get_underlying_func(func)
        f.parent_extractor_type = self.parent_extractor_type
        return func


class FeatureKind:
    """A decorator to specify the kind of a feature.

    This decorator attaches a "feature kind" (e.g., univariate, bivariate)
    to a feature extraction function.

    Parameters
    ----------
    feature_kind : ~eegdash.features.extractors.MultivariateFeature
        An instance of a feature kind class, such as
        :class:`~eegdash.features.extractors.UnivariateFeature` or
        :class:`~eegdash.features.extractors.BivariateFeature`.

    """

    def __init__(self, feature_kind: MultivariateFeature):
        self.feature_kind = feature_kind

    def __call__(self, func: Callable) -> Callable:
        """Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The feature extraction function to decorate.

        Returns
        -------
        callable
            The decorated function with the `feature_kind` attribute set.

        """
        f = _get_underlying_func(func)
        f.feature_kind = self.feature_kind
        return func


# Syntax sugar
univariate_feature = FeatureKind(UnivariateFeature())
"""Decorator to mark a feature as univariate.

This is a convenience instance of :class:`~eegdash.features.decorators.FeatureKind` pre-configured for
univariate features.
"""


def bivariate_feature(func: Callable, directed: bool = False) -> Callable:
    """Decorator to mark a feature as bivariate.

    This decorator specifies that the feature operates on pairs of channels.

    Parameters
    ----------
    func : callable
        The feature extraction function to decorate.
    directed : bool, default False
        If True, the feature is directed (e.g., connectivity from channel A
        to B is different from B to A). If False, the feature is undirected.

    Returns
    -------
    callable
        The decorated function with the appropriate bivariate feature kind
        attached.

    """
    if directed:
        kind = DirectedBivariateFeature()
    else:
        kind = BivariateFeature()
    return FeatureKind(kind)(func)


multivariate_feature = FeatureKind(MultivariateFeature())
"""Decorator to mark a feature as multivariate.

This is a convenience instance of :class:`~eegdash.features.decorators.FeatureKind` pre-configured for
multivariate features, which operate on all channels simultaneously.
"""
