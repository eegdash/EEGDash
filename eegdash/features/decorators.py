"""
This module defines decorators for feature extraction functions.

These decorators are used to classify and organize feature extractors,
making it easier to manage and apply them in a structured way.
"""
from collections.abc import Callable
from typing import List, Type

from .extractors import (
    BivariateFeature,
    DirectedBivariateFeature,
    FeatureExtractor,
    MultivariateFeature,
    UnivariateFeature,
    _get_underlying_func,
)


class FeaturePredecessor:
    """A decorator to specify the parent extractor type for a feature.

    This decorator is used to associate a feature extractor with its parent
    extractor type, which is useful for organizing and chaining feature
    extractors.

    Parameters
    ----------
    *parent_extractor_type : list[Type]
        A list of parent extractor types.
    """

    def __init__(self, *parent_extractor_type: List[Type]):
        parent_cls = parent_extractor_type
        if not parent_cls:
            parent_cls = [FeatureExtractor]
        for p_cls in parent_cls:
            assert issubclass(p_cls, FeatureExtractor)
        self.parent_extractor_type = parent_cls

    def __call__(self, func: Callable):
        """Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The function to decorate.

        Returns
        -------
        callable
            The decorated function.
        """
        f = _get_underlying_func(func)
        f.parent_extractor_type = self.parent_extractor_type
        return func


class FeatureKind:
    """A decorator to specify the kind of a feature.

    This decorator is used to classify a feature as univariate, bivariate, or
    multivariate, which helps in applying the correct processing logic.

    Parameters
    ----------
    feature_kind : MultivariateFeature
        The kind of the feature.
    """

    def __init__(self, feature_kind: MultivariateFeature):
        self.feature_kind = feature_kind

    def __call__(self, func):
        """Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The function to decorate.

        Returns
        -------
        callable
            The decorated function.
        """
        f = _get_underlying_func(func)
        f.feature_kind = self.feature_kind
        return func


# A decorator for univariate features.
univariate_feature = FeatureKind(UnivariateFeature())


def bivariate_feature(func, directed=False):
    """A decorator for bivariate features.

    Parameters
    ----------
    func : callable
        The function to decorate.
    directed : bool
        Whether the feature is directed.

    Returns
    -------
    callable
        The decorated function.
    """
    if directed:
        kind = DirectedBivariateFeature()
    else:
        kind = BivariateFeature()
    return FeatureKind(kind)(func)


# A decorator for multivariate features.
multivariate_feature = FeatureKind(MultivariateFeature())
