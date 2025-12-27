r"""
Feature Metadata Decorators.

This module provides  decorators used to annotate feature extraction
functions with structural metadata. These annotations define the dependency
graph (via predecessors) and the data format (via feature kinds).

The module provides the following decorators:
- :class:`FeaturePredecessor` — Specifies the required input transformation 
  for a feature.
- :class:`FeatureKind` — Defines the dimensionality of the feature output.
- :func:`univariate_feature` — Sugar for per-channel features.
- :func:`bivariate_feature` — Sugar for channel-pair features.
- :func:`multivariate_feature` — Sugar for global/all-channel features.

"""
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
    r"""Decorator to specify parent extractors for a feature function.

    This decorator attaches a list of immediate parent preprocessing steps to 
    a feature extraction function. This metadata is used by the 
    :class:`~eegdash.features.extractors.FeatureExtractor` to validate the 
    execution tree.

    Parameters
    ----------
    *parent_extractor_type : list of callable or None
        A list of preprocessing functions that this feature immediately 
        depends on. Use ``None`` to indicate that the feature can operate 
        directly on raw signal arrays.

    Attributes
    ----------
    parent_extractor_type : list of callable or None
        The stored list of predecessor functions.

    Notes
    -----
    A feature can have multiple potential predecessors.

    """

    def __init__(self, *parent_extractor_type: List[Callable | None]):
        parent_func = parent_extractor_type
        if not parent_func:
            parent_func = [None]
        for p_func in parent_func:
            assert p_func is None or callable(p_func)
        self.parent_extractor_type = parent_func

    def __call__(self, func: Callable) -> Callable:
        r"""Apply the decorator to a function.

        Parameters
        ----------
        func : callable
            The feature extraction function to decorate.

        Returns
        -------
        callable
            The decorated function with the `parent_extractor_type` 
            attribute attached.
        
        """
        f = _get_underlying_func(func)
        f.parent_extractor_type = self.parent_extractor_type
        return func


class FeatureKind:
    r"""Decorator to specify the operational dimensionality of a feature.

    This decorator attaches a "feature kind" instance to a function, 
    determining how the :class:`~eegdash.features.extractors.FeatureExtractor` 
    should map the resulting numerical arrays to channel names.

    Parameters
    ----------
    feature_kind : ~eegdash.features.extractors.MultivariateFeature
        An instance of a feature kind class, such as 
        :class:`~eegdash.features.extractors.UnivariateFeature` or 
        :class:`~eegdash.features.extractors.BivariateFeature`.

    Attributes
    ----------
    feature_kind : ~eegdash.features.extractors.MultivariateFeature
        The stored kind instance used for output formatting.
    """

    def __init__(self, feature_kind: MultivariateFeature):
        self.feature_kind = feature_kind

    def __call__(self, func: Callable) -> Callable:
        r"""Apply the decorator to a function.

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
r"""Decorator to mark a feature as univariate.

Indicates that the feature is computed for each channel independently. 
The output will be formatted as a dictionary with keys matching the 
original channel names.
"""


def bivariate_feature(func: Callable, directed: bool = False) -> Callable:
    r"""Decorator to mark a feature as bivariate.

    Specifies that the feature operates on pairs of channels.

    Parameters
    ----------
    func : callable
        The feature extraction function to decorate.
    directed : bool, default False
        If True, the feature is treated as directed. 
        If False, only unique, unordered pairs are computed.

    Returns
    -------
    callable
        The decorated function with either a 
        :class:`~eegdash.features.extractors.BivariateFeature` or 
        :class:`~eegdash.features.extractors.DirectedBivariateFeature` 
        kind attached.

    """
    if directed:
        kind = DirectedBivariateFeature()
    else:
        kind = BivariateFeature()
    return FeatureKind(kind)(func)


multivariate_feature = FeatureKind(MultivariateFeature())
r"""Decorator to mark a feature as multivariate.

Indicates that the feature operates on all channels simultaneously. The 
output naming convention is determined by the feature's internal logic 
rather than channel labels.
"""
