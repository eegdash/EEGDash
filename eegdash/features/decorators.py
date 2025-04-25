from typing import List, Type
from collections.abc import Callable

from .extractors import (
    FeatureExtractor,
    UnivariateFeature,
    BivariateFeature,
    DirectedBivariateFeature,
    MultivariateFeature,
)
from .extractors import _get_underlying_func


class FeaturePredecessor:
    def __init__(self, *parent_extractor_type: List[Type]):
        parent_cls = parent_extractor_type
        if not parent_cls:
            parent_cls = [FeatureExtractor]
        for p_cls in parent_cls:
            assert issubclass(p_cls, FeatureExtractor)
        self.parent_extractor_type = parent_cls

    def __call__(self, func: Callable):
        f = _get_underlying_func(func)
        f.parent_extractor_type = self.parent_extractor_type
        return func


class FeatureKind:
    def __init__(self, feature_kind: MultivariateFeature):
        self.feature_kind = feature_kind

    def __call__(self, func):
        f = _get_underlying_func(func)
        f.feature_kind = self.feature_kind
        return func


# Syntax sugar
univariate_feature = FeatureKind(UnivariateFeature())
bivariate_feature = FeatureKind(BivariateFeature())
directed_bivariate_feature = FeatureKind(DirectedBivariateFeature())
multivariate_feature = FeatureKind(MultivariateFeature())
