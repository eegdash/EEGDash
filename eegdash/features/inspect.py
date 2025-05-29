from collections.abc import Callable
import inspect

from .extractors import FeatureExtractor, MultivariateFeature, _get_underlying_func
from . import feature_bank, extractors


def get_feature_predecessors(feature_or_extractor: Callable):
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
    return _get_underlying_func(feature).feature_kind


def get_all_features():
    def isfeature(x):
        return hasattr(_get_underlying_func(x), "feature_kind")

    return inspect.getmembers(feature_bank, isfeature)


def get_all_feature_extractors():
    def isfeatureextractor(x):
        return inspect.isclass(x) and issubclass(x, FeatureExtractor)

    return [
        ("FeatureExtractor", FeatureExtractor),
        *inspect.getmembers(feature_bank, isfeatureextractor),
    ]


def get_all_feature_kinds():
    def isfeaturekind(x):
        return inspect.isclass(x) and issubclass(x, MultivariateFeature)

    return inspect.getmembers(extractors, isfeaturekind)
