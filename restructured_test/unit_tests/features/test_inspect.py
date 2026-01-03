"""Test for features module Python 3.10+ compatibility."""

from eegdash.features import (
    get_all_feature_kinds,
    get_all_feature_preprocessors,
    get_all_features,
)


def test_features_basic_functionality():
    """Test basic features module functionality."""
    # These should return lists without errors
    features = get_all_features()
    assert isinstance(features, list)

    extractors = get_all_feature_preprocessors()
    assert isinstance(extractors, list)

    kinds = get_all_feature_kinds()
    assert isinstance(kinds, list)


def test_inspect_gaps():
    from eegdash.features.inspect import (
        get_all_features,
        get_feature_kind,
        get_feature_predecessors,
    )

    # get_all_features
    feats = get_all_features()
    assert isinstance(feats, list)

    if feats:
        name, func = feats[0]
        # get_feature_kind
        get_feature_kind(func)
        # get_feature_predecessors
        preds = get_feature_predecessors(func)
        assert isinstance(preds, list)

    # 45: get_feature_predecessors with FeatureExtractor
    from eegdash.features.extractors import FeatureExtractor

    fe = FeatureExtractor({"f": lambda x: x})
    get_feature_predecessors(fe)

    # 113-117, 132-140, 156-159
    from eegdash.features.inspect import (
        get_all_feature_extractors,
        get_all_feature_kinds,
        get_all_feature_preprocessors,
    )

    all_extractors = get_all_feature_extractors()
    assert isinstance(all_extractors, list)
    all_preprocs = get_all_feature_preprocessors()
    assert isinstance(all_preprocs, list)
    all_kinds = get_all_feature_kinds()
    assert isinstance(all_kinds, list)
