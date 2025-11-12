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
