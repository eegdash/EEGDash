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


def test_get_feature_predecessors_none():
    """Test get_feature_predecessors with None input."""
    from eegdash.features.inspect import get_feature_predecessors

    result = get_feature_predecessors(None)
    assert result == [None]


def test_get_feature_predecessors_multiple():
    """Test get_feature_predecessors with multiple predecessors."""
    from eegdash.features.inspect import get_feature_predecessors

    # Create a mock feature with multiple predecessors
    def mock_feature():
        pass

    def pred1():
        pass

    def pred2():
        pass

    pred1.parent_extractor_type = [None]
    pred2.parent_extractor_type = [None]
    mock_feature.parent_extractor_type = [pred1, pred2]

    result = get_feature_predecessors(mock_feature)
    assert mock_feature in result


def test_get_all_feature_preprocessors():
    """Test getting all feature preprocessors."""
    from eegdash.features.inspect import get_all_feature_preprocessors

    result = get_all_feature_preprocessors()
    assert isinstance(result, list)


def test_get_all_feature_kinds():
    """Test getting all feature kinds."""
    from eegdash.features.inspect import get_all_feature_kinds

    result = get_all_feature_kinds()
    assert isinstance(result, list)
    # Should find at least MultivariateFeature subclasses
    assert len(result) > 0


def test_get_feature_kind():
    """Test get_feature_kind function."""
    from eegdash.features.extractors import UnivariateFeature
    from eegdash.features.inspect import get_feature_kind

    # Create a function with feature_kind attribute
    def mock_feature():
        pass

    mock_feature.feature_kind = UnivariateFeature()

    result = get_feature_kind(mock_feature)
    assert isinstance(result, UnivariateFeature)


def test_get_all_features():
    """Test get_all_features function."""
    from eegdash.features.inspect import get_all_features

    result = get_all_features()
    assert isinstance(result, list)
    # Should find at least some features in the feature bank
    assert len(result) > 0


def test_get_feature_predecessors_with_extractor():
    """Test get_feature_predecessors with FeatureExtractor instance."""
    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.inspect import get_feature_predecessors

    def preproc(x):
        return x

    preproc.parent_extractor_type = [None]

    def feat(x):
        return x

    feat.parent_extractor_type = [preproc]

    extractor = FeatureExtractor({"feat": feat}, preprocessor=preproc)
    result = get_feature_predecessors(extractor)
    assert isinstance(result, list)
