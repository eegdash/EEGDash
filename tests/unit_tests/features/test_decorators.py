def test_bivariate_feature():
    """Test bivariate_feature creates BivariateFeature (not directed)."""
    from eegdash.features.decorators import BivariateFeature, bivariate_feature

    def test_func(x, y):
        return x + y

    # Apply decorator with directed=False (line 132)
    decorated = bivariate_feature(test_func)

    # Check that the function has been decorated
    assert hasattr(decorated, "feature_kind")
    assert isinstance(decorated.feature_kind, BivariateFeature)


def test_feature_predecessor_empty():
    """Test FeaturePredecessor with no args."""
    from eegdash.features.decorators import FeaturePredecessor

    @FeaturePredecessor()
    def my_func(x):
        return x

    from eegdash.features.base_utils import get_underlying_func

    assert get_underlying_func(my_func).parent_extractor_type == [None]


def test_feature_kind_decorator():
    """Test FeatureKind decorator."""
    from eegdash.features.decorators import FeatureKind
    from eegdash.features.kinds import UnivariateFeature

    kind_instance = UnivariateFeature()

    @FeatureKind(kind_instance)
    def my_feature(x):
        return x

    from eegdash.features.base_utils import get_underlying_func

    assert get_underlying_func(my_feature).feature_kind is kind_instance
