def test_bivariate_feature():
    """Test bivariate_feature creates BivariateFeature (not directed)."""
    from eegdash.features.decorators import bivariate_feature
    from eegdash.features.kinds import BivariateFeature

    def test_func(x, y):
        return x + y

    # Apply decorator with directed=False (line 132)
    decorated = bivariate_feature(test_func)

    # Check that the function has been decorated
    assert hasattr(decorated, "feature_kind")
    assert isinstance(decorated.feature_kind, BivariateFeature)


def test_feature_predecessor_empty():
    """Test FeaturePredecessor with no args."""
    from eegdash.features.decorators import feature_predecessor
    from eegdash.features.output_types import SignalOutputType

    @feature_predecessor()
    def my_func(x):
        return x

    from eegdash.features.base_utils import get_underlying_func

    assert get_underlying_func(my_func).parent_extractor_type == [SignalOutputType]


def test_feature_kind_decorator():
    """Test FeatureKind decorator."""
    from eegdash.features.decorators import feature_kind
    from eegdash.features.kinds import UnivariateFeature

    kind_instance = UnivariateFeature()

    @feature_kind(kind_instance)
    def my_feature(x):
        return x

    from eegdash.features.base_utils import get_underlying_func

    assert get_underlying_func(my_feature).feature_kind is kind_instance
