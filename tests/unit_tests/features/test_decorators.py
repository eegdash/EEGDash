def test_bivariate_feature_undirected():
    """Test bivariate_feature creates BivariateFeature (not directed)."""
    from eegdash.features.decorators import BivariateFeature, bivariate_feature

    def test_func(x, y):
        return x + y

    # Apply decorator with directed=False (line 132)
    decorated = bivariate_feature(test_func, directed=False)

    # Check that the function has been decorated
    assert hasattr(decorated, "feature_kind")
    assert isinstance(decorated.feature_kind, BivariateFeature)


def test_bivariate_feature_directed():
    """Test bivariate_feature with directed=True (line 132)."""
    from eegdash.features.decorators import bivariate_feature
    from eegdash.features.extractors import DirectedBivariateFeature

    @bivariate_feature
    def dummy_undirected(x):
        return x

    @bivariate_feature
    def dummy_directed(x):
        return x

    # Apply with directed=True
    dummy_directed_applied = bivariate_feature(lambda x: x, directed=True)

    from eegdash.features.extractors import _get_underlying_func

    kind = _get_underlying_func(dummy_directed_applied).feature_kind
    assert isinstance(kind, DirectedBivariateFeature)


def test_feature_predecessor_empty():
    """Test FeaturePredecessor with no args."""
    from eegdash.features.decorators import FeaturePredecessor

    @FeaturePredecessor()
    def my_func(x):
        return x

    from eegdash.features.extractors import _get_underlying_func

    assert _get_underlying_func(my_func).parent_extractor_type == [None]


def test_feature_kind_decorator():
    """Test FeatureKind decorator."""
    from eegdash.features.decorators import FeatureKind
    from eegdash.features.extractors import UnivariateFeature

    kind_instance = UnivariateFeature()

    @FeatureKind(kind_instance)
    def my_feature(x):
        return x

    from eegdash.features.extractors import _get_underlying_func

    assert _get_underlying_func(my_feature).feature_kind is kind_instance
