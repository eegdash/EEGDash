"""Test for features module Python 3.10+ compatibility."""


def test_multivariate_feature_dict_input():
    """Test MultivariateFeature with dict input."""
    import numpy as np

    from eegdash.features.kinds import UnivariateFeature

    uf = UnivariateFeature()
    _metadata = {"info": {"ch_names": ["ch1", "ch2"]}}

    # Create dict input
    x = {"key": np.array([[1, 2], [3, 4]])}
    result = uf(x, _metadata=_metadata)
    assert isinstance(result, dict)


def test_bivariate_feature_channel_names():
    """Test BivariateFeature channel name generation."""
    from eegdash.features.base_utils import BivariateIterator
    from eegdash.features.kinds import BivariateFeature

    bf = BivariateFeature()
    _metadata = {
        "info": {"ch_names": ["A", "B", "C"]},
        "ch_pair_iterator": BivariateIterator(3),
    }
    result = bf.feature_channel_names(_metadata=_metadata)
    # Should have 3 pairs: A<>B, A<>C, B<>C
    assert len(result) == 3
    assert "A<>B" in result


def test_array_to_dict_empty_channels():
    """Test _array_to_dict with empty channel list."""
    import numpy as np

    from eegdash.features.kinds import MultivariateFeature

    x = np.array([[1, 2, 3]])
    result = MultivariateFeature._array_to_dict(x, [], "test")
    assert "test" in result
