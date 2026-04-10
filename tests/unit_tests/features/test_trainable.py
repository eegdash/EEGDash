"""Test for features module Python 3.10+ compatibility."""

import pytest


def test_extractors_gaps():
    from functools import partial

    from eegdash.features.base_utils import get_underlying_func
    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.trainable import TrainableFeature

    # _get_underlying_func gaps
    def dummy():
        pass

    p = partial(dummy)
    assert get_underlying_func(p) == dummy

    # TrainableFeature gaps
    class MyTrainable(TrainableFeature):
        def clear(self):
            pass

        def partial_fit(self, *x, y=None):
            pass

    mt = MyTrainable()
    with pytest.raises(RuntimeError, match="trained first"):
        mt()
    mt.fit()
    # Now it shouldn't raise RuntimeError, but maybe it fails because it's an ABC without __call__ implementation in parent?
    # Actually TrainableFeature.__call__ is implemented in extractors.py as we saw.

    # FeatureExtractor validate logic (142-150)
    # We need a mismatched preprocessor
    def pre1(x):
        return x

    def pre2(x):
        return x

    def feat(x):
        return x

    # mock parent_extractor_type
    feat.parent_extractor_type = [pre1]

    with pytest.raises(TypeError, match="cannot be a child of"):
        FeatureExtractor({"f": feat}, preprocessor=pre2)


def test_trainable_feature_interface():
    """Test TrainableFeature clear and partial_fit methods."""
    import numpy as np

    from eegdash.features.trainable import TrainableFeature

    class ConcreteTrainable(TrainableFeature):
        def __init__(self):
            self._is_fitted = False
            self._is_trained = False

        def clear(self):
            self._is_trained = False

        def partial_fit(self, *x, y=None):
            self._is_trained = True

    tf = ConcreteTrainable()
    # Test clear method
    tf.clear()
    assert not tf._is_trained

    # Test partial_fit
    tf.partial_fit(np.array([1, 2, 3]))
    assert tf._is_trained

    # Test fit method
    tf.fit()
    assert tf._is_trained


def test_feature_extractor_with_trainable():
    """Test FeatureExtractor with trainable features."""
    import numpy as np

    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def dummy_feature(x):
        return np.mean(x, axis=-1, keepdims=True)

    # Add parent_extractor_type attribute
    dummy_feature.parent_extractor_type = [SignalOutputType]
    dummy_feature.feature_kind = None

    fe = FeatureExtractor({"dummy": dummy_feature})
    assert not fe._is_trainable
