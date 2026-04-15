"""Test for features module Python 3.10+ compatibility."""

from functools import partial

import numpy as np
import pytest

from eegdash import features
from eegdash.features import (
    FeatureExtractor,
    FeaturesConcatDataset,
    extract_features,
)


@pytest.fixture(scope="module")
def feature_dict(windows_ds):
    """Fixture to create a feature extraction tree."""
    sfreq = windows_ds.datasets[0].raw.info["sfreq"]
    raw_kwargs = windows_ds.datasets[0].raw_preproc_kwargs
    filter_freqs = None

    # Handle list of preprocessors (standard braindecode/eegdash structure)
    if isinstance(raw_kwargs, list):
        for item in raw_kwargs:
            if isinstance(item, dict) and item.get("fn") == "filter":
                filter_freqs = item.get("kwargs")
                break

    # Fallback/Debug
    if filter_freqs is None:
        try:
            # Old behavior or different structure?
            filter_freqs = dict(raw_kwargs)["filter"]
        except Exception:
            # Default if not found (or raise error if critical)
            # But the test depends on it, so let's default to typical vals if missing?
            # Or fail gracefully.
            # Based on debug output: {'l_freq': 1, 'h_freq': 55}
            pass

    if filter_freqs is None:
        raise ValueError(
            f"Could not find filter parameters in raw_preproc_kwargs: {raw_kwargs}"
        )

    feats = {
        "sig": features.FeatureExtractor(
            {
                "mean": features.signal_mean,
                "var": features.signal_variance,
                "std": features.signal_std,
                "skew": features.signal_skewness,
                "kurt": features.signal_kurtosis,
                "rms": features.signal_root_mean_square,
                "ptp": features.signal_peak_to_peak,
                "quan.1": partial(features.signal_quantile, q=0.1),
                "quan.9": partial(features.signal_quantile, q=0.9),
                "line_len": features.signal_line_length,
                "zero_x": features.signal_zero_crossings,
            },
        ),
        "spec": features.FeatureExtractor(
            preprocessor=partial(
                features.spectral_preprocessor,
                fs=sfreq,
                f_min=filter_freqs["l_freq"],
                f_max=filter_freqs["h_freq"],
                nperseg=2 * sfreq,
                noverlap=int(1.5 * sfreq),
            ),
            feature_extractors={
                "rtot_power": features.spectral_root_total_power,
                "band_power": partial(
                    features.spectral_bands_power,
                    bands={
                        "theta": (4.5, 8),
                        "alpha": (8, 12),
                        "beta": (12, 30),
                    },
                ),
                0: features.FeatureExtractor(
                    preprocessor=features.spectral_normalized_preprocessor,
                    feature_extractors={
                        "moment": features.spectral_moment,
                        "entropy": features.spectral_entropy,
                        "edge": partial(features.spectral_edge, edge=0.9),
                    },
                ),
                1: features.FeatureExtractor(
                    preprocessor=features.spectral_db_preprocessor,
                    feature_extractors={
                        "slope": features.spectral_slope,
                    },
                ),
            },
        ),
    }
    return feats


@pytest.fixture(scope="module")
def feature_extractor(feature_dict):
    """Fixture to create a feature extractor."""
    feats = FeatureExtractor(feature_dict)
    return feats


@pytest.mark.benchmark(group="features")
def test_feature_extraction_benchmark(
    benchmark, windows_ds, feature_extractor, batch_size=512, n_jobs=1
):
    """Benchmark feature extraction function.

    Note: This test requires pytest-benchmark plugin. Run with:
        pytest --benchmark-enable tests/unit_tests/features/test_extractors.py
    """
    feats = benchmark(
        extract_features,
        windows_ds,
        feature_extractor,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )
    assert isinstance(feats, FeaturesConcatDataset)
    assert len(windows_ds.datasets) == len(feats.datasets)


@pytest.fixture(scope="module")
def features_ds(windows_ds, feature_extractor, batch_size=512, n_jobs=1):
    """Fixture to create a features dataset."""
    feats = extract_features(
        windows_ds, feature_extractor, batch_size=batch_size, n_jobs=n_jobs
    )
    return feats


def test_feature_extractor_basic():
    # Test basic FeatureExtractor
    try:
        ext = FeatureExtractor(
            feature_extractors={"mean": lambda x: np.mean(x, axis=-1)}
        )
        data = np.random.randn(2, 4, 100)
        feats = ext(
            data,
            _metadata={
                "batch_size": data.shape[0],
                "info": {"ch_names": ["C1", "C2", "C3", "C4"]},
            },
        )
        assert "mean" in feats
    except Exception as e:
        print(f"Error in test_feature_extractor_basic: {e}")
        # If it fails, we want to know why
        raise


def test_features_pipeline_basic():
    # Use FeatureExtractor instead of FeaturesPipeline
    ext = FeatureExtractor({"mean": lambda x: np.mean(x, axis=-1)})
    data = np.random.randn(2, 4, 100)
    feats = ext(
        data,
        _metadata={
            "batch_size": data.shape[0],
            "info": {"ch_names": ["C1", "C2", "C3", "C4"]},
        },
    )
    assert "mean" in feats


def test_extractors_more():
    from eegdash.features.extractors import (
        FeatureExtractor,
    )
    from eegdash.features.kinds import (
        BivariateFeature,
        MultivariateFeature,
    )
    from eegdash.features.output_types import SignalOutputType

    mv = MultivariateFeature()
    # verify it has feature_channel_names
    assert mv.feature_channel_names(["ch1"]) == []

    # bivariate
    bv = BivariateFeature()
    assert bv.channel_pair_format is None

    # FeatureExtractor more gaps
    from functools import partial

    def f1(x, a=1):
        return {"f1": x.mean(axis=(-1, -2))}

    f1.parent_extractor_type = [SignalOutputType]

    # 127, 130, 132: features_kwargs
    fe = FeatureExtractor({"feat": partial(f1, a=2)})
    assert "feat" in fe.features_kwargs
    assert fe.features_kwargs["feat"] == {"a": 2}

    def pre(x, a=1):
        return x

    f1.parent_extractor_type = [SignalOutputType, pre]  # add pre to parent types
    fe_pre = FeatureExtractor({"feat": f1}, preprocessor=partial(pre, a=2))
    assert "preprocess_kwargs" in fe_pre.features_kwargs
    # 181: call with preprocessor
    fe_pre(
        np.array([[[1.0]]]), _metadata={"batch_size": 1, "info": {"ch_names": ["ch1"]}}
    )
    fe_inner = FeatureExtractor({"inner": f1})
    fe_outer = FeatureExtractor({"outer": fe_inner})
    assert "outer" in fe_outer.features_kwargs
    assert fe_outer.features_kwargs["outer"] == fe_inner.features_kwargs

    # 159, 161: _check_is_trainable
    fe_outer._check_is_trainable({"f": fe_inner})

    class MyTrainable(FeatureExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_trainable = True

        def clear(self):
            pass

        def partial_fit(self, *x, y=None):
            pass

        def fit(self):
            self._is_fitted = True

    def feat(x):
        return x

    feat.parent_extractor_type = [SignalOutputType]
    trainable_fe = MyTrainable({"f": feat})
    fe_outer._check_is_trainable({"f": trainable_fe})

    # 158-159, 161, 181 ...
    # circular import check bypass usually works

    # Test Dispatcher unwrapping
    from numba import njit
    from numba.core.dispatcher import Dispatcher

    @njit
    def jitted(x):
        return x

    from eegdash.features.base_utils import get_underlying_func

    # If NUMBA_DISABLE_JIT=1, jitted is not a Dispatcher
    if isinstance(jitted, Dispatcher):
        assert get_underlying_func(jitted) == jitted.py_func
    else:
        assert get_underlying_func(jitted) == jitted


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


def test_feature_extractor_clear_non_trainable():
    """Test that clear() on non-trainable extractor does nothing."""
    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def simple_feature(x):
        return x

    simple_feature.parent_extractor_type = [SignalOutputType]

    fe = FeatureExtractor({"simple": simple_feature})
    # Should not raise
    fe.clear()


def test_feature_extractor_partial_fit_non_trainable():
    """Test that partial_fit on non-trainable extractor does nothing."""
    import numpy as np

    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def simple_feature(x):
        return x

    simple_feature.parent_extractor_type = [SignalOutputType]

    fe = FeatureExtractor({"simple": simple_feature})
    # Should not raise
    fe.partial_fit(np.array([[1, 2, 3]]), _metadata={})


def test_feature_extractor_fit_non_trainable():
    """Test that fit on non-trainable extractor does nothing."""
    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def simple_feature(x):
        return x

    simple_feature.parent_extractor_type = [SignalOutputType]

    fe = FeatureExtractor({"simple": simple_feature})
    # Should not raise
    fe.fit()


def test_feature_extractor_with_partial_preprocessor():
    """Test FeatureExtractor stores kwargs from partial preprocessor."""
    from functools import partial

    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def preprocessor(x, scale=1.0):
        return x * scale

    preprocessor.parent_extractor_type = [SignalOutputType]

    def simple_feature(x):
        return x

    simple_feature.parent_extractor_type = [preprocessor]

    partial_preproc = partial(preprocessor, scale=2.0)

    fe = FeatureExtractor({"simple": simple_feature}, preprocessor=partial_preproc)
    assert "preprocess_kwargs" in fe.features_kwargs


def test_feature_extractor_nested_check_trainable():
    """Test _check_is_trainable with nested FeatureExtractor."""
    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def inner_feat(x):
        return x

    inner_feat.parent_extractor_type = [SignalOutputType]

    inner_extractor = FeatureExtractor({"inner": inner_feat})

    def outer_preproc(x):
        return x

    outer_preproc.parent_extractor_type = [SignalOutputType]

    # Mark inner extractor's preprocessor
    inner_extractor.preprocessor = None
    inner_extractor.parent_extractor_type = [SignalOutputType]

    outer_extractor = FeatureExtractor({"nested": inner_extractor})
    assert not outer_extractor._is_trainable


def test_feature_extractor_preprocess_none():
    """Test preprocess method with no preprocessor."""
    import numpy as np

    from eegdash.features.extractors import FeatureExtractor
    from eegdash.features.output_types import SignalOutputType

    def feat(x):
        return x

    feat.parent_extractor_type = [SignalOutputType]

    extractor = FeatureExtractor({"feat": feat}, preprocessor=None)
    result = extractor.preprocess(np.array([1, 2, 3]), _metadata={})
    assert isinstance(result, tuple)
