"""Test for features module Python 3.10+ compatibility."""

from functools import partial
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from eegdash import features
from eegdash.features import (
    FeatureExtractor,
    FeaturesConcatDataset,
    extract_features,
    TrainableFeature,
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


def test_feature_extraction_benchmark(
    benchmark, windows_ds, feature_extractor, batch_size=512, n_jobs=1
):
    """Benchmark feature extraction function."""
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
        feats = ext(data, _batch_size=data.shape[0], _ch_names=["C1", "C2", "C3", "C4"])
        assert "mean" in feats
    except Exception as e:
        print(f"Error in test_feature_extractor_basic: {e}")
        # If it fails, we want to know why
        raise


def test_features_pipeline_basic():
    # Use FeatureExtractor instead of FeaturesPipeline
    ext = FeatureExtractor({"mean": lambda x: np.mean(x, axis=-1)})
    data = np.random.randn(2, 4, 100)
    feats = ext(data, _batch_size=data.shape[0], _ch_names=["C1", "C2", "C3", "C4"])
    assert "mean" in feats


def test_extractors_gaps():
    from functools import partial

    from eegdash.features.extractors import (
        FeatureExtractor,
        TrainableFeature,
        _get_underlying_func,
    )

    # _get_underlying_func gaps
    def dummy():
        pass

    p = partial(dummy)
    assert _get_underlying_func(p) == dummy

    # TrainableFeature gaps
    class MyTrainable(TrainableFeature):
        def clear(self):
            pass

        def partial_fit(self, *x, y=None):
            pass

    mt = MyTrainable()
    with pytest.raises(RuntimeError, match="fitted first"):
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


def test_extractors_more():
    from eegdash.features.extractors import (
        BivariateFeature,
        DirectedBivariateFeature,
        FeatureExtractor,
        MultivariateFeature,
    )

    mv = MultivariateFeature()
    # verify it has feature_channel_names
    assert mv.feature_channel_names(["ch1"]) == []

    # bivariate
    bv = BivariateFeature()
    assert bv.channel_pair_format == "{}<>{}"

    # directed bivariate
    dbv = DirectedBivariateFeature()
    assert dbv.channel_pair_format == "{}<>{}"

    # FeatureExtractor more gaps
    from functools import partial

    def f1(x, a=1):
        return {"f1": x.mean(axis=(-1, -2))}

    f1.parent_extractor_type = [None]

    # 127, 130, 132: features_kwargs
    fe = FeatureExtractor({"feat": partial(f1, a=2)})
    assert "feat" in fe.features_kwargs
    assert fe.features_kwargs["feat"] == {"a": 2}

    def pre(x, a=1):
        return x

    f1.parent_extractor_type = [None, pre]  # add pre to parent types
    fe_pre = FeatureExtractor({"feat": f1}, preprocessor=partial(pre, a=2))
    assert "preprocess_kwargs" in fe_pre.features_kwargs
    # 181: call with preprocessor
    fe_pre(np.array([[[1.0]]]), _batch_size=1, _ch_names=["ch1"])
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

    feat.parent_extractor_type = [None]
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

    from eegdash.features.extractors import _get_underlying_func

    # If NUMBA_DISABLE_JIT=1, jitted is not a Dispatcher
    if isinstance(jitted, Dispatcher):
        assert _get_underlying_func(jitted) == jitted.py_func
    else:
        assert _get_underlying_func(jitted) == jitted


def test_extractors_partial_fit_gap():
    # FeatureExtractor.partial_fit
    # It delegates to _check_is_trainable?
    # Ensure we call it in a way that iterates features

    # We need a trainable child
    child = MagicMock(spec=TrainableFeature)
    fe = FeatureExtractor({"c": child})
    fe.partial_fit(np.array([[[1]]]))
    assert child.partial_fit.called


def test_extractor_preprocess_tuple_gap():
    # Cover line 251 in extractors.py: if not isinstance(z, tuple): z = (z,)

    # Mock FeatureExtractor with a preprocess that returns a single item
    class SingleRetExtractor(FeatureExtractor):
        def preprocess(self, x):
            return x  # Single item, not tuple

    child = MagicMock(spec=TrainableFeature)
    fe = SingleRetExtractor({"c": child})

    # partial_fit calls preprocess
    fe.partial_fit("data")

    # Child should receive it wrapped
    assert child.partial_fit.call_args[0][0] == "data"
