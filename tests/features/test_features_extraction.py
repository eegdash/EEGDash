"""Test for features module Python 3.10+ compatibility."""

from functools import partial

import pytest

from eegdash import features
from eegdash.features import FeatureExtractor, FeaturesConcatDataset, extract_features


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
