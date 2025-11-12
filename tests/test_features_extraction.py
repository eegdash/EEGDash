"""Test for features module Python 3.10+ compatibility."""

import shutil
from functools import partial
from pathlib import Path

import pytest

from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from eegdash import EEGDashDataset, features
from eegdash.features import FeatureExtractor, FeaturesConcatDataset, extract_features
from eegdash.hbn.preprocessing import hbn_ec_ec_reannotation
from eegdash.logging import logger


@pytest.fixture(scope="module")
def eeg_dash_dataset(cache_dir: Path):
    """Fixture to create an instance of EEGDashDataset."""
    return EEGDashDataset(
        query={
            "dataset": "ds005514",
            "task": "RestingState",
            "subject": "NDARDB033FW5",
        },
        cache_dir=cache_dir,
    )


@pytest.fixture(scope="module")
def preprocess_instance(eeg_dash_dataset, cache_dir: Path):
    """Fixture to create an instance of EEGDashDataset with preprocessing."""
    selected_channels = [
        "E22",
        "E9",
        "E33",
        "E24",
        "E11",
        "E124",
        "E122",
        "E29",
        "E6",
        "E111",
        "E45",
        "E36",
        "E104",
        "E108",
        "E42",
        "E55",
        "E93",
        "E58",
        "E52",
        "E62",
        "E92",
        "E96",
        "E70",
        "Cz",
    ]
    pre_processed_dir = cache_dir / "preprocessed"
    pre_processed_dir.mkdir(parents=True, exist_ok=True)
    try:
        eeg_dash_dataset = load_concat_dataset(
            pre_processed_dir,
            preload=True,
        )
        return eeg_dash_dataset

    except Exception as e:
        logger.warning(f"Failed to load dataset creating a new instance: {e}. ")
        if pre_processed_dir.exists():
            # folder with issue, erasing and creating again
            shutil.rmtree(pre_processed_dir)
            pre_processed_dir.mkdir(parents=True, exist_ok=True)

        preprocessors = [
            hbn_ec_ec_reannotation(),
            Preprocessor(
                "pick_channels",
                ch_names=selected_channels,
            ),
            Preprocessor("resample", sfreq=128),
            Preprocessor("filter", l_freq=1, h_freq=55),
        ]

        eeg_dash_dataset = preprocess(
            eeg_dash_dataset, preprocessors, n_jobs=-1, save_dir=pre_processed_dir
        )

        return eeg_dash_dataset


@pytest.fixture(scope="module")
def windows_ds(preprocess_instance):
    """Fixture to create windows from the preprocessed EEG dataset."""
    windows = create_windows_from_events(
        preprocess_instance,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=256,
        preload=True,
    )
    return windows


@pytest.fixture(scope="module")
def feature_dict(windows_ds):
    """Fixture to create a feature extraction tree."""
    sfreq = windows_ds.datasets[0].raw.info["sfreq"]
    filter_freqs = dict(windows_ds.datasets[0].raw_preproc_kwargs)["filter"]

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


def test_feature_extraction(windows_ds, feature_extractor, batch_size=128, n_jobs=2):
    """Test the feature extraction function."""
    features = extract_features(
        windows_ds, feature_extractor, batch_size=batch_size, n_jobs=n_jobs
    )
    assert isinstance(features, FeaturesConcatDataset)
    assert len(windows_ds.datasets) == len(features.datasets)


@pytest.fixture(scope="module")
def features_ds(windows_ds, feature_extractor, batch_size=512, n_jobs=1):
    """Fixture to create a features dataset."""
    features = extract_features(
        windows_ds, feature_extractor, batch_size=batch_size, n_jobs=n_jobs
    )
    return features


def benchmark_extract_features(benchmark, batch_size=512, n_jobs=1):
    """Benchmark feature extraction function."""
    benchmark(test_feature_extraction, batch_size=batch_size, n_jobs=n_jobs)
