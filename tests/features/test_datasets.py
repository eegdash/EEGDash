import os

import numpy as np
import pandas as pd
import pytest

from eegdash.features.datasets import FeaturesConcatDataset, FeaturesDataset


@pytest.fixture
def sample_feature_dataset():
    # Create sample data
    n_samples = 10
    n_features = 3

    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    metadata = pd.DataFrame(
        {
            "target": np.random.randint(0, 2, n_samples),
            "i_window_in_trial": np.arange(n_samples),  # simplified
            "i_start_in_trial": np.zeros(n_samples),
            "i_stop_in_trial": np.zeros(n_samples) + 100,
        }
    )

    description = pd.Series({"subject": "sub-01"})

    return FeaturesDataset(
        features=features, metadata=metadata, description=description
    )


def test_features_dataset_init(sample_feature_dataset):
    ds = sample_feature_dataset
    assert ds.n_features == 3
    assert len(ds) == 10

    # Check getitem
    X, y, crop = ds[0]
    assert X.shape == (3,)
    assert isinstance(y, int) or isinstance(y, np.integer)


def test_features_concat_dataset_stats(sample_feature_dataset):
    # Create valid concat dataset with 2 datasets
    ds1 = sample_feature_dataset

    # Second dataset with offset values to test mean/var
    features2 = ds1.features.copy() + 10.0
    metadata2 = ds1.metadata.copy()
    desc2 = pd.Series({"subject": "sub-02"})

    ds2 = FeaturesDataset(features2, metadata2, desc2)

    concat_ds = FeaturesConcatDataset([ds1, ds2])

    assert len(concat_ds) == 20

    # Test count
    counts = concat_ds.count()
    assert (counts == 20).all()

    # Test mean
    # ds1 mean approx 0, ds2 mean approx 10 -> concat mean approx 5
    means = concat_ds.mean()
    np.testing.assert_allclose(means.values, 5.0, atol=1.0)  # tolerant statistics

    # Test var (pooled variance)
    # var of combined should be approx var of one + (diff/2)^2 = 1 + 25 = 26
    vars_ = concat_ds.var()
    np.testing.assert_allclose(vars_.values, 26.0, atol=5.0)


def test_features_concat_dataset_save(tmp_path, sample_feature_dataset):
    concat_ds = FeaturesConcatDataset([sample_feature_dataset])
    save_dir = tmp_path / "saved_ds"
    os.makedirs(save_dir, exist_ok=True)

    concat_ds.save(str(save_dir))

    # Verify files
    assert (save_dir / "0").is_dir()
    assert (save_dir / "0" / "0-feat.parquet").exists()
    assert (save_dir / "0" / "metadata_df.pkl").exists()
    assert (save_dir / "0" / "description.json").exists()


def test_features_concat_dataset_split(sample_feature_dataset):
    ds1 = sample_feature_dataset
    features2 = ds1.features.copy()
    metadata2 = ds1.metadata.copy()
    desc2 = pd.Series({"subject": "sub-02"})  # Distinct subject
    ds2 = FeaturesDataset(features2, metadata2, desc2)

    concat_ds = FeaturesConcatDataset([ds1, ds2])

    splits = concat_ds.split("subject")
    assert "sub-01" in splits
    assert "sub-02" in splits
    assert len(splits["sub-01"]) == 10

    # Split by dict
    splits_dict = concat_ds.split({"train": [0], "test": [1]})
    assert "train" in splits_dict
    assert len(splits_dict["train"]) == 10


def test_features_concat_dataset_to_dataframe(sample_feature_dataset):
    concat_ds = FeaturesConcatDataset([sample_feature_dataset])

    df = concat_ds.to_dataframe(include_metadata=True, include_target=True)
    assert "target" in df.columns
    assert "i_window_in_trial" in df.columns
    assert "feat_0" in df.columns
    assert len(df) == 10
