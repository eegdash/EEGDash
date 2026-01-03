import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from eegdash.dataset.dataset import EEGDashDataset
from eegdash.downloader import (
    download_s3_file,
)
from eegdash.features.datasets import (
    FeaturesConcatDataset,
    FeaturesDataset,
    _compute_stats,
    _pooled_var,
)
from eegdash.features.extractors import (
    FeatureExtractor,
)
from eegdash.features.feature_bank.csp import (
    CommonSpatialPattern,
)
from eegdash.features.feature_bank.spectral import spectral_edge, spectral_preprocessor
from eegdash.features.serialization import (
    load_features_concat_dataset,
)
from eegdash.features.utils import (
    extract_features,
    fit_feature_extractors,
)
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
)
from eegdash.paths import (
    get_default_cache_dir,
)
from eegdash.features.feature_bank.complexity import (
    complexity_approx_entropy,
    complexity_entropy_preprocessor,
    complexity_lempel_ziv,
    complexity_sample_entropy,
    complexity_svd_entropy,
)
from eegdash.features.feature_bank.dimensionality import (
    dimensionality_detrended_fluctuation_analysis,
    dimensionality_higuchi_fractal_dim,
    dimensionality_hurst_exp,
    dimensionality_katz_fractal_dim,
    dimensionality_petrosian_fractal_dim,
)


def test_serialization_gap(tmp_path, features_dataset):
    # Trigger serialization.py 67, 124
    p = tmp_path / "ser"
    p.mkdir()
    # Save a real dataset there
    ds_concat = FeaturesConcatDataset([features_dataset])
    ds_concat.save(str(p))

    res = load_features_concat_dataset(str(p))
    assert len(res.datasets) == 1
