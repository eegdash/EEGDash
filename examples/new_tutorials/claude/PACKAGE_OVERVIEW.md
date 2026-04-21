# EEGDash Package Overview

**Version:** 0.6.0 | **License:** BSD-3-Clause | **Python:** 3.10+

EEGDash is a data-sharing archive and analysis toolkit for MEEG (EEG/MEG) recordings. It provides a unified interface for querying 25+ labs and 27,000+ participants worth of EEG data, preprocessing pipelines integrated with **braindecode**, and a composable feature extraction framework for machine learning.

---

## Package Structure

```
eegdash/
├── api.py                  # EEGDash REST API client
├── dataset/                # Dataset loading & querying
├── features/               # Feature extraction framework  ← primary focus
├── hbn/                    # HBN-specific preprocessing
├── bids_metadata.py        # BIDS metadata utilities
├── downloader.py           # S3/HTTP download utilities
├── schemas.py              # Pydantic metadata validation
└── const.py                # Dataset constants & field mappings
```

**Top-level public API (lazy-loaded):**
- `EEGDash` — REST API client
- `EEGDashDataset` — Main dataset loader (braindecode subclass)
- `EEGChallengeDataset` — Challenge-specific dataset
- `DataIntegrityError` — Exception for data access failures

---

## Braindecode Integration

EEGDash is built on top of **braindecode**, a PyTorch-native EEG deep learning library. This means:

- `EEGDashDataset` extends `braindecode.BaseConcatDataset`
- Individual recordings are `braindecode.RawDataset` instances
- Fully compatible with `braindecode.preprocessing.Preprocessor` and `create_fixed_length_windows()`
- `FeaturesDataset` wraps `braindecode.EEGWindowsDataset`
- All datasets are PyTorch `IterableDataset`-compatible and work with `DataLoader`

**Standard preprocessing workflow:**
```python
from eegdash import EEGDashDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_fixed_length_windows

ds = EEGDashDataset(dataset="ds002718", subject="012", cache_dir="./data")

preprocessors = [
    Preprocessor("pick_channels", ch_names=["Cz", "Pz", "Oz"]),
    Preprocessor("resample", sfreq=128),
    Preprocessor("filter", l_freq=1.0, h_freq=55.0),
]
preprocess(ds, preprocessors, n_jobs=-1)

windows_ds = create_fixed_length_windows(
    ds,
    window_size_samples=256,
    window_stride_samples=128,
)
```

---

## `eegdash/features/` — Feature Extraction Framework

The features module implements a composable, declarative pipeline architecture. Features are organized into a dependency graph where each feature function may require a specific preprocessor to have run first.

### Core Classes

#### `FeatureExtractor` (extractors.py)

The central orchestrator for multi-stage feature extraction. Can be nested to define complex pipelines.

```python
from eegdash.features import FeatureExtractor, signal_mean, spectral_preprocessor, spectral_entropy

extractor = FeatureExtractor({
    "temporal": FeatureExtractor({
        "mean": signal_mean,
    }),
    "spectral": FeatureExtractor(
        {"entropy": spectral_entropy},
        preprocessor=spectral_preprocessor,
    ),
})
```

Key methods:
- `__call__(X, _metadata)` — extract features from a batch
- `fit()` / `partial_fit()` / `clear()` — training interface for trainable features
- `to_json()` / `to_yaml()` / `to_hocon()` — serialization
- `from_json()` / `from_yaml()` — deserialization

#### `FeaturesDataset` (datasets.py)

Stores extracted features for a single recording.

- `features` — `pd.DataFrame` of shape `(n_windows, n_features)`, columns are named feature outputs
- `metadata` — window-level metadata including target labels
- `raw_info` — MNE `Info` object (channel names, sampling rate, etc.)
- `crop_inds` — window start/stop indices in the original trial

#### `FeaturesConcatDataset` (datasets.py)

Manages a collection of `FeaturesDataset` objects (one per recording).

- `split(by_field)` — split by metadata field
- Statistics computed across all recordings
- Serialization via safetensors format

#### `TrainableFeature` (trainable.py)

Abstract base class for learnable features (e.g., CSP). Requires implementing `clear()`, `partial_fit()`, and `fit()`.

---

### Decorator System (decorators.py)

Decorators annotate feature functions with structural metadata that EEGDash uses to validate and build the execution graph.

| Decorator | Purpose |
|---|---|
| `@feature_predecessor(fn)` | Specifies required preprocessor input; `None` means raw signal |
| `@univariate_feature` | Feature returns one value per channel |
| `@bivariate_feature` | Feature returns one value per channel pair |
| `@multivariate_feature` | Feature returns a global/vector value |
| `@metadata_preprocessor` | Marks preprocessors that modify `_metadata` |
| `@preprocessor_output_type(T)` | Specifies the data format output by a preprocessor |

---

### Feature "Kinds" System (kinds.py)

Handles mapping raw numeric outputs to named dictionaries using channel names.

- `UnivariateFeature` — `{ch_name: value}` for each channel
- `BivariateFeature` — `{ch1<>ch2: value}` for each pair, using `BivariateIterator`
- `MultivariateFeature` — `{feature_name: value}` for global outputs

---

### Main Extraction Functions (utils.py)

```python
from eegdash.features import extract_features, fit_feature_extractors

# Fit trainable features (e.g., CSP) on training data
fitted_extractor = fit_feature_extractors(
    windows_ds, extractor, batch_size=8192
)

# Extract features for all recordings
features_ds = extract_features(
    windows_ds, extractor, batch_size=512, n_jobs=-1
)
# Returns FeaturesConcatDataset
```

---

### Introspection (inspect.py)

```python
from eegdash.features import get_all_features, get_feature_kind, get_feature_predecessors

all_features = get_all_features()          # dict of name → function
kind = get_feature_kind(signal_mean)        # UnivariateFeature
preds = get_feature_predecessors(spectral_entropy)  # [spectral_preprocessor]
```

---

## `feature_bank/` — Built-In Feature Library

~70 pre-built feature functions organized by domain.

### Signal Features (signal.py) — univariate, raw signal input

| Feature | Description |
|---|---|
| `signal_mean`, `signal_variance`, `signal_std` | Descriptive statistics |
| `signal_skewness`, `signal_kurtosis` | Higher-order moments |
| `signal_peak_to_peak`, `signal_root_mean_square` | Amplitude measures |
| `signal_zero_crossings`, `signal_line_length` | Temporal activity |
| `signal_hjorth_activity/mobility/complexity` | Hjorth parameters |
| `signal_decorrelation_time` | Temporal correlation decay |

**Preprocessors:**
- `signal_filter_preprocessor` — bandpass filter
- `signal_hilbert_preprocessor` — analytic signal envelope

### Spectral Features (spectral.py) — requires `spectral_preprocessor`

Computed via Welch's PSD method.

| Feature | Description |
|---|---|
| `spectral_root_total_power` | Total signal power |
| `spectral_bands_power` | Power in canonical EEG bands |
| `spectral_slope` | 1/f exponent |
| `spectral_entropy` | Shannon entropy of power spectrum |
| `spectral_hjorth_activity/mobility/complexity` | Spectral Hjorth |
| `spectral_moment` | Spectral moments |
| `spectral_edge` | Spectral edge frequency |

**Preprocessors:**
- `spectral_preprocessor(f_min, f_max)` → `(freq, power)` tuple
- `spectral_normalized_preprocessor()` → normalized PDF-like power
- `spectral_db_preprocessor()` → dB-scaled power

### Connectivity Features (connectivity.py) — bivariate

| Feature | Description |
|---|---|
| `connectivity_magnitude_square_coherence` | MSC between channel pairs |
| `connectivity_imaginary_coherence` | Imaginary part of coherence |
| `connectivity_lagged_coherence` | Phase-lagged coherence |

**Preprocessor:** `connectivity_coherency_preprocessor()` → complex coherency matrix

### Complexity Features (complexity.py) — univariate, requires `complexity_entropy_preprocessor`

| Feature | Description |
|---|---|
| `complexity_approx_entropy` | Approximate entropy |
| `complexity_sample_entropy` | Sample entropy |
| `complexity_svd_entropy` | SVD-based entropy |
| `complexity_multiscale_entropy` | Multiscale entropy |
| `complexity_lempel_ziv` | Lempel-Ziv complexity |

### Dimensionality / Fractal Features (dimensionality.py) — univariate

| Feature | Description |
|---|---|
| `dimensionality_higuchi_fractal_dim` | Higuchi fractal dimension |
| `dimensionality_petrosian_fractal_dim` | Petrosian fractal dimension |
| `dimensionality_katz_fractal_dim` | Katz fractal dimension |
| `dimensionality_hurst_exp` | Hurst exponent |
| `dimensionality_detrended_fluctuation_analysis` | DFA scaling exponent |

### Trainable Feature (csp.py)

`CommonSpatialPattern(n_select=...)` — CSP spatial filters, requires `fit_feature_extractors()` before extraction.

---

## Full End-to-End Example

```python
from eegdash import EEGDashDataset
from eegdash.features import (
    FeatureExtractor, extract_features, fit_feature_extractors,
    signal_mean, signal_variance,
    spectral_preprocessor, spectral_entropy, spectral_bands_power,
    complexity_entropy_preprocessor, complexity_sample_entropy,
    connectivity_coherency_preprocessor, connectivity_magnitude_square_coherence,
)
from braindecode.preprocessing import Preprocessor, preprocess, create_fixed_length_windows

# 1. Load dataset
ds = EEGDashDataset(dataset="ds002718", subject=["012", "013"], cache_dir="./data")

# 2. Preprocess with braindecode
preprocess(ds, [
    Preprocessor("pick_types", eeg=True),
    Preprocessor("resample", sfreq=128),
    Preprocessor("filter", l_freq=1.0, h_freq=55.0),
], n_jobs=-1)

# 3. Create windows
windows_ds = create_fixed_length_windows(ds, window_size_samples=256, window_stride_samples=128)

# 4. Define feature extractor
extractor = FeatureExtractor({
    "temporal": FeatureExtractor({
        "mean": signal_mean,
        "var": signal_variance,
    }),
    "spectral": FeatureExtractor(
        {"entropy": spectral_entropy, "bands": spectral_bands_power},
        preprocessor=spectral_preprocessor,
    ),
    "complexity": FeatureExtractor(
        {"sample_ent": complexity_sample_entropy},
        preprocessor=complexity_entropy_preprocessor,
    ),
    "connectivity": FeatureExtractor(
        {"msc": connectivity_magnitude_square_coherence},
        preprocessor=connectivity_coherency_preprocessor,
    ),
})

# 5. Extract features
features_ds = extract_features(windows_ds, extractor, batch_size=512, n_jobs=-1)

# 6. Access results
for rec_features in features_ds.datasets:
    df = rec_features.features      # pd.DataFrame (n_windows × n_features)
    meta = rec_features.metadata    # window metadata with labels
    print(df.shape, df.columns[:5].tolist())
```

---

## Serialization

Extractors and feature datasets can be saved and reloaded:

```python
# Save extractor config
extractor.to_json("extractor_config.json")

# Reload
from eegdash.features.serialization import load_feature_extractor_from_json
extractor = load_feature_extractor_from_json("extractor_config.json")

# Save extracted features
features_ds.save("./features_output/")

# Reload
from eegdash.features.serialization import load_features_concat_dataset
features_ds = load_features_concat_dataset("./features_output/")
```

---

## Key Design Patterns

1. **Lazy loading (PEP 562):** Top-level imports don't trigger braindecode/torch until used.
2. **Decorator-driven graph:** Feature dependencies (`@feature_predecessor`) are metadata, not runtime logic.
3. **Composable extractors:** Nest `FeatureExtractor` to build arbitrarily complex pipelines.
4. **Metadata-aware functions:** Feature functions receive `_metadata` with MNE Info, channel names, batch size.
5. **Parallelization-ready:** `extract_features()` uses joblib across recordings.
6. **Unified serialization:** Config, parameters, and feature data all persist together.

---

## Existing Tutorials

Located in `examples/tutorials/`:
- `noplot_tutorial_feature_extraction.py` — Complete classification example (sex prediction)
- `noplot_tutorial_pfactor_features.py` — Feature engineering for clinical prediction
- `tutorial_transfer_learning.py` — Deep learning with braindecode integration
