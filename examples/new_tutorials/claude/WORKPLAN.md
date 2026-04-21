# Tutorial Workplan

## Status

| Tutorial | Status | Notes |
|---|---|---|
| T1 — Basic Pipeline | **Ready to draft** | Datasets confirmed |
| T2 — Custom Features | **Ready to draft** | No blockers |
| T3 — Avalanche Toolbox | **Structure only** (no dataset yet) | Dataset TBD; decorated file is a draft |
| T4 — Non-BIDS Data | **Structure only** (no dataset yet) | Dataset TBD |

---

## Confirmed API (verified from source, not from outdated examples)

```python
# Correct imports
from eegdash.features import (
    FeatureExtractor,
    extract_features,
    fit_feature_extractors,
    feature_predecessor,       # decorator
    univariate_feature,        # decorator
    bivariate_feature,         # decorator
    multivariate_feature,      # decorator
    preprocessor_output_type,  # decorator
    metadata_preprocessor,     # decorator
    SignalOutputType,
)

# FeatureExtractor constructor
FeatureExtractor(
    feature_extractors: dict[str, callable],
    preprocessor: callable | None = None,
)

# extract_features — batch_size and n_jobs are keyword-only
extract_features(concat_dataset, features, *, batch_size=512, n_jobs=1)

# fit_feature_extractors — for trainable features (e.g., CSP)
fit_feature_extractors(concat_dataset, features, batch_size=8192)
```

**Data shape convention (Time-Last):**
- Feature function input: `(batch, channels, time)`
- Univariate feature output: `(batch, channels)` — collapse last axis
- Multivariate feature output: `(batch, n_output_features)`

**Decorator contract for a custom univariate feature:**
```python
@feature_predecessor()           # depends on raw signal
@univariate_feature              # one value per channel
def my_feature(x):               # x: (batch, channels, time)
    return x.mean(axis=-1)       # → (batch, channels)
```

**Decorator contract for a custom preprocessor:**
```python
@feature_predecessor()
@preprocessor_output_type(SignalOutputType)
def my_preprocessor(x, /):
    return transformed_x         # same shape as input
```

---

## Tutorial 1 — Basic Pipeline

**File:** `tutorial_1_basic_pipeline.ipynb`

**Sections:**
1. Data loading — `EEGDashDataset` with HBN resting-state task
2. Preprocessing — braindecode: notch 60 Hz, average re-reference, bandpass 1–55 Hz
3. Windowing — `create_fixed_length_windows` (4 s epochs, 2 s stride)
4. Feature extraction — spectral + connectivity features via `extract_features`
5. Data leakage guard — subject-level train/test split *before* feature aggregation
6. LightGBM model — predict age, report MAE and R²
7. (Extension) SHAP interpretation — `shap.TreeExplainer` + beeswarm plot

**Features used (all from feature_bank):**
- `spectral_preprocessor` → `spectral_bands_power`, `spectral_entropy`, `spectral_slope`
- `connectivity_coherency_preprocessor` → `connectivity_magnitude_square_coherence`

**Datasets:** `ds005505` (release 1) + `ds005506` (release 2), resting state only.

**Remaining open:**
- Train/test split strategy (random subject split vs. age-stratified)
- Whether SHAP extension is a runnable cell or a stub

---

## Tutorial 2 — Custom Features

**File:** `tutorial_2_custom_features.ipynb`

**Sections:**
1. The decorator system — what `@feature_predecessor` and `@univariate_feature` do
2. Shape conventions — input/output contract with a toy batch
3. Re-implementing `zero_crossings` and `line_length` with decorators
4. New feature: `my_mean_absolute_deviation` (not in feature_bank)
5. Custom spectral preprocessor + `my_spectral_mean_frequency` feature
6. Validation — compare outputs to feature_bank equivalents

**5 features to implement:**

| Name | Kind | Predecessor | Note |
|---|---|---|---|
| `my_zero_crossings` | univariate | raw signal | re-implement existing |
| `my_line_length` | univariate | raw signal | re-implement existing |
| `my_mean_absolute_deviation` | univariate | raw signal | new feature |
| `my_spectral_preprocessor` | preprocessor | raw signal | thin welch wrapper |
| `my_spectral_mean_frequency` | univariate | `my_spectral_preprocessor` | new feature |

**Pending:**
- Confirm whether to show custom `BasePreprocessorOutputType` subclass (advanced) or reuse `SignalOutputType`

---

## Tutorial 3 — Avalanche Toolbox

**File:** `tutorial_3_avalanche.ipynb`

**Sections:**
1. Scientific background — what are neural avalanches, why do they matter
2. Single-recording demo — original functions, plots (branching parameter, power-law fit)
3. The scaling problem — why manual loops over 1000 recordings fail
4. Preprocessor vs. feature distinction — methodological framing
5. Shape fix — original 2D `(channels, time)` → EEGDash 3D `(batch, channels, time)`
6. Correct decorators — fixing `FeaturePredecessor` → `feature_predecessor`, adding `@multivariate_feature`
7. Full pipeline — `FeatureExtractor` + `extract_features` on full dataset
8. Analysis — group comparison (epilepsy vs. healthy or sleep stages), violin plot

**Known issues in `avalanches_decorated.py` to fix before using in tutorial:**
- `@feat.FeaturePredecessor(...)` → `@feature_predecessor(avalanche_preprocessor)`
- Strip all `save_dir` / `save_prefix` parameters — saving is out of scope for the tutorial

**Status:** Build full notebook structure and decorated code now; leave dataset-dependent cells (sections 2 and 8)
as clearly marked stubs. Dataset TBD.

---

## Tutorial 4 — Non-BIDS Data

**File:** `tutorial_4_non_bids.ipynb`

**Sections:**
1. Motivation — feature extraction is domain-agnostic if data is `(windows, channels, samples)`
2. The bridge — `mne.RawArray` → braindecode windowing → `BaseConcatDataset`
3. Feature extraction — general signal + dimensionality features (no EEG-specific assumptions)
4. Optional user-built feature — `my_range` (max − min) to tie back to Tutorial 2

**Proposed data path (in-memory, no real file required):**
```python
# Simulated non-BIDS recording
raw_array = np.random.randn(32, 7680)  # 32 ch, 30 s @ 256 Hz
info = mne.create_info(ch_names=[f"ch{i}" for i in range(32)], sfreq=256, ch_types="misc")
raw = mne.io.RawArray(raw_array, info)
# → create_fixed_length_windows → extract_features
```

**Status:** Build full notebook structure now; leave data-loading cells as clearly marked stubs. Dataset TBD.
