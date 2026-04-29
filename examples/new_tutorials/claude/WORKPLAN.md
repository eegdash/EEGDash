# Tutorial Workplan

## Status

| Tutorial | Status | Notes |
|---|---|---|
| T1 — Basic Pipeline | **Draft complete** | R1 + R2 corrections applied (2026-04-29) |
| T2 — Custom Features | **Draft complete** | R1 corrections applied (2026-04-29) |
| T3 — Avalanche Toolbox | **Draft complete** | R1 corrections applied (2026-04-29); dataset TBD |
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

**R1 corrections applied (2026-04-29):**
- Removed all `[?]_#` review markers (notes 1–3 resolved)
- Added eyes-open/eyes-closed note to data loading section
- Justified 50% windowing overlap for feature-based ML
- Added `spectral_entropy` and `spectral_slope` to extractor; updated markdown and inspect cell

**R2 corrections applied (2026-04-29):**
- Removed R1 eyes-open/closed blockquote from data loading cell
- Windowing cell: replaced `create_fixed_length_windows` with `create_windows_from_events`; extracts only eyes-closed intervals via `mapping={"instructed_toCloseEyes": 0}` and `trial_stop_offset_samples = 40 * sfreq`
- Feature extraction cell: split into three branches — non-alpha spectral (all channels), occipital alpha (`pick_channels_preprocessor` + nested spectral extractor with `ALPHA_BAND`), connectivity (MSC)
- Applied `preprocessor=` before `feature_extractors=` convention throughout (also applied to T2, T3)

**Remaining open:**
- `OCCIPITAL_CHS = []` — must be filled with HBN EGI-128 occipital channel names
- Train/test split strategy (random subject split vs. age-stratified)
- Whether SHAP extension is a runnable cell or a stub
- Dataset size (~280 subjects) — team decision
- ICA pipeline — consult Tom and Oren

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

**R1 corrections applied (2026-04-29):**
- Rewrote intro: fixed "seven features" → "six features + one preprocessor"; added explicit learning goals
- Rewrote decorator system section: converted to a table, added mandatory-order callout, added axis=-1 diagram
- Changed `my_zero_crossings` implementation to product-sign approach (pedagogically distinct from bank)
- Changed `my_line_length` to two-step explicit form with geometric explanation
- Emphasised `axis=-1` and decorator order throughout sections 1 and 2

**Remaining open:**
- Note [?]_5 ("rephrase the sentence") — marker absent from notebook; sentence not identified. Team needs to locate and mark the target sentence.
- "Tutorials from different types of variation" — currently only univariate features; consider adding one bivariate example in a future pass.
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

**R1 corrections applied (2026-04-29):**
- Note 4: Removed "No caching" row from The Scaling Problem table; updated "solves all four" → "all three"
- Note 6: Shape fix section updated to use `axis=-1`/`axis=-2` with real avalanche code snippet and rule-of-thumb explanation
- Note 7: Removed OLD API code block from section 5 decorator cheat-sheet
- Note 8: `@metadata_preprocessor` description now distinguishes reading (no decorator needed) from writing (decorator required)
- Note 9: Changed `""` intermediate keys to `0`; updated diagram and extractor code; explained non-string keys suppress name prefixing
- Note 10: Added `partial()` explanation block in extractor cell
- Note 12: DCC explanation updated — "each node has exactly one predecessor; feature nodes cannot take other feature nodes as input"
- Note 13: DCC computation changed from a loop to a single `pd.concat` + vectorised column operation
- Code: Removed `t_max_method="lab"` option from `tau_exponent` (simplification as noted)

**Remaining open (deferred to team):**
- Note 1: Section 0 intro — Gal will rewrite
- Note 2: Section 1 original code — "change to original or remove" — decision needed
- Note 3: Part 2 needs real dataset (MEG?)
- Note 5: Function/feature name changes — team to confirm current vs. desired names
- Note 11: "Change FeatureExtractor order of preprocessors and dict" — unclear intent
- Code: `StandardizedSignalType` / split standardization — not in current code; source unclear
- Code: `_fit_power_law` simplification — no simpler replacement identified yet
- Code: `starts`/`ends` list vs array representation — decision needed
- Code: `gamma_exponent` base-e / linear fit — marked as "consider"; deferred

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
