# Instructions for the eegdash tuturials

## main
- All new files should be written only in the `../examples/new_tutorials` folder.
- Claude is authorized to read every file in the main folder, but is not authorized to changes anything outside `new_tutorials`.

## goals
The main goal of the package is to allow users to download quality brain-related data (mainly eeg), preform preprocessing and extract features (using the `../eegdash/features` sub-module) in order to train ML or DL models. The tutorials should be a user-friendly introduction to the package, focusing mainly on features extruction.
- The `../eegdash/features/feature_bank` submodule includes basic features.

## tutorials
There should be 4 tutorials:
1. _Basic Pipeline_: very basic usage pipiling (download -> preprocess -> feature extraction -> train model -> conclude), with two extentions (advanced preprocessing + SHAP).
2. _Costume made features_: Show users how to recreate basic features from the `feature_bank` as costume made code and integrate them to the feature extraction mechanism.
3. _Advanced feature coding_: Tutorial to show adnvanced feature engieneering using the example of avalanche analysis.
4. _Non-BIDS data_: Show users how to create a dataloader for their own data so they can use the feature extraction kit.

## full details

### TUTORIAL 1
Tutorial 1: Basic Usage for ML models
Main Idea: Use the EEGDASH platform to train a Machine Learning model.
Feature-Based learning - focus on spectral features. Use interpretable model (lightGBM) to predict subjects’ age in the HBN dataset

Goal: Show users the basics of the EEG-DASH package - dataset downloading (EEG-DASH), basic preprocessing (BRAINDECODE), feature extraction (EEG-DASH); ML learning and interpretation for academic purposes.

Steps:
- Load one (or more) of the HBN datasets. Or: load data from `resting state` only from multiple HBN datasets. 
- Perform basic preprocessing using `Braindecode` - notch filter (60 Hz), re-referencing, bandpass filter.
- Extract features of two types - purely spectral, and conectivity-like (use features from features_bank only for simplicity). 
Use `lightGBM` to predict subjects’ ages from features. Make sure no data leakage.

Extension: (ideas to maybe implement)
- preprocess with ICA and ASR
- use SHAP

#### Claude’s thoughts

**Plan:**
The notebook will follow a strict 5-section structure: Load → Preprocess → Window → Extract → Train.

*Data loading:* Use `EEGDashDataset` to query the HBN dataset filtered to `task="RestEyesOpen"` (resting state). Load a
modest subset (e.g., 50–100 subjects) to keep the tutorial runnable. Target variable is `age` from the subject metadata.

*Preprocessing (braindecode):* Apply three `Preprocessor` steps — `notch_filter(60)`, `set_eeg_reference(‘average’)`,
`filter(l_freq=1.0, h_freq=55.0)`. Then `create_fixed_length_windows` (e.g., 4-second epochs, 2-second stride, no overlap
at subject boundaries).

*Feature extraction:* Build a `FeatureExtractor` with two sub-extractors:
- Spectral: `spectral_bands_power`, `spectral_entropy`, `spectral_slope` — wrapped under a `spectral_preprocessor`.
- Connectivity: `connectivity_magnitude_square_coherence` — wrapped under `connectivity_coherency_preprocessor`.
Call `extract_features(windows_ds, extractor, batch_size=512, n_jobs=-1)`.

*Data leakage note:* Crucially, split **by subject** (not by window) before any feature extraction. The tutorial will make
this explicit — pooling all windows then splitting would leak subject-level information.

*ML model:* Average per-subject features across windows, then fit a `lightgbm.LGBMRegressor` for age prediction. Report
MAE and R².

*Extension — SHAP:* Add a collapsed cell showing `shap.TreeExplainer` on the fitted LightGBM model and a beeswarm plot.
Extension — ICA/ASR: mention as a note; these require `mne` / `mne_icalabel` outside the braindecode API and would
complicate the tutorial. Suggest linking to a separate notebook.

**Questions for the team:**
1. Which specific HBN dataset identifier(s) should be used? (e.g., `ds004186`?) And is there a recommended subject count
   for a "light" tutorial run?
2. Should the train/test split be a simple random subject split, or stratified by age group?
3. Is the SHAP extension meant to be a runnable cell or a "further reading" stub?


### TUTORIAL 2
Tutorial 2: Costume made features
Main Idea: Show BASIC user-made features that implement existing simple features in banks. Up to 5-6 features and at least one preprocessor (spectral?)

Notes:
- Verify decorators logic to wrap the functions
- Verify data-size conventions
- Make sure to use correct dependencies.

features to write:
- zero crossing
- signal length
- ... maybe something that doesn't exist in the `feature_bank`?

#### Claude's thoughts

**Plan:**
This tutorial is primarily pedagogical — it teaches the decorator contract by re-implementing known features, then extends
with one original feature.

*Data shape convention to emphasize:* EEGDash uses **Time-Last** → input is `(batch, channels, time)`. Univariate features
must return `(batch, channels)` — i.e., collapse the last axis. This is the most common mistake new users make.

*Features to implement (5 total):*

1. **`my_zero_crossings`** (univariate, raw signal) — count sign changes along the time axis.
   Decorators: `@feature_predecessor()`, `@univariate_feature`.

2. **`my_line_length`** (univariate, raw signal) — sum of absolute first differences.
   Decorators: same as above.

3. **`my_mean_absolute_deviation`** (univariate, raw signal) — `mean(|x - mean(x)|)` — **not in the feature_bank**,
   a natural complement to `signal_std`. Good new addition.
   Decorators: same as above.

4. **`my_spectral_preprocessor`** (preprocessor, signal → (freq, power)) — thin wrapper around `scipy.signal.welch`.
   Decorators: `@feature_predecessor()`, `@preprocessor_output_type(SignalOutputType)`.
   _Note: output type should actually be a custom type or reuse `SignalOutputType`; needs to match what the child
   features expect. Will verify against `spectral_preprocessor` source._

5. **`my_spectral_mean_frequency`** (univariate, spectral) — weighted mean frequency: `sum(f * P) / sum(P)` — **not in
   the feature_bank**. Requires `my_spectral_preprocessor` as predecessor.
   Decorators: `@feature_predecessor(my_spectral_preprocessor)`, `@univariate_feature`.

*Structure of the tutorial:*
- Section 1: explain the decorator system (what `@feature_predecessor` and `@univariate_feature` do, what `_metadata` is)
- Section 2: implement and test each function in isolation with a toy `np.random.randn(2, 4, 256)` batch
- Section 3: integrate into a `FeatureExtractor` and run `extract_features` on a real (or mock) windowed dataset
- Section 4: compare outputs to the equivalent feature_bank function to validate correctness

**Questions for the team:**
1. For the custom spectral preprocessor — should it reuse `SignalOutputType` as its output type, or should we show users
   how to create a custom `BasePreprocessorOutputType` subclass? The latter is more complete but harder.
2. Is `my_mean_absolute_deviation` a good "new" feature to add, or do you have something more interesting in mind?


### TUTORIAL 3
Tutorial 3: Avalanche Toolbox

NOTES: Explore avalanche analysis for single recording (with plots etc.), and later employ decorators and possible fixes to handle correct data shape - in order to use `EEGDash.Features` logic for fast, big-data analysis.

- Code for the original (non-eegdash) functions: `../new_tutorials/avalacnhe_code/avalacnhe_original/`
- Code decorated and matched for EEG-DASH at: `../new_tutorials/avalacnhe_code/avalacnhe_decorated`. This code includes ‘save’ functionalities taht are not required for the tutorial.

Main Idea: Use the EEGDASH platform to extract features from a given dataset - using user made features. Avalanche analysis is a great example of lab-specific features that are not in the features bank.
The package has built-in decorators to transform user-made functions into features that can be extracted with the eegdash feature extraction pipeline.
Use a dataset that is relevant for avalanche analysis (e.g., epilepsy vs healthy, sleep). Maybe show correlation or build a binary classifier?

Goal: Explain how to use your own features as EEG-DASH features in a simple way. Show how flexible and modular the package is, and how simple it is to create a feature extraction pipeline.

Steps:
- Explain the main idea (markdown): The user wrote functions to analyze avalanches from EEG data, and wants to employ the ‘EEGDash.Feature’ module for the extraction of features using said functions.
- Methodological Remarks: Discrimination between preprocessing steps (no ‘feature’ output) and feature extracting steps. Understanding the dependencies of features on preprocessing steps. Multi- /Uni- /Bi- Variate features (avalanches are multivariate).
- Show the user’s initial functions and the fixations needed (mainly shape compatibility).
Show the final, EEG-Dash compatible function, with correct decorators.
- Show a few examples of the avalanche analysis in EEG-Dash form. Explain the differences between choices of bin size, k (multiplier), and dataset (healthy/epileptic).

Optional: Show scientific work and analysis for one recording, expend using eegdash.

#### Claude’s thoughts

**Code review of the decorated avalanche files:**
I read both `avalanche_original/` and `avalanches_decorated.py`. A few issues to fix before writing the tutorial:

1. **Wrong decorator name:** The decorated file uses `@feat.FeaturePredecessor(...)` — this does NOT exist in the current
   API. The correct decorator is `@feature_predecessor(avalanche_preprocessor)` (lowercase, from `eegdash.features`).
2. **Save parameters:** The decorated functions include `save_dir` / `save_prefix` parameters for optional disk output.
   These are fine to keep but should be hidden from the tutorial narrative (they’re an implementation detail, not a
   teaching point).
3. **Shape contract:** The original functions operate on 2D `(channels, time)` arrays. The EEGDash convention is
   `(batch, channels, time)`. The decorated versions iterate over the batch dimension — the tutorial should make this
   transition explicit and explain WHY it’s necessary (vectorized batch processing).

**Tutorial structure:**

- **Part 1 — Single recording:** Show the original avalanche analysis on one raw recording (binarize → bin → detect
  avalanche events → branching parameter, exponents). Include plots. This gives scientific context.
- **Part 2 — The problem:** Explain that running this manually on 1000+ recordings is untenable. Introduce the idea of
  wrapping the functions in EEGDash’s decorator system.
- **Part 3 — Methodological framing:** Preprocessor vs. feature. Multivariate vs. univariate. The avalanche preprocessor
  produces a binarized, binned representation → this is a preprocessor. Branching parameter, alpha/tau/gamma → these
  are multivariate features (one scalar per window, computed over all channels jointly).
- **Part 4 — Decorated code:** Show the shape fix (`for i in range(batch_size): ...`) and the correct decorators.
  Show the final `FeatureExtractor` definition and `extract_features` call.
- **Part 5 — Results:** Compare branching parameter distributions between two groups (e.g., epilepsy vs. healthy). A
  violin plot or simple t-test to show the pipeline produces scientifically meaningful output.

**Questions for the team:**
1. The `avalanches_decorated.py` file — is this authoritative, or should I treat it as a rough draft that needs
   decorator corrections? Specifically: should `save_dir`/`save_prefix` functionality be kept in the tutorial?
2. Which dataset should Tutorial 3 use for the two-group comparison? (epilepsy vs. healthy, or sleep stages?)
3. The file paths in the instructions say `avalacnhe_code/avalacnhe_original/` (misspelled) but the actual directory is
   `avalanche_code/avalanche_original/` — please confirm the correct path.


### TUTORIAL 4
Tutorial 4: Non-BIDS data

Main Idea: The `EEGDaSh.Features` module could be relevant for many types of data - from Multielectrode recording to well-defined 2-photon imaging. If the data can be expressed as (windows x channels x samples) - it fits the feature extraction toolbox.

Show how to convert any type of data to BIDS and feed it into the feature extraction pipeline. We’ll focus on general features that are not EEG-specific.

Goal: Explain the BIDS conversion pipeline and expand our user base to other sub-fields in neuroscience.

Steps: 
- Show a type of (eeg) recording that is not in BIDS format.
- Show how to build corresponding `DataLoaders` for casting the data into a `WindowedDataSet` type.
- Use ` EEGDash.Features` for analysis.
- Maybe add some user-built features?

#### Claude’s thoughts

**Key design question — BIDS conversion vs. direct DataLoader:**
The title says "BIDS conversion" but the description says "build DataLoaders for casting data into a `WindowedDataSet`."
These are two different things. My read: the tutorial should show the **DataLoader route** (the simpler, more broadly
applicable path), because: (a) BIDS conversion is a full data engineering task outside the package’s scope, and (b) the
feature extraction pipeline only requires a `BaseConcatDataset` of `WindowsDataset`/`EEGWindowsDataset` objects — it
does not care how you got there.

**Proposed approach — build a `WindowsDataset` from a numpy array:**

```python
# Simulated non-BIDS data: 10 recordings, 32 channels, 30 seconds @ 256 Hz
import numpy as np
from braindecode.datasets import WindowsDataset, BaseConcatDataset
import pandas as pd, mne

raw_arrays = [np.random.randn(32, 7680) for _ in range(10)]  # (channels, time)
sfreq = 256

# 1. Wrap each array in an mne.RawArray
# 2. Create EEGWindowsDataset from it (via create_from_X_y or manual windowing)
# 3. Stack into BaseConcatDataset
# 4. Pass to extract_features(...)
```

The tutorial will show MNE’s `RawArray` → `mne.Epochs` → braindecode’s `create_windows_from_events` or
`create_fixed_length_windows` as the bridge. This is the minimal-friction path.

**Features to use:** General signal features only (no EEG-specific assumptions) — `signal_mean`, `signal_variance`,
`signal_line_length`, `dimensionality_higuchi_fractal_dim`. This reinforces that the feature extraction toolkit
is domain-agnostic.

**Optional user-built feature:** Could add a simple `my_range` (max − min) as a univariate feature to tie back to
Tutorial 2’s decorator lesson.

**Questions for the team:**
1. Should the tutorial use a **real** non-BIDS dataset (e.g., from PhysioNet or a lab-recorded .edf file), or is a
   simulated numpy array sufficient for teaching purposes?
2. Is the goal BIDS conversion (write files to disk in BIDS format) OR just building a compatible DataLoader in memory?
   These have very different scopes.
3. Should this tutorial include any ML output (like Tutorial 1), or is it purely about data loading?
