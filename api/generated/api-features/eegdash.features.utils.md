# eegdash.features.utils

Feature Extraction Utilities.

This module provides the primary entry points for applying feature extraction
pipelines to windowed datasets.

The module provides the following functions:

- `extract_features()` — The main interface for computing features
  across an entire concatenated dataset.
- `fit_feature_extractors()` — Fits trainable features using a
  representative dataset.

<!-- !! processed by numpydoc !! -->

### Functions

| `extract_features`(concat_dataset, features, \*)   | Extract features from a collection of windowed recordings.   |
|----------------------------------------------------|--------------------------------------------------------------|
| `fit_feature_extractors`(concat_dataset, features) | Fit trainable feature extractors on a concatenated dataset.  |

### eegdash.features.utils.extract_features(concat_dataset: BaseConcatDataset, features: FeatureExtractor | Dict[str, Callable] | List[Callable], , batch_size: int = 512, n_jobs: int = 1) → FeaturesConcatDataset

Extract features from a collection of windowed recordings.

This function applies a feature extraction pipeline to every
individual recording in a `BaseConcatDataset`.

* **Parameters:**
  * **concat_dataset** (*BaseConcatDataset*) – A concatenated dataset of `WindowsDataset` or
    `EEGWindowsDataset` instances.
  * **features** (*FeatureExtractor* *or* *dict* *or* *list*) – The feature extractor(s) to apply. Can be a
    `FeatureExtractor` instance,
    a dictionary of named feature functions, or a list of feature
    functions.
  * **batch_size** (*int* *,* *default 512*) – The size of batches used for feature extraction within each recording.
  * **n_jobs** (*int* *,* *default 1*) – The number of parallel jobs to use for processing different
    recordings simultaneously.
* **Returns:**
  A unified collection of feature datasets corresponding to the
  input recordings.
* **Return type:**
  *FeaturesConcatDataset*

<!-- !! processed by numpydoc !! -->

### eegdash.features.utils.fit_feature_extractors(concat_dataset: BaseConcatDataset, features: FeatureExtractor | Dict[str, Callable] | List[Callable], batch_size: int = 8192) → FeatureExtractor

Fit trainable feature extractors on a concatenated dataset.

Scans the provided feature pipeline for components that require training
(subclasses of `TrainableFeature`).
If found, the function iterates through the dataset in batches to
perform partial fitting before finalization.

* **Parameters:**
  * **concat_dataset** (*BaseConcatDataset*) – The dataset used to train the feature extractors.
  * **features** (*FeatureExtractor* *or* *dict* *or* *list*) – The feature extractor pipeline(s) to fit.
  * **batch_size** (*int* *,* *default 8192*) – The batch size to use when streaming data through the
    `partial_fit()` phase.
* **Returns:**
  The fitted feature extractor instance, ready for feature extraction.
* **Return type:**
  *FeatureExtractor*

### Notes

If the provided extractors are not trainable, the function returns
the original input without modification.

<!-- !! processed by numpydoc !! -->
