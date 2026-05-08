# Feature Package Overview

The `eegdash.features` namespace re-exports feature extractors,
decorators, and dataset utilities from the underlying submodules so callers can
import the most common helpers from a single place. To avoid duplicated
documentation in the API reference, the classes themselves are documented in
their defining modules (see the links below). This page focuses on the
high-level orchestration helpers that only live in the package `__init__`.

## High-level discovery helpers

### eegdash.features.get_all_features() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]]

Get a list of all available feature functions.

Scans the `feature_bank` module for functions
that have been decorated with a feature_kind.

* **Returns:**
  A list of (name, function) tuples for all discovered feature functions.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.get_feature_kind(feature: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → eegdash.features.extractors.MultivariateFeature

Get the ‘kind’ of a feature function.

Identifies whether a feature is univariate, bivariate, or multivariate
using decorators.

* **Parameters:**
  **feature** (*callable*) – The feature function to inspect.
* **Returns:**
  An instance of the feature kind.
* **Return type:**
  `MultivariateFeature`

<!-- !! processed by numpydoc !! -->

### eegdash.features.get_feature_predecessors(feature_or_extractor: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)

Get the dependency hierarchy for a feature or feature extractor.

This function recursively traverses the parent_extractor_type attribute
of a feature or extractor to build a list representing its dependency lineage.

* **Parameters:**
  **feature_or_extractor** (*callable*) – The feature function or
  `FeatureExtractor` instance
  to inspect.
* **Returns:**
  A nested list representing the dependency tree. For a simple linear
  chain, this will be a flat list from the specific feature up to the
  base signal input. For multiple dependencies, it contains tuples
  of sub-dependencies.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list)

### Notes

The traversal stops when it reaches a predecessor of `None`, which
typically represents the raw signal.

### Examples

```pycon
>>> # Example: Linear dependency with a branching dependency
>>> print(get_feature_predecessors(feature_bank.spectral_entropy))
    [<function spectral_entropy at 0x...>,
    <function spectral_normalized_preprocessor at 0x...>,
    <function spectral_preprocessor at 0x...>,
    (None, [<function signal_hilbert_preprocessor at 0x...>, None])]
```

<!-- !! processed by numpydoc !! -->

### eegdash.features.get_all_feature_kinds() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[TypeAliasForwardRef('eegdash.features.extractors.MultivariateFeature')]]]

Get a list of all available feature ‘kind’ classes.

Scans the `kinds` module for all classes
that subclass `MultivariateFeature`.

* **Returns:**
  A list of (name, class) tuples for all discovered feature kinds.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.get_all_preprocessor_output_types() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[BasePreprocessorOutputType]]]

Get a list of all available preprocessor output type classes.

Scans the `feature_bank` module for all classes
that subclass `BasePreprocessorOutputType`.

* **Returns:**
  A list of (name, class) tuples for all discovered preprocessor output types.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

## Dataset and extraction utilities

### eegdash.features.extract_features(concat_dataset: [BaseConcatDataset](https://braindecode.org/stable/generated/braindecode.datasets.BaseConcatDataset.html#braindecode.datasets.BaseConcatDataset), features: FeatureExtractor | [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)], , batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 512, n_jobs: [int](https://docs.python.org/3/library/functions.html#int) = 1) → FeaturesConcatDataset

Extract features from a collection of windowed recordings.

This function applies a feature extraction pipeline to every
individual recording in a `BaseConcatDataset`.

* **Parameters:**
  * **concat_dataset** (*BaseConcatDataset*) – A concatenated dataset of `WindowsDataset` or
    `EEGWindowsDataset` instances.
  * **features** (*FeatureExtractor* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list)) – The feature extractor(s) to apply. Can be a
    `FeatureExtractor` instance,
    a dictionary of named feature functions, or a list of feature
    functions.
  * **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *default 512*) – The size of batches used for feature extraction within each recording.
  * **n_jobs** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *default 1*) – The number of parallel jobs to use for processing different
    recordings simultaneously.
* **Returns:**
  A unified collection of feature datasets corresponding to the
  input recordings.
* **Return type:**
  *FeaturesConcatDataset*

<!-- !! processed by numpydoc !! -->

### eegdash.features.fit_feature_extractors(concat_dataset: [BaseConcatDataset](https://braindecode.org/stable/generated/braindecode.datasets.BaseConcatDataset.html#braindecode.datasets.BaseConcatDataset), features: FeatureExtractor | [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)], batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 8192) → FeatureExtractor

Fit trainable feature extractors on a concatenated dataset.

Scans the provided feature pipeline for components that require training
(subclasses of `TrainableFeature`).
If found, the function iterates through the dataset in batches to
perform partial fitting before finalization.

* **Parameters:**
  * **concat_dataset** (*BaseConcatDataset*) – The dataset used to train the feature extractors.
  * **features** (*FeatureExtractor* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *or* [*list*](https://docs.python.org/3/library/stdtypes.html#list)) – The feature extractor pipeline(s) to fit.
  * **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *default 8192*) – The batch size to use when streaming data through the
    `partial_fit()` phase.
* **Returns:**
  The fitted feature extractor instance, ready for feature extraction.
* **Return type:**
  *FeatureExtractor*

### Notes

If the provided extractors are not trainable, the function returns
the original input without modification.

<!-- !! processed by numpydoc !! -->

### eegdash.features.load_features_concat_dataset(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), ids_to_load: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, n_jobs: [int](https://docs.python.org/3/library/functions.html#int) = 1) → eegdash.features.datasets.FeaturesConcatDataset

Load a stored `FeaturesConcatDataset` from a directory.

This function reconstructs a concatenated dataset by loading individual
`FeaturesDataset` instances from numbered subdirectories.

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *or* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The root directory where the dataset was previously saved. This
    directory should contain numbered subdirectories.
  * **ids_to_load** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – A list of specific recording IDs (subdirectory names) to load.
    If **None**, all numbered subdirectories found in the path are
    loaded in ascending numerical order.
  * **n_jobs** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *default=1*) – The number of CPU cores to use for parallel loading. Set to -1 to
    use all available processors.
* **Returns:**
  A unified concatenated dataset containing the loaded recordings.
* **Return type:**
  FeaturesConcatDataset

#### SEE ALSO
[`braindecode.datautil.load_concat_dataset`](https://braindecode.org/stable/generated/braindecode.datautil.load_concat_dataset.html#braindecode.datautil.load_concat_dataset)

### Notes

The function expects the directory structure generated by
`FeaturesConcatDataset.save()`. It automatically reconstructs
the feature DataFrames (safetensors), metadata (Pickle), recording info
(FIF), and preprocessing keyword arguments (JSON).

<!-- !! processed by numpydoc !! -->

## See also

- `eegdash.features.extractors` for the feature-extraction base classes
  such as `FeatureExtractor`.
- `eegdash.features.datasets` for dataset wrappers like
  `FeaturesConcatDataset`.
- `eegdash.features.feature_bank.*` for the concrete feature families
  (complexity, connectivity, spectral, and more).
