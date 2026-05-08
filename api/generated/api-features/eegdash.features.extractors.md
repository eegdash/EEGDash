# eegdash.features.extractors

Core Feature Extraction Orchestration.

This module defines the fundamental building blocks for creating feature
extraction pipelines.

The module provides the base class:

- `FeatureExtractor` - The central pipeline for execution trees.

<!-- !! processed by numpydoc !! -->

### Classes

| `FeatureExtractor`(feature_extractors[, ...])   | Pipeline for multi-stage feature extraction.   |
|-------------------------------------------------|------------------------------------------------|

### *class* eegdash.features.extractors.FeatureExtractor(feature_extractors: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)], preprocessor: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `TrainableFeature`

Pipeline for multi-stage feature extraction.

This class manages a collection of feature extraction functions or nested
extractors. It handles the application of shared preprocessing, validates
the dependency graph between components, and aggregates results into a
named dictionary compatible with `FeaturesDataset`.

* **Parameters:**
  * **feature_extractors** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *callable* *]*) – A dictionary where keys are the base names for the features and
    values are the extraction functions or other `FeatureExtractor`
    instances.
  * **preprocessor** (*callable* *,* *optional*) – A shared preprocessing function applied to the input data
    before it is passed to child extractors.

#### preprocessor

The shared preprocessing stage for this extractor.

* **Type:**
  callable or None

#### feature_extractors_dict

The validated dictionary of child extractors.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### features_kwargs

A collection of all keyword arguments used by the preprocessor and
child functions, preserved for metadata tracking.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

### Notes

The extractor automatically detects if any child components are
trainable and will require a `fit()` phase before
extraction can occur.

### Examples

```pycon
>>> # Create a simple extractor
>>> fe = FeatureExtractor(
...     feature_extractors={'mean': signal_mean, 'std': signal_std}
... )
```

```pycon
>>> # Extract from a batch (2 windows, 3 channels, 100 samples)
>>> X = np.random.randn(2, 3, 100)
>>> results = fe(X, _batch_size=2, _ch_names=['O1', 'Oz', 'O2'])
```

<!-- !! processed by numpydoc !! -->

#### preprocess(\*x, \_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

Apply the shared preprocessor to the input data.

* **Parameters:**
  * **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* *ndarray*) – The input data batch.
  * **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.
* **Returns:**
  * *tuple* – The preprocessed data passed as a tuple to support multi-output
    preprocessors.
  * **\_metadata** (*dict*) – The preprocessed metadata. Only relevant for metadata preprocessors.

<!-- !! processed by numpydoc !! -->

#### clear()

Clear the state of all trainable sub-features.

<!-- !! processed by numpydoc !! -->

#### partial_fit(\*x, y=None, \_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict))

Propagate partial fitting to all trainable children.

* **Parameters:**
  * **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* *ndarray*) – The input data batch.
  * **y** (*ndarray* *,* *optional*) – Target labels for supervised training.
  * **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.

<!-- !! processed by numpydoc !! -->

#### fit()

Fit all trainable sub-features.

<!-- !! processed by numpydoc !! -->

#### *property* feature_names *: [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*

A list of full feature names (without the channel names).

<!-- !! processed by numpydoc !! -->

#### to_dict() → [dict](https://docs.python.org/3/library/stdtypes.html#dict)

Dumps the feature extractor to a dictionary.

* **Returns:**
  A dictionary representing the feature extractor, with
  `"feature_extractors"` and `"preprocessor"` fields (if applicable).
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### SEE ALSO
`feature_extractor_from_dict`

### Notes

Feature extractors including non-function callables are not supported.

<!-- !! processed by numpydoc !! -->

#### to_json(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path))

Dumps the feature extractor to a json file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the json file.

#### SEE ALSO
`load_feature_extractor_from_json`, `FeatureExtractor.to_dict`

### Notes

Feature extractors including non-function callables are not supported.

<!-- !! processed by numpydoc !! -->

#### to_yaml(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path))

Dumps the feature extractor to a yaml file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the yaml file.

#### SEE ALSO
`load_feature_extractor_from_yaml`, `FeatureExtractor.to_dict`

### Notes

- Feature extractors including non-function callables are not
  : supported.
- Requires the pyyaml package.

<!-- !! processed by numpydoc !! -->

#### to_hocon(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path))

Dumps the feature extractor to a HOCON’s conf file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the conf file.

#### SEE ALSO
`load_feature_extractor_from_hocon`, `FeatureExtractor.to_dict`

### Notes

- Feature extractors including non-function callables are not
  : supported.
- Requires the pyhocon package.

<!-- !! processed by numpydoc !! -->
