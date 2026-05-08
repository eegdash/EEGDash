# eegdash.features.serialization

Serialization Utilities for Feature Datasets.

This module provides functions for reconstructing feature datasets
from disk. It serves as the inverse of the saving logic implemented in
`FeaturesConcatDataset` and `FeatureExtractor`, allowing
for efficient, parallelized reloading of processed features and their
associated metadata.

<!-- !! processed by numpydoc !! -->

### Functions

| `feature_extractor_from_dict`(fe_dict)      | Get a feature extractor from a dictionary.              |
|---------------------------------------------|---------------------------------------------------------|
| `load_feature_extractor_from_hocon`(path)   | Reads a feature extractor from a HOCON's conf file.     |
| `load_feature_extractor_from_json`(path)    | Reads a feature extractor from a json file.             |
| `load_feature_extractor_from_yaml`(path)    | Reads a feature extractor from a yaml file.             |
| `load_features_concat_dataset`(path[, ...]) | Load a stored `FeaturesConcatDataset` from a directory. |

### eegdash.features.serialization.feature_extractor_from_dict(fe_dict: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → eegdash.features.extractors.FeatureExtractor

Get a feature extractor from a dictionary.

Get a feature extractor object from a dictionary saved by
`FeatureExtractor.to_dict()`.

* **Parameters:**
  **fe_dict** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary representing the feature extractor, with
  `"feature_extractors"` and `"preprocessor"` fields (if applicable).
* **Returns:**
  A feature extractor
* **Return type:**
  FeatureExtractor

#### SEE ALSO
`FeatureExtractor.to_dict`

### Notes

- Only `feature_bank` features and preprocessors
  : are supported.
- Feature extractors including non-function callables are not supported.

<!-- !! processed by numpydoc !! -->

### eegdash.features.serialization.load_feature_extractor_from_hocon(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) → eegdash.features.extractors.FeatureExtractor

Reads a feature extractor from a HOCON’s conf file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the conf file.

#### SEE ALSO
`FeatureExtractor.to_hocon`, `feature_extractor_from_dict`

### Notes

- Only `feature_bank` features and
  : preprocessors are supported.
- Feature extractors including non-function callables are not
  : supported.
- Requires the pyhocon package.

<!-- !! processed by numpydoc !! -->

### eegdash.features.serialization.load_feature_extractor_from_json(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) → eegdash.features.extractors.FeatureExtractor

Reads a feature extractor from a json file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the json file.

#### SEE ALSO
`FeatureExtractor.to_json`, `feature_extractor_from_dict`

### Notes

- Only `feature_bank` features and
  : preprocessors are supported.
- Feature extractors including non-function callables are not
  : supported.

<!-- !! processed by numpydoc !! -->

### eegdash.features.serialization.load_feature_extractor_from_yaml(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) → eegdash.features.extractors.FeatureExtractor

Reads a feature extractor from a yaml file.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – The path to the yaml file.

### Notes

- Only `feature_bank` features and
  : preprocessors are supported.
- Feature extractors including non-function callables are not
  : supported.
- Requires the yaml package.

#### SEE ALSO
`FeatureExtractor.to_yaml`, `feature_extractor_from_dict`

<!-- !! processed by numpydoc !! -->

### eegdash.features.serialization.load_features_concat_dataset(path: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path), ids_to_load: [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, n_jobs: [int](https://docs.python.org/3/library/functions.html#int) = 1) → eegdash.features.datasets.FeaturesConcatDataset

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
