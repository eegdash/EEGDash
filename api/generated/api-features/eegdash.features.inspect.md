# eegdash.features.inspect

Feature Bank Inspection and Discovery.

This module provides utilities for introspecting the feature extraction
registry. It allows users and system components to discover available
features, identify their kinds, and traverse the preprocessing dependency
graph.

The module provides the following utilities:

- `get_all_features()` — Lists all final feature functions.
- `get_all_feature_preprocessors()` — Lists all available preprocessing
  steps.
- `get_feature_kind()` — Identifies the dimensionality of a feature.
- `get_feature_predecessors()` — Traces the dependency lineage of a
  feature.
- `get_all_feature_kinds()` — Lists all valid feature categories.

<!-- !! processed by numpydoc !! -->

### Functions

| `get_all_feature_preprocessors`()                | Get a list of all available preprocessor functions.              |
|--------------------------------------------------|------------------------------------------------------------------|
| `get_all_feature_kinds`()                        | Get a list of all available feature 'kind' classes.              |
| `get_all_features`()                             | Get a list of all available feature functions.                   |
| `get_all_preprocessor_output_types`()            | Get a list of all available preprocessor output type classes.    |
| `get_feature_kind`(feature)                      | Get the 'kind' of a feature function.                            |
| `get_feature_predecessors`(feature_or_extractor) | Get the dependency hierarchy for a feature or feature extractor. |

### eegdash.features.inspect.get_all_feature_preprocessors() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]]

Get a list of all available preprocessor functions.

Scans the `feature_bank` module for all functions
that participate in the dependency graph but do not produce final
features (e.g., lack a feature_kind).

* **Returns:**
  A list of (name, function) tuples for all discovered feature
  preprocessors.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.inspect.get_all_feature_kinds() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[TypeAliasForwardRef('eegdash.features.extractors.MultivariateFeature')]]]

Get a list of all available feature ‘kind’ classes.

Scans the `kinds` module for all classes
that subclass `MultivariateFeature`.

* **Returns:**
  A list of (name, class) tuples for all discovered feature kinds.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.inspect.get_all_features() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]]

Get a list of all available feature functions.

Scans the `feature_bank` module for functions
that have been decorated with a feature_kind.

* **Returns:**
  A list of (name, function) tuples for all discovered feature functions.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.inspect.get_all_preprocessor_output_types() → [list](https://docs.python.org/3/library/stdtypes.html#list)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [type](https://docs.python.org/3/library/functions.html#type)[BasePreprocessorOutputType]]]

Get a list of all available preprocessor output type classes.

Scans the `feature_bank` module for all classes
that subclass `BasePreprocessorOutputType`.

* **Returns:**
  A list of (name, class) tuples for all discovered preprocessor output types.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.inspect.get_feature_kind(feature: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → eegdash.features.extractors.MultivariateFeature

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

### eegdash.features.inspect.get_feature_predecessors(feature_or_extractor: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [None](https://docs.python.org/3/library/constants.html#None)) → [list](https://docs.python.org/3/library/stdtypes.html#list)

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
