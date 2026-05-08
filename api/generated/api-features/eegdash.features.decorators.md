# eegdash.features.decorators

Feature Metadata Decorators.

This module provides  decorators used to annotate feature extraction
functions with structural metadata. These annotations define the dependency
graph (via predecessors) and the data format (via feature kinds).

The module provides the following decorators:

- `feature_predecessor()` — Specifies the required input transformation
  for a feature.
- `feature_kind()` — Defines the dimensionality of the feature output.
  - `univariate_feature()` — Sugar for per-channel features.
  - `bivariate_feature()` — Sugar for per channel-pair features.
  - `multivariate_feature()` — Sugar for global/all-channel features.
- `metadata_preprocessor()` — Specifies a preprocessor returning a modified
  metadata instance.
- `channel_pairer()` — Specifies a preprocessor that creates channel pairs.
  - `channel_pairer_undirected()` — Sugar for undirected pairs.
  - `channel_pairer_directed()` — Sugar for directed pairs.

<!-- !! processed by numpydoc !! -->

### Module Attributes

| `univariate_feature`(func, \*, kind)              | Apply the `feature_kind()` decorator to a function.   |
|---------------------------------------------------|-------------------------------------------------------|
| `bivariate_feature`(func, \*, kind)               | Apply the `feature_kind()` decorator to a function.   |
| `multivariate_feature`(func, \*, kind)            | Apply the `feature_kind()` decorator to a function.   |
| `channel_pairer_undirected`(func, \*[, directed]) | Apply the `channel_pairer()` decorator to a function. |
| `channel_pairer_directed`(func, \*[, directed])   | Apply the `channel_pairer()` decorator to a function. |

### Functions

| `bivariate_feature`(func, \*, kind)               | Apply the `feature_kind()` decorator to a function.                 |
|---------------------------------------------------|---------------------------------------------------------------------|
| `channel_pairer`([directed])                      | Decorator to set a feature preprocessor as a channel pairer.        |
| `channel_pairer_directed`(func, \*[, directed])   | Apply the `channel_pairer()` decorator to a function.               |
| `channel_pairer_undirected`(func, \*[, directed]) | Apply the `channel_pairer()` decorator to a function.               |
| `feature_kind`(kind)                              | Decorator to specify the operational dimensionality of a feature.   |
| `feature_predecessor`(\*parent_extractor_type)    | Decorator to specify parent extractors for a feature function.      |
| `metadata_preprocessor`(func)                     | Decorator to set a feature preprocessor as a metadata preprocessor. |
| `multivariate_feature`(func, \*, kind)            | Apply the `feature_kind()` decorator to a function.                 |
| `preprocessor_output_type`(output_type)           | Decorator to specify the expected output type of a preprocessor.    |
| `univariate_feature`(func, \*, kind)              | Apply the `feature_kind()` decorator to a function.                 |

### eegdash.features.decorators.bivariate_feature(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), \*, kind: MultivariateFeature = <eegdash.features.kinds.BivariateFeature object>) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to mark a feature as bivariate.

Specifies that the feature operates on pairs of channels.
The output will be formatted as a dictionary with keys matching the
original channel name pairs.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.channel_pairer(directed: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to set a feature preprocessor as a channel pairer.

This decorator lets a feature preprocessor get an additional `pairs`
keyword argument, and sets a metadata field named `'ch_pair_iterator'`
containing a `BivariateIterator`
accordingly before calling the underlying preprocessor.

* **Parameters:**
  **directed** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – Whether the preprocessor assumes *directed* or *undirected* bivariate
  iteration.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.channel_pairer_directed(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), , directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.channel_pairer_undirected(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), , directed: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to mark a feature preprocessor as an undirected channel pairer.

Specifies that the feature preprocessor operates on undirected pairs of
channels.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.feature_kind(kind: MultivariateFeature) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to specify the operational dimensionality of a feature.

This decorator attaches a “feature kind” instance to a function,
determining how the `FeatureExtractor`
should map the resulting numerical arrays to channel names.

* **Parameters:**
  **kind** (*MultivariateFeature*) – An instance of a feature kind class, such as
  `UnivariateFeature` or
  `BivariateFeature`.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.feature_predecessor(\*parent_extractor_type: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [Type](https://docs.python.org/3/library/typing.html#typing.Type)]) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to specify parent extractors for a feature function.

This decorator attaches a list of immediate parent preprocessing steps to
a feature extraction function. This metadata is used by the
`FeatureExtractor` to validate the
execution tree.

* **Parameters:**
  **\*parent_extractor_type** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* *callable* *or* [*type*](https://docs.python.org/3/library/functions.html#type)) – A list of preprocessing functions that this feature immediately
  depends on.
  Default is [`SignalOutputType`].

### Notes

A feature can have multiple potential predecessors.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.metadata_preprocessor(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to set a feature preprocessor as a metadata preprocessor.

A metadata preprocessor must get a keyword argument named `"_metadata"`
and return a copy of it as its last output argument.

* **Parameters:**
  **func** (*callable*) – The feature preprocessor function to decorate.
* **Returns:**
  The decorated function with the metadata_preprocessor attribute set.
* **Return type:**
  callable

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.multivariate_feature(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), \*, kind: MultivariateFeature = <eegdash.features.kinds.MultivariateFeature object>) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to mark a feature as multivariate.

Indicates that the feature operates on all channels simultaneously. The
output naming convention is determined by the feature’s internal logic
rather than channel labels.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.preprocessor_output_type(output_type: [Type](https://docs.python.org/3/library/typing.html#typing.Type)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to specify the expected output type of a preprocessor.

* **Parameters:**
  **output_type** (*Type*) – The expected output type for the preprocessor. Must be a
  `BasePreprocessorOutputType`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If the provided output_type does not inherit from
  `BasePreprocessorOutputType`.

<!-- !! processed by numpydoc !! -->

### eegdash.features.decorators.univariate_feature(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), \*, kind: MultivariateFeature = <eegdash.features.kinds.UnivariateFeature object>) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Decorator to mark a feature as univariate.

Indicates that the feature is computed for each channel independently.
The output will be formatted as a dictionary with keys matching the
original channel names.

<!-- !! processed by numpydoc !! -->
