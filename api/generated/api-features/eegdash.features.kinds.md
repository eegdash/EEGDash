# eegdash.features.kinds

Feature Channel-processing Kinds.

This module defines the fundamental feature-processing kinds and the
logic to map raw arrays to named features.

The module provides the classes:

- `UnivariateFeature`
- `BivariateFeature`
- `MultivariateFeature`

<!-- !! processed by numpydoc !! -->

### Classes

| `BivariateFeature`(\*args[, channel_pair_format])   | Feature kind for operations on pairs of channels.                    |
|-----------------------------------------------------|----------------------------------------------------------------------|
| `MultivariateFeature`()                             | Logic wrapper for features that operate on one or more EEG channels. |
| `UnivariateFeature`()                               | Feature kind for operations applied to each channel independently.   |

### *class* eegdash.features.kinds.BivariateFeature(\*args, channel_pair_format: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Bases: `MultivariateFeature`

Feature kind for operations on pairs of channels.

Designed for undirected relationship measures between two signals.

* **Parameters:**
  **channel_pair_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – A format string used to create feature names from pairs of
  channel names. Default is “{}<>{}” for undirected bivariate features
  or “{}->{}” for directed bivariate features.

<!-- !! processed by numpydoc !! -->

#### feature_channel_names(\_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Generate feature names for each unique pair of channels.

* **Parameters:**
  **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.
* **Returns:**
  Formatted strings representing channel pairs (e.g., ‘F3<>F4’).
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [str](https://docs.python.org/3/library/stdtypes.html#str)

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.kinds.MultivariateFeature

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Logic wrapper for features that operate on one or more EEG channels.

This class defines the logic for mapping raw numerical results into
structured, named dictionaries. It determines the “kind” of a feature
(e.g., univariate, bivariate) and handles the association of feature
values with specific channels or channel groupings.

### Notes

Subclasses should override `feature_channel_names()` to define
specific naming conventions for the extracted features.

<!-- !! processed by numpydoc !! -->

#### feature_channel_names(\_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Generate feature-specific names based on input channels.

* **Parameters:**
  **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.
* **Returns:**
  A list of strings defining the naming for each output feature.
  Returns an empty list in the base implementation.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [str](https://docs.python.org/3/library/stdtypes.html#str)

<!-- !! processed by numpydoc !! -->

### *class* eegdash.features.kinds.UnivariateFeature

Bases: `MultivariateFeature`

Feature kind for operations applied to each channel independently.

Used when a single feature value is produced per channel.

<!-- !! processed by numpydoc !! -->

#### feature_channel_names(\_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Return the channel names themselves as feature names.

* **Parameters:**
  **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.
* **Returns:**
  A list of channel names.
* **Return type:**
  [list](https://docs.python.org/3/library/stdtypes.html#list) of [str](https://docs.python.org/3/library/stdtypes.html#str)

<!-- !! processed by numpydoc !! -->
