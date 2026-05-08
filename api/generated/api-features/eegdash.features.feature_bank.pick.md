# eegdash.features.feature_bank.pick

## Channel-picking feature preprocessors

This module provides the ability to pick specific channels or channel
pairs for further processing.

### Data Shape Convention

By default, this module follows a **Channel-panultimate** convention:

* **Input:** `(..., channel, :)`
* **Output:** same as input

The choice of the channel dimension can be adjusted using the `axis`
parameter.

<!-- !! processed by numpydoc !! -->

### Functions

| `pick_channel_pairs_preprocessor`(\*x, pairs, ...)   | Pick a subset of channel pairs for further processing steps.   |
|------------------------------------------------------|----------------------------------------------------------------|
| `pick_channels_preprocessor`(\*x, channels, ...)     | Pick a subset of channels for further processing steps.        |

### eegdash.features.feature_bank.pick.pick_channel_pairs_preprocessor(\*x, pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]], \_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict), index: [int](https://docs.python.org/3/library/functions.html#int) | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = -1, c_index: [int](https://docs.python.org/3/library/functions.html#int) | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, x_index: [int](https://docs.python.org/3/library/functions.html#int) | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, y_index: [int](https://docs.python.org/3/library/functions.html#int) | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, axis: [int](https://docs.python.org/3/library/functions.html#int) = -2, c_axis: [int](https://docs.python.org/3/library/functions.html#int) = -2)

Pick a subset of channel pairs for further processing steps.

Must follow a preprocessor decorated with `channel_pairer` (or
`channel_directed_pairer`).

* **Parameters:**
  * **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *[**ndarray* *]*) – Input batch.
  * **pairs** (*Iterable* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – A list of channel pairs to pick.
  * **index** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *Iterable* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]*) – The index (or indices) of the input ndarray[s] to pick channel pairs
    from. Default is -1.
  * **c_index** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *Iterable* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]*) – The index (or indices) of the input ndarray[s] to pick channels from.
    Default is [].
  * **x_index** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *Iterable* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]*) – The index (or indices) of the input ndarray[s] to pick pair-first
    channels from. Default is [].
  * **y_index** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *Iterable* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]*) – The index (or indices) of the input ndarray[s] to pick pair-second
    channels from. Default is [].
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The channel pairs axis of the input batch at index `index`. Default
    is -2.
  * **c_axis** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The channels axis of the input batch at index `c_index` or
    `x_index` or `y_index`. Default is -2.
* **Returns:**
  *  *\*ndarray* – Sliced input batch containing only the picked channels.
  * **\_metadata** (*dict*) – Updated metadata dictionary.

### Notes

Picking by index pair, e.g., `x[i, j]`, is not directly supported because
the result may not be an numpy.ndarray. It is preferred to use a pair
axis. It is possible, however, to pick just by `x_index` with
`c_axis=0`, then pick again just by `y_index` with `c_index=1` (or
vice versa) to effectively pick the indices intersection of such an
numpy.ndarray.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.pick.pick_channels_preprocessor(\*x, channels: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)], \_metadata: [dict](https://docs.python.org/3/library/stdtypes.html#dict), index: [int](https://docs.python.org/3/library/functions.html#int) | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[int](https://docs.python.org/3/library/functions.html#int)] = -1, axis: [int](https://docs.python.org/3/library/functions.html#int) = -2)

Pick a subset of channels for further processing steps.

* **Parameters:**
  * **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *[**ndarray* *]*) – Input batch.
  * **channels** (*Iterable* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – A list of channels to pick.
  * **index** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *Iterable* *[*[*int*](https://docs.python.org/3/library/functions.html#int) *]*) – The index (or indices) of the input ndarray[s] to pick channels from.
    Default is -1.
  * **axis** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The channels axis of the input batch. Default is -2.
* **Returns:**
  *  *\*ndarray* – Sliced input batch containing only the picked channels.
  * **\_metadata** (*dict*) – Updated metadata dictionary.

<!-- !! processed by numpydoc !! -->
