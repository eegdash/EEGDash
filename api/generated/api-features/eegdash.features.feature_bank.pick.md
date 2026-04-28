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

### eegdash.features.feature_bank.pick.pick_channel_pairs_preprocessor(\*x, pairs: Iterable[Tuple[str, str]], \_metadata: dict, index: int | Iterable[int] | None = -1, c_index: int | Iterable[int] | None = None, x_index: int | Iterable[int] | None = None, y_index: int | Iterable[int] | None = None, axis: int = -2, c_axis: int = -2)

Pick a subset of channel pairs for further processing steps.

Must follow a preprocessor decorated with `channel_pairer` (or
`channel_directed_pairer`).

* **Parameters:**
  * **\*x** (*tuple* *[**ndarray* *]*) – Input batch.
  * **pairs** (*Iterable* *[**str* *]*) – A list of channel pairs to pick.
  * **index** (*int* *|* *Iterable* *[**int* *]*) – The index (or indices) of the input ndarray[s] to pick channel pairs
    from. Default is -1.
  * **c_index** (*int* *|* *Iterable* *[**int* *]*) – The index (or indices) of the input ndarray[s] to pick channels from.
    Default is [].
  * **x_index** (*int* *|* *Iterable* *[**int* *]*) – The index (or indices) of the input ndarray[s] to pick pair-first
    channels from. Default is [].
  * **y_index** (*int* *|* *Iterable* *[**int* *]*) – The index (or indices) of the input ndarray[s] to pick pair-second
    channels from. Default is [].
  * **axis** (*int*) – The channel pairs axis of the input batch at index `index`. Default
    is -2.
  * **c_axis** (*int*) – The channels axis of the input batch at index `c_index` or
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

### eegdash.features.feature_bank.pick.pick_channels_preprocessor(\*x, channels: Iterable[str], \_metadata: dict, index: int | Iterable[int] = -1, axis: int = -2)

Pick a subset of channels for further processing steps.

* **Parameters:**
  * **\*x** (*tuple* *[**ndarray* *]*) – Input batch.
  * **channels** (*Iterable* *[**str* *]*) – A list of channels to pick.
  * **index** (*int* *|* *Iterable* *[**int* *]*) – The index (or indices) of the input ndarray[s] to pick channels from.
    Default is -1.
  * **axis** (*int*) – The channels axis of the input batch. Default is -2.
* **Returns:**
  *  *\*ndarray* – Sliced input batch containing only the picked channels.
  * **\_metadata** (*dict*) – Updated metadata dictionary.

<!-- !! processed by numpydoc !! -->
