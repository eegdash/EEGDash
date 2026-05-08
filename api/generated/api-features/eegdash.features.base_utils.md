# eegdash.features.base_utils

Basic Feature Extraction Utilities

This module defines basic utilities for feature extraction.

<!-- !! processed by numpydoc !! -->

### Functions

| `channel_names_to_indices`(channels, ch_names)   | Converts a list of channel names to channel indices in another list.   |
|--------------------------------------------------|------------------------------------------------------------------------|
| `get_underlying_func`(func)                      | Retrieve the original Python function from a potential wrapper.        |

### Classes

| `BivariateIterator`(pairs[, directed])   | Pairs iterator for iterating pairs of channels.   |
|------------------------------------------|---------------------------------------------------|

### *class* eegdash.features.base_utils.BivariateIterator(pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]] | [int](https://docs.python.org/3/library/functions.html#int), directed=False)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Pairs iterator for iterating pairs of channels.

* **Parameters:**
  * **pairs** (*Iterable* *[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/library/functions.html#int) *,* [*int*](https://docs.python.org/3/library/functions.html#int) *]* *]*  *|* [*int*](https://docs.python.org/3/library/functions.html#int)) – If an iterable of tuples is given, it represents the channel index
    pairs to iterate
    If an integer `n` is given, iterate through all unique pairs
    out of `n` channels.
  * **directed** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If an integer was given in `pairs`, this parameter controls whether
    all directed pairs should be iterated.
    Otherwise this parameter is ignored.
    Default is False.

<!-- !! processed by numpydoc !! -->

#### get_pair_iterators() → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)]

Get indices for pairs of channels.

Computes the upper triangle indices of an (n, n) matrix,
excluding the diagonal.

* **Returns:**
  The row and column indices for the unique pairs.
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple) of ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.base_utils.channel_names_to_indices(channels: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], ch_names: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]

Converts a list of channel names to channel indices in another list.

* **Parameters:**
  * **channels** (*List* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – A list of channel names.
  * **ch_names** (*List* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]*) – A list of existing channel names to take indices from.
* **Returns:**
  A list of channel indices.
* **Return type:**
  List[[int](https://docs.python.org/3/library/functions.html#int)]
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If the channel name was not found in the existing channels list.

<!-- !! processed by numpydoc !! -->

### eegdash.features.base_utils.get_underlying_func(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) → [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

Retrieve the original Python function from a potential wrapper.

* **Parameters:**
  **func** (*callable*) – The function to unwrap. Typically a raw function, a
  `functools.partial` object, or a Numba `Dispatcher`.
* **Returns:**
  The underlying Python function.
* **Return type:**
  callable

### Notes

This utility specifically handles:
\* **functools.partial**: Returns the `.func` attribute.
\* **numba.Dispatcher**: Returns the `.py_func` attribute.

<!-- !! processed by numpydoc !! -->
