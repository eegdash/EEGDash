# eegdash.features.feature_bank.utils

## Feature Extraction Utilities

This module provides the following helper functions:
- get_valid_freq_band: Validates and returns frequency boundaries based on Nyquist and resolution.
- slice_freq_band: Slices frequency vector and associated data arrays to a specific range.
- reduce_freq_bands: Reduces spectral data into discrete frequency bands by aggregating bins

<!-- !! processed by numpydoc !! -->

### Functions

| `get_valid_freq_band`(fs, n[, f_min, f_max])    | Validate and return frequency boundaries based on Nyquist and resolution.   |
|-------------------------------------------------|-----------------------------------------------------------------------------|
| `preprocessor_as_feature`(\*x)                  | A pass-through feature, returning its preprocessor output as is.            |
| `reduce_freq_bands`(f, x, bands[, reduce_func]) | Reduce spectral data into discrete frequency bands by aggregating bins.     |
| `set_spectral_default_kwargs`(kwargs, metadata) | Sets default parameters for spectral preprocecssors.                        |
| `slice_freq_band`(f, \*x[, f_min, f_max])       | Slice frequency vector and associated data arrays to a specific range.      |
| `spectral_kwargs`(func)                         | A decorator for functions receiving spectral-like parameters.               |

### eegdash.features.feature_bank.utils.get_valid_freq_band(fs, n, f_min=None, f_max=None)

Validate and return frequency boundaries based on Nyquist and resolution.

* **Parameters:**
  * **fs** ([*float*](https://docs.python.org/3/library/functions.html#float)) – The sampling frequency in Hz.
  * **n** ([*int*](https://docs.python.org/3/library/functions.html#int)) – The number of points in the signal/window.
  * **f_min** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Requested minimum frequency. Defaults to 2 \* resolution (f0).
  * **f_max** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Requested maximum frequency. Defaults to Nyquist frequency.
* **Returns:**
  **f_min, f_max** – The validated frequency boundaries.
* **Return type:**
  [float](https://docs.python.org/3/library/functions.html#float)
* **Raises:**
  * [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError) – If f_min is below the minimum resolvable frequency.
  * [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError) – If f_max is above the Nyquist frequency.

### Examples

```pycon
>>> get_valid_freq_band(fs=100, n=1000)
(0.2, 50.0)
>>> get_valid_freq_band(fs=200, n=500, f_min=1, f_max=80)
(1, 80)
```

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.utils.preprocessor_as_feature(\*x)

A pass-through feature, returning its preprocessor output as is.

Use if the preprocessor is a feature by itself, and it should also be treated as a
feature.

* **Parameters:**
  **\*x** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – Any preprocessor output.
* **Returns:**
  **\*x** – The input (as is).
* **Return type:**
  [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.utils.reduce_freq_bands(f, x, bands, reduce_func=<function sum>)

Reduce spectral data into discrete frequency bands by aggregating bins.

This function identifies the frequency indices belonging to specific
bands and applies a reduction function (like sum or mean) to collapse
the frequency axis.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **x** (*ndarray*) – Spectral data. Can be multi-dimensional.
    The last dimension must match the length of f.
  * **bands** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Mapping of band names to (min, max) frequency tuples.
  * **reduce_func** (*callable* *,* *optional*) – Function to aggregate the values. Default is np.sum.
* **Returns:**
  **x_bands** – Dictionary where keys are the band names from bands and values
  are the reduced arrays. The last dimension of the input x
  is removed.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)
* **Raises:**
  [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError) – If a band name is not a string.
  If a band limit tuple does not contain exactly two values or
  if min > max.
  If the requested band limits fall outside the range of the
  available frequency vector f.

### Examples

```pycon
>>> f = np.array([0, 2, 4, 6, 8, 10])
>>> x = np.array([
...     [1, 2, 3, 4, 5, 6],
...     [60, 50, 40, 30, 20, 10],
... ])
>>> bands = {'low': (0, 5), 'high': (5, 11)} # check assertion
>>> results = reduce_freq_bands(f, x, bands, reduce_func=np.sum)
>>> results['low']
array([6, 150])
>>> results['high']
array([15, 60])
```

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.utils.set_spectral_default_kwargs(kwargs, metadata)

Sets default parameters for spectral preprocecssors.

- Set the default frequency limits to the bandpass frequencies (if available).
- Set the default sampling frequency to freq in MNE’s info.
- Use window_size_in_sec if nperseg is not provided. Defaults to 4 seconds.
- Use overlap_in_sec if nperseg and noverlap are not provided.
  : Defaults to half the window size.
- Set the axis to -1

* **Parameters:**
  * **kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of keyword arguments.
  * **metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dictionary of record and batch metadata.
* **Returns:**
  * **f_min** (*float*) – Minimum frequency.
  * **f_max** (*float*) – Maximum frequency.
  * **kwargs** (*dict*) – A dictionary of keyword arguments.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.utils.slice_freq_band(f, \*x, f_min=None, f_max=None)

Slice frequency vector and associated data arrays to a specific range.

* **Parameters:**
  * **f** (*ndarray*) – The frequency vector.
  * **\*x** (*ndarray*) – One or more data arrays to be sliced along the frequency axis.
    The last dimension of each array must match the length of f.
  * **f_min** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Lower frequency bound.
  * **f_max** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Upper frequency bound.
* **Returns:**
  * **f** (*ndarray*) – The cropped frequency vector.
  * **\*xl** (*ndarray*) – The cropped data arrays.

### Examples

```pycon
>>> # Create 0-10 Hz frequencies
>>> freqs = np.array([0, 2, 4, 6, 8, 10])
```

```pycon
>>> # Create data: (2 channels, 6 frequency bins)
>>> data = np.array([[10, 20, 30, 40, 50, 60],
...                  [15, 25, 35, 45, 55, 65]])
```

```pycon
>>> # Keep only the range 4Hz to 8Hz
>>> f_s, d_s = slice_freq_band(freqs, data, f_min=4, f_max=8)
```

```pycon
>>> f_s
array([4, 6, 8])~
>>> d_s
array([[30, 40, 50],
       [35, 45, 55]])
```

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.utils.spectral_kwargs(func: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable))

A decorator for functions receiving spectral-like parameters.

* **Parameters:**
  **func** (*Callable*) – A function receiving spectral-like parameters.
* **Returns:**
  A wrapped function with extra parameters and a suitable docstring.
* **Return type:**
  Callable

<!-- !! processed by numpydoc !! -->
