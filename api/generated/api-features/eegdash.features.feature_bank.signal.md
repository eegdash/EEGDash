# eegdash.features.feature_bank.signal

## Signal-Level Feature Extraction

This module provides temporal and statistical features computed
directly from time-series data.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Module Attributes

| `signal_hjorth_activity`(x, /, \*\*kwargs)   | Calculate the Hjorth Activity of the signal.   |
|----------------------------------------------|------------------------------------------------|

### Functions

| `signal_envelope_preprocessor`(x, /)               | Compute the amplitude envelope of the signal using Hilbert transform.   |
|----------------------------------------------------|-------------------------------------------------------------------------|
| `signal_filter_preprocessor`(x, /, \*, ...[, ...]) | Linear-phase FIR band-pass filter.                                      |
| `signal_decorrelation_time`(x, /, \*, \_metadata)  | Calculate the Decorrelation Time of the signal.                         |
| `signal_hjorth_activity`(x, /, \*\*kwargs)         | Calculate the Hjorth Activity of the signal.                            |
| `signal_hjorth_complexity`(x, /)                   | Calculate the Hjorth Complexity of the signal.                          |
| `signal_hjorth_mobility`(x, /)                     | Calculate the Hjorth Mobility of the signal.                            |
| `signal_kurtosis`(x, /, \*\*kwargs)                | Compute the temporal kurtosis of the signal.                            |
| `signal_line_length`(x, /)                         | Calculate the Mean Signal Line Length.                                  |
| `signal_mean`(x, /)                                | Compute the temporal mean of the signal.                                |
| `signal_peak_to_peak`(x, /, \*\*kwargs)            | Calculate the peak-to-peak (maximum range) of the signal.               |
| `signal_quantile`(x, /[, q])                       | Compute the q-th quantile of the signal.                                |
| `signal_root_mean_square`(x, /)                    | Calculate the Root Mean Square (RMS) magnitude.                         |
| `signal_skewness`(x, /, \*\*kwargs)                | Compute the temporal skewness of the signal.                            |
| `signal_std`(x, /, \*\*kwargs)                     | Compute the temporal standard deviation of the signal.                  |
| `signal_variance`(x, /, \*\*kwargs)                | Compute the temporal variance of the signal.                            |
| `signal_zero_crossings`(x, /[, threshold])         | Count the number of times the signal crosses the zero axis.             |

### eegdash.features.feature_bank.signal.signal_envelope_preprocessor(x,)

Compute the amplitude envelope of the signal using Hilbert transform.

* **Parameters:**
  **x** (*ndarray*) – Input signal
* **Returns:**
  The signal envelope, with the same shape as the input.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_filter_preprocessor(x, , , \_metadata, f_min, f_max, num_taps=None)

Linear-phase FIR band-pass filter.

* **Parameters:**
  * **x** (*ndarray*) – Input signal
  * **f_min** ([*float*](https://docs.python.org/3/library/functions.html#float)) – Low cutoff frequency (Hz)
  * **f_max** ([*float*](https://docs.python.org/3/library/functions.html#float)) – High cutoff frequency (Hz)
  * **num_taps** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of filter taps (must be odd for exact linear phase)
* **Returns:**
  The band-pass filtered signal, with the same shape as the input.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_decorrelation_time(x, , , \_metadata)

Calculate the Decorrelation Time of the signal.

This function computes the time it takes for the signal to
decorrelate, defined as the first time lag where the autocorrelation
function drops to zero.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The time (in seconds) until the signal decorrelates.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

This function uses the [Wiener-Khinchin Theorem](https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem) to
compute the autocorrelation via the inverse FFT of the power spectrum.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_hjorth_activity(x, , \*\*kwargs)

Calculate the Hjorth Activity of the signal.

Activity is defined as the variance of the signal itself.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The Hjorth Activity value. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

The activity is calculated using the following formula:

$$
\text{Activity}\left(x\left(t\right)\right) = \operatorname{Var}\left(x\left(t\right)\right)

$$

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

for more details, see the [Wikipedia entry](https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Activity).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_hjorth_complexity(x,)

Calculate the Hjorth Complexity of the signal.

Complexity represents the change in frequency. The parameter
compares the signal’s similarity to a pure sine wave, where value
of 1 indicates a perfect sine wave.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The complexity value. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

#### SEE ALSO
`signal_hjorth_mobility`

### Notes

The complexity is calculated using the following formula:

$$
\text{Complexity}\left(x\left(t\right)\right) = \frac{\text{Mobility}\left(\frac{\mathrm{d}x\left(t\right)}{\mathrm{dt}}\right)}{\text{Mobility}\left(x\left(t\right)\right)}

$$

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

For more details, see the [Wikipedia entry](https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Complexity).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_hjorth_mobility(x,)

Calculate the Hjorth Mobility of the signal.

Mobility is defined as the standard deviation of the signal’s first
derivative normalized by the standard deviation of the signal itself.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The Hjorth Mobility value. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

#### SEE ALSO
`signal_hjorth_activity`

### Notes

The mobility is calculated using the following formula:

$$
\text{Mobility}\left(x\left(t\right)\right) = \sqrt{\frac{\text{Var}\left(\frac{\mathrm{d}x\left(t\right)}{\mathrm{dt}}\right)}{\text{Var}\left(x\left(t\right)\right)}}

$$

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

for more details, see the [Wikipedia entry](https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Mobility).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_kurtosis(x, , \*\*kwargs)

Compute the temporal kurtosis of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to [`scipy.stats.kurtosis()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html#scipy.stats.kurtosis).
* **Returns:**
  The kurtosis of the signal along the temporal axis.
  Shape is `x.shape[:-1]`
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_line_length(x,)

Calculate the Mean Signal Line Length.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The mean absolute vertical distance between consecutive samples.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_mean(x,)

Compute the temporal mean of the signal.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The mean of the signal along the temporal axis.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_peak_to_peak(x, , \*\*kwargs)

Calculate the peak-to-peak (maximum range) of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to [`numpy.ptp()`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html#numpy.ptp).
* **Returns:**
  The peak-to-peak amplitude.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

This function wraps [`numpy.ptp()`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html#numpy.ptp); see the NumPy documentation for
details on additional keyword arguments.

For a theoretical overview of Peak-To-Peak amplitude in signal analysis,
see the [Wikipedia entry](https://en.wikipedia.org/wiki/Peak-to-peak).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_quantile(x, , q: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) = 0.5, \*\*kwargs)

Compute the q-th quantile of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **q** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* *array_like* *,* *optional*) – The quantile to compute. 0.5 (default) is the median.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to [`numpy.quantile()`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile).
* **Returns:**
  The quantile values.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

This function wraps [`numpy.quantile()`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile); see the NumPy documentation for
details on additional keyword arguments.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_root_mean_square(x,)

Calculate the Root Mean Square (RMS) magnitude.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The RMS amplitude of the signal.
  Shape is `x.shape[:-1]`
* **Return type:**
  ndarray

### Notes

For the RMS definition, see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Root_mean_square).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_skewness(x, , \*\*kwargs)

Compute the temporal skewness of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to [`scipy.stats.skew()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html#scipy.stats.skew).
* **Returns:**
  The skewness of the signal along the temporal axis.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_std(x, , \*\*kwargs)

Compute the temporal standard deviation of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to `np.std()`.
* **Returns:**
  The standard deviation of the signal along the temporal axis.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_variance(x, , \*\*kwargs)

Compute the temporal variance of the signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Additional keyword arguments passed to `np.var()`.
* **Returns:**
  The variance of the signal along the temporal axis.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.signal.signal_zero_crossings(x, , threshold=1e-15)

Count the number of times the signal crosses the zero axis.

This function identifies points where the signal changes sign or
enters/leaves a defined noise floor (threshold).

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **threshold** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – A small epsilon value to treat values near zero as exactly zero,
    preventing false counts due to floating-point noise.
* **Returns:**
  The count of zero crossings.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

For a theoretical overview of zero-crossing rate in signal analysis,
see the [Wikipedia entry](https://en.wikipedia.org/wiki/Zero_crossing).

<!-- !! processed by numpydoc !! -->
