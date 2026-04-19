# eegdash.features.feature_bank.dimensionality

## Dimensionality Features Extraction

This module provides functions to compute various dimensionality features
from signals.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Functions

| `dimensionality_higuchi_fractal_dim`(x, /[, ...])     | Calculate Higuchi's Fractal Dimension (HFD).   |
|-------------------------------------------------------|------------------------------------------------|
| `dimensionality_petrosian_fractal_dim`(x, /)          | Calculate Petrosian Fractal Dimension (PFD).   |
| `dimensionality_katz_fractal_dim`(x, /)               | Calculate Katz Fractal Dimension (KFD).        |
| `dimensionality_hurst_exp`(x, /)                      | Estimate the Hurst Exponent.                   |
| `dimensionality_detrended_fluctuation_analysis`(x, /) | Calculate the Scaling Exponent via DFA.        |

### eegdash.features.feature_bank.dimensionality.dimensionality_higuchi_fractal_dim(x, , k_max=10, eps=1e-07)

Calculate Higuchi’s Fractal Dimension (HFD).

Higuchi’s Fractal Dimension [[1]](#r81c8ba91077a-1) [[2]](#r81c8ba91077a-2) estimates the complexity of a time series
by measuring the mean length of the curve at different time scales $k$. It is
highly robust for non-stationary signals.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **k_max** (*int* *,* *optional*) – Maximum time interval (delay) used for calculating curve lengths.
  * **eps** (*float* *,* *optional*) – A small constant to avoid log of zero during regression.
* **Returns:**
  The Higuchi’s Fractal Dimension values.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

Optimized with Numba.

For a theoretical overview of Higuchi’s Fractal Dimension, see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Higuchi_dimension).

### References

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.dimensionality.dimensionality_petrosian_fractal_dim(x,)

Calculate Petrosian Fractal Dimension (PFD).

Petrosian Fractal Dimension [[1]](#r3e02912f8ce5-1) [[2]](#r3e02912f8ce5-2) provides a fast estimate of fractal
dimension by analyzing the number of sign changes in the signal’s
first derivative.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The Petrosian Fractal Dimension values.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### References

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.dimensionality.dimensionality_katz_fractal_dim(x,)

Calculate Katz Fractal Dimension (KFD).

Katz Fractal Dimension [[1]](#re99721b5c64b-1) [[2]](#re99721b5c64b-2) is calculated as the ratio between the total
path length and the maximum planar distance from the first point to any other
point.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The Katz Fractal Dimension values.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### References

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.dimensionality.dimensionality_hurst_exp(x,)

Estimate the Hurst Exponent.

The Hurst exponent quantifies the long-term memory and predictability of
a time series. It indicates whether a process is purely random, tends to
trend in the same direction (persistent), or tends to reverse its direction
(anti-persistent).

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The estimated Hurst Exponents.
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

This function calculate the Gamma Function Ratios and Bias Correction Factors
to apply the Anis-Lloyd correction for small sample sizes.

For more details on the Hurst Exponent and R/S analysis, visit the
[Wikipedia entry](https://en.wikipedia.org/wiki/Hurst_exponent#Rescaled_range_(R/S)_analysis).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.dimensionality.dimensionality_detrended_fluctuation_analysis(x,)

Calculate the Scaling Exponent via DFA.

Detrended Fluctuation Analysis (DFA) is a method used to detect long-range
temporal correlations (LRTC) in non-stationary signals. It is a more robust
way to estimate the Hurst exponent when the data is noisy or has shifting trends.

* **Parameters:**
  **x** (*ndarray*) – The input signal.
* **Returns:**
  The DFA scaling exponents ($alpha$).
  Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

Optimized with Numba.

For a theoretical overview of Detrended Fluctuation Analysis, see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis).

<!-- !! processed by numpydoc !! -->
