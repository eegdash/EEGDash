# eegdash.features.feature_bank.complexity

## Complexity Feature Extraction

This module provides functions to compute various complexity features from signals.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Functions

| `complexity_entropy_preprocessor`(x, /[, m, r, l])   | Precompute neighbor counts for Approximate and Sample Entropy.   |
|------------------------------------------------------|------------------------------------------------------------------|
| `complexity_approx_entropy`(counts_m, ...)           | Calculate Approximate Entropy (ApEn).                            |
| `complexity_multiscale_entropy`(x, /[, m, r, ...])   | Calculate Multiscale Entropy (MSE).                              |
| `complexity_sample_entropy`(counts_m, ...)           | Calculate Sample Entropy (SampEn).                               |
| `complexity_svd_entropy`(x, /[, m, tau])             | Calculate Singular Value Decomposition (SVD) Entropy.            |
| `complexity_lempel_ziv`(x, /[, threshold, ...])      | Calculate Lempel-Ziv Complexity (LZC).                           |
| `complexity_hurst_exp`(x, /)                         | Estimate the Hurst Exponent.                                     |
| `complexity_detrended_fluctuation_analysis`(x, /)    | Calculate the Scaling Exponent via DFA.                          |

### eegdash.features.feature_bank.complexity.complexity_entropy_preprocessor(x, , m=2, r=0.2, l=1)

Precompute neighbor counts for Approximate and Sample Entropy.

This function creates a delay-embedding of the signal and uses a KDTree
to count how many vectors are within a distance ‘r’ of each other.
It computes counts for both dimension ‘m’ and ‘m+1’.

* **Parameters:**
  * **x** (*ndarray*) – The input signal of shape (…, n_times).
  * **m** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Embedding dimension (length of compared sequences).
  * **r** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Tolerance threshold, expressed as a fraction of the signal
    standard deviation.
  * **l** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – The lag or delay between successive embedding vectors.
* **Returns:**
  * **counts_m** (*ndarray*) – Neighbor counts for embedding dimension m.
  * **counts_mp1** (*ndarray*) – Neighbor counts for embedding dimension m + 1.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_approx_entropy(counts_m, counts_mp1,)

Calculate Approximate Entropy (ApEn).

Approximate Entropy quantifies the amount of regularity and the
unpredictability of fluctuations over time-series data. Smaller values
indicate more regular signals.

* **Parameters:**
  * **counts_m** (*ndarray*) – Neighbor counts for embedding dimension m.
  * **counts_mp1** (*ndarray*) – Neighbor counts for embedding dimension m + 1.
* **Returns:**
  Approximate Entropy values. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_multiscale_entropy(x, , m=2, r=0.2, l_max=16)

Calculate Multiscale Entropy (MSE).

Computes the sample entropy (SampEn) for multiple timescales (from 1
to `l_max`), then calculate the integral of the SampEn as a function
of timescale.

* **Parameters:**
  * **x** (*ndarray*) – The input signal of shape (…, n_times).
  * **m** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Embedding dimension (length of compared sequences).
  * **r** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Tolerance threshold, expressed as a fraction of the signal
    standard deviation.
  * **l_max** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – The maximal lag or delay between successive embedding vectors.
* **Returns:**
  MSE values. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_sample_entropy(counts_m, counts_mp1,)

Calculate Sample Entropy (SampEn).

A refinement of Approximate Entropy that is more consistent and less
dependent on signal length. It measures the likelihood that similar
patterns of data will remain similar when the window size increases.

* **Parameters:**
  * **counts_m** (*ndarray*) – Neighbor counts for embedding dimension m.
  * **counts_mp1** (*ndarray*) – Neighbor counts for embedding dimension m + 1.
* **Returns:**
  SampEn values. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_svd_entropy(x, , m=10, tau=1)

Calculate Singular Value Decomposition (SVD) Entropy.

SVD Entropy measures the complexity of the signal’s embedding space.
It indicates the number of independent components required to
reconstruct the signal. Higher values suggest a more complex signal.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **m** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – The embedding dimension.
  * **tau** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – The time delay for embedding.
* **Returns:**
  SVD Entropy values. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_lempel_ziv(x, , threshold=None, normalize=True)

Calculate Lempel-Ziv Complexity (LZC).

LZC evaluates the randomness of a sequence by counting the number
of distinct patterns it contains.

* **Parameters:**
  * **x** (*ndarray*) – The input signal.
  * **threshold** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – Value used to binarize the signal. If None, the median is used.
  * **normalize** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – If True, normalizes the result by:
* **Returns:**
  LZC values. Shape is `x.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

- The implementation follows the constructive algorithm for
  production complexity as described by Kaspar and Schuster [[1]](#r60a22090cb8c-1).
- Optimized with Numba.

### References

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.complexity.complexity_hurst_exp(x,)

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

### eegdash.features.feature_bank.complexity.complexity_detrended_fluctuation_analysis(x,)

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
