# eegdash.features.feature_bank.spectral

## Spectral Feature Extraction

This module provides functions to compute various spectral features from signals.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Functions

| `spectral_preprocessor`(x, /, \*, \_metadata[, ...])   | Compute the Power Spectral Density (PSD) using Welch's method.    |
|--------------------------------------------------------|-------------------------------------------------------------------|
| `spectral_normalized_preprocessor`(f, p, /)            | Normalize the PSD so that the total power equals 1.               |
| `spectral_db_preprocessor`(f, p, /[, eps])             | Convert the PSD to decibels.                                      |
| `spectral_root_total_power`(f, p, /)                   | Calculate the square root of the total spectral power.            |
| `spectral_moment`(f, p, /)                             | Calculate the first spectral moment ('Weighted' Mean Frequency).  |
| `spectral_entropy`(f, p, /)                            | Calculate Spectral Entropy of thepower spectrum.                  |
| `spectral_edge`(f, p, /[, edge])                       | Calculate the Spectral Edge Frequency (SEF).                      |
| `spectral_slope`(f, p, /)                              | Estimate the $1/f$ spectral slope using least-squares regression. |
| `spectral_bands_power`(f, p, /[, bands])               | Calculate total power within specified frequency bands.           |
| `spectral_hjorth_activity`(f, p, /)                    | Calculate Hjorth Activity in the frequency domain.                |
| `spectral_hjorth_mobility`(f, p, /)                    | Calculate Hjorth Mobility in the frequency domain.                |
| `spectral_hjorth_complexity`(f, p, /)                  | Calculate Hjorth Complexity in the frequency domain.              |

### eegdash.features.feature_bank.spectral.spectral_preprocessor(x, , , \_metadata, f_min: float | None = None, f_max: float | None = None, fs: int | None = None, window_size_in_sec: float | None = 4, overlap_in_sec: float | None = None, \*\*kwargs)

Compute the Power Spectral Density (PSD) using Welch’s method.

* **Parameters:**
  * **x** (*ndarray*) – The input signal (shape: …, n_times).
  * **\*\*kwargs** (*dict*) – Supports any scipy.signal.welch arguments like ‘nperseg’ and ‘noverlap’.
  * **fs** (*int* *|* *None*) – Sampling frequency. Defaults to sfreq in MNE’s info. Do not use unless you know what you are doing.
  * **f_min** (*float* *|* *None*) – The minimum frequency. Use None for half the window length. Defaults to the highpass frequency used to MNE’s:meth:~mne.io.Raw.filter.
  * **f_max** (*float* *|* *None*) – The maximum frequency. Use None for Nyquist. Defaults to the lowpass frequency used to MNE’s `filter()`.
  * **window_size_in_sec** (*float* *|* *None*) – Window size in seconds, replacing nperseg. Only used if nperseg is not provided. Defaults to 4 seconds.
  * **overlap_in_sec** (*float* *|* *None*) – Window overlap in seconds, replacing noverlap. Only used if nperseg and noverlap are not provided.defaults to half of window_size_in_sec.
* **Returns:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_normalized_preprocessor(f, p,)

Normalize the PSD so that the total power equals 1.

This is equivalent to treating the PSD as a Probability Density
Function (PDF), which is required for Spectral Entropy.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density.
* **Returns:**
  * **f** (*ndarray*) – Frequency vector (unchanged).
  * **p** (*ndarray*) – Normalized Power Spectral Density.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_db_preprocessor(f, p, , eps=1e-15)

Convert the PSD to decibels.

Calculated as:

$$
10 \cdot \log_{10}\left(P\left(f\right) + \epsilon\right).

$$

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density.
  * **eps** (*float* *,* *optional*) – A small constant to prevent log of zero (default: 1e-15).
* **Returns:**
  * **f** (*ndarray*) – Frequency vector (unchanged).
  * *ndarray* – Power Spectral Density in decibels. Shape is `p.shape`.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_root_total_power(f, p,)

Calculate the square root of the total spectral power.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density (PSD).
* **Returns:**
  The root total power. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_moment(f, p,)

Calculate the first spectral moment (‘Weighted’ Mean Frequency).

When applied to a normalized PSD, this represents the “center of mass”
of the power spectrum.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Normalized Power Spectral Density.
* **Returns:**
  The mean frequency of the signal. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_entropy(f, p,)

Calculate Spectral Entropy of thepower spectrum.

Spectral Entropy (SE) measures the complexity or “disorder” of a signal.
A high SE indicates a flat, broad spectrum (e.g., white noise), while a
low SE indicates a spectrum concentrated in a few frequency components.

It is calculated as:

$$
SE = -\sum_f P\left(f\right) \ln\left(P\left(f\right)\right)

$$

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Normalized Power Spectral Density (treated as a PDF).
* **Returns:**
  The entropy values. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_edge(f, p, , edge=0.9)

Calculate the Spectral Edge Frequency (SEF).

The frequency below which a certain percentage (e.g., 90%) of
the total power is contained.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Normalized Power Spectral Density (treated as a PDF).
  * **edge** (*float* *,* *optional*) – The fraction of total power (default is 0.9 for SEF90).
* **Returns:**
  The spectral edge frequency. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

### Notes

Optimized with Numba `fastmath` for rapid scanning of cumulative power.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_slope(f, p,)

Estimate the $1/f$ spectral slope using least-squares regression.

This measures the slope and intercept of the PSD in log-log space.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density in decibels.
* **Returns:**
  A dictionary containing:
  - `'exp'`: The slope/exponent (scaling).
  - `'int'`: The y-intercept (offset).
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_bands_power(f, p, , bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4.5), 'theta': (4.5, 8)})

Calculate total power within specified frequency bands.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density (PSD).
  * **bands** (*dict* *,* *optional*) – Mapping of band names to (min, max) frequency tuples.
* **Returns:**
  The summed power for each band.
* **Return type:**
  dict

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_hjorth_activity(f, p,)

Calculate Hjorth Activity in the frequency domain.

Activity represents the total power of the signal, calculated here
as the integral (sum) of the Power Spectral Density.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density.
* **Returns:**
  Total spectral power. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_hjorth_mobility(f, p,)

Calculate Hjorth Mobility in the frequency domain.

Mobility is an estimate of the mean frequency.
For a normalized PSD, it is calculated as:

$$
\sqrt{\sum_f f^2 P(f)},

$$

where $\sum P(f) = 1$ (since the PSD is normalized).

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Normalized Power Spectral Density.
* **Returns:**
  Spectral mobility. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.spectral.spectral_hjorth_complexity(f, p,)

Calculate Hjorth Complexity in the frequency domain.

Complexity measures the bandwidth or the “irregularity” of the spectrum.

For a normalized PSD, it is calculated as:

$$
\frac{\sqrt{\sum_f f^4 P(f)}}{\sum_f f^2 P(f)}.

$$

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Normalized Power Spectral Density.
* **Returns:**
  Spectral complexity. Shape is `p.shape[:-1]`.
* **Return type:**
  ndarray

### References

- Hjorth, B. (1970). EEG analysis based on time domain properties.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

<!-- !! processed by numpydoc !! -->
