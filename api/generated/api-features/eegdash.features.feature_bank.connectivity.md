# eegdash.features.feature_bank.connectivity

## Connectivity Feature Extraction

This module computes bivariate connectivity features based on the complex
coherency between pairs of channels.

### Data Shape Convention

This module follows a **Time-Last** convention:

* **Input:** `(..., time)`
* **Output:** `(...,)`

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).

<!-- !! processed by numpydoc !! -->

### Functions

| `connectivity_coherency_preprocessor`(x, /, \*, ...)   | Compute Complex Coherency for all unique channel pairs.   |
|--------------------------------------------------------|-----------------------------------------------------------|
| `connectivity_magnitude_square_coherence`(f, c, /)     | Calculate Magnitude Squared Coherence (MSC).              |
| `connectivity_imaginary_coherence`(f, c, /[, ...])     | Calculate Imaginary Coherence (iCOH).                     |
| `connectivity_lagged_coherence`(f, c, /[, bands])      | Calculate Lagged Coherence.                               |

### eegdash.features.feature_bank.connectivity.connectivity_coherency_preprocessor(x, , , \_metadata, f_min: float | None = None, f_max: float | None = None, fs: int | None = None, window_size_in_sec: float | None = 4, overlap_in_sec: float | None = None, pairs: Iterable[Tuple[str, str]] | None = None, \*\*kwargs)

Compute Complex Coherency for all unique channel pairs.

The Complex Coherency is calculated by estimating the Cross-Spectral
Densities (CSD) between pairs of channels and normalizing it by the
auto-spectral densities.

* **Parameters:**
  * **x** (*ndarray*) – The input signal of shape (n_trials, n_channels, n_times).
  * **\*\*kwargs** (*dict*) – Supports any `scipy.signal.csd()` arguments like ‘nperseg’
    and ‘noverlap’.
  * **fs** (*int* *|* *None*) – Sampling frequency. Defaults to sfreq in MNE’s info. Do not use unless you know what you are doing.
  * **f_min** (*float* *|* *None*) – The minimum frequency. Use None for half the window length. Defaults to the highpass frequency used to MNE’s:meth:~mne.io.Raw.filter.
  * **f_max** (*float* *|* *None*) – The maximum frequency. Use None for Nyquist. Defaults to the lowpass frequency used to MNE’s `filter()`.
  * **window_size_in_sec** (*float* *|* *None*) – Window size in seconds, replacing nperseg. Only used if nperseg is not provided. Defaults to 4 seconds.
  * **overlap_in_sec** (*float* *|* *None*) – Window overlap in seconds, replacing noverlap. Only used if nperseg and noverlap are not provided.defaults to half of window_size_in_sec.
  * **pairs** (*Optional* *[**Iterable* *[**Tuple* *[**str* *,* *str* *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  * **f** (*ndarray*) – Frequency vector of shape (n_frequencies,).
  * **c** (*ndarray*) – Complex coherency array of shape (n_trials, n_pairs, n_frequencies).
    Values are complex numbers where:
    - Absolute value $|c|$ is the coherence magnitude (0 to 1).
    - Angle $\arg(c)$ is the phase lag.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_magnitude_square_coherence(f, c, , bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4.5), 'theta': (4.5, 8)})

Calculate Magnitude Squared Coherence (MSC).

MSC measures the linear correlation between two signals in the frequency
domain. It is defined as the squared magnitude of the complex coherency,
$|c|^2$, where $c$ is the complex coherency.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **c** (*ndarray*) – Complex coherency array.
  * **bands** (*dict* *,* *optional*) – Frequency bands to aggregate (defaults to DEFAULT_FREQ_BANDS).
* **Returns:**
  Mean MSC for each frequency band.
* **Return type:**
  dict

### References

[Brainstorm - Connectivity](https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_imaginary_coherence(f, c, , bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4.5), 'theta': (4.5, 8)})

Calculate Imaginary Coherence (iCOH).

Imaginary coherence captures only the non-zero phase-lagged
synchronization. It is defined as $\operatorname{Im}(c)$,
where $c$ is the complex coherency.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **c** (*ndarray*) – Complex coherency array.
  * **bands** (*dict* *,* *optional*) – Frequency bands to aggregate.
* **Returns:**
  Mean Imaginary Coherence for each frequency band.
* **Return type:**
  dict

### References

[Brainstorm - Connectivity](https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_lagged_coherence(f, c, , bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4.5), 'theta': (4.5, 8)})

Calculate Lagged Coherence.

Lagged coherence further refines the synchronization measure by
normalizing the imaginary part of the coherency, effectively removing
all instantaneous (zero-lag) contributions. It is defined as
$\operatorname{Im}(c)/\sqrt{1 - \left(\operatorname{Re}(c)\right)^2}$,
where $c$ is the complex coherency.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **c** (*ndarray*) – Complex coherency array.
  * **bands** (*dict* *,* *optional*) – Frequency bands to aggregate.
* **Returns:**
  Mean Lagged Coherence for each frequency band.
* **Return type:**
  dict

### References

[Brainstorm - Connectivity](https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity)

<!-- !! processed by numpydoc !! -->
