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

| `connectivity_coherency_preprocessor`(x, /, \*, ...)        | Compute Complex Coherency for all unique channel pairs.                           |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `connectivity_phase_diff_preprocessor`(x, /, \*)            | Compute complex exponent of phase difference for all unique channel pairs.        |
| `connectivity_temporal_correlation`(x, /, \*[, ...])        | Compute the temporal correlation between channel pairs.                           |
| `connectivity_spectral_correlation`(f, p, /, \*)            | Compute the spectral correlation between channel pairs.                           |
| `connectivity_max_cross_correlation`(x, /, \*, ...)         | Compute the maximum cross correlation between channel pairs.                      |
| `connectivity_magnitude_square_coherence`(f, c, /)          | Calculate Magnitude Squared Coherence (MSC).                                      |
| `connectivity_imaginary_coherence`(f, c, /[, ...])          | Calculate Imaginary Coherence (iCOH).                                             |
| `connectivity_lagged_coherence`(f, c, /[, bands])           | Calculate Lagged Coherence.                                                       |
| `connectivity_phase_locking_value`(exp_dphi, /)             | Compute the Phase Locking Value (PLV) of each channel pair.                       |
| `connectivity_corrected_imaginary_phase_locking_value`(...) | Compute the corrected imaginary Phase Locking Value (ciPLV) of each channel pair. |
| `connectivity_phase_lag_index`(exp_dphi, /)                 | Compute the Phase Lag Index (PLI) of each channel pair.                           |
| `connectivity_weighted_phase_lag_index`(...)                | Compute the weighted Phase Lag Index (wPLI) of each channel pair.                 |
| `connectivity_directed_phase_lag_index`(...)                | Compute the directed Phase Lag Index (dPLI) of each channel pair.                 |

### eegdash.features.feature_bank.connectivity.connectivity_coherency_preprocessor(x, , , \_metadata, f_min: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, f_max: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, fs: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, window_size_in_sec: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = 4, overlap_in_sec: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, \*\*kwargs)

Compute Complex Coherency for all unique channel pairs.

The Complex Coherency is calculated by estimating the Cross-Spectral
Densities (CSD) between pairs of channels and normalizing it by the
auto-spectral densities.

* **Parameters:**
  * **x** (*ndarray*) – The input signal of shape (n_trials, n_channels, n_times).
  * **\*\*kwargs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Supports any [`scipy.signal.csd()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html#scipy.signal.csd) arguments like ‘nperseg’
    and ‘noverlap’.
  * **fs** ([*int*](https://docs.python.org/3/library/functions.html#int) *|* *None*) – Sampling frequency. Defaults to sfreq in MNE’s info. Do not use unless you know what you are doing.
  * **f_min** ([*float*](https://docs.python.org/3/library/functions.html#float) *|* *None*) – The minimum frequency. Use None for half the window length. Defaults to the highpass frequency used to MNE’s:meth:~mne.io.Raw.filter.
  * **f_max** ([*float*](https://docs.python.org/3/library/functions.html#float) *|* *None*) – The maximum frequency. Use None for Nyquist. Defaults to the lowpass frequency used to MNE’s [`filter()`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter).
  * **window_size_in_sec** ([*float*](https://docs.python.org/3/library/functions.html#float) *|* *None*) – Window size in seconds, replacing nperseg. Only used if nperseg is not provided. Defaults to 4 seconds.
  * **overlap_in_sec** ([*float*](https://docs.python.org/3/library/functions.html#float) *|* *None*) – Window overlap in seconds, replacing noverlap. Only used if nperseg and noverlap are not provided.defaults to half of window_size_in_sec.
  * **pairs** ([*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional) *[*[*Iterable*](https://docs.python.org/3/library/typing.html#typing.Iterable) *[*[*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  * **f** (*ndarray*) – Frequency vector of shape (n_frequencies,).
  * **c** (*ndarray*) – Complex coherency array of shape (n_trials, n_pairs, n_frequencies).
    Values are complex numbers where:
    - Absolute value $|c|$ is the coherence magnitude (0 to 1).
    - Angle $\arg(c)$ is the phase lag.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_phase_diff_preprocessor(x, , , pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, \_metadata)

Compute complex exponent of phase difference for all unique channel pairs.

For each pair of channels $l, m$, calculate:

$$
e^{i\left(\varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}.

$$

The instantanous phases are calculated via Hilbert transform.

#### NOTE
This preprocessor should follow a narrow-band filter, otherwise the
Hilbert transform cannot yield meaningful phases.

* **Parameters:**
  * **x** (*ndarray*) – The input signal of shape (n_trials, n_channels, n_times).
  * **pairs** ([*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional) *[*[*Iterable*](https://docs.python.org/3/library/typing.html#typing.Iterable) *[*[*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  Complex exponents of phase diffs.
* **Return type:**
  ndarray

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_temporal_correlation(x, , , pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, \_metadata)

Compute the temporal correlation between channel pairs.

* **Parameters:**
  * **x** ([*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – The input signal of shape (n_trials, n_channels, n_times).
  * **pairs** ([*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional) *[*[*Iterable*](https://docs.python.org/3/library/typing.html#typing.Iterable) *[*[*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  The channel pairwise temporal Pearson correlation of shape
  (n_trials, n_pairs).
* **Return type:**
  [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_spectral_correlation(f, p, , , pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, \_metadata)

Compute the spectral correlation between channel pairs.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **p** (*ndarray*) – Power Spectral Density (PSD).
  * **pairs** ([*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional) *[*[*Iterable*](https://docs.python.org/3/library/typing.html#typing.Iterable) *[*[*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  The channel pairwise Pearson correlation between power spectra of shape
  (n_trials, n_pairs).
* **Return type:**
  [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_max_cross_correlation(x, , , \_metadata, pairs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, eps=1e-15)

Compute the maximum cross correlation between channel pairs.

* **Parameters:**
  * **x** ([*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)) – The input signal of shape (n_trials, n_channels, n_times).
  * **eps** ([*float*](https://docs.python.org/3/library/functions.html#float) *,* *optional*) – A small constant to prevent log of zero (default: 1e-15).
  * **pairs** ([*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional) *[*[*Iterable*](https://docs.python.org/3/library/typing.html#typing.Iterable) *[*[*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple) *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *]* *]*) – A list of channel pairs to pick.
* **Returns:**
  The channel pairs maximum cross correlation of shape
  (n_trials, n_pairs).
* **Return type:**
  [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

### Notes

This function computes the cross correlation via multiplication of the
Fourier transformed signals.

### References

For more details, see the [Wikipedia entry](https://en.wikipedia.org/wiki/Cross-correlation#Normalization).

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_magnitude_square_coherence(f, c, , bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4.5), 'theta': (4.5, 8)})

Calculate Magnitude Squared Coherence (MSC).

MSC measures the linear correlation between two signals in the frequency
domain. It is defined as the squared magnitude of the complex coherency,
$|c|^2$, where $c$ is the complex coherency.

* **Parameters:**
  * **f** (*ndarray*) – Frequency vector.
  * **c** (*ndarray*) – Complex coherency array.
  * **bands** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Frequency bands to aggregate (defaults to DEFAULT_FREQ_BANDS).
* **Returns:**
  Mean MSC for each frequency band.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

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
  * **bands** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Frequency bands to aggregate.
* **Returns:**
  Mean Imaginary Coherence for each frequency band.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

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
  * **bands** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict) *,* *optional*) – Frequency bands to aggregate.
* **Returns:**
  Mean Lagged Coherence for each frequency band.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

### References

[Brainstorm - Connectivity](https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity)

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_phase_locking_value(exp_dphi,)

Compute the Phase Locking Value (PLV) of each channel pair.

$$
PLV_{lm} = \left|\left\langle e^{i\left(\varphi_l\left(t\right)
           - \varphi_m\left(t\right)\right)}\right\rangle_t\right|

$$

* **Parameters:**
  **exp_dphi** (*ndarray*) – Complex exponents of phase diffs.
* **Returns:**
  The PLV of each channel pair.
* **Return type:**
  ndarray

### References

- Lachaux, JP. et al. (1999). Measuring phase synchrony in brain signals.
  Hum Brain Mapp, 8(4), 194-208.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_corrected_imaginary_phase_locking_value(exp_dphi,)

Compute the corrected imaginary Phase Locking Value (ciPLV) of each channel pair.

$$
ciPLV_{lm} = \frac{\Im{\left(\left\langle e^{i\left(
             \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
             \right\rangle_t\right)}}{\sqrt{1 - \left(\Re{\left(
             \left\langle e^{i\left(\varphi_l\left(t\right)
             - \varphi_m\left(t\right)\right)}\right\rangle_t\right)}
             \right)^2}}
$$

* **Parameters:**
  **exp_dphi** (*ndarray*) – Complex exponents of phase diffs.
* **Returns:**
  The ciPLV of each channel pair.
* **Return type:**
  ndarray

### References

- Bruña, R. et al. (2018). Phase locking value revisited: teaching new
  tricks to an old dog. Journal of Neural Engineering, 15(5), 056011.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_phase_lag_index(exp_dphi,)

Compute the Phase Lag Index (PLI) of each channel pair.

$$
PLI_{lm} = \left|\left\langle\operatorname{sign}{\left(\Im{\left(
           e^{i\left(\varphi_l\left(t\right) - \varphi_m\left(t\right)
           \right)}\right)}\right)}\right\rangle_t\right|
$$

* **Parameters:**
  **exp_dphi** (*ndarray*) – Complex exponents of phase diffs.
* **Returns:**
  The PLI of each channel pair.
* **Return type:**
  ndarray

### References

- Stam, CJ. et al. (2007). A. Phase lag index: assessment of functional
  connectivity from multi channel EEG and MEG with diminished bias from
  common sources. Hum Brain Mapp, 28(11), 1178-93.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_weighted_phase_lag_index(exp_dphi,)

Compute the weighted Phase Lag Index (wPLI) of each channel pair.

$$
wPLI_{lm} = \frac{\left|\left\langle\Im{\left(e^{i\left(
            \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
            \right)}\right\rangle_t\right|}{\left\langle\left|
            \Im{\left(e^{i\left(\varphi_l\left(t\right)
            - \varphi_m\left(t\right)\right)}\right)}\right|
            \right\rangle_t}
$$

* **Parameters:**
  **exp_dphi** (*ndarray*) – Complex exponents of phase diffs.
* **Returns:**
  The wPLI of each channel pair.
* **Return type:**
  ndarray

### References

- Vinck, M. et al. (2011). An improved index of phase-synchronization for
  electrophysiological data in the presence of volume-conduction, noise
  and sample-size bias. NeuroImage, 55(4), 1548-1565.

<!-- !! processed by numpydoc !! -->

### eegdash.features.feature_bank.connectivity.connectivity_directed_phase_lag_index(exp_dphi,)

Compute the directed Phase Lag Index (dPLI) of each channel pair.

$$
dPLI_{lm} = \left|\left\langle\Theta{\left(\Im{\left(e^{i\left(
            \varphi_l\left(t\right) - \varphi_m\left(t\right)\right)}
            \right)}\right)}\right\rangle_t\right|,
$$

where $\Theta\left(\cdot\right)$ is Heaviside’s step function.

* **Parameters:**
  **exp_dphi** (*ndarray*) – Complex exponents of phase diffs.
* **Returns:**
  The dPLI of each channel pair.
* **Return type:**
  ndarray

### References

- Stam, CJ. et al. (2012). Go with the flow: Use of a directed phase lag
  index (dPLI) to characterize patterns of phase relations in a
  large-scale model of brain dynamics. NeuroImage, 62(3), 1415-1428.

<!-- !! processed by numpydoc !! -->
