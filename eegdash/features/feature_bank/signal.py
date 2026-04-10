r"""Signal-Level Feature Extraction
===============================

This module provides temporal and statistical features computed
directly from time-series data.

Data Shape Convention
---------------------
This module follows a **Time-Last** convention:

* **Input:** ``(..., time)``
* **Output:** ``(...,)``

All functions collapse the last dimension (time), returning an ndarray of
features corresponding to the leading dimensions (e.g., subjects, channels).
"""

import numbers

import numpy as np
from scipy import signal, stats

from ..decorators import (
    feature_predecessor,
    preprocessor_output_type,
    univariate_feature,
)
from ..output_types import BasePreprocessorOutputType

__all__ = [
    "SignalOutputType",
    "signal_hilbert_preprocessor",
    "signal_filter_preprocessor",
    "SIGNAL_PREDECESSORS",
    "signal_decorrelation_time",
    "signal_hjorth_activity",
    "signal_hjorth_complexity",
    "signal_hjorth_mobility",
    "signal_kurtosis",
    "signal_line_length",
    "signal_mean",
    "signal_peak_to_peak",
    "signal_quantile",
    "signal_root_mean_square",
    "signal_skewness",
    "signal_std",
    "signal_variance",
    "signal_zero_crossings",
]


class SignalOutputType(BasePreprocessorOutputType):
    pass


SIGNAL_PREDECESSORS = [None, SignalOutputType]


@feature_predecessor(*SIGNAL_PREDECESSORS)
@preprocessor_output_type(SignalOutputType)
def signal_hilbert_preprocessor(x, /):
    r"""Compute the amplitude envelope of the analytic signal.

    Parameters
    ----------
    x : ndarray
        Input signal

    Returns
    -------
    ndarray
        The signal envelope, with the same shape as the input.

    """
    return np.abs(signal.hilbert(x - x.mean(axis=-1, keepdims=True), axis=-1))


@feature_predecessor(*SIGNAL_PREDECESSORS)
@preprocessor_output_type(SignalOutputType)
def signal_filter_preprocessor(x, /, *, _metadata, f_min, f_max, num_taps=None):
    """Linear-phase FIR band-pass filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    f_min : float
        Low cutoff frequency (Hz)
    f_max : float
        High cutoff frequency (Hz)
    num_taps : int
        Number of filter taps (must be odd for exact linear phase)

    Returns
    -------
    ndarray
        The band-pass filtered signal, with the same shape as the input.

    """
    fs = _metadata["info"]["sfreq"]
    if num_taps is None:
        num_taps = int(fs * 1.5)  # rule of thumb for choosing the filter order
    if num_taps % 2 == 0:  # ensure odd
        num_taps += 1
    taps = signal.firwin(
        numtaps=num_taps,
        cutoff=[f_min, f_max],
        fs=fs,
        pass_zero=False,
        window="hamming",
    )

    return signal.filtfilt(
        taps, [1.0], x, padlen=min(3 * num_taps, x.shape[-1] - 1), axis=-1
    )  # Zero-phase application (preserves waveform shape)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_mean(x, /):
    r"""Compute the temporal mean of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The mean of the signal along the temporal axis.
        Shape is ``x.shape[:-1]``.

    """
    return x.mean(axis=-1)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_variance(x, /, **kwargs):
    r"""Compute the temporal variance of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to :func:`np.var`.

    Returns
    -------
    ndarray
        The variance of the signal along the temporal axis.
        Shape is ``x.shape[:-1]``.

    """
    return x.var(axis=-1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_std(x, /, **kwargs):
    r"""Compute the temporal standard deviation of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to :func:`np.std`.

    Returns
    -------
    ndarray
        The standard deviation of the signal along the temporal axis.
        Shape is ``x.shape[:-1]``.

    """
    return x.std(axis=-1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_skewness(x, /, **kwargs):
    r"""Compute the temporal skewness of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to :func:`scipy.stats.skew`.

    Returns
    -------
    ndarray
        The skewness of the signal along the temporal axis.
        Shape is ``x.shape[:-1]``.

    """
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_kurtosis(x, /, **kwargs):
    r"""Compute the temporal kurtosis of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to :func:`scipy.stats.kurtosis`.

    Returns
    -------
    ndarray
        The kurtosis of the signal along the temporal axis.
        Shape is ``x.shape[:-1]``

    """
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_root_mean_square(x, /):
    r"""Calculate the Root Mean Square (RMS) magnitude.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The RMS amplitude of the signal.
        Shape is ``x.shape[:-1]``

    Notes
    -----
    For the RMS definition, see the
    `Wikipedia entry <https://en.wikipedia.org/wiki/Root_mean_square>`__.

    """
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_peak_to_peak(x, /, **kwargs):
    r"""Calculate the peak-to-peak (maximum range) of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to :func:`numpy.ptp`.

    Returns
    -------
    ndarray
        The peak-to-peak amplitude.
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    This function wraps :func:`numpy.ptp`; see the NumPy documentation for
    details on additional keyword arguments.

    For a theoretical overview of Peak-To-Peak amplitude in signal analysis,
    see the `Wikipedia entry <https://en.wikipedia.org/wiki/Peak-to-peak>`__.

    """
    return np.ptp(x, axis=-1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_quantile(x, /, q: numbers.Number = 0.5, **kwargs):
    r"""Compute the q-th quantile of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    q : float or array_like, optional
        The quantile to compute. 0.5 (default) is the median.
    **kwargs : dict
        Additional keyword arguments passed to :func:`numpy.quantile`.

    Returns
    -------
    ndarray
        The quantile values.
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    This function wraps :func:`numpy.quantile`; see the NumPy documentation for
    details on additional keyword arguments.

    """
    return np.quantile(x, q=q, axis=-1, **kwargs)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_line_length(x, /):
    r"""Calculate the Mean Signal Line Length.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The mean absolute vertical distance between consecutive samples.
        Shape is ``x.shape[:-1]``.

    """
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_zero_crossings(x, /, threshold=1e-15):
    r"""Count the number of times the signal crosses the zero axis.

    This function identifies points where the signal changes sign or
    enters/leaves a defined noise floor (threshold).

    Parameters
    ----------
    x : ndarray
        The input signal.
    threshold : float, optional
        A small epsilon value to treat values near zero as exactly zero,
        preventing false counts due to floating-point noise.

    Returns
    -------
    ndarray
        The count of zero crossings.
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    For a theoretical overview of zero-crossing rate in signal analysis,
    see the `Wikipedia entry <https://en.wikipedia.org/wiki/Zero_crossing>`__.

    """
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_mobility(x, /):
    r"""Calculate the Hjorth Mobility of the signal.

    Mobility is defined as the standard deviation of the signal's first
    derivative normalized by the standard deviation of the signal itself.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The Hjorth Mobility value. Shape is ``x.shape[:-1]``.

    See Also
    --------
    signal_hjorth_activity

    Notes
    -----
    The mobility is calculated using the following formula:

    .. math:: \text{Mobility}\left(x\left(t\right)\right) = \sqrt{\frac{\text{Var}\left(\frac{\mathrm{d}x\left(t\right)}{\mathrm{dt}}\right)}{\text{Var}\left(x\left(t\right)\right)}}

    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

    for more details, see the `Wikipedia entry
    <https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Mobility>`__.


    """
    return np.diff(x, axis=-1).std(axis=-1) / x.std(axis=-1)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_complexity(x, /):
    r"""Calculate the Hjorth Complexity of the signal.

    Complexity represents the change in frequency. The parameter
    compares the signal's similarity to a pure sine wave, where value
    of 1 indicates a perfect sine wave.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The complexity value. Shape is ``x.shape[:-1]``.

    See Also
    --------
    signal_hjorth_mobility

    Notes
    -----
    The complexity is calculated using the following formula:

    .. math:: \text{Complexity}\left(x\left(t\right)\right) = \frac{\text{Mobility}\left(\frac{\mathrm{d}x\left(t\right)}{\mathrm{dt}}\right)}{\text{Mobility}\left(x\left(t\right)\right)}

    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

    For more details, see the `Wikipedia entry
    <https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Complexity>`__.

    """
    return (np.diff(x, 2, axis=-1).std(axis=-1) * x.std(axis=-1)) / np.diff(
        x, axis=-1
    ).var(axis=-1)


@feature_predecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_decorrelation_time(x, /, *, _metadata):
    r"""Calculate the Decorrelation Time of the signal.

    This function computes the time it takes for the signal to
    decorrelate, defined as the first time lag where the autocorrelation
    function drops to zero.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The time (in seconds or samples) until the signal decorrelates.
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    This function uses the `Wiener-Khinchin Theorem
    <https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem>`_ to
    compute the autocorrelation via the inverse FFT of the power spectrum.

    """
    f = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / _metadata["info"]["sfreq"]


# =================================  Aliases  =================================
signal_hjorth_activity = signal_variance
r"""Calculate the Hjorth Activity of the signal.

    Activity is defined as the variance of the signal itself.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The Hjorth Activity value. Shape is ``x.shape[:-1]``.

    Notes
    -----
    The activity is calculated using the following formula:

    .. math:: \text{Activity}\left(x\left(t\right)\right) = \operatorname{Var}\left(x\left(t\right)\right)

    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

    for more details, see the `Wikipedia entry
    <https://en.wikipedia.org/wiki/Hjorth_parameters#Hjorth_Activity>`__.
    """
