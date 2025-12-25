import numbers

import numpy as np
from scipy import signal, stats

from ..decorators import FeaturePredecessor, univariate_feature

__all__ = [
    "signal_hilbert_preprocessor",
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


@FeaturePredecessor()
def signal_hilbert_preprocessor(x, /):
    """Compute the amplitude envelope of the analytic signal.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The magnitude of the Hilbert transform after mean subtraction.
    
    """
    return np.abs(signal.hilbert(x - x.mean(axis=-1, keepdims=True), axis=-1))

# SIGNAL_PREDECESSORS allows features to accept raw data (None) 
# or the Hilbert-transformed envelope.
SIGNAL_PREDECESSORS = [None, signal_hilbert_preprocessor]


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_mean(x, /):
    """Compute the temporal mean of the signal.
    
    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The mean of the signal along the temporal axis. Shape is ``x.shape[:-1]``.

    """
    return x.mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_variance(x, /, **kwargs):
    """Compute the temporal variance of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to `np.var`.

    Returns
    -------
    ndarray
        The variance of the signal along the temporal axis. Shape is ``x.shape[:-1]``.

    """
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_std(x, /, **kwargs):
    """Compute the temporal standard deviation of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to `np.std`.

    Returns
    -------
    ndarray
        The standard deviation of the signal along the temporal axis. Shape 
        is ``x.shape[:-1]``
    
    """
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_skewness(x, /, **kwargs):
    """Compute the temporal skewness of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to `scipy.stats.skew`.

    Returns
    -------
    ndarray
        The skewness of the signal along the temporal axis. Shape is ``x.shape[:-1]``
    
    """
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_kurtosis(x, /, **kwargs):
    """Measure the temporal kurtosis of the signal distribution.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to `scipy.stats.kurtosis`.

    Returns
    -------
    ndarray
        The kurtosis of the signal along the temporal axis. Shape is ``x.shape[:-1]``
        
    """
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_root_mean_square(x, /):
    """Calculate the Root Mean Square (RMS) magnitude.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The RMS amplitude of the signal. Shape is ``x.shape[:-1]``
    """
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_peak_to_peak(x, /, **kwargs):
    """Calculate the peak-to-peak (maximum range) of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    **kwargs : dict
        Additional keyword arguments passed to `np.ptp`.

    Returns
    -------
    ndarray
        The peak-to-peak amplitude. Shape is ``x.shape[:-1]``.
    """
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_quantile(x, /, q: numbers.Number = 0.5, **kwargs):
    """Compute the q-th quantile of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    q : float or array_like, optional
        The quantile to compute. 0.5 (default) is the median.
    **kwargs : dict
        Additional keyword arguments passed to `numpy.quantile`.

    Returns
    -------
    ndarray
        The requested quantile value(s). Shape is ``x.shape[:-1]``.
    """
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_line_length(x, /):
    """Calculate the average line length (waveform complexity).

    The mean absolute vertical distance between consecutive samples.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The line length of the signal. Shape is ``x.shape[:-1]``.
    
    """
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_zero_crossings(x, /, threshold=1e-15):
    """Count the number of times the signal crosses the zero axis.

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
        The count of zero crossings. Shape is ``x.shape[:-1]``, 
        dtype is typically ``int``.
    
    """
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_mobility(x, /):
    """Calculate Hjorth Mobility.

    Mobility represents the mean frequency of the power spectrum. 
    It is calculated as the square root of the ratio of the variance 
    of the first derivative to the variance of the signal.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The mobility value. Shape is ``x.shape[:-1]``.
    """
    return np.diff(x, axis=-1).std(axis=-1) / x.std(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_complexity(x, /):
    """Calculate Hjorth Complexity.

    Complexity (or form factor) indicates how similar the signal 
    is to a pure sine wave. A value of 1 represents a pure sinusoid; 
    higher values indicate a more complex, irregular signal.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The complexity value. Shape is ``x.shape[:-1]``.
    
    """
    return (np.diff(x, 2, axis=-1).std(axis=-1) * x.std(axis=-1)) / np.diff(
        x, axis=-1
    ).var(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_decorrelation_time(x, /, fs=1):
    """Estimate the decorrelation time of the signal.

    This function calculates the autocorrelation function (ACF) using FFT 
    and finds the first time point where the ACF drops to or below zero.

    Parameters
    ----------
    x : ndarray
        The input signal.
    fs : float, optional
        The sampling frequency in Hz. If 1 (default), the result is 
        returned in samples. If fs is provided, the result is in seconds.

    Returns
    -------
    ndarray
        The time (in seconds or samples) until the signal decorrelates. 
        Shape is ``x.shape[:-1]``.

    """
    f = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / fs


# =================================  Aliases  =================================

#: In Hjorth parameters, Activity represents the total power of the signal.
signal_hjorth_activity = signal_variance
"""a
"""
