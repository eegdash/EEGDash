"""
This module provides functions for calculating basic signal features from EEG
data.

It includes a variety of statistical, time-domain, and complexity features
that are commonly used in EEG analysis.
"""
import numbers

import numpy as np
from scipy import signal, stats

from ..decorators import FeaturePredecessor, univariate_feature
from ..extractors import FeatureExtractor

__all__ = [
    "HilbertFeatureExtractor",
    "signal_mean",
    "signal_variance",
    "signal_skewness",
    "signal_kurtosis",
    "signal_std",
    "signal_root_mean_square",
    "signal_peak_to_peak",
    "signal_quantile",
    "signal_zero_crossings",
    "signal_line_length",
    "signal_hjorth_activity",
    "signal_hjorth_mobility",
    "signal_hjorth_complexity",
    "signal_decorrelation_time",
]


@FeaturePredecessor(FeatureExtractor)
class HilbertFeatureExtractor(FeatureExtractor):
    """A feature extractor that applies the Hilbert transform.

    This extractor computes the analytic signal using the Hilbert transform and
    returns its absolute value, which represents the envelope of the signal.
    """

    def preprocess(self, x):
        """Apply the Hilbert transform to the input data.

        Parameters
        ----------
        x : np.ndarray
            The input time series.

        Returns
        -------
        np.ndarray
            The envelope of the signal.
        """
        return np.abs(signal.hilbert(x - x.mean(axis=-1, keepdims=True), axis=-1))


SIGNAL_PREDECESSORS = [FeatureExtractor, HilbertFeatureExtractor]


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_mean(x):
    """Calculate the mean of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.

    Returns
    -------
    np.ndarray
        The mean of the signal for each channel.
    """
    return x.mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_variance(x, **kwargs):
    """Calculate the variance of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    **kwargs
        Keyword arguments for `numpy.var`.

    Returns
    -------
    np.ndarray
        The variance of the signal for each channel.
    """
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_std(x, **kwargs):
    """Calculate the standard deviation of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    **kwargs
        Keyword arguments for `numpy.std`.

    Returns
    -------
    np.ndarray
        The standard deviation of the signal for each channel.
    """
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_skewness(x, **kwargs):
    """Calculate the skewness of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    **kwargs
        Keyword arguments for `scipy.stats.skew`.

    Returns
    -------
    np.ndarray
        The skewness of the signal for each channel.
    """
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_kurtosis(x, **kwargs):
    """Calculate the kurtosis of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    **kwargs
        Keyword arguments for `scipy.stats.kurtosis`.

    Returns
    -------
    np.ndarray
        The kurtosis of the signal for each channel.
    """
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_root_mean_square(x):
    """Calculate the root mean square (RMS) of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.

    Returns
    -------
    np.ndarray
        The RMS of the signal for each channel.
    """
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_peak_to_peak(x, **kwargs):
    """Calculate the peak-to-peak amplitude of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    **kwargs
        Keyword arguments for `numpy.ptp`.

    Returns
    -------
    np.ndarray
        The peak-to-peak amplitude for each channel.
    """
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_quantile(x, q: numbers.Number = 0.5, **kwargs):
    """Calculate the q-th quantile of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    q : float, default 0.5
        The quantile to compute.
    **kwargs
        Keyword arguments for `numpy.quantile`.

    Returns
    -------
    np.ndarray
        The q-th quantile for each channel.
    """
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_line_length(x):
    """Calculate the line length of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.

    Returns
    -------
    np.ndarray
        The line length for each channel.
    """
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_zero_crossings(x, threshold=1e-15):
    """Calculate the number of zero-crossings in the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    threshold : float, default 1e-15
        The threshold for detecting zero-crossings.

    Returns
    -------
    np.ndarray
        The number of zero-crossings for each channel.
    """
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_mobility(x):
    """Calculate the Hjorth mobility of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.

    Returns
    -------
    np.ndarray
        The Hjorth mobility for each channel.
    """
    return np.diff(x, axis=-1).std(axis=-1) / x.std(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_complexity(x):
    """Calculate the Hjorth complexity of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.

    Returns
    -------
    np.ndarray
        The Hjorth complexity for each channel.
    """
    return (np.diff(x, 2, axis=-1).std(axis=-1) * x.std(axis=-1)) / np.diff(
        x, axis=-1
    ).var(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_decorrelation_time(x, fs=1):
    """Calculate the decorrelation time of the signal.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    fs : int, default 1
        The sampling frequency.

    Returns
    -------
    np.ndarray
        The decorrelation time for each channel.
    """
    f = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / fs


# =================================  Aliases  =================================

signal_hjorth_activity = signal_variance
