import numbers

import numpy as np
from scipy import signal, stats

from ..decorators import FeaturePredecessor, univariate_feature, PreprocessorOutputType
from ..extractors import BasePreprocessorOutputType

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


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@PreprocessorOutputType(SignalOutputType)
def signal_hilbert_preprocessor(x, /):
    return np.abs(signal.hilbert(x - x.mean(axis=-1, keepdims=True), axis=-1))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@PreprocessorOutputType(SignalOutputType)
def signal_filter_preprocessor(x, /, fs, f_min, f_max, num_taps=None):
    """Linear-phase FIR band-pass filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency (Hz)
    f_min : float
        Low cutoff frequency (Hz)
    f_max : float
        High cutoff frequency (Hz)
    num_taps : int
        Number of filter taps (must be odd for exact linear phase)

    Returns
    -------
    ndarray
    """
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
        taps, [1.0], x
    )  # Zero-phase application (preserves waveform shape)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_mean(x, /):
    return x.mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_variance(x, /, **kwargs):
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_std(x, /, **kwargs):
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_skewness(x, /, **kwargs):
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_kurtosis(x, /, **kwargs):
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_root_mean_square(x, /):
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_peak_to_peak(x, /, **kwargs):
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_quantile(x, /, q: numbers.Number = 0.5, **kwargs):
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_line_length(x, /):
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_zero_crossings(x, /, threshold=1e-15):
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_mobility(x, /):
    return np.diff(x, axis=-1).std(axis=-1) / x.std(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_complexity(x, /):
    return (np.diff(x, 2, axis=-1).std(axis=-1) * x.std(axis=-1)) / np.diff(
        x, axis=-1
    ).var(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_decorrelation_time(x, /, fs=1):
    f = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / fs


# =================================  Aliases  =================================

signal_hjorth_activity = signal_variance
