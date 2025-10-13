import numbers

import numpy as np
from scipy import signal, stats

from ..decorators import FeaturePredecessor, univariate_feature
from ..extractors import FeatureExtractor

__all__ = [
    "HilbertFeatureExtractor",
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


@FeaturePredecessor(FeatureExtractor)
class HilbertFeatureExtractor(FeatureExtractor):
    def preprocess(self, x):
        return np.abs(signal.hilbert(x - x.mean(axis=-1, keepdims=True), axis=-1))


SIGNAL_PREDECESSORS = [FeatureExtractor, HilbertFeatureExtractor]


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_mean(x):
    return x.mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_variance(x, **kwargs):
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_std(x, **kwargs):
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_skewness(x, **kwargs):
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_kurtosis(x, **kwargs):
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_root_mean_square(x):
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_peak_to_peak(x, **kwargs):
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_quantile(x, q: numbers.Number = 0.5, **kwargs):
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_line_length(x):
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_zero_crossings(x, threshold=1e-15):
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_mobility(x):
    return np.diff(x, axis=-1).std(axis=-1) / x.std(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_hjorth_complexity(x):
    return (np.diff(x, 2, axis=-1).std(axis=-1) * x.std(axis=-1)) / np.diff(
        x, axis=-1
    ).var(axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def signal_decorrelation_time(x, fs=1):
    f = np.fft.fft(x - x.mean(axis=-1, keepdims=True), axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / fs


# =================================  Aliases  =================================

signal_hjorth_activity = signal_variance
