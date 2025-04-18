import numbers
import numpy as np
from scipy import stats

from .extractors import FeatureExtractor, ByChannelFeatureExtractor, FeaturePredecessor


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_mean(x):
    return x.mean(axis=-1)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_variance(x, **kwargs):
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_std(x, **kwargs):
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_skewness(x, **kwargs):
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_kurtosis(x, **kwargs):
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_root_mean_square(x):
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_peak_to_peak(x, **kwargs):
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_quantile(x, q: numbers.Number = 0.5, **kwargs):
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_line_length(x):
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_zero_crossings(x, threshold=1e-15):
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(FeatureExtractor, ByChannelFeatureExtractor)
class HjorthFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, x):
        return (x, np.diff(x, axis=-1), x.std(axis=-1))


@FeaturePredecessor(HjorthFeatureExtractor)
def signal_hjorth_mobility(x, dx, x_std):
    return dx.std(axis=-1) / x_std


@FeaturePredecessor(HjorthFeatureExtractor)
def signal_hjorth_complexity(x, dx, x_std):
    return np.diff(dx, axis=-1).std(axis=-1) / x_std


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_decorrelation_time(x, fs=1):
    f = np.fft.fft(x, axis=-1)
    ac = np.fft.ifft(f.real**2 + f.imag**2, axis=-1)[..., : x.shape[-1] // 2]
    dct = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        dct[i] = np.searchsorted(ac[i] <= 0, True)
    return dct / fs


# =================================  Aliases  =================================

signal_hjorth_activity = signal_variance
