import numbers
import numpy as np
from scipy import stats
from .extractors import ByChannelFeatureExtractor, Feature


@Feature(ByChannelFeatureExtractor)
def signal_mean(x):
    return x.mean(axis=-1)


@Feature(ByChannelFeatureExtractor)
def signal_variance(x, **kwargs):
    return x.var(axis=-1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_std(x, **kwargs):
    return x.std(axis=-1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_skewness(x, **kwargs):
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_kurtosis(x, **kwargs):
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_rms(x):
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@Feature(ByChannelFeatureExtractor)
def signal_amp_ptp(x, **kwargs):
    return np.ptp(x, axis=-1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_quantile(x, q: numbers.Number = 0.5, **kwargs):
    return np.quantile(x, q=q, axis=-1, **kwargs)


@Feature(ByChannelFeatureExtractor)
def signal_line_length(x):
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)
