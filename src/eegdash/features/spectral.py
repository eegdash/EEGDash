import numpy as np
from scipy.signal import welch
from scipy.stats import linregress
from .extractors import ByChannelFeatureExtractor, Feature


@Feature()
class SpectralFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, x, **kwargs):
        f, p = welch(x, **kwargs)
        return f, p


@Feature(SpectralFeatureExtractor)
class NormalizedSpectralFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, *x, **kwargs):
        return (*x[:-1], x[-1] / x[-1].sum(axis=-1, keepdims=True))


@Feature(SpectralFeatureExtractor)
def root_total_power(f, p):
    return np.sqrt(p.sum(axis=-1))


@Feature(NormalizedSpectralFeatureExtractor)
def spectral_moment(f, p):
    return np.sum(f * p, axis=-1)


@Feature(NormalizedSpectralFeatureExtractor)
def spectral_entropy(f, p):
    idx = p > 0
    plogp = np.zeros_like(p)
    plogp[idx] = p[idx] * np.log(p[idx])
    return -np.sum(plogp, axis=-1)


@Feature(NormalizedSpectralFeatureExtractor)
def spectral_edge(f, p, edge=0.9):
    ps = p.cumsum(axis=-1)
    se = np.empty(ps.shape[0])
    for i in range(ps.shape[0]):
        se[i] = f[np.searchsorted(ps[i], edge)]
    return se


@Feature(SpectralFeatureExtractor)
def spectral_slope(f, p):
    exponent, intercept = np.empty(p.shape[0]), np.empty(p.shape[0])
    for i in range(p.shape[0]):
        ind = np.logical_and(f > 0, p[i] > 0)
        if ind.sum() > 1:
            r = linregress(np.log(f[ind]), np.log(p[i, ind]))
            exponent[i], intercept[i] = r.slope, r.intercept
        else:
            exponent[i], intercept[i] = np.nan, np.nan
    return {"exp": exponent, "int": intercept}
