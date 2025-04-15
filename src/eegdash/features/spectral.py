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
class DBSpectralFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, *x, **kwargs):
        return (*x[:-1], 10 * np.log10(x[-1]))


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


@Feature(DBSpectralFeatureExtractor)
def spectral_slope(f, p):
    log_f = np.log(f)
    exponent, intercept = np.empty(p.shape[0]), np.empty(p.shape[0])
    for i in range(p.shape[0]):
        ind = np.logical_and(f > 0, p[i] > 0)
        if ind.sum() > 1:
            r = linregress(log_f[ind], p[i, ind])
            exponent[i], intercept[i] = r.slope, r.intercept
        else:
            exponent[i], intercept[i] = np.nan, np.nan
    return {"exp": exponent, "int": intercept}


@Feature(
    SpectralFeatureExtractor,
    NormalizedSpectralFeatureExtractor,
    DBSpectralFeatureExtractor,
)
def spectral_bands_power(
    f,
    p,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    bands_power = dict()
    for k, v in bands.items():
        assert isinstance(k, str)
        assert isinstance(v, tuple)
        assert len(v) == 2
        mask = np.logical_and(f > v[0], f < v[1])
        power = p[:, mask].sum(axis=-1)
        bands_power[k] = power
    return bands_power
