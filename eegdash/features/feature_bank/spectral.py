import numba as nb
import numpy as np
from scipy.signal import welch

from ..decorators import FeaturePredecessor, univariate_feature
from ..extractors import FeatureExtractor
from . import utils

__all__ = [
    "SpectralFeatureExtractor",
    "NormalizedSpectralFeatureExtractor",
    "DBSpectralFeatureExtractor",
    "spectral_root_total_power",
    "spectral_moment",
    "spectral_entropy",
    "spectral_edge",
    "spectral_slope",
    "spectral_bands_power",
    "spectral_hjorth_activity",
    "spectral_hjorth_mobility",
    "spectral_hjorth_complexity",
]


class SpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, x, **kwargs):
        f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
        f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
        assert "fs" in kwargs
        kwargs["axis"] = -1
        f, p = welch(x, **kwargs)
        f_min, f_max = utils.get_valid_freq_band(
            kwargs["fs"], x.shape[-1], f_min, f_max
        )
        f, p = utils.slice_freq_band(f, p, f_min=f_min, f_max=f_max)
        return f, p


@FeaturePredecessor(SpectralFeatureExtractor)
class NormalizedSpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, *x):
        return (*x[:-1], x[-1] / x[-1].sum(axis=-1, keepdims=True))


@FeaturePredecessor(SpectralFeatureExtractor)
class DBSpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, *x, eps=1e-15):
        return (*x[:-1], 10 * np.log10(x[-1] + eps))


@FeaturePredecessor(SpectralFeatureExtractor)
@univariate_feature
def spectral_root_total_power(f, p):
    return np.sqrt(p.sum(axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_moment(f, p):
    return np.sum(f * p, axis=-1)


@FeaturePredecessor(SpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_activity(f, p):
    return np.sum(p, axis=-1)


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_mobility(f, p):
    return np.sqrt(np.sum(np.power(f, 2) * p, axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_complexity(f, p):
    return np.sqrt(np.sum(np.power(f, 4) * p, axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_entropy(f, p):
    idx = p > 0
    plogp = np.zeros_like(p)
    plogp[idx] = p[idx] * np.log(p[idx])
    return -np.sum(plogp, axis=-1)


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def spectral_edge(f, p, edge=0.9):
    se = np.empty(p.shape[:-1])
    for i in np.ndindex(p.shape[:-1]):
        se[i] = f[np.searchsorted(np.cumsum(p[i]), edge)]
    return se


@FeaturePredecessor(DBSpectralFeatureExtractor)
@univariate_feature
def spectral_slope(f, p):
    log_f = np.vstack((np.log(f), np.ones(f.shape[0]))).T
    r = np.linalg.lstsq(log_f, p.reshape(-1, p.shape[-1]).T)[0]
    r = r.reshape(2, *p.shape[:-1])
    return {"exp": r[0], "int": r[1]}


@FeaturePredecessor(
    SpectralFeatureExtractor,
    NormalizedSpectralFeatureExtractor,
    DBSpectralFeatureExtractor,
)
@univariate_feature
def spectral_bands_power(f, p, bands=utils.DEFAULT_FREQ_BANDS):
    return utils.reduce_freq_bands(f, p, bands, np.sum)
