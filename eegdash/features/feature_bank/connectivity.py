from itertools import chain
import numpy as np
from scipy.signal import csd

from ..extractors import FeatureExtractor, BivariateFeature
from ..decorators import FeaturePredecessor, bivariate_feature


__all__ = [
    "CoherenceFeatureExtractor",
    "connectivity_magnitude_square_coherence",
    "connectivity_imaginary_coherence",
    "connectivity_lagged_coherence",
]


class CoherenceFeatureExtractor(FeatureExtractor):
    def preprocess(self, x, **kwargs):
        f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
        f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
        kwargs["axis"] = -1
        n = x.shape[1]
        idx_x, idx_y = BivariateFeature.get_pair_iterators(n)
        ix, iy = list(chain(range(n), idx_x)), list(chain(range(n), idx_y))
        f, s = csd(x[:, ix], x[:, iy], **kwargs)
        if f_min is not None or f_max is not None:
            f_min_idx = f > f_min if f_min is not None else True
            f_max_idx = f < f_max if f_max is not None else True
            idx = np.logical_and(f_min_idx, f_max_idx)
            f = f[idx]
            s = s[..., idx]
        sx, sxy = np.split(s, [n], axis=1)
        sxx, syy = sx[:, idx_x].real, sx[:, idx_y].real
        c = sxy / np.sqrt(sxx * syy)
        return f, c


def _avg_over_bands(f, x, bands):
    bands_avg = dict()
    for k, v in bands.items():
        assert isinstance(k, str)
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert v[0] < v[1]
        mask = np.logical_and(f > v[0], f < v[1])
        avg = x[..., mask].mean(axis=-1)
        bands_avg[k] = avg
    return bands_avg


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_magnitude_square_coherence(
    f,
    c,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.real**2 + c.imag**2
    return _avg_over_bands(f, coher, bands)


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_imaginary_coherence(
    f,
    c,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag
    return _avg_over_bands(f, coher, bands)


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_lagged_coherence(
    f,
    c,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag / np.sqrt(1 - c.real)
    return _avg_over_bands(f, coher, bands)
