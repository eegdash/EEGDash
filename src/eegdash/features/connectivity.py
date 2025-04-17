from itertools import chain
import numpy as np
from scipy.signal import csd
from .extractors import FeatureExtractor, ByChannelPairFeatureExtractor, Feature


@Feature(FeatureExtractor, ByChannelPairFeatureExtractor)
class CoherenceFeatureExtractor(ByChannelPairFeatureExtractor):
    def preprocess(self, x, **kwargs):
        f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
        f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
        n = x.shape[0]
        idx_x, idx_y = ByChannelPairFeatureExtractor.get_pair_iterators(n)
        ix, iy = list(chain(range(n), idx_x)), list(chain(range(n), idx_y))
        f, s = csd(x[ix], x[iy], **kwargs)
        if f_min is not None or f_max is not None:
            f_min_idx = f > f_min if f_min is not None else True
            f_max_idx = f < f_max if f_max is not None else True
            idx = np.logical_and(f_min_idx, f_max_idx)
            f = f[idx]
            s = s[:, idx]
        sx, sxy = np.split(s, [n], axis=0)
        sxx, syy = sx[idx_x].real, sx[idx_y].real
        return f, sxx, syy, sxy


def _avg_over_bands(f, x, bands):
    bands_avg = dict()
    for k, v in bands.items():
        assert isinstance(k, str)
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert v[0] < v[1]
        mask = np.logical_and(f > v[0], f < v[1])
        avg = x[:, mask].mean(axis=-1)
        bands_avg[k] = avg
    return bands_avg


@Feature(CoherenceFeatureExtractor)
def connectivity_magnitude_square_coherence(
    f,
    sxx,
    syy,
    sxy,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    coherence = ((sxy.real**2) + (sxy.imag**2)) / (sxx * syy)
    return _avg_over_bands(f, coherence, bands)
