from itertools import chain

import numpy as np
from scipy.signal import csd

from ..decorators import FeaturePredecessor, bivariate_feature
from ..extractors import BivariateFeature
from . import utils
from .signal import SIGNAL_PREDECESSORS

__all__ = [
    "connectivity_coherency_preprocessor",
    "connectivity_magnitude_square_coherence",
    "connectivity_imaginary_coherence",
    "connectivity_lagged_coherence",
]


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
def connectivity_coherency_preprocessor(x, /, **kwargs):
    f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
    f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
    assert "fs" in kwargs and "nperseg" in kwargs
    kwargs["axis"] = -1
    n = x.shape[1]
    idx_x, idx_y = BivariateFeature.get_pair_iterators(n)
    ix, iy = list(chain(range(n), idx_x)), list(chain(range(n), idx_y))
    f, s = csd(x[:, ix], x[:, iy], **kwargs)
    f_min, f_max = utils.get_valid_freq_band(kwargs["fs"], x.shape[-1], f_min, f_max)
    f, s = utils.slice_freq_band(f, s, f_min=f_min, f_max=f_max)
    p, sxy = np.split(s, [n], axis=1)
    sxx, syy = p[:, idx_x].real, p[:, idx_y].real
    c = sxy / np.sqrt(sxx * syy)
    return f, c


@FeaturePredecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_magnitude_square_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.real**2 + c.imag**2
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_imaginary_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_lagged_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag / np.sqrt(1 - c.real)
    return utils.reduce_freq_bands(f, coher, bands, np.mean)
