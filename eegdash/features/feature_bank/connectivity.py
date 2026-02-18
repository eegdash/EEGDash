"""
Connectivity Feature Extraction
===============================

This module computes bivariate connectivity features based on the complex 
coherency between pairs of channels.

Data Shape Convention
---------------------
This module follows a **Time-Last** convention:

* **Input:** ``(..., time)``
* **Output:** ``(...,)``

All functions collapse the last dimension (time), returning an ndarray of 
features corresponding to the leading dimensions (e.g., subjects, channels).
"""
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
    r"""Compute Complex Coherency for all unique channel pairs.

    The Complex Coherency is calculated by estimating the Cross-Spectral Densities 
    (CSD) between pairs of channels and normalizing it by the auto-spectral densities. 

    Parameters
    ----------
    x : ndarray
        The input signal of shape (n_trials, n_channels, n_times).
    **kwargs : dict
        Additional keyword arguments to pass to `scipy.signal.csd`. Must include
        'fs' (sampling frequency) and 'nperseg' (length of each segment for CSD estimation).
        Optional keys include 'f_min' and 'f_max' to specify frequency band limits.

    Returns
    -------
    f : ndarray
        Frequency vector of shape (n_frequencies,).
    c : ndarray
        Complex coherency array of shape (n_trials, n_pairs, n_frequencies).
        Values are complex numbers where:
        - Absolute value |c| is the coherence (0 to 1).
        - Angle arg(c) is the phase lag.

    Assertions
    ----------
    - 'fs' and 'nperseg' must be provided in kwargs.
    
    """
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
    r"""Calculate Magnitude Squared Coherence (MSC).

    MSC measures the linear correlation between two signals in the frequency 
    domain. It is defined as the squared magnitude of the complex coherency: 
    .. math::|c|^2, where :math:`c` is the complex coherency.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate (defaults to DEFAULT_FREQ_BANDS).

    Returns
    -------
    dict
        Mean MSC for each frequency band.
    
    See also
    --------
    `https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity`
    
    """
    coher = c.real**2 + c.imag**2
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_imaginary_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate Imaginary Coherence (iCOH).

    iCOH captures only the non-zero phase-lagged synchronization.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate.

    Returns
    -------
    dict
        Mean Imaginary Coherence for each frequency band.
    
    See also
    --------
    `https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity`
    
    """
    coher = c.imag
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(connectivity_coherency_preprocessor)
@bivariate_feature
def connectivity_lagged_coherence(f, c, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate Lagged Coherence.

    Lagged coherence further refines the synchronization measure by 
    normalizing the imaginary part of the coherency, effectively removing 
    all instantaneous (zero-lag) contributions.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    c : ndarray
        Complex coherency array.
    bands : dict, optional
        Frequency bands to aggregate.

    Returns
    -------
    dict
        Mean Lagged Coherence for each frequency band.
    
    See also
    --------
    `https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity`
    
    """
    coher = c.imag / np.sqrt(1 - c.real) # chnage to: c.imag / np.sqrt(1 - c.real**2)
    return utils.reduce_freq_bands(f, coher, bands, np.mean)
