"""
This module provides functions for calculating connectivity features from EEG
signals.

It includes a feature extractor for coherence-based measures and functions for
calculating magnitude-squared coherence, imaginary coherence, and lagged
coherence.
"""
from itertools import chain

import numpy as np
from scipy.signal import csd

from ..decorators import FeaturePredecessor, bivariate_feature
from ..extractors import BivariateFeature, FeatureExtractor
from . import utils

__all__ = [
    "CoherenceFeatureExtractor",
    "connectivity_magnitude_square_coherence",
    "connectivity_imaginary_coherence",
    "connectivity_lagged_coherence",
]


class CoherenceFeatureExtractor(FeatureExtractor):
    """A feature extractor for coherence-based connectivity measures."""

    def preprocess(self, x, **kwargs):
        """Preprocess the data for coherence calculation.

        This method calculates the cross-spectral density (CSD) between all
        pairs of channels and then computes the coherence.

        Parameters
        ----------
        x : np.ndarray
            The input data, with shape (n_epochs, n_channels, n_times).
        **kwargs
            Keyword arguments for `scipy.signal.csd`.

        Returns
        -------
        tuple
            A tuple of (frequencies, coherence).
        """
        f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
        f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
        assert "fs" in kwargs and "nperseg" in kwargs
        kwargs["axis"] = -1
        n = x.shape[1]
        idx_x, idx_y = BivariateFeature.get_pair_iterators(n)
        ix, iy = list(chain(range(n), idx_x)), list(chain(range(n), idx_y))
        f, s = csd(x[:, ix], x[:, iy], **kwargs)
        f_min, f_max = utils.get_valid_freq_band(
            kwargs["fs"], x.shape[-1], f_min, f_max
        )
        f, s = utils.slice_freq_band(f, s, f_min=f_min, f_max=f_max)
        p, sxy = np.split(s, [n], axis=1)
        sxx, syy = p[:, idx_x].real, p[:, idx_y].real
        c = sxy / np.sqrt(sxx * syy)
        return f, c


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_magnitude_square_coherence(f, c, bands=utils.DEFAULT_FREQ_BANDS):
    """Calculate the magnitude-squared coherence.

    Parameters
    ----------
    f : np.ndarray
        The frequencies.
    c : np.ndarray
        The coherence values.
    bands : dict, optional
        The frequency bands to average over.

    Returns
    -------
    dict
        A dictionary of coherence values for each frequency band.
    """
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.real**2 + c.imag**2
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_imaginary_coherence(f, c, bands=utils.DEFAULT_FREQ_BANDS):
    """Calculate the imaginary part of the coherence.

    Parameters
    ----------
    f : np.ndarray
        The frequencies.
    c : np.ndarray
        The coherence values.
    bands : dict, optional
        The frequency bands to average over.

    Returns
    -------
    dict
        A dictionary of imaginary coherence values for each frequency band.
    """
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag
    return utils.reduce_freq_bands(f, coher, bands, np.mean)


@FeaturePredecessor(CoherenceFeatureExtractor)
@bivariate_feature
def connectivity_lagged_coherence(f, c, bands=utils.DEFAULT_FREQ_BANDS):
    """Calculate the lagged coherence.

    Parameters
    ----------
    f : np.ndarray
        The frequencies.
    c : np.ndarray
        The coherence values.
    bands : dict, optional
        The frequency bands to average over.

    Returns
    -------
    dict
        A dictionary of lagged coherence values for each frequency band.
    """
    # https://neuroimage.usc.edu/brainstorm/Tutorials/Connectivity
    coher = c.imag / np.sqrt(1 - c.real)
    return utils.reduce_freq_bands(f, coher, bands, np.mean)
