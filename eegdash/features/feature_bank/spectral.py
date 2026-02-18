r"""
Spectral Feature Extraction
===========================

This module provides functions to compute various spectral features from signals.

Data Shape Convention
---------------------
This module follows a **Time-Last** convention:

* **Input:** ``(..., time)``
* **Output:** ``(...,)``

All functions collapse the last dimension (time), returning an ndarray of 
features corresponding to the leading dimensions (e.g., subjects, channels).
"""
import numba as nb
import numpy as np
from scipy.signal import welch

from ..decorators import FeaturePredecessor, univariate_feature
from . import utils
from .signal import SIGNAL_PREDECESSORS

__all__ = [
    "spectral_preprocessor",
    "spectral_normalized_preprocessor",
    "spectral_db_preprocessor",
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


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
def spectral_preprocessor(x, /, **kwargs):
    r"""Compute the Power Spectral Density (PSD) using Welch's method.

    Parameters
    ----------
    x : ndarray
        The input signal (shape: ..., n_times).
    **kwargs : dict
        Must include 'fs' (sampling frequency). Supports standard 
        `scipy.signal.welch` arguments like 'nperseg' and 'noverlap'.
        Can also include 'f_min' and 'f_max' for frequency slicing.

    Returns
    -------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density.

    Assertsions
    -----------
    - 'fs' must be provided in kwargs.
    
    """
    f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
    f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
    assert "fs" in kwargs
    kwargs["axis"] = -1
    f, p = welch(x, **kwargs)
    f_min, f_max = utils.get_valid_freq_band(kwargs["fs"], x.shape[-1], f_min, f_max)
    f, p = utils.slice_freq_band(f, p, f_min=f_min, f_max=f_max)
    return f, p


@FeaturePredecessor(spectral_preprocessor)
def spectral_normalized_preprocessor(f, p, /):
    r"""Normalize the PSD so that the total power equals 1.
    
    This is equivalent to treating the PSD as a Probability Density 
    Function (PDF), which is required for Spectral Entropy.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density.

    Returns
    -------
    f : ndarray
        Frequency vector (unchanged).
    p : ndarray
        Normalized Power Spectral Density.
    """
    return f, p / p.sum(axis=-1, keepdims=True)


@FeaturePredecessor(spectral_preprocessor)
def spectral_db_preprocessor(f, p, /, eps=1e-15):
    r"""Convert the PSD to decibels. 
    
    Calculated as:

    $10 \cdot \log_{10}(P + \epsilon)$. 
    
    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density.
    eps : float, optional
        A small constant to prevent log of zero (default: 1e-15).
    
    Returns
    -------
    f : ndarray
        Frequency vector (unchanged).
    ndarray
        Power Spectral Density in decibels. Shape is ``p.shape``.
    """
    return f, 10 * np.log10(p + eps)


@FeaturePredecessor(spectral_preprocessor)
@univariate_feature
def spectral_root_total_power(f, p, /):
    r"""Calculate the square root of the total spectral power.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density (PSD).

    Returns
    -------
    ndarray
        The root total power. Shape is ``p.shape[:-1]``.
    
    """
    return np.sqrt(p.sum(axis=-1))


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_moment(f, p, /):
    r"""Calculate the first spectral moment ('Weighted' Mean Frequency).

    When applied to a normalized PSD, this represents the "center of mass" 
    of the power spectrum.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Normalized Power Spectral Density.

    Returns
    -------
    ndarray
        The mean frequency of the signal. Shape is ``p.shape[:-1]``.
    
    """
    return np.sum(f * p, axis=-1)


@FeaturePredecessor(spectral_preprocessor)
@univariate_feature
def spectral_hjorth_activity(f, p, /):
    r"""Calculate Hjorth Activity in the frequency domain.

    Activity represents the total power of the signal, calculated here 
    as the integral (sum) of the Power Spectral Density.

    Parameters
    ----------
    f : ndarray
            Frequency vector.
    p : ndarray
        Power Spectral Density.
    
    Returns
    -------
    ndarray
        Total spectral power. Shape is ``p.shape[:-1]``.
    
    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return np.sum(p, axis=-1)


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_hjorth_mobility(f, p, /):
    r"""Calculate Hjorth Mobility in the frequency domain.

    Mobility is an estimate of the mean frequency. 
    For a normalized PSD, it is calculated as:

    .. math:: \sqrt{\frac{\sum f^2 P(f)}{\sum P(f)}}

    Where: 
    
    .. math:: \sum P(f) = 1$$ (since the PSD is normalized)

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Normalized Power Spectral Density.
    
    Returns
    -------
    ndarray
        Spectral mobility. Shape is ``p.shape[:-1]``.
    
    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    return np.sqrt(np.sum(np.power(f, 2) * p, axis=-1))


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_hjorth_complexity(f, p, /):
    r"""Calculate Hjorth Complexity in the frequency domain.

    Complexity measures the bandwidth or the "irregularity" of the spectrum.

    For a normalized PSD, it is calculated as:

    .. math:: \frac{\sqrt{\sum f^4 P(f)}}{\sum f^2 P(f)}

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Normalized Power Spectral Density.
    
    Returns
    -------
    ndarray
        Spectral complexity. Shape is ``p.shape[:-1]``.

    References
    ----------
    - Hjorth, B. (1970). EEG analysis based on time domain properties.
      Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
    """
    
    return np.sqrt(np.sum(np.power(f, 4) * p, axis=-1))

    # return np.sqrt(np.sum(np.power(f, 4) * p, axis=-1)) / np.sum(np.power(f, 2) * p, axis=-1)


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_entropy(f, p, /):
    r"""Calculate Spectral Entropy of thepower spectrum.

    Spectral Entropy (SE) measures the complexity or "disorder" of a signal.
    A high SE indicates a flat, broad spectrum (e.g., white noise), while a 
    low SE indicates a spectrum concentrated in a few frequency components.

    It is calculated as:
    
    .. math:: SE = -\sum_{i=1}^{N} p_i \ln(p_i)

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Normalized Power Spectral Density (treated as a PDF).
    
    Returns
    -------
    ndarray
        The entropy values. Shape is ``p.shape[:-1]``.
    
    """
    idx = p > 0
    plogp = np.zeros_like(p)
    plogp[idx] = p[idx] * np.log(p[idx])
    return -np.sum(plogp, axis=-1)


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def spectral_edge(f, p, /, edge=0.9):
    r"""Calculate the Spectral Edge Frequency (SEF).

    The frequency below which a certain percentage (e.g., 90%) of 
    the total power is contained.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Normalized Power Spectral Density (treated as a PDF).
    edge : float, optional
        The fraction of total power (default is 0.9 for SEF90).

    Returns
    -------
    ndarray
        The spectral edge frequency. Shape is ``p.shape[:-1]``.

    Notes
    -----
    Optimized with Numba ``fastmath`` for rapid scanning of cumulative power.

    """
    se = np.empty(p.shape[:-1])
    for i in np.ndindex(p.shape[:-1]):
        se[i] = f[np.searchsorted(np.cumsum(p[i]), edge)]
    return se


@FeaturePredecessor(spectral_db_preprocessor)
@univariate_feature
def spectral_slope(f, p, /):
    r"""Estimate the $1/f$ spectral slope using least-squares regression.

    This measures the slope and intercept of the PSD in log-log space.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density in decibels.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'exp': The slope/exponent (scaling).
        - 'int': The y-intercept (offset).
    
    """
    log_f = np.vstack((np.log(f), np.ones(f.shape[0]))).T
    r = np.linalg.lstsq(log_f, p.reshape(-1, p.shape[-1]).T)[0]
    r = r.reshape(2, *p.shape[:-1])
    return {"exp": r[0], "int": r[1]}


@FeaturePredecessor(
    spectral_preprocessor,
    spectral_normalized_preprocessor,
    spectral_db_preprocessor,
)
@univariate_feature
def spectral_bands_power(f, p, /, bands=utils.DEFAULT_FREQ_BANDS):
    r"""Calculate total power within specified frequency bands.

    Parameters
    ----------
    f : ndarray
        Frequency vector.
    p : ndarray
        Power Spectral Density (PSD).
    bands : dict, optional
        Mapping of band names to (min, max) frequency tuples.

    Returns
    -------
    dict
        The summed power for each band.
    
    """
    return utils.reduce_freq_bands(f, p, bands, np.sum)
