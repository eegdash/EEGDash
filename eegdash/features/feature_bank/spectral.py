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
    """Compute the Power Spectral Density (PSD) using Welch's method.

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
    """Normalize the PSD so that the total power equals 1.
    
    This is equivalent to treating the PSD as a Probability Density 
    Function (PDF), which is required for Spectral Entropy.

    """
    return f, p / p.sum(axis=-1, keepdims=True)


@FeaturePredecessor(spectral_preprocessor)
def spectral_db_preprocessor(f, p, /, eps=1e-15):
    r"""Convert the PSD to decibels: $10 \cdot \log_{10}(P + \epsilon)$. """
    return f, 10 * np.log10(p + eps)


@FeaturePredecessor(spectral_preprocessor)
@univariate_feature
def spectral_root_total_power(f, p, /):
    """Calculate the square root of the total spectral power.

    This is the frequency-domain equivalent of the Root Mean Square (RMS) 
    amplitude in the time domain.

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
    """Calculate the first spectral moment ('Weighted' Mean Frequency).

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
    """Calculate Hjorth Activity in the frequency domain.

    Activity represents the total power of the signal, calculated here 
    as the integral (sum) of the Power Spectral Density.

    Returns
    -------
    ndarray
        Total spectral power. Shape is ``p.shape[:-1]``.
    
    """
    return np.sum(p, axis=-1)


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_hjorth_mobility(f, p, /):
    """Calculate Hjorth Mobility in the frequency domain.

    Mobility is an estimate of the mean frequency. In the frequency domain, 
    it is calculated as the square root of the second spectral moment 
    of the normalized PSD.

    Returns
    -------
    ndarray
        Spectral mobility. Shape is ``p.shape[:-1]``.
    
    """
    return np.sqrt(np.sum(np.power(f, 2) * p, axis=-1))


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_hjorth_complexity(f, p, /):
    """Calculate Hjorth Complexity in the frequency domain.

    Complexity measures the bandwidth or the "irregularity" of the 
    spectrum. It is calculated using the fourth spectral moment of 
    the normalized PSD.

    Returns
    -------
    ndarray
        Spectral complexity. Shape is ``p.shape[:-1]``.
    """
    
    return np.sqrt(np.sum(np.power(f, 4) * p, axis=-1))


@FeaturePredecessor(spectral_normalized_preprocessor)
@univariate_feature
def spectral_entropy(f, p, /):
    """Calculate Spectral Entropy.

    Measures the complexity or "flatness" of the power spectrum. A 
    rhythmic signal has low entropy, while white noise has high entropy.

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
    """Calculate the Spectral Edge Frequency (SEF).

    The frequency below which a certain percentage (e.g., 90%) of 
    the total power is contained.

    Parameters
    ----------
    edge : float, optional
        The fraction of total power (default is 0.9 for SEF90).

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
    """Estimate the $1/f$ spectral slope using least-squares regression.

    This measures the slope and intercept of the PSD in log-log space.

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
    """Calculate total power within specified frequency bands.

    Parameters
    ----------
    bands : dict, optional
        Mapping of band names to (min, max) frequency tuples.

    Returns
    -------
    dict
        The summed power for each band.
    
    """
    return utils.reduce_freq_bands(f, p, bands, np.sum)
