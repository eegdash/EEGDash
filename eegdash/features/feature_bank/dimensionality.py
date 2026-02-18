r"""
Dimensionality Features Extraction
==================================

This module provides functions to compute various dimensionality features
from signals.

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
from scipy import special

from ..decorators import FeaturePredecessor, univariate_feature
from .signal import SIGNAL_PREDECESSORS, signal_zero_crossings

__all__ = [
    "dimensionality_higuchi_fractal_dim",
    "dimensionality_petrosian_fractal_dim",
    "dimensionality_katz_fractal_dim",
    "dimensionality_hurst_exp",
    "dimensionality_detrended_fluctuation_analysis",
]


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def dimensionality_higuchi_fractal_dim(x, /, k_max=10, eps=1e-7):
    r"""Calculate Higuchi's Fractal Dimension (HFD).

    Higuchi's Fractal Dimension [1]_ [2]_ estimates the complexity of a time series
    by measuring the mean length of the curve at different time scales $k$. It is 
    highly robust for non-stationary signals. 

    Parameters
    ----------
    x : ndarray
        The input signal.
    k_max : int, optional
        Maximum time interval (delay) used for calculating curve lengths.
    eps : float, optional
        A small constant to avoid log of zero during regression.

    Returns
    -------
    ndarray
        The Higuchi's Fractal Dimension values. 
        Shape is ``x.shape[:-1]``.

    Notes
    ----------
    Optimized with Numba.

    For a theoretical overview of Higuchi's Fractal Dimension, see the 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Higuchi_dimension>`_.

    References
    ----------
    .. [1] Higuchi, T., 1988. Approach to an irregular time series on the basis of 
           the fractal theory. Physica D: Nonlinear Phenomena, 31(2), pp.277-283.
    .. [2] Esteller, R., Vachtsevanos, G., Echauz, J. and Litt, B., 2001. 
           A comparison of waveform fractal dimension algorithms. IEEE Transactions 
           on Circuits and Systems I: Fundamental Theory and Applications, 48(2), pp.177-183.
    """
    N = x.shape[-1]
    hfd = np.empty(x.shape[:-1])
    log_k = np.vstack((-np.log(np.arange(1, k_max + 1)), np.ones(k_max))).T
    L_km = np.empty(k_max)
    L_k = np.empty(k_max)
    for i in np.ndindex(x.shape[:-1]):
        for k in range(1, k_max + 1):
            for m in range(k):
                L_km[m] = np.mean(np.abs(np.diff(x[i + (slice(m, None),)], n=k)))
            L_k[k - 1] = (N - 1) * np.sum(L_km[:k]) / (k**3)
        L_k = np.maximum(L_k, eps)
        hfd[i] = np.linalg.lstsq(log_k, np.log(L_k))[0][0]
    return hfd


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_petrosian_fractal_dim(x, /):
    r"""Calculate Petrosian Fractal Dimension (PFD).

    Petrosian Fractal Dimension [1]_ [2]_ provides a fast estimate of fractal 
    dimension by analyzing the number of sign changes in the signal's 
    first derivative.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The Petrosian Fractal Dimension values. 
        Shape is ``x.shape[:-1]``.

    References
    ----------
    .. [1] Petrosian, A., 1995. Kolmogorov complexity of finite sequences and 
           recognition of different preictal EEG patterns. In Proceedings of the 
           1995 IEEE International Symposium on Circuits and Systems (Vol. 2, pp. 86-89). IEEE.
    .. [2] Esteller, R., Vachtsevanos, G., Echauz, J. and Litt, B., 2001. 
           A comparison of waveform fractal dimension algorithms. IEEE Transactions 
           on Circuits and Systems I: Fundamental Theory and Applications, 48(2), pp.177-183.
    """
    nd = signal_zero_crossings(np.diff(x, axis=-1))
    log_n = np.log(x.shape[-1])
    return log_n / (np.log(nd) + log_n)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_katz_fractal_dim(x, /):
    r"""Calculate Katz Fractal Dimension (KFD).

    KFD[1]_ [2]_ is calculated as the ratio between the total path length and the 
    maximum planar distance from the first point to any other point.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The Katz Fractal Dimension values. 
        Shape is ``x.shape[:-1]``.
    
    References
    ----------
    .. [1] Katz, M. J. (1988). Fractals and the analysis of waveforms.
           Computers in Biology and Medicine, 18(3), 145-156.
    .. [2] Esteller, R., Vachtsevanos, G., Echauz, J. and Litt, B., 2001. 
           A comparison of waveform fractal dimension algorithms. IEEE Transactions 
           on Circuits and Systems I: Fundamental Theory and Applications, 48(2), pp.177-183.
    """
    dists = np.abs(np.diff(x, axis=-1))
    L = dists.sum(axis=-1)
    a = dists.mean(axis=-1)
    log_n = np.log(L / a)
    d = np.abs(x[..., 1:] - x[..., 0, None]).max(axis=-1)
    return log_n / (np.log(d / L) + log_n)


@nb.njit(cache=True, fastmath=True)
def _hurst_exp(x, ns, a, gamma_ratios, log_n):
    r"""Internal helper to calculate the Hurst Exponent. 
    
    The Hurst Exponent is calculated using the Rescaled range (R/S) analysis
    method, expanded with Anis-Lloyd correction.
    

    Parameters
    ----------
    x : ndarray
        The input signal.
    ns : ndarray
        The array of window sizes (time scales).
    a : ndarray
        Precomputed bias correction factors for the Anis-Lloyd correction.
    gamma_ratios : ndarray
        Precomputed Gamma function ratios for the Anis-Lloyd correction.
    log_n : ndarray
        The natural logarithm of the window sizes (ns).

    Returns
    -------
    ndarray
        The estimated Hurst exponent for each channel/trial. 
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    Optimized with Numba.

    References
    ----------
    For more details on the Hurst Exponent and R/S analysis, visit the
    `Wikipedia entry <https://en.wikipedia.org/wiki/Hurst_exponent#Rescaled_range_(R/S)_analysis>`_.
    """
    h = np.empty(x.shape[:-1])
    rs = np.empty((ns.shape[0], x.shape[-1] // ns[0]))
    log_rs = np.empty(ns.shape[0])
    for i in np.ndindex(x.shape[:-1]):
        t0 = 0
        for j, n in enumerate(ns):
            for k, t0 in enumerate(range(0, x.shape[-1], n)):
                xj = x[i][t0 : t0 + n]
                m = np.mean(xj)
                y = xj - m
                z = np.cumsum(y)
                r = np.ptp(z)
                s = np.sqrt(np.mean(y**2))
                if s == 0.0:
                    rs[j, k] = np.nan
                else:
                    rs[j, k] = r / s
            log_rs[j] = np.log(np.nanmean(rs[j, : x.shape[1] // n]))
            log_rs[j] -= np.log(np.sum(np.sqrt((n - a[:n]) / a[:n])) * gamma_ratios[j])
        h[i] = 0.5 + np.linalg.lstsq(log_n, log_rs)[0][0]
    return h


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_hurst_exp(x, /):
    r"""Estimate the Hurst Exponent.

    The Hurst exponent quantifies the long-term memory and predictability of 
    a time series. It indicates whether a process is purely random, tends to 
    trend in the same direction (persistent), or tends to reverse its direction 
    (anti-persistent).

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The estimated Hurst Exponents. 
        Shape is ``x.shape[:-1]``.

    Notes
    -----
    This function calculate the Gamma Function Ratios and Bias Correction Factors
    to apply the Anis-Lloyd correction for small sample sizes.

    For more details on the Hurst Exponent and R/S analysis, visit the
    `Wikipedia entry <https://en.wikipedia.org/wiki/Hurst_exponent#Rescaled_range_(R/S)_analysis>`_.
    """
    ns = np.unique(np.power(2, np.arange(2, np.log2(x.shape[-1]) - 1)).astype(int))
    idx = ns > 340
    gamma_ratios = np.empty(ns.shape[0])
    gamma_ratios[idx] = 1 / np.sqrt(ns[idx] / 2)
    gamma_ratios[~idx] = special.gamma((ns[~idx] - 1) / 2) / special.gamma(ns[~idx] / 2)
    gamma_ratios /= np.sqrt(np.pi)
    log_n = np.vstack((np.log(ns), np.ones(ns.shape[0]))).T
    a = np.arange(1, ns[-1], dtype=float)
    return _hurst_exp(x, ns, a, gamma_ratios, log_n)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def dimensionality_detrended_fluctuation_analysis(x, /):
    r"""Calculate the Scaling Exponent via DFA.

    Detrended Fluctuation Analysis (DFA) is a method used to detect long-range 
    temporal correlations (LRTC) in non-stationary signals. It is a more robust 
    way to estimate the Hurst exponent when the data is noisy or has shifting trends.

    Parameters
    ----------
    x : ndarray
        The input signal.

    Returns
    -------
    ndarray
        The DFA scaling exponents ($\alpha$). 
        Shape is ``x.shape[:-1]``.
        
    Notes
    -----
    Optimized with Numba.

    For a theoretical overview of Detrended Fluctuation Analysis, see the 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis>`_.
    """
    ns = np.unique(np.floor(np.power(2, np.arange(2, np.log2(x.shape[-1]) - 1))))
    a = np.vstack((np.arange(ns[-1]), np.ones(int(ns[-1])))).T
    log_n = np.vstack((np.log(ns), np.ones(ns.shape[0]))).T
    Fn = np.empty(ns.shape[0])
    alpha = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        X = np.cumsum(x[i] - np.mean(x[i]))
        for j, n in enumerate(ns):
            n = int(n)
            Z = np.reshape(X[: n * (X.shape[0] // n)], (n, X.shape[0] // n))
            Fni2 = np.linalg.lstsq(a[:n], Z)[1] / n
            Fn[j] = np.sqrt(np.mean(Fni2))
        alpha[i] = np.linalg.lstsq(log_n, np.log(Fn))[0][0]
    return alpha
