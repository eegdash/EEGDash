r"""Dimensionality Features Extraction
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

from ..decorators import feature_predecessor, univariate_feature
from .signal import signal_zero_crossings

__all__ = [
    "dimensionality_higuchi_fractal_dim",
    "dimensionality_petrosian_fractal_dim",
    "dimensionality_katz_fractal_dim",
]


@feature_predecessor()
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
    -----
    Optimized with Numba.

    For a theoretical overview of Higuchi's Fractal Dimension, see the
    `Wikipedia entry <https://en.wikipedia.org/wiki/Higuchi_dimension>`__.

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
    L_k = np.empty(k_max)
    for i in np.ndindex(x.shape[:-1]):
        for k in range(1, k_max + 1):
            L_km = np.empty(k)
            for m in range(k):
                # Correct logic: subsample with stride k, then take linear diffs
                # N_m = floor((N - m - 1) / k) * k
                # We need length of curve for this k and m
                # y = x[i, m::k]
                # distinct points count is y.shape[0]
                # normalization factor (N-1) / (floor(...) * k)

                # y = x[i][m::k] is strided, make it contiguous for np.diff
                y = np.ascontiguousarray(x[i][m::k])
                if y.shape[0] < 2:
                    L_km[m] = 0.0
                    continue

                n_max = ((N - m - 1) // k) * k
                norm_factor = (N - 1) / (n_max * k) if n_max > 0 else 0
                L_m = np.sum(np.abs(np.diff(y))) * norm_factor
                L_km[m] = L_m
            L_k[k - 1] = np.mean(L_km)
        L_k = np.maximum(L_k, eps)
        hfd[i] = np.linalg.lstsq(log_k, np.log(L_k))[0][0]
    return hfd


@feature_predecessor()
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
    return log_n / (2 * log_n - np.log(nd))


@feature_predecessor()
@univariate_feature
def dimensionality_katz_fractal_dim(x, /):
    r"""Calculate Katz Fractal Dimension (KFD).

    Katz Fractal Dimension [1]_ [2]_ is calculated as the ratio between the total
    path length and the maximum planar distance from the first point to any other
    point.

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
