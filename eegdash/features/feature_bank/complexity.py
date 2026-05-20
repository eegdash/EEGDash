r"""Complexity Feature Extraction
===============================

This module provides functions to compute various complexity features from signals.

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
from sklearn.neighbors import KDTree

from ..decorators import feature_predecessor, univariate_feature

__all__ = [
    "complexity_entropy_preprocessor",
    "complexity_approx_entropy",
    "complexity_multiscale_entropy",
    "complexity_sample_entropy",
    "complexity_svd_entropy",
    "complexity_lempel_ziv",
]


@nb.njit(cache=True, fastmath=True)
def _create_embedding(x, dim, lag):
    r"""Create a delay-coordinate embedding of the signal.

    Parameters
    ----------
    x : ndarray
        1D signal array.
    dim : int
        Embedding dimension.
    lag : int
        Time lag.

    Returns
    -------
    ndarray
        Embedded signal of shape ((x.shape[-1] - dim + 1) // lag, dim).

    Notes
    -----
    Optimized with Numba.

    """
    n_epochs = (x.shape[-1] - dim + 1) // lag
    y = np.empty((n_epochs, dim))
    for i in range(n_epochs):
        curr = i * lag
        y[i] = x[curr : curr + dim]
    return y


def _channel_app_samp_entropy_counts(x, m, r, l):
    r"""Helper to compute neighbor counts for a single channel using KDTree.

    Parameters
    ----------
    x : ndarray
        1D signal array.
    m : int
        Embedding dimension.
    r : float
        Tolerance threshold.
    l : int
        Time lag.

    Returns
    -------
    ndarray
        Neighbor counts for the given embedding dimension.

    """
    x_emb = _create_embedding(x, m, l)
    kdtree = KDTree(x_emb, metric="chebyshev")
    return kdtree.query_radius(x_emb, r, count_only=True)


@feature_predecessor()
def complexity_entropy_preprocessor(x, /, m=2, r=0.2, l=1):
    r"""Precompute neighbor counts for Approximate and Sample Entropy.

    This function creates a delay-embedding of the signal and uses a KDTree
    to count how many vectors are within a distance 'r' of each other.
    It computes counts for both dimension 'm' and 'm+1'.

    Parameters
    ----------
    x : ndarray
        The input signal of shape (..., n_times).
    m : int, optional
        Embedding dimension (length of compared sequences).
    r : float, optional
        Tolerance threshold, expressed as a fraction of the signal
        standard deviation.
    l : int, optional
        The lag or delay between successive embedding vectors.

    Returns
    -------
    counts_m : ndarray
        Neighbor counts for embedding dimension m.
    counts_mp1 : ndarray
        Neighbor counts for embedding dimension m + 1.

    """
    rr = r * x.std(axis=-1)
    counts_m = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // l))
    counts_mp1 = np.empty((*x.shape[:-1], (x.shape[-1] - m) // l))
    for i in np.ndindex(x.shape[:-1]):
        counts_m[i + (slice(None),)] = _channel_app_samp_entropy_counts(
            x[i], m, rr[i], l
        )
        counts_mp1[i + (slice(None),)] = _channel_app_samp_entropy_counts(
            x[i], m + 1, rr[i], l
        )
    return counts_m, counts_mp1


@feature_predecessor(complexity_entropy_preprocessor)
@univariate_feature
def complexity_approx_entropy(counts_m, counts_mp1, /):
    r"""Calculate Approximate Entropy (ApEn).

    Approximate Entropy quantifies the amount of regularity and the
    unpredictability of fluctuations over time-series data. Smaller values
    indicate more regular signals.

    Parameters
    ----------
    counts_m : ndarray
        Neighbor counts for embedding dimension m.
    counts_mp1 : ndarray
        Neighbor counts for embedding dimension m + 1.

    Returns
    -------
    ndarray
        Approximate Entropy values. Shape is ``x.shape[:-1]``.

    """
    phi_m = np.log(counts_m / counts_m.shape[-1]).mean(axis=-1)
    phi_mp1 = np.log(counts_mp1 / counts_mp1.shape[-1]).mean(axis=-1)
    return phi_m - phi_mp1


@feature_predecessor(complexity_entropy_preprocessor)
@univariate_feature
def complexity_sample_entropy(counts_m, counts_mp1, /):
    r"""Calculate Sample Entropy (SampEn).

    A refinement of Approximate Entropy that is more consistent and less
    dependent on signal length. It measures the likelihood that similar
    patterns of data will remain similar when the window size increases.

    Parameters
    ----------
    counts_m : ndarray
        Neighbor counts for embedding dimension m.
    counts_mp1 : ndarray
        Neighbor counts for embedding dimension m + 1.

    Returns
    -------
    ndarray
        SampEn values. Shape is ``x.shape[:-1]``.

    """
    A = np.sum(counts_mp1 - 1, axis=-1)
    B = np.sum(counts_m - 1, axis=-1)
    return -np.log(A / B)


@feature_predecessor()
@univariate_feature
def complexity_multiscale_entropy(x, /, m=2, r=0.2, l_max=16):
    r"""Calculate Multiscale Entropy (MSE).

    Computes the sample entropy (SampEn) for multiple timescales (from 1
    to ``l_max``), then calculate the integral of the SampEn as a function
    of timescale.

    Parameters
    ----------
    x : ndarray
        The input signal of shape (..., n_times).
    m : int, optional
        Embedding dimension (length of compared sequences).
    r : float, optional
        Tolerance threshold, expressed as a fraction of the signal
        standard deviation.
    l_max : int, optional
        The maximal lag or delay between successive embedding vectors.

    Returns
    -------
    ndarray
        MSE values. Shape is ``x.shape[:-1]``.

    """
    samp_en = np.empty((*x.shape[:-1], l_max))
    for l in range(l_max):
        samp_en[..., l] = complexity_sample_entropy(
            *complexity_entropy_preprocessor(x, m=m, r=r, l=l + 1)
        )
    samp_en[~np.isfinite(samp_en)] = 0
    return np.trapezoid(samp_en, dx=1, axis=-1) / l_max


@feature_predecessor()
@univariate_feature
def complexity_svd_entropy(x, /, m=10, tau=1):
    r"""Calculate Singular Value Decomposition (SVD) Entropy.

    SVD Entropy measures the complexity of the signal's embedding space.
    It indicates the number of independent components required to
    reconstruct the signal. Higher values suggest a more complex signal.

    Parameters
    ----------
    x : ndarray
        The input signal.
    m : int, optional
        The embedding dimension.
    tau : int, optional
        The time delay for embedding.

    Returns
    -------
    ndarray
        SVD Entropy values. Shape is ``x.shape[:-1]``.

    """
    x_emb = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // tau, m))
    for i in np.ndindex(x.shape[:-1]):
        x_emb[i + (slice(None), slice(None))] = _create_embedding(x[i], m, tau)
    s = np.linalg.svd(x_emb, compute_uv=False)
    s /= s.sum(axis=-1, keepdims=True)
    return -np.sum(s * np.log(s), axis=-1)


@feature_predecessor()
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def complexity_lempel_ziv(x, /, threshold=None, normalize=True):
    r"""Calculate Lempel-Ziv Complexity (LZC).

    LZC evaluates the randomness of a sequence by counting the number
    of distinct patterns it contains.

    Parameters
    ----------
    x : ndarray
        The input signal.
    threshold : float, optional
        Value used to binarize the signal. If None, the median is used.
    normalize : bool, optional
        If True, normalizes the result by :math:`\log_2(n) / n`, where ``n``
        is the signal length.

    Returns
    -------
    ndarray
        LZC values. Shape is ``x.shape[:-1]``.

    Notes
    -----
    - The implementation follows the constructive algorithm for
      production complexity as described by Kaspar and Schuster [1]_.
    - Optimized with Numba.

    References
    ----------
    .. [1] Kaspar, F., & Schuster, H. G. (1987). Easily calculable measure for the
           complexity of spatiotemporal patterns. Physical Review A, 36(2), 842.

    """
    lzc = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        t = np.median(x[i]) if threshold is None else threshold
        s = x[i] > t
        n = s.shape[0]
        j, k, l = 0, 1, 1
        k_max = 1
        lzc[i] = 1
        while True:
            if s[j + k - 1] == s[l + k - 1]:
                k += 1
                if l + k > n:
                    lzc[i] += 1
                    break
            else:
                k_max = np.maximum(k, k_max)
                j += 1
                if j == l:
                    lzc[i] += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    j, k, k_max = 0, 1, 1
                else:
                    k = 1
        if normalize:
            lzc[i] *= np.log2(n) / n
    return lzc
