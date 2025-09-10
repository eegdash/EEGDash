"""
This module provides functions for calculating complexity features from EEG
signals.

It includes implementations of various entropy measures and Lempel-Ziv
complexity, which are commonly used to quantify the complexity of time series
data.
"""
import numba as nb
import numpy as np
from sklearn.neighbors import KDTree

from ..decorators import FeaturePredecessor, univariate_feature
from ..extractors import FeatureExtractor
from .signal import SIGNAL_PREDECESSORS

__all__ = [
    "EntropyFeatureExtractor",
    "complexity_approx_entropy",
    "complexity_sample_entropy",
    "complexity_svd_entropy",
    "complexity_lempel_ziv",
]


@nb.njit(cache=True, fastmath=True)
def _create_embedding(x, dim, lag):
    """Create a time-delayed embedding of a time series.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    dim : int
        The embedding dimension.
    lag : int
        The time lag.

    Returns
    -------
    np.ndarray
        The embedded time series.
    """
    y = np.empty(((x.shape[-1] - dim + 1) // lag, dim))
    for i in range(0, x.shape[-1] - dim + 1, lag):
        y[i] = x[i : i + dim]
    return y


def _channel_app_samp_entropy_counts(x, m, r, l):
    """Calculate the number of similar vectors for approximate and sample entropy.

    Parameters
    ----------
    x : np.ndarray
        The input time series for a single channel.
    m : int
        The embedding dimension.
    r : float
        The tolerance radius.
    l : int
        The time lag.

    Returns
    -------
    np.ndarray
        The counts of similar vectors.
    """
    x_emb = _create_embedding(x, m, l)
    kdtree = KDTree(x_emb, metric="chebyshev")
    return kdtree.query_radius(x_emb, r, count_only=True)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
class EntropyFeatureExtractor(FeatureExtractor):
    """A feature extractor for various entropy measures."""

    def preprocess(self, x, m=2, r=0.2, l=1):
        """Preprocess the data for entropy calculation.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        m : int, default 2
            The embedding dimension.
        r : float, default 0.2
            The tolerance radius as a fraction of the standard deviation.
        l : int, default 1
            The time lag.

        Returns
        -------
        tuple
            A tuple of (counts_m, counts_mp1).
        """
        rr = r * x.std(axis=-1)
        counts_m = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // l))
        counts_mp1 = np.empty((*x.shape[:-1], (x.shape[-1] - m) // l))
        for i in np.ndindex(x.shape[:-1]):
            counts_m[*i, :] = _channel_app_samp_entropy_counts(x[i], m, rr[i], l)
            counts_mp1[*i, :] = _channel_app_samp_entropy_counts(x[i], m + 1, rr[i], l)
        return counts_m, counts_mp1


@FeaturePredecessor(EntropyFeatureExtractor)
@univariate_feature
def complexity_approx_entropy(counts_m, counts_mp1):
    """Calculate the approximate entropy.

    Parameters
    ----------
    counts_m : np.ndarray
        The counts of similar vectors for embedding dimension `m`.
    counts_mp1 : np.ndarray
        The counts of similar vectors for embedding dimension `m+1`.

    Returns
    -------
    np.ndarray
        The approximate entropy for each channel.
    """
    phi_m = np.log(counts_m / counts_m.shape[-1]).mean(axis=-1)
    phi_mp1 = np.log(counts_mp1 / counts_mp1.shape[-1]).mean(axis=-1)
    return phi_m - phi_mp1


@FeaturePredecessor(EntropyFeatureExtractor)
@univariate_feature
def complexity_sample_entropy(counts_m, counts_mp1):
    """Calculate the sample entropy.

    Parameters
    ----------
    counts_m : np.ndarray
        The counts of similar vectors for embedding dimension `m`.
    counts_mp1 : np.ndarray
        The counts of similar vectors for embedding dimension `m+1`.

    Returns
    -------
    np.ndarray
        The sample entropy for each channel.
    """
    A = np.sum(counts_mp1 - 1, axis=-1)
    B = np.sum(counts_m - 1, axis=-1)
    return -np.log(A / B)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def complexity_svd_entropy(x, m=10, tau=1):
    """Calculate the Singular Value Decomposition (SVD) entropy.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    m : int, default 10
        The embedding dimension.
    tau : int, default 1
        The time lag.

    Returns
    -------
    np.ndarray
        The SVD entropy for each channel.
    """
    x_emb = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // tau, m))
    for i in np.ndindex(x.shape[:-1]):
        x_emb[*i, :, :] = _create_embedding(x[i], m, tau)
    s = np.linalg.svdvals(x_emb)
    s /= s.sum(axis=-1, keepdims=True)
    return -np.sum(s * np.log(s), axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def complexity_lempel_ziv(x, threshold=None, normalize=True):
    """Calculate the Lempel-Ziv complexity.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    threshold : float, optional
        The binarization threshold. If None, the median is used.
    normalize : bool, default True
        Whether to normalize the complexity score.

    Returns
    -------
    np.ndarray
        The Lempel-Ziv complexity for each channel.
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
