import numba as nb
import numpy as np
from sklearn.neighbors import KDTree

from ..decorators import FeaturePredecessor, univariate_feature
from .signal import SIGNAL_PREDECESSORS

__all__ = [
    "complexity_entropy_preprocessor",
    "complexity_approx_entropy",
    "complexity_sample_entropy",
    "complexity_svd_entropy",
    "complexity_lempel_ziv",
]


@nb.njit(cache=True, fastmath=True)
def _create_embedding(x, dim, lag):
    n_epochs = (x.shape[-1] - dim + 1) // lag
    y = np.empty((n_epochs, dim))
    for i in range(n_epochs):
        curr = i * lag
        y[i] = x[curr : curr + dim]
    return y


def _channel_app_samp_entropy_counts(x, m, r, l):
    x_emb = _create_embedding(x, m, l)
    kdtree = KDTree(x_emb, metric="chebyshev")
    return kdtree.query_radius(x_emb, r, count_only=True)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
def complexity_entropy_preprocessor(x, /, m=2, r=0.2, l=1):
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


@FeaturePredecessor(complexity_entropy_preprocessor)
@univariate_feature
def complexity_approx_entropy(counts_m, counts_mp1, /):
    phi_m = np.log(counts_m / counts_m.shape[-1]).mean(axis=-1)
    phi_mp1 = np.log(counts_mp1 / counts_mp1.shape[-1]).mean(axis=-1)
    return phi_m - phi_mp1


@FeaturePredecessor(complexity_entropy_preprocessor)
@univariate_feature
def complexity_sample_entropy(counts_m, counts_mp1, /):
    A = np.sum(counts_mp1 - 1, axis=-1)
    B = np.sum(counts_m - 1, axis=-1)
    return -np.log(A / B)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def complexity_svd_entropy(x, /, m=10, tau=1):
    x_emb = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // tau, m))
    for i in np.ndindex(x.shape[:-1]):
        x_emb[i + (slice(None), slice(None))] = _create_embedding(x[i], m, tau)
    s = np.linalg.svd(x_emb, compute_uv=False)
    s /= s.sum(axis=-1, keepdims=True)
    return -np.sum(s * np.log(s), axis=-1)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def complexity_lempel_ziv(x, /, threshold=None, normalize=True):
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
