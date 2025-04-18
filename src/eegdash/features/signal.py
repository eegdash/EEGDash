import numbers
import numpy as np
import numba as nb
from scipy import stats, special
from sklearn.neighbors import KDTree

from .extractors import FeatureExtractor, ByChannelFeatureExtractor, FeaturePredecessor


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_mean(x):
    return x.mean(axis=-1)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_variance(x, **kwargs):
    return x.var(axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_std(x, **kwargs):
    return x.std(axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_skewness(x, **kwargs):
    return stats.skew(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_kurtosis(x, **kwargs):
    return stats.kurtosis(x, axis=x.ndim - 1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_rms(x):
    return np.sqrt(np.power(x, 2).mean(axis=-1))


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_amp_ptp(x, **kwargs):
    return np.ptp(x, axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_quantile(x, q: numbers.Number = 0.5, **kwargs):
    return np.quantile(x, q=q, axis=-1, **kwargs)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_line_length(x):
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_zero_crossings(x, threshold=1e-15):
    zero_ind = np.logical_and(x > -threshold, x < threshold)
    zero_cross = np.diff(zero_ind, axis=-1).astype(int).sum(axis=-1)
    y = x.copy()
    y[zero_ind] = 0
    zero_cross += np.sum(np.signbit(y[..., :-1]) != np.signbit(y[..., 1:]), axis=-1)
    return zero_cross


@FeaturePredecessor(FeatureExtractor, ByChannelFeatureExtractor)
class HjorthFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, x):
        return (x, np.diff(x, axis=-1), x.std(axis=-1))


@FeaturePredecessor(HjorthFeatureExtractor)
def signal_hjorth_mobility(x, dx, x_std):
    return dx.std(axis=-1) / x_std


@FeaturePredecessor(HjorthFeatureExtractor)
def signal_hjorth_complexity(x, dx, x_std):
    return np.diff(dx, axis=-1).std(axis=-1) / x_std


@FeaturePredecessor(ByChannelFeatureExtractor)
@nb.njit(cache=True, fastmath=True)
def signal_higuchi_fractal_dim(x, k_max=10, eps=1e-7):
    N = x.shape[-1]
    hfd = np.empty(x.shape[:-1])
    log_k = np.vstack((-np.log(np.arange(1, k_max + 1)), np.ones(k_max))).T
    L_km = np.empty(k_max)
    L_k = np.empty(k_max)
    for i in np.ndindex(x.shape[:-1]):
        for k in range(1, k_max + 1):
            for m in range(k):
                L_km[m] = np.mean(np.abs(np.diff(x[*i, m:], n=k)))
            L_k[k - 1] = (N - 1) * np.sum(L_km[:k]) / (k**3)
        L_k = np.maximum(L_k, eps)
        hfd[i] = np.linalg.lstsq(log_k, np.log(L_k))[0][0]
    return hfd


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_katz_fractal_dim(x, eps=1e-7):
    dists = np.abs(np.diff(x, axis=-1))
    L = dists.sum(axis=-1)
    a = dists.mean(axis=-1)
    log_n = np.log(L / a)
    d = np.abs(x[..., 1:].T - x[..., 0].T).T.max(axis=-1)
    return log_n / (np.log(d / L) + log_n)


@nb.njit(cache=True, fastmath=True)
def _create_embedding(x, dim, lag):
    y = np.empty(((x.shape[-1] - dim + 1) // lag, dim))
    for i in range(0, x.shape[-1] - dim + 1, lag):
        y[i] = x[i : i + dim]
    return y


def _channel_app_samp_entropy_counts(x, m, r, l):
    x_emb = _create_embedding(x, m, l)
    kdtree = KDTree(x_emb, metric="chebyshev")
    return kdtree.query_radius(x_emb, r, count_only=True)


@FeaturePredecessor(FeatureExtractor, ByChannelFeatureExtractor)
class EntropyFeatureExtractor(ByChannelFeatureExtractor):
    def preprocess(self, x, m=2, r=0.2, l=1):
        rr = r * x.std(axis=-1)
        counts_m = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // l))
        counts_mp1 = np.empty((*x.shape[:-1], (x.shape[-1] - m) // l))
        for i in np.ndindex(x.shape[:-1]):
            counts_m[*i, :] = _channel_app_samp_entropy_counts(x[i], m, rr[i], l)
            counts_mp1[*i, :] = _channel_app_samp_entropy_counts(x[i], m + 1, rr[i], l)
        return counts_m, counts_mp1


@FeaturePredecessor(EntropyFeatureExtractor)
def signal_app_entropy(counts_m, counts_mp1):
    phi_m = np.log(counts_m / counts_m.shape[-1]).mean(axis=-1)
    phi_mp1 = np.log(counts_mp1 / counts_mp1.shape[-1]).mean(axis=-1)
    return phi_m - phi_mp1


@FeaturePredecessor(EntropyFeatureExtractor)
def signal_samp_entropy(counts_m, counts_mp1):
    A = np.sum(counts_mp1 - 1, axis=-1)
    B = np.sum(counts_m - 1, axis=-1)
    return -np.log(A / B)


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_svd_entropy(x, m=10, tau=1):
    x_emb = np.empty((*x.shape[:-1], (x.shape[-1] - m + 1) // tau, m))
    for i in np.ndindex(x.shape[:-1]):
        x_emb[i, :, :] = _create_embedding(x[i], m, tau)
    s = np.linalg.svdvals(x_emb)
    s /= s.sum(axis=-1, keepdims=True)
    return -np.sum(s * np.log(s), axis=-1)


@FeaturePredecessor(ByChannelFeatureExtractor)
@nb.njit(cache=True, fastmath=True)
def _hurst_exp(x, ns, a, gamma_ratios, log_n):
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


@FeaturePredecessor(ByChannelFeatureExtractor)
def signal_hurst_exp(x):
    ns = np.unique(np.power(2, np.arange(2, np.log2(x.shape[-1]) - 1)).astype(int))
    idx = ns > 340
    gamma_ratios = np.empty(ns.shape[0])
    gamma_ratios[idx] = 1 / np.sqrt(ns[idx] / 2)
    gamma_ratios[~idx] = special.gamma((ns[~idx] - 1) / 2) / special.gamma(ns[~idx] / 2)
    gamma_ratios /= np.sqrt(np.pi)
    log_n = np.vstack((np.log(ns), np.ones(ns.shape[0]))).T
    a = np.arange(1, ns[-1], dtype=float)
    return _hurst_exp(x, ns, a, gamma_ratios, log_n)


@FeaturePredecessor(ByChannelFeatureExtractor)
@nb.njit(cache=True, fastmath=True)
def signal_dfa(x):
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


# =================================  Aliases  =================================

signal_hjorth_activity = signal_variance
