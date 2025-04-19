import numpy as np
import numba as nb
from scipy import special

from ..extractors import ByChannelFeatureExtractor, FeaturePredecessor
from .signal import signal_zero_crossings


__all__ = [
    "dimensionality_higuchi_fractal_dim",
    "dimensionality_petrosian_fractal_dim",
    "dimensionality_katz_fractal_dim",
    "dimensionality_hurst_exp",
    "dimensionality_detrended_fluctuation_analysis",
]


@FeaturePredecessor(ByChannelFeatureExtractor)
@nb.njit(cache=True, fastmath=True)
def dimensionality_higuchi_fractal_dim(x, k_max=10, eps=1e-7):
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
def dimensionality_petrosian_fractal_dim(x):
    nd = signal_zero_crossings(np.diff(x, axis=-1))
    log_n = np.log(x.shape[-1])
    return log_n / (np.log(nd) + log_n)


@FeaturePredecessor(ByChannelFeatureExtractor)
def dimensionality_katz_fractal_dim(x):
    dists = np.abs(np.diff(x, axis=-1))
    L = dists.sum(axis=-1)
    a = dists.mean(axis=-1)
    log_n = np.log(L / a)
    d = np.abs(x[..., 1:] - x[..., 0, None]).max(axis=-1)
    return log_n / (np.log(d / L) + log_n)


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
def dimensionality_hurst_exp(x):
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
def dimensionality_detrended_fluctuation_analysis(x):
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
