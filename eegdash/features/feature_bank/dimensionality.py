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


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_petrosian_fractal_dim(x, /):
    nd = signal_zero_crossings(np.diff(x, axis=-1))
    log_n = np.log(x.shape[-1])
    return log_n / (2 * log_n - np.log(nd))


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_katz_fractal_dim(x, /):
    dists = np.abs(np.diff(x, axis=-1))
    L = dists.sum(axis=-1)
    a = dists.mean(axis=-1)
    log_n = np.log(L / a)
    d = np.abs(x[..., 1:] - x[..., 0, None]).max(axis=-1)
    return log_n / (np.log(d / L) + log_n)


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


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@univariate_feature
def dimensionality_hurst_exp(x, /):
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
    ns = np.unique(np.floor(np.power(2, np.arange(2, np.log2(x.shape[-1]) - 1))))
    a = np.vstack((np.arange(ns[-1]), np.ones(int(ns[-1])))).T
    log_n = np.vstack((np.log(ns), np.ones(ns.shape[0]))).T
    Fn = np.empty(ns.shape[0])
    alpha = np.empty(x.shape[:-1])
    for i in np.ndindex(x.shape[:-1]):
        X = np.cumsum(x[i] - np.mean(x[i]))
        for j, n in enumerate(ns):
            n = int(n)
            # Correct reshape to get windows as columns
            # Take truncation
            limit = n * (X.shape[0] // n)
            # reshape(-1, n).T ensures [0..n-1] is col 0, [n..2n-1] is col 1
            Z = np.reshape(X[:limit], (-1, n)).T
            # a[:n] is (n, 2)
            # Z is (n, num_windows)
            # lstsq returns residuals sum of squares for each column
            Fni2 = np.linalg.lstsq(a[:n], Z)[1] / n
            Fn[j] = np.sqrt(np.mean(Fni2))
        alpha[i] = np.linalg.lstsq(log_n, np.log(Fn))[0][0]
    return alpha
