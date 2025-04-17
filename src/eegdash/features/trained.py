import numpy as np
import numba as nb
import scipy
import scipy.linalg

from .extractors import FeaturePredecessor, FitableFeature


@nb.njit(cache=True, fastmath=True, parallel=True)
def _update_mean_cov(count, mean, cov, x):
    alpha2 = 1 / count
    alpha1 = 1 - alpha2
    cov[:] = alpha1 * (cov + np.outer(mean, mean)) + alpha2 * np.outer(x, x)
    mean[:] = alpha1 * mean + alpha2 * x
    cov[:] -= np.outer(mean, mean)


@FeaturePredecessor()
class CSP(FitableFeature):
    def __init__(self):
        super().__init__()

    def clear(self):
        self._labels = None
        self._counts = [0, 0]
        self._means = [None, None]
        self._covs = [None, None]
        self._eigvals = None
        self._weights = None

    def _update_labels(self, label):
        if self._labels is None:
            self._labels = np.array([label])
        elif label not in self._labels:
            assert self._labels.shape[0] < 2
            self._labels = np.append(self._labels, label)
        return self._labels

    def _update_stats(self, l, x):
        if self._counts[l] == 0:
            self._counts[l] = 1
            self._means[l] = x.copy()
            self._covs[l] = np.zeros((x.shape[0], x.shape[0]))
        else:
            self._counts[l] += 1
            _update_mean_cov(self._counts[l], self._means[l], self._covs[l], x)

    def partial_fit(self, x, y=None):
        labels = self._update_labels(y)
        self._update_stats((labels == y).nonzero()[0][0], x.ravel())

    def fit(self):
        l, w = scipy.linalg.eig(self._covs[0], self._covs[0] + self._covs[1])
        ord = np.abs(l - 0.5).argsort()[::-1]
        self._eigvals = l[ord]
        self._weights = w[:, ord]
        return super().fit()

    def __call__(self, x, n_select=None, crit_select=None):
        super().__call__()
        w = self._weights
        if n_select:
            w = w[:, :n_select]
        if crit_select:
            sel = 0.5 - np.abs(self._eigvals - 0.5) < crit_select
            w = w[:, sel]
        if w.shape[-1] == 0:
            raise RuntimeError(
                "CSP weights selection criterion is too strict,"
                + "all weights were filtered out."
            )
        proj = w.T @ x.ravel()[:, None]
        return {f'{i}': proj[i] for i in range(proj.shape[0])}
