import numba as nb
import numpy as np
import scipy
import scipy.linalg

from ..decorators import multivariate_feature
from ..extractors import FitableFeature

__all__ = [
    "CommonSpatialPattern",
]


@nb.njit(cache=True, fastmath=True, parallel=True)
def _update_mean_cov(count, mean, cov, x_count, x_mean, x_cov):
    alpha2 = x_count / count
    alpha1 = 1 - alpha2
    cov[:] = alpha1 * (cov + np.outer(mean, mean))
    cov[:] += alpha2 * (x_cov + np.outer(x_mean, x_mean))
    mean[:] = alpha1 * mean + alpha2 * x_mean
    cov[:] -= np.outer(mean, mean)


@multivariate_feature
class CommonSpatialPattern(FitableFeature):
    def __init__(self):
        super().__init__()

    def clear(self):
        self._labels = None
        self._counts = np.array([0, 0])
        self._means = np.array([None, None])
        self._covs = np.array([None, None])
        self._mean = None
        self._eigvals = None
        self._weights = None

    def _update_labels(self, labels):
        if self._labels is None:
            self._labels = labels
        else:
            for label in labels:
                if label not in self._labels:
                    self._labels = np.append(self._labels, label)
        assert self._labels.shape[0] < 3
        return self._labels

    def _update_stats(self, l, x):
        x_count, x_mean, x_cov = x.shape[0], x.mean(axis=0), np.cov(x.T, ddof=0)
        if self._counts[l] == 0:
            self._counts[l] = x_count
            self._means[l] = x_mean
            self._covs[l] = x_cov
        else:
            self._counts[l] += x_count
            _update_mean_cov(
                self._counts[l], self._means[l], self._covs[l], x_count, x_mean, x_cov
            )

    def partial_fit(self, x, y=None):
        labels = self._update_labels(np.unique(y))
        for i, l in enumerate(labels):
            ind = (y == l).nonzero()[0]
            if ind.shape[0] > 0:
                xl = self.transform_input(x[ind])
                self._update_stats(i, xl)

    @staticmethod
    def transform_input(x):
        return x.swapaxes(1, 2).reshape(-1, x.shape[1])

    def fit(self):
        alphas = self._counts / self._counts.sum()
        self._mean = np.sum(alphas * self._means)
        for l in range(len(self._labels)):
            self._covs[l] *= self._counts[l] / (self._counts[1] - 1)
        l, w = scipy.linalg.eig(self._covs[0], self._covs[0] + self._covs[1])
        l = l.real
        ind = l > 0
        l, w = l[ind], w[:, ind]
        ord = np.abs(l - 0.5).argsort()[::-1]
        self._eigvals = l[ord]
        self._weights = w[:, ord]
        super().fit()

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
        proj = (self.transform_input(x) - self._mean) @ w
        proj = proj.reshape(x.shape[0], x.shape[2], -1).mean(axis=1)
        return {f"{i}": proj[:, i] for i in range(proj.shape[-1])}
