r"""
Common Spatial Pattern Features Extraction
==========================================
This module provides the Common Spatial Pattern (CSP) feature extractor
for signal classification.

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
import scipy
import scipy.linalg

from ..decorators import FeaturePredecessor, multivariate_feature
from ..extractors import TrainableFeature
from .signal import SIGNAL_PREDECESSORS

__all__ = [
    "CommonSpatialPattern",
]


@nb.njit(cache=True, fastmath=True, parallel=True)
def _update_mean_cov(count, mean, cov, x_count, x_mean, x_cov):
    r"""Online update of running mean and covariance matrix.

    Combine existing statistics with a new batch of data without 
    storing the entire dataset in memory.

    Parameters
    ----------
    count : int
        Total number of samples after the update.
    mean : ndarray
        Running mean vector to be updated.
    cov : ndarray
        Running covariance matrix to be updated.
    x_count : int
        Number of samples in the new batch.
    x_mean : ndarray
        Mean vector of the new batch.
    x_cov : ndarray
        Covariance matrix of the new batch.

    Notes
    -----
    Optimized with Numba.

    This function modifies `mean` and `cov` in place.   
    """
    alpha2 = x_count / count
    alpha1 = 1 - alpha2
    cov[:] = alpha1 * (cov + np.outer(mean, mean))
    cov[:] += alpha2 * (x_cov + np.outer(x_mean, x_mean))
    mean[:] = alpha1 * mean + alpha2 * x_mean
    cov[:] -= np.outer(mean, mean)


@FeaturePredecessor(*SIGNAL_PREDECESSORS)
@multivariate_feature
class CommonSpatialPattern(TrainableFeature):
    r"""Common Spatial Pattern (CSP) for binary signal classification.

    CSP finds spatial filters that maximize the variance for one class while 
    minimizing it for the other. It transforms multi-channel signals into a
    subspace where the differences between two conditions are most prominent.

    Attributes
    ----------
    _weights : ndarray
        The spatial filter matrix.
    _eigvals : ndarray
        The eigenvalues representing the variance ratio for class 0.
    _means : ndarray
        The class-wise means used for centering.
    _covs : ndarray
        The class-wise covariance matrices.

    Notes
    -----
    This implementation supports online learning through ``partial_fit``, 
    allowing the model to be updated with new batches.

    For a theoretical overview of Common Spatial Patterns, see the 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Common_spatial_pattern>`_.
    """
    def __init__(self):
        super().__init__()

    def clear(self):
        r"""Reset the internal state of the feature extractor. 

        See also
        --------
        :func:`eegdash.features.extractors.TrainableFeature.clear`
        """
        self._labels = None
        self._counts = np.array([0, 0])
        self._means = np.array([None, None])
        self._covs = np.array([None, None])
        self._mean = None
        self._eigvals = None
        self._weights = None

    def _update_labels(self, labels):
        r"""Update and validate the set of unique class labels.

        Parameters
        ----------
        labels : ndarray
            Labels to be added to the tracked set ``_labels``.

        Returns
        -------
        ndarray
            The updated array of tracked labels.

        Raises
        ------
        AssertionError
            If more than two unique labels are tracked.
        """
        if self._labels is None:
            self._labels = labels
        else:
            for label in labels:
                if label not in self._labels:
                    self._labels = np.append(self._labels, label)
        assert self._labels.shape[0] < 3
        return self._labels

    def _update_stats(self, l, x):
        r"""Update the running mean and covariance for a specific class.

        This method calculates the batch statistics and merges them with 
        the existing class-wise counts, means, and covariances.

        Parameters
        ----------
        l : int
            The index of the class being updated (0 or 1).
        x : ndarray
            The input data for this specific class.
        """
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
        r"""Incrementally update class-wise mean and covariance statistics.

        Parameters
        ----------
        x : ndarray
            Input array of shape (n_epochs, n_channels, n_times).
        y : ndarray
            Class labels for each epoch (must contain exactly two classes).

        Raises
        ------
        AssertionError
            If more than two unique labels are detected across all 
            partial fits.
        """
        labels = self._update_labels(np.unique(y))
        for i, l in enumerate(labels):
            ind = (y == l).nonzero()[0]
            if ind.shape[0] > 0:
                xl = self.transform_input(x[ind])
                self._update_stats(i, xl)

    @staticmethod
    def transform_input(x):
        r"""Reshape and transpose epoch data for matrix operations.

        Converts 3D epoch data into a 2D format suitable for covariance 
        estimation and spatial filtering. The temporal dimension is 
        collapsed into the samples dimension.

        Parameters
        ----------
        x : ndarray
            Input array of shape (n_epochs, n_channels, n_times).

        Returns
        -------
        ndarray
            Reshaped array of shape (n_epochs * n_times, n_channels).
        
        """
        return x.swapaxes(1, 2).reshape(-1, x.shape[1])

    def fit(self):
        r"""Solve the generalized eigenvalue problem to find spatial filters.

        Calculates the filters $W$ such that the ratio of variances between 
        the two classes is maximized. Filters are sorted by their 
        discriminative power (distance from 0.5 eigenvalue).

        See also
        --------
        :func:`~eegdash.features.extractors.TrainableFeature.fit`.

        Notes
        -----
        For more details on the CSP algorithm, visit the
        `Wikipedia entry <https://en.wikipedia.org/wiki/Common_spatial_pattern>`_.
        """
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
        # TODO: Verify correctness of docstring - return description
        r"""Apply CSP filters and return the log-variance of the projections.

        Parameters
        ----------
        x : ndarray
            Input array of shape (n_epochs, n_channels, n_times).
        n_select : int, optional
            Number of top filters to select.
        crit_select : float, optional
            Threshold for selecting filters based on eigenvalue distance 
            from 0.5. Filters with $\left| \lambda - 0.5 \right| > {crit}_{select}$ 
            are kept.

        Returns
        -------
        dict
            A dictionary where keys are string indices of the selected components 
            and values are 1D arrays of shape (n_epochs,) representing the 
            variance of that component.

        Raises
        ------
        RuntimeError
            If selection criteria result in zero filters.

        See also
        -------
        :func:`~eegdash.features.extractors.TrainableFeature.__call__` 

        Notes
        -----
        For more details on the CSP algorithm, visit the
        `Wikipedia entry <https://en.wikipedia.org/wiki/Common_spatial_pattern>`_.
        """
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
        proj = proj.reshape(x.shape[0], x.shape[2], -1).var(axis=1)
        return {f"{i}": proj[:, i] for i in range(proj.shape[-1])}
