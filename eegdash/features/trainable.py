r"""Core Trainable Feature Interface.

This module defines the interface for creating trainable features.

The module provides the base class:

- :class:`TrainableFeature` - The interface for features requiring a
  fitting phase.
"""

from abc import ABC, abstractmethod

__all__ = [
    "TrainableFeature",
]


class TrainableFeature(ABC):
    r"""Abstract base class for features requiring a training phase.

    This class provides the interface for features that must be
    fitted on a representative dataset before they can process new samples.

    Attributes
    ----------
    _is_trained : bool
        Internal flag indicating whether the feature has completed
        its training phase.

    """

    def __init__(self):
        self._is_trained = False
        self.clear()

    @abstractmethod
    def clear(self):
        r"""Reset the internal state of the feature.

        This method must be implemented by subclasses to clear any learned
        parameters, statistics, or buffers.
        """
        pass

    @abstractmethod
    def partial_fit(self, *x, y=None):
        r"""Update the extractor's state using a single batch of data.

        This method allows for incremental learning, making it possible to
        train on datasets that are too large to fit into memory at once.

        Parameters
        ----------
        *x : tuple of ndarray
            The input data batch.
        y : ndarray, optional
            Target labels associated with the batch, required for supervised
            feature extraction methods.

        """
        pass

    def fit(self):
        r"""Finalize the training of the feature extractor.

        This method should be called after the entire training set has been
        processed via :meth:`partial_fit`. It transitions the object to a
        "trained" state, enabling the :meth:`__call__` method.
        """
        self._is_trained = True

    def __call__(self, *args, **kwargs):
        r"""Validate the fitted state before execution.

        Raises
        ------
        RuntimeError
            If the feature is called before :meth:`fit` has been executed.

        """
        if not self._is_trained:
            raise RuntimeError(
                f"{self.__class__} cannot be called, it has to be trained first."
            )
