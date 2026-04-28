r"""Feature Channel-processing Kinds.

This module defines the fundamental feature-processing kinds and the
logic to map raw arrays to named features.

The module provides the classes:

- :class:`UnivariateFeature`
- :class:`BivariateFeature`
- :class:`MultivariateFeature`
"""

import numpy as np

__all__ = [
    "BivariateFeature",
    "MultivariateFeature",
    "UnivariateFeature",
]

DEFAULT_BIVARIATE_FORMAT_BY_DIRECTED = {
    False: "{}<>{}",
    True: "{}->{}",
}


class MultivariateFeature:
    r"""Logic wrapper for features that operate on one or more EEG channels.

    This class defines the logic for mapping raw numerical results into
    structured, named dictionaries. It determines the "kind" of a feature
    (e.g., univariate, bivariate) and handles the association of feature
    values with specific channels or channel groupings.

    Notes
    -----
    Subclasses should override :meth:`feature_channel_names` to define
    specific naming conventions for the extracted features.

    """

    def __call__(self, x: np.ndarray, _metadata: dict) -> dict | np.ndarray:
        r"""Convert a raw feature array into a named dictionary.

        Parameters
        ----------
        x : numpy.ndarray
            The computed feature array from the extraction function.
        _metadata : dict
            A dictionary of record and batch metadata.

        Returns
        -------
        dict or numpy.ndarray
            A dictionary where keys are formatted feature names and values
            are feature arrays. Returns the original array if channel names
            cannot be resolved.

        """
        f_channels = self.feature_channel_names(_metadata)
        if isinstance(x, dict):
            r = dict()
            for k, v in x.items():
                r.update(self._array_to_dict(v, f_channels, k))
            return r
        return self._array_to_dict(x, f_channels)

    @staticmethod
    def _array_to_dict(
        x: np.ndarray, f_channels: list[str], name: str = ""
    ) -> dict | np.ndarray:
        r"""Map a numpy array to a dictionary with named keys.

        Parameters
        ----------
        x : numpy.ndarray
            The feature values to be mapped.
        f_channels : list of str
            The list of generated feature channel names.
        name : str, default=""
            A prefix for the feature name.

        Returns
        -------
        dict or numpy.ndarray
            A dictionary of named features or the original array if
            `f_channels` is empty.

        """
        if not f_channels:
            return {name: x} if name else x
        assert x.shape[1] == len(f_channels), f"{x.shape[1]} != {len(f_channels)}"
        x = x.swapaxes(0, 1)
        prefix = f"{name}_" if name else ""
        names = [f"{prefix}{ch}" for ch in f_channels]
        return dict(zip(names, x))

    def feature_channel_names(self, _metadata: dict) -> list[str]:
        r"""Generate feature-specific names based on input channels.

        Parameters
        ----------
        _metadata : dict
            A dictionary of record and batch metadata.

        Returns
        -------
        list of str
            A list of strings defining the naming for each output feature.
            Returns an empty list in the base implementation.

        """
        return []


class UnivariateFeature(MultivariateFeature):
    r"""Feature kind for operations applied to each channel independently.

    Used when a single feature value is produced per channel.
    """

    def feature_channel_names(self, _metadata: dict) -> list[str]:
        r"""Return the channel names themselves as feature names.

        Parameters
        ----------
        _metadata : dict
            A dictionary of record and batch metadata.

        Returns
        -------
        list of str
            A list of channel names.

        """
        return _metadata["info"]["ch_names"]


class BivariateFeature(MultivariateFeature):
    r"""Feature kind for operations on pairs of channels.

    Designed for undirected relationship measures between two signals.

    Parameters
    ----------
    channel_pair_format : str
        A format string used to create feature names from pairs of
        channel names. Default is "{}<>{}" for undirected bivariate features
        or "{}->{}" for directed bivariate features.

    """

    def __init__(self, *args, channel_pair_format: str | None = None):
        super().__init__(*args)
        self.channel_pair_format = channel_pair_format

    def feature_channel_names(self, _metadata: dict) -> list[str]:
        r"""Generate feature names for each unique pair of channels.

        Parameters
        ----------
        _metadata : dict
            A dictionary of record and batch metadata.

        Returns
        -------
        list of str
            Formatted strings representing channel pairs (e.g., 'F3<>F4').

        """
        ch_names = _metadata["info"]["ch_names"]
        pair_format = (
            self.channel_pair_format
            if self.channel_pair_format is not None
            else DEFAULT_BIVARIATE_FORMAT_BY_DIRECTED[
                _metadata["ch_pair_iterator"].directed
            ]
        )
        return [
            pair_format.format(ch_names[i], ch_names[j])
            for i, j in zip(*_metadata["ch_pair_iterator"].get_pair_iterators())
        ]
