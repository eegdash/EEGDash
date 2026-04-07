r"""Basic Feature Extraction Utilities

This module defines basic utilities for feature extraction.

"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Iterable, List, Tuple

import numpy as np
from numba.core.dispatcher import Dispatcher

__all__ = [
    "BivariateIterator",
    "channel_names_to_indices",
    "get_underlying_func",
]


def get_underlying_func(func: Callable) -> Callable:
    r"""Retrieve the original Python function from a potential wrapper.

    Parameters
    ----------
    func : callable
        The function to unwrap. Typically a raw function, a
        :class:`functools.partial` object, or a Numba :class:`Dispatcher`.

    Returns
    -------
    callable
        The underlying Python function.

    Notes
    -----
    This utility specifically handles:
    * **functools.partial**: Returns the ``.func`` attribute.
    * **numba.Dispatcher**: Returns the ``.py_func`` attribute.

    """
    f = func
    if isinstance(f, partial):
        f = f.func
    if isinstance(f, Dispatcher):
        f = f.py_func
    return f


class BivariateIterator:
    r"""Pairs iterator for iterating pairs of channels.

    Parameters
    ----------
    pairs : Iterable[tuple[int, int]] | int
        If an iterable of tuples is given, it represents the channel index
        pairs to iterate
        If an integer ``n`` is given, iterate through all unique pairs
        out of ``n`` channels.
    directed : bool
        If an integer was given in ``pairs``, this parameter controls whether
        all directed pairs should be iterated.
        Otherwise this parameter is ignored.
        Default is False.

    """

    def __init__(self, pairs: Iterable[Tuple[int, int]] | int, directed=False):
        if isinstance(pairs, int):
            if not directed:
                pairs = list(zip(*np.triu_indices(pairs, 1)))
            else:
                pairs = list(zip(*np.triu_indices(pairs, 1))) + list(
                    zip(*np.tril_indices(pairs, -1))
                )
        self.pairs = list(pairs)

    def get_pair_iterators(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Get indices for pairs of channels.

        Computes the upper triangle indices of an (n, n) matrix,
        excluding the diagonal.

        Returns
        -------
        tuple of ndarray
            The row and column indices for the unique pairs.

        """
        return tuple([np.array(x) for x in zip(*self.pairs)])


def channel_names_to_indices(channels: List[str], ch_names: List[str]) -> List[int]:
    r"""Converts a list of channel names to channel indices in another list.

    Parameters
    ----------
    channels : List[str]
        A list of channel names.
    ch_names : List[str]
        A list of existing channel names to take indices from.

    Returns
    -------
    List[int]
        A list of channel indices.

    Raises
    ------
    ValueError
        If the channel name was not found in the existing channels list.

    """
    channel_idx = []
    for channel in channels:
        if channel in ch_names:
            channel_idx.append(ch_names.index(channel))
        else:
            raise ValueError(
                f"Channel {channel} not found in metadata channels: {ch_names}."
            )
    return channel_idx
