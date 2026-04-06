r"""Channel-picking feature preprocessors
=====================================

This module provides the ability to pick specific channels or channel
pairs for further processing.

Data Shape Convention
---------------------
By default, this module follows a **Channel-panultimate** convention:

* **Input:** ``(..., channel, :)``
* **Output:** same as input

The choice of the channel dimension can be adjusted using the ``axis``
parameter.
"""

from typing import Iterable, Tuple

import mne

from ..base_utils import BivariateIterator, channel_names_to_indices
from ..decorators import (
    FeaturePredecessor,
    metadata_preprocessor,
)
from ..output_types import AsInputOutputType

__all__ = [
    "pick_channel_pairs_preprocessor",
    "pick_channels_preprocessor",
]


@FeaturePredecessor(AsInputOutputType)
@metadata_preprocessor
def pick_channels_preprocessor(
    *x,
    channels: Iterable[str],
    _metadata: dict,
    index: int | Iterable[int] = -1,
    axis: int = -2,
):
    r"""Pick a subset of channels for further processing steps.

    Parameters
    ----------
    *x : tuple[ndarray]
        Input batch.
    channels : Iterable[str]
        A list of channels to pick.
    index: int | Iterable[int]
        The index (or indices) of the input ndarray[s] to pick channels from.
        Default is -1.
    axis : int
        The channels axis of the input batch. Default is -2.

    Returns
    -------
    *ndarray
        Sliced input batch containing only the picked channels.
    _metadata : dict
        Updated metadata dictionary.

    """
    if isinstance(index, int):
        index = [index]
    pick_idx = channel_names_to_indices(channels, _metadata["info"]["ch_names"])
    y = list(x)
    for i in index:
        y[i] = x[i].take(pick_idx, axis=axis)
    _metadata["info"] = mne.pick_info(_metadata["info"], pick_idx, copy=True)
    return *y, _metadata


@FeaturePredecessor(AsInputOutputType)
@metadata_preprocessor
def pick_channel_pairs_preprocessor(
    *x,
    pairs: Iterable[Tuple[str, str]],
    _metadata: dict,
    index: int | Iterable[int] | None = -1,
    c_index: int | Iterable[int] | None = None,
    x_index: int | Iterable[int] | None = None,
    y_index: int | Iterable[int] | None = None,
    axis: int = -2,
    c_axis: int = -2,
):
    r"""Pick a subset of channel pairs for further processing steps.

    Must follow a preprocessor decorated with ``channel_pairer`` (or
    ``channel_directed_pairer``).

    Parameters
    ----------
    *x : tuple[ndarray]
        Input batch.
    channels : Iterable[str]
        A list of channels to pick.
    index: int | Iterable[int]
        The index (or indices) of the input ndarray[s] to pick channel pairs
        from. Default is -1.
    c_index: int | Iterable[int]
        The index (or indices) of the input ndarray[s] to pick channels from.
        Default is [].
    x_index: int | Iterable[int]
        The index (or indices) of the input ndarray[s] to pick pair-first
        channels from. Default is [].
    y_index: int | Iterable[int]
        The index (or indices) of the input ndarray[s] to pick pair-second
        channels from. Default is [].
    axis : int
        The channel pairs axis of the input batch at index ``index``. Default
        is -2.
    c_axis : int
        The channels axis of the input batch at index ``c_index`` or
        ``x_index`` or ``y_index``. Default is -2.

    Returns
    -------
    *ndarray
        Sliced input batch containing only the picked channels.
    _metadata : dict
        Updated metadata dictionary.

    Note
    ----
    Picking by index pair, e.g., ``x[i, j]``, is not directly supported because
    the result may not be an `numpy.ndarray`. It is preferred to use a pair
    axis. It is possible, however, to pick just by ``x_index`` with
    ``c_axis=0``, then pick again just by ``y_index`` with ``c_index=1`` (or
    vice versa) to effectively pick the indices intersection of such an
    `numpy.ndarray`.

    """
    assert index or c_index or x_index or y_index
    if index is None:
        index = []
    elif isinstance(index, int):
        index = [index]
    if c_index is None:
        c_index = []
    elif isinstance(c_index, int):
        c_index = [c_index]
    if x_index is None:
        x_index = []
    elif isinstance(x_index, int):
        x_index = [x_index]
    if y_index is None:
        y_index = []
    elif isinstance(y_index, int):
        y_index = [y_index]
    xy_index = set(x_index).intersection(set(y_index))
    c_index = list(set(c_index).union(xy_index))
    pick_pairs = list(
        zip(
            *[
                channel_names_to_indices(p, _metadata["info"]["ch_names"])
                for p in zip(*pairs)
            ]
        )
    )
    assert not set(pick_pairs) - set(_metadata["ch_pair_iterator"].pairs)
    pick_idx = [_metadata["ch_pair_iterator"].pairs.index(p) for p in pick_pairs]
    x_it, y_it = _metadata["ch_pair_iterator"].get_pair_iterators()
    y = list(x)
    for i in index:
        y[i] = x[i].take(pick_idx, axis=axis)
    for i in c_index:
        y[i] = x[i].take(list(set(x_it[pick_idx] + y_it[pick_idx])), axis=c_axis)
    for i in x_index:
        y[i] = x[i].take(list(set(x_it[pick_idx])), axis=c_axis)
    for i in y_index:
        y[i] = x[i].take(list(set(y_it[pick_idx])), axis=c_axis)
    _metadata["ch_pair_iterator"] = BivariateIterator(pick_pairs)
    return *y, _metadata
