r"""Channel-picking feature preprocessors
=====================================

This module provides the ability to pick specific channels for further
processing.

Data Shape Convention
---------------------
By default, this module follows a **panultimate-Channel** convention:

* **Input:** ``(..., channel, :)``
* **Output:** same as input

The choice of the channel dimension can be adjusted using the ``axis``
parameter.
"""

from typing import Iterable, List

import mne

from ..decorators import FeaturePredecessor, metadata_perprocessor
from ..extractors import AsInputOutputType

__all__ = [
    "pick_channels_preprocessor",
]


def _channel_names_to_indices(channels: List[str], ch_names: List[str]) -> List[int]:
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


@FeaturePredecessor(AsInputOutputType)
@metadata_perprocessor
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
        The index of the input ndarray[s] to pick channels from. Default is -1.
    axis : int
        The channels axis of the input batch. Default is -2.

    Returns
    -------
    ndarray
        Sliced input batch containing only the picked channels.
    _metadata : dict
        Updated metadata dictionary.

    """
    if isinstance(index, int):
        index = [index]
    pick_idx = _channel_names_to_indices(channels, _metadata["info"]["ch_names"])
    y = list(x)
    for i in index:
        y[i] = x[i].swapaxes(0, axis)[pick_idx].swapaxes(0, axis)
    _metadata["info"] = mne.pick_info(_metadata["info"], pick_idx, copy=True)
    return *y, _metadata
