import mne
import numpy as np
import pytest

from eegdash.features.base_utils import BivariateIterator
from eegdash.features.feature_bank.pick import (
    pick_channel_pairs_preprocessor,
    pick_channels_preprocessor,
)


def _metadata(ch_names=("C3", "C4", "Pz"), directed=False):
    info = mne.create_info(
        list(ch_names), sfreq=128.0, ch_types=["eeg"] * len(ch_names)
    )
    return {
        "info": info,
        "ch_pair_iterator": BivariateIterator(len(ch_names), directed=directed),
    }


@pytest.mark.parametrize(
    "index,axis,expected_shape",
    [
        (-1, -2, (2, 2, 5)),
        (0, 1, (2, 2, 5)),
        ([0, -1], -2, (2, 2, 5)),
    ],
)
def test_pick_channels_preprocessor_selects_requested_channels(
    index, axis, expected_shape
):
    md = _metadata()
    x0 = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    x1 = np.arange(2 * 3 * 5).reshape(2, 3, 5) + 1000

    y0, y1, out_md = pick_channels_preprocessor(
        x0, x1, channels=["C3", "Pz"], _metadata=md, index=index, axis=axis
    )

    if index in (0, [0, -1]):
        assert y0.shape == expected_shape
    if index in (-1, [0, -1]):
        assert y1.shape == expected_shape
    assert out_md["info"]["ch_names"] == ["C3", "Pz"]


def test_pick_channels_preprocessor_raises_for_unknown_channel():
    md = _metadata()
    x = np.zeros((1, 3, 10))
    with pytest.raises(ValueError, match="Channel O1 not found"):
        pick_channels_preprocessor(x, channels=["O1"], _metadata=md)


def test_pick_channel_pairs_preprocessor_requires_at_least_one_target_index():
    md = _metadata()
    x = np.zeros((1, 3, 10))
    with pytest.raises(AssertionError):
        pick_channel_pairs_preprocessor(
            x,
            pairs=[("C3", "C4")],
            _metadata=md,
            index=None,
            c_index=None,
            x_index=None,
            y_index=None,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": 0},
        {"index": None, "c_index": -1},
        {"index": None, "x_index": -1, "y_index": -1},
        {"index": 0, "c_index": -1, "x_index": -1, "y_index": -1},
    ],
    ids=["pair-axis-only", "channel-union-only", "x-and-y-only", "all-targets"],
)
def test_pick_channel_pairs_preprocessor_updates_outputs_and_metadata(kwargs):
    md = _metadata()
    # 3 channels -> 3 undirected pairs
    pair_tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    channel_tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4) + 100

    out_pair, out_ch, out_md = pick_channel_pairs_preprocessor(
        pair_tensor,
        channel_tensor,
        pairs=[("C3", "C4"), ("C4", "Pz")],
        _metadata=md,
        axis=-2,
        c_axis=-2,
        **kwargs,
    )

    # Pair selection should reduce pair axis to selected pairs where index targets are used.
    if kwargs.get("index") is not None:
        assert out_pair.shape[1] == 2
    # Channel selection paths can reduce channels to unique channels from selected pairs.
    if any(k in kwargs for k in ("c_index", "x_index", "y_index")):
        assert out_ch.shape[1] <= 3
        assert out_ch.shape[1] >= 1

    assert len(out_md["ch_pair_iterator"].pairs) == 2
