"""Recordings whose BIDS sidecar says ``RecordingType: "epoched"``.

``mne_bids.read_raw_bids`` refuses them and asks for
``mne_bids.read_epochs_bids``, which returns :class:`mne.Epochs` with the
sidecars applied (``channels.tsv`` types, units and bads, electrodes,
``participants.tsv``). :class:`~eegdash.dataset.base.EEGDashRaw` serves a
:class:`mne.io.Raw`, so :func:`epochs_to_raw` lays the epochs end to end in a
:class:`mne.io.RawArray` that keeps that ``info`` and marks each epoch and each
join with annotations.
"""

from __future__ import annotations

import mne
import numpy as np

# The RuntimeError mne-bids (>= 0.20) raises from read_raw_bids.
EPOCHED_ERROR = 'RecordingType is "epoched"'

# MNE's markers for a discontinuity (as written by mne.concatenate_raws).
_BOUNDARY = ("BAD boundary", "EDGE boundary")


def epochs_to_raw(epochs: mne.BaseEpochs) -> mne.io.RawArray:
    """Concatenate ``epochs`` into one continuous :class:`mne.io.RawArray`.

    Epoch ``i`` occupies samples ``[i * n, (i + 1) * n)`` of the result, with
    ``n = len(epochs.times)``. ``epochs.info`` is kept as is (channel types,
    units, bad channels, digitization, measurement date). Annotations:

    - one per epoch, named after its event (``epochs.event_id``), starting at the
      event (``-tmin`` into the epoch, clipped to the epoch) and lasting until
      the end of the epoch;
    - ``"BAD boundary"`` and ``"EDGE boundary"`` (zero duration) at each join
      between two epochs, as :func:`mne.concatenate_raws` marks them, since the
      signal is not continuous there.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        The epochs, e.g. from :func:`mne_bids.read_epochs_bids`.

    Returns
    -------
    mne.io.RawArray
        The epochs as one continuous recording.

    Raises
    ------
    ValueError
        If ``epochs`` holds no epoch.

    """
    data = epochs.get_data(copy=False)
    n_epochs, n_channels, n_per = data.shape
    if n_epochs == 0:
        raise ValueError("no epoch to concatenate (all dropped or none in the file)")
    # (epochs, channels, times) -> (channels, epochs * times), epoch after epoch.
    continuous = np.ascontiguousarray(data.transpose(1, 0, 2)).reshape(
        n_channels, n_epochs * n_per
    )
    del data
    raw = mne.io.RawArray(continuous, epochs.info.copy(), verbose="ERROR")

    sfreq = raw.info["sfreq"]
    length = n_per / sfreq
    starts = np.arange(n_epochs) * length
    offset = float(np.clip(-epochs.tmin, 0.0, length))
    names = {code: name for name, code in epochs.event_id.items()}
    onset = list(starts + offset)
    duration = [length - offset] * n_epochs
    description = [names.get(int(code), str(int(code))) for code in epochs.events[:, 2]]
    for start in starts[1:]:
        for marker in _BOUNDARY:
            onset.append(start)
            duration.append(0.0)
            description.append(marker)
    raw.set_annotations(
        mne.Annotations(onset, duration, description, orig_time=raw.info["meas_date"])
    )
    return raw


__all__ = ["EPOCHED_ERROR", "epochs_to_raw"]
