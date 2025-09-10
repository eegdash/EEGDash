"""
This module provides preprocessing functions for the HBN (Healthy Brain Network)
dataset.

It includes a preprocessor for re-annotating the raw data to create new events
for eyes open and eyes closed, which is useful for cleaning and segmenting the
data.
"""
import logging

import mne
import numpy as np

from braindecode.preprocessing import Preprocessor

logger = logging.getLogger("eegdash")


class hbn_ec_ec_reannotation(Preprocessor):
    """Preprocessor to re-annotate HBN data for eyes open/closed events.

    This preprocessor is specifically designed for HBN datasets to create new
    annotations for "eyes_open" and "eyes_closed" events based on the original
    "instructed_toOpenEyes" and "instructed_toCloseEyes" events.

    This is useful for creating epochs of clean, artifact-free data for further
    analysis.
    """

    def __init__(self):
        super().__init__(fn=self.transform, apply_on_array=False)

    def transform(self, raw):
        """Re-annotate the raw data for eyes open and eyes closed events.

        This function modifies the raw MNE object by creating new events based on
        the existing annotations for "instructed_toCloseEyes" and "instructed_toOpenEyes".
        It generates new events every 2 seconds within specified time ranges after
        the original events, and replaces the existing annotations with these new events.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw MNE object containing EEG data and annotations.

        Returns
        -------
        mne.io.Raw
            The modified raw MNE object with the new annotations.
        """
        events, event_id = mne.events_from_annotations(raw)

        logger.info("Original events found with ids: %s", event_id)

        # Create new events array for 2-second segments
        new_events = []
        sfreq = raw.info["sfreq"]
        for event in events[events[:, 2] == event_id["instructed_toCloseEyes"]]:
            # For each original event, create events every 2 seconds from 15s to 29s after
            start_times = event[0] + np.arange(15, 29, 2) * sfreq
            new_events.extend([[int(t), 0, 1] for t in start_times])

        for event in events[events[:, 2] == event_id["instructed_toOpenEyes"]]:
            # For each original event, create events every 2 seconds from 5s to 19s after
            start_times = event[0] + np.arange(5, 19, 2) * sfreq
            new_events.extend([[int(t), 0, 2] for t in start_times])

        # replace events in raw
        new_events = np.array(new_events)

        annot_from_events = mne.annotations_from_events(
            events=new_events,
            event_desc={1: "eyes_closed", 2: "eyes_open"},
            sfreq=raw.info["sfreq"],
        )

        raw.set_annotations(annot_from_events)

        return raw
