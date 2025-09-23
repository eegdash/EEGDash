# Authors: The EEGDash contributors.
# License: GNU General Public License
# Copyright the EEGDash contributors.

"""Preprocessing utilities specific to the Healthy Brain Network dataset.

This module contains preprocessing classes and functions designed specifically for
HBN EEG data, including specialized annotation handling for eyes-open/eyes-closed
paradigms and other HBN-specific preprocessing steps.
"""

import mne
import numpy as np

from braindecode.preprocessing import Preprocessor

from ..logging import logger


class hbn_ec_ec_reannotation(Preprocessor):
    """Preprocessor to reannotate the raw data for eyes open and eyes closed events.

    This processor is designed for HBN datasets.

    """

    def __init__(self):
        super().__init__(fn=self.transform, apply_on_array=False)

    def transform(self, raw):
        """Reannotate the raw data to create new events for eyes open and eyes closed

        This function modifies the raw MNE object by creating new events based on
        the existing annotations for "instructed_toCloseEyes" and "instructed_toOpenEyes".
        It generates new events every 2 seconds within specified time ranges after
        the original events, and replaces the existing annotations with these new events.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw MNE object containing EEG data and annotations.

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
