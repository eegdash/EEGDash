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
    """Preprocessor to reannotate HBN data for eyes-open/eyes-closed events.

    This preprocessor is specifically designed for Healthy Brain Network (HBN)
    datasets. It identifies existing annotations for "instructed_toCloseEyes"
    and "instructed_toOpenEyes" and creates new, regularly spaced annotations
    for "eyes_closed" and "eyes_open" segments, respectively.

    This is useful for creating windowed datasets based on these new, more
    precise event markers.

    Notes
    -----
    This class inherits from :class:`braindecode.preprocessing.Preprocessor`
    and is intended to be used within a braindecode preprocessing pipeline.

    """

    def __init__(self):
        super().__init__(fn=self.transform, apply_on_array=False)

    def transform(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Create new annotations for eyes-open and eyes-closed periods.

        This function finds the original "instructed_to..." annotations and
        generates new annotations every 2 seconds within specific time ranges
        relative to the original markers:
        - "eyes_closed": 15s to 29s after "instructed_toCloseEyes"
        - "eyes_open": 5s to 19s after "instructed_toOpenEyes"

        The original annotations in the `mne.io.Raw` object are replaced by
        this new set of annotations.

        Parameters
        ----------
        raw : mne.io.Raw
            The raw MNE object containing the HBN data and original annotations.

        Returns
        -------
        mne.io.Raw
            The raw MNE object with the modified annotations.

        """
        events, event_id = mne.events_from_annotations(raw)

        logger.info("Original events found with ids: %s", event_id)

        # Create new events array for 2-second segments
        new_events = []
        sfreq = raw.info["sfreq"]

        close_event_id = event_id.get("instructed_toCloseEyes")
        if close_event_id:
            for event in events[events[:, 2] == close_event_id]:
                # For each original event, create events every 2s from 15s to 29s after
                start_times = event[0] + np.arange(15, 29, 2) * sfreq
                new_events.extend([[int(t), 0, 1] for t in start_times])

        open_event_id = event_id.get("instructed_toOpenEyes")
        if open_event_id:
            for event in events[events[:, 2] == open_event_id]:
                # For each original event, create events every 2s from 5s to 19s after
                start_times = event[0] + np.arange(5, 19, 2) * sfreq
                new_events.extend([[int(t), 0, 2] for t in start_times])

        if not new_events:
            logger.warning(
                "Could not find 'instructed_toCloseEyes' or 'instructed_toOpenEyes' "
                "annotations. No new events created."
            )
            return raw

        # replace events in raw
        new_events = np.array(new_events)

        annot_from_events = mne.annotations_from_events(
            events=new_events,
            event_desc={1: "eyes_closed", 2: "eyes_open"},
            sfreq=raw.info["sfreq"],
            orig_time=raw.info.get("meas_date"),
        )

        raw.set_annotations(annot_from_events)

        return raw
