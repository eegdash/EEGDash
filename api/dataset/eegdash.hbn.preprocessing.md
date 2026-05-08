# eegdash.hbn.preprocessing module

Preprocessing utilities specific to the Healthy Brain Network dataset.

This module contains preprocessing classes and functions designed specifically for
HBN EEG data, including specialized annotation handling for eyes-open/eyes-closed
paradigms and other HBN-specific preprocessing steps.

<!-- !! processed by numpydoc !! -->

### *class* eegdash.hbn.preprocessing.hbn_ec_ec_reannotation

Bases: [`Preprocessor`](https://braindecode.org/stable/generated/braindecode.preprocessing.Preprocessor.html#braindecode.preprocessing.Preprocessor)

Preprocessor to reannotate HBN data for eyes-open/eyes-closed events.

This preprocessor is specifically designed for Healthy Brain Network (HBN)
datasets. It identifies existing annotations for “instructed_toCloseEyes”
and “instructed_toOpenEyes” and creates new, regularly spaced annotations
for “eyes_closed” and “eyes_open” segments, respectively.

This is useful for creating windowed datasets based on these new, more
precise event markers.

### Notes

This class inherits from [`braindecode.preprocessing.Preprocessor`](https://braindecode.org/stable/generated/braindecode.preprocessing.Preprocessor.html#braindecode.preprocessing.Preprocessor)
and is intended to be used within a braindecode preprocessing pipeline.

<!-- !! processed by numpydoc !! -->

#### transform(raw: [Raw](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)) → [Raw](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)

Create new annotations for eyes-open and eyes-closed periods.

This function finds the original “instructed_to…” annotations and
generates new annotations every 2 seconds within specific time ranges
relative to the original markers:
- “eyes_closed”: 15s to 29s after “instructed_toCloseEyes”
- “eyes_open”: 5s to 19s after “instructed_toOpenEyes”

The original annotations in the mne.io.Raw object are replaced by
this new set of annotations.

* **Parameters:**
  **raw** ([*mne.io.Raw*](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)) – The raw MNE object containing the HBN data and original annotations.
* **Returns:**
  The raw MNE object with the modified annotations.
* **Return type:**
  [mne.io.Raw](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw)

<!-- !! processed by numpydoc !! -->
