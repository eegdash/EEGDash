# eegdash.hbn package

## Submodules

* [eegdash.hbn.preprocessing module](eegdash.hbn.preprocessing.md)
* [eegdash.hbn.windows module](eegdash.hbn.windows.md)

## Module contents

Healthy Brain Network (HBN) specific utilities and preprocessing.

This module provides specialized functions for working with the Healthy Brain Network
dataset, including preprocessing pipelines, annotation handling, and windowing utilities
tailored for HBN EEG data analysis.

<!-- !! processed by numpydoc !! -->

### eegdash.hbn.add_aux_anchors(raw: Raw, stim_desc: str = 'stimulus_anchor', resp_desc: str = 'response_anchor') → Raw

Add auxiliary annotations for stimulus and response onsets.

This function inspects existing “contrast_trial_start” annotations and
adds new, zero-duration “anchor” annotations at the precise onsets of
stimuli and responses for each trial.

* **Parameters:**
  * **raw** (*mne.io.Raw*) – The raw data object with “contrast_trial_start” annotations.
  * **stim_desc** (*str* *,* *default "stimulus_anchor"*) – The description for the new stimulus annotations.
  * **resp_desc** (*str* *,* *default "response_anchor"*) – The description for the new response annotations.
* **Returns:**
  The raw object with the auxiliary annotations added.
* **Return type:**
  mne.io.Raw

<!-- !! processed by numpydoc !! -->

### eegdash.hbn.add_extras_columns(windows_concat_ds: BaseConcatDataset, original_concat_ds: BaseConcatDataset, desc: str = 'contrast_trial_start', keys: tuple = ('target', 'rt_from_stimulus', 'rt_from_trialstart', 'stimulus_onset', 'response_onset', 'correct', 'response_type')) → BaseConcatDataset

Add columns from annotation extras to a windowed dataset’s metadata.

This function propagates trial-level information stored in the extras
of annotations to the metadata DataFrame of a WindowsDataset.

* **Parameters:**
  * **windows_concat_ds** (*BaseConcatDataset*) – The windowed dataset whose metadata will be updated.
  * **original_concat_ds** (*BaseConcatDataset*) – The original (non-windowed) dataset containing the raw data and
    annotations with the extras to be added.
  * **desc** (*str* *,* *default "contrast_trial_start"*) – The description of the annotations to source the extras from.
  * **keys** (*tuple* *,* *default* *(* *...* *)*) – The keys to extract from each annotation’s extras dictionary and
    add as columns to the metadata.
* **Returns:**
  The windows_concat_ds with updated metadata.
* **Return type:**
  BaseConcatDataset

<!-- !! processed by numpydoc !! -->

### eegdash.hbn.annotate_trials_with_target(raw: Raw, target_field: str = 'rt_from_stimulus', epoch_length: float = 2.0, require_stimulus: bool = True, require_response: bool = True) → Raw

Create trial annotations with a specified target value.

This function reads the BIDS events file associated with the raw object,
builds a trial table, and creates new MNE annotations for each trial.
The annotations are labeled “contrast_trial_start” and their extras
dictionary is populated with trial metrics, including a “target” key.

* **Parameters:**
  * **raw** (*mne.io.Raw*) – The raw data object. Must have a single associated file name from
    which the BIDS path can be derived.
  * **target_field** (*str* *,* *default "rt_from_stimulus"*) – The column from the trial table to use as the “target” value in the
    annotation extras.
  * **epoch_length** (*float* *,* *default 2.0*) – The duration to set for each new annotation.
  * **require_stimulus** (*bool* *,* *default True*) – If True, only include trials that have a recorded stimulus event.
  * **require_response** (*bool* *,* *default True*) – If True, only include trials that have a recorded response event.
* **Returns:**
  The raw object with the new annotations set.
* **Return type:**
  mne.io.Raw
* **Raises:**
  **KeyError** – If target_field is not a valid column in the built trial table.

<!-- !! processed by numpydoc !! -->

### eegdash.hbn.build_trial_table(events_df: DataFrame) → DataFrame

Build a table of contrast trials from an events DataFrame.

This function processes a DataFrame of events (typically from a BIDS
events.tsv file) to identify contrast trials and extract relevant
metrics like stimulus onset, response onset, and reaction times.

* **Parameters:**
  **events_df** (*pandas.DataFrame*) – A DataFrame containing event information, with at least “onset” and
  “value” columns.
* **Returns:**
  A DataFrame where each row represents a single contrast trial, with
  columns for onsets, reaction times, and response correctness.
* **Return type:**
  pandas.DataFrame

<!-- !! processed by numpydoc !! -->

### *class* eegdash.hbn.hbn_ec_ec_reannotation

Bases: `Preprocessor`

Preprocessor to reannotate HBN data for eyes-open/eyes-closed events.

This preprocessor is specifically designed for Healthy Brain Network (HBN)
datasets. It identifies existing annotations for “instructed_toCloseEyes”
and “instructed_toOpenEyes” and creates new, regularly spaced annotations
for “eyes_closed” and “eyes_open” segments, respectively.

This is useful for creating windowed datasets based on these new, more
precise event markers.

### Notes

This class inherits from `braindecode.preprocessing.Preprocessor`
and is intended to be used within a braindecode preprocessing pipeline.

<!-- !! processed by numpydoc !! -->

#### transform(raw: Raw) → Raw

Create new annotations for eyes-open and eyes-closed periods.

This function finds the original “instructed_to…” annotations and
generates new annotations every 2 seconds within specific time ranges
relative to the original markers:
- “eyes_closed”: 15s to 29s after “instructed_toCloseEyes”
- “eyes_open”: 5s to 19s after “instructed_toOpenEyes”

The original annotations in the mne.io.Raw object are replaced by
this new set of annotations.

* **Parameters:**
  **raw** (*mne.io.Raw*) – The raw MNE object containing the HBN data and original annotations.
* **Returns:**
  The raw MNE object with the modified annotations.
* **Return type:**
  mne.io.Raw

<!-- !! processed by numpydoc !! -->

### eegdash.hbn.keep_only_recordings_with(desc: str, concat_ds: BaseConcatDataset) → BaseConcatDataset

Filter a concatenated dataset to keep only recordings with a specific annotation.

* **Parameters:**
  * **desc** (*str*) – The description of the annotation that must be present in a recording
    for it to be kept.
  * **concat_ds** (*BaseConcatDataset*) – The concatenated dataset to filter.
* **Returns:**
  A new concatenated dataset containing only the filtered recordings.
* **Return type:**
  BaseConcatDataset

<!-- !! processed by numpydoc !! -->
