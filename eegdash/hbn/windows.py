# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Windowing and trial processing utilities for HBN datasets.

This module provides functions for building trial tables, adding auxiliary anchors,
annotating trials with targets, and filtering recordings based on various criteria.
These utilities are specifically designed for working with HBN EEG data structures
and experimental paradigms.
"""

import logging

import mne
import numpy as np
import pandas as pd
from mne_bids import get_bids_path_from_fname

from braindecode.datasets.base import BaseConcatDataset


def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build a table of contrast trials from an events DataFrame.

    This function processes a DataFrame of events (typically from a BIDS
    `events.tsv` file) to identify contrast trials and extract relevant
    metrics like stimulus onset, response onset, and reaction times.

    Parameters
    ----------
    events_df : pandas.DataFrame
        A DataFrame containing event information, with at least "onset" and
        "value" columns.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a single contrast trial, with
        columns for onsets, reaction times, and response correctness.

    """
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[
        events_df["value"].isin(["left_buttonPress", "right_buttonPress"])
    ].copy()
    end_events = events_df[events_df["value"].eq("end_experiment")].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
    if len(trials) and not end_events.empty:
        last_end = float(end_events["onset"].iloc[-1])
        last_idx = trials.index[-1]
        if pd.isna(trials.loc[last_idx, "next_onset"]) and last_end > float(
            trials.loc[last_idx, "onset"]
        ):
            trials.loc[last_idx, "next_onset"] = last_end
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"])
        end = float(tr["next_onset"])

        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        if not np.isnan(stim_onset):
            resp_block = responses[
                (responses["onset"] >= stim_onset) & (responses["onset"] < end)
            ]
        else:
            resp_block = responses[
                (responses["onset"] >= start) & (responses["onset"] < end)
            ]

        if resp_block.empty:
            resp_onset = np.nan
            resp_type = None
            feedback = None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type = resp_block.iloc[0]["value"]
            feedback = resp_block.iloc[0].get("feedback")

        rt_from_stim = (
            (resp_onset - stim_onset)
            if (not np.isnan(stim_onset) and not np.isnan(resp_onset))
            else np.nan
        )
        rt_from_trial = (resp_onset - start) if not np.isnan(resp_onset) else np.nan

        correct = None
        if isinstance(feedback, str):
            if feedback == "smiley_face":
                correct = True
            elif feedback == "sad_face":
                correct = False

        rows.append(
            {
                "trial_start_onset": start,
                "trial_stop_onset": end,
                "stimulus_onset": stim_onset,
                "response_onset": resp_onset,
                "rt_from_stimulus": rt_from_stim,
                "rt_from_trialstart": rt_from_trial,
                "response_type": resp_type,
                "correct": correct,
            }
        )

    return pd.DataFrame(rows)


def _to_float_or_none(x):
    """Safely convert a value to float or None."""
    return None if pd.isna(x) else float(x)


def _to_int_or_none(x):
    """Safely convert a value to int or None."""
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def _to_str_or_none(x):
    """Safely convert a value to string or None."""
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def annotate_trials_with_target(
    raw: mne.io.Raw,
    target_field: str = "rt_from_stimulus",
    epoch_length: float = 2.0,
    require_stimulus: bool = True,
    require_response: bool = True,
) -> mne.io.Raw:
    """Create trial annotations with a specified target value.

    This function reads the BIDS events file associated with the `raw` object,
    builds a trial table, and creates new MNE annotations for each trial.
    The annotations are labeled "contrast_trial_start" and their `extras`
    dictionary is populated with trial metrics, including a "target" key.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object. Must have a single associated file name from
        which the BIDS path can be derived.
    target_field : str, default "rt_from_stimulus"
        The column from the trial table to use as the "target" value in the
        annotation extras.
    epoch_length : float, default 2.0
        The duration to set for each new annotation.
    require_stimulus : bool, default True
        If True, only include trials that have a recorded stimulus event.
    require_response : bool, default True
        If True, only include trials that have a recorded response event.

    Returns
    -------
    mne.io.Raw
        The `raw` object with the new annotations set.

    Raises
    ------
    KeyError
        If `target_field` is not a valid column in the built trial table.

    """
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath

    events_df = (
        pd.read_csv(events_file, sep="\t")
        .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
        .sort_values("onset", kind="mergesort")
        .reset_index(drop=True)
    )

    trials = build_trial_table(events_df)

    if require_stimulus:
        trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:
        trials = trials[trials["response_onset"].notna()].copy()

    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)

    onsets = trials["trial_start_onset"].to_numpy(float)
    durations = np.full(len(trials), float(epoch_length), dtype=float)
    descs = ["contrast_trial_start"] * len(trials)

    extras = []
    for i, v in enumerate(targets):
        row = trials.iloc[i]
        extras.append(
            {
                "target": _to_float_or_none(v),
                "rt_from_stimulus": _to_float_or_none(row["rt_from_stimulus"]),
                "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
                "stimulus_onset": _to_float_or_none(row["stimulus_onset"]),
                "response_onset": _to_float_or_none(row["response_onset"]),
                "correct": _to_int_or_none(row["correct"]),
                "response_type": _to_str_or_none(row["response_type"]),
            }
        )

    new_ann = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descs,
        orig_time=raw.info.get("meas_date"),
        extras=extras,
    )
    raw.set_annotations(new_ann, verbose=False)
    return raw


def add_aux_anchors(
    raw: mne.io.Raw,
    stim_desc: str = "stimulus_anchor",
    resp_desc: str = "response_anchor",
) -> mne.io.Raw:
    """Add auxiliary annotations for stimulus and response onsets.

    This function inspects existing "contrast_trial_start" annotations and
    adds new, zero-duration "anchor" annotations at the precise onsets of
    stimuli and responses for each trial.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object with "contrast_trial_start" annotations.
    stim_desc : str, default "stimulus_anchor"
        The description for the new stimulus annotations.
    resp_desc : str, default "response_anchor"
        The description for the new response annotations.

    Returns
    -------
    mne.io.Raw
        The `raw` object with the auxiliary annotations added.

    """
    ann = raw.annotations
    mask = ann.description == "contrast_trial_start"
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex.get("stimulus_onset")
        resp_t = ex.get("response_onset")

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex.get("rt_from_trialstart")
            rts = ex.get("rt_from_stimulus")
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex.get("rt_from_trialstart")
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if stim_t is not None and not (isinstance(stim_t, float) and np.isnan(stim_t)):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if resp_t is not None and not (isinstance(resp_t, float) and np.isnan(resp_t)):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc] * len(stim_onsets) + [resp_desc] * len(resp_onsets),
            orig_time=raw.info.get("meas_date"),
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw


def add_extras_columns(
    windows_concat_ds: BaseConcatDataset,
    original_concat_ds: BaseConcatDataset,
    desc: str = "contrast_trial_start",
    keys: tuple = (
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
) -> BaseConcatDataset:
    """Add columns from annotation extras to a windowed dataset's metadata.

    This function propagates trial-level information stored in the `extras`
    of annotations to the `metadata` DataFrame of a `WindowsDataset`.

    Parameters
    ----------
    windows_concat_ds : BaseConcatDataset
        The windowed dataset whose metadata will be updated.
    original_concat_ds : BaseConcatDataset
        The original (non-windowed) dataset containing the raw data and
        annotations with the `extras` to be added.
    desc : str, default "contrast_trial_start"
        The description of the annotations to source the extras from.
    keys : tuple, default (...)
        The keys to extract from each annotation's `extras` dictionary and
        add as columns to the metadata.

    Returns
    -------
    BaseConcatDataset
        The `windows_concat_ds` with updated metadata.

    """
    float_cols = {
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
    }

    for win_ds, base_ds in zip(windows_concat_ds.datasets, original_concat_ds.datasets):
        ann = base_ds.raw.annotations
        idx = np.where(ann.description == desc)[0]
        if idx.size == 0:
            continue

        per_trial = [
            {
                k: (
                    ann.extras[i][k]
                    if ann.extras is not None and k in ann.extras[i]
                    else None
                )
                for k in keys
            }
            for i in idx
        ]

        md = win_ds.metadata.copy()
        first = md["i_window_in_trial"].to_numpy() == 0
        trial_ids = first.cumsum() - 1
        n_trials = trial_ids.max() + 1 if len(trial_ids) else 0
        assert n_trials == len(per_trial), (
            f"Trial mismatch: {n_trials} vs {len(per_trial)}"
        )

        for k in keys:
            vals = [per_trial[t][k] if t < len(per_trial) else None for t in trial_ids]
            if k == "correct":
                ser = pd.Series(
                    [None if v is None else int(bool(v)) for v in vals],
                    index=md.index,
                    dtype="Int64",
                )
            elif k in float_cols:
                ser = pd.Series(
                    [np.nan if v is None else float(v) for v in vals],
                    index=md.index,
                    dtype="Float64",
                )
            else:  # response_type
                ser = pd.Series(vals, index=md.index, dtype="string")

            md[k] = ser

        win_ds.metadata = md.reset_index(drop=True)
        if hasattr(win_ds, "y"):
            y_np = win_ds.metadata["target"].astype(float).to_numpy()
            win_ds.y = y_np[:, None]  # (N, 1)

    return windows_concat_ds


def keep_only_recordings_with(
    desc: str, concat_ds: BaseConcatDataset
) -> BaseConcatDataset:
    """Filter a concatenated dataset to keep only recordings with a specific annotation.

    Parameters
    ----------
    desc : str
        The description of the annotation that must be present in a recording
        for it to be kept.
    concat_ds : BaseConcatDataset
        The concatenated dataset to filter.

    Returns
    -------
    BaseConcatDataset
        A new concatenated dataset containing only the filtered recordings.

    """
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            logging.warning(
                f"Recording {ds.raw.filenames[0]} does not contain event '{desc}'"
            )
    return BaseConcatDataset(kept)
