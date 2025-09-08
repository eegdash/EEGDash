import logging

import mne
import numpy as np
import pandas as pd
from mne_bids import get_bids_path_from_fname

from braindecode.datasets.base import BaseConcatDataset

logger = logging.getLogger("eegdash")


def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast trial with stimulus/response metrics."""
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[
        events_df["value"].isin(["left_buttonPress", "right_buttonPress"])
    ].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
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
            feedback = resp_block.iloc[0]["feedback"]

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


# Aux functions to inject the annot
def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)


def _to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def annotate_trials_with_target(
    raw,
    target_field="rt_from_stimulus",
    epoch_length=2.0,
    require_stimulus=True,
    require_response=True,
):
    """Create 'contrast_trial_start' annotations with float target in extras."""
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
        orig_time=raw.info["meas_date"],
        extras=extras,
    )
    raw.set_annotations(new_ann, verbose=False)
    return raw


def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann = raw.annotations
    mask = ann.description == "contrast_trial_start"
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex["stimulus_onset"]
        resp_t = ex["response_onset"]

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex["rt_from_trialstart"]
            rts = ex["rt_from_stimulus"]
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex["rt_from_trialstart"]
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if (stim_t is not None) and not (
            isinstance(stim_t, float) and np.isnan(stim_t)
        ):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if (resp_t is not None) and not (
            isinstance(resp_t, float) and np.isnan(resp_t)
        ):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc] * len(stim_onsets) + [resp_desc] * len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw


def add_extras_columns(
    windows_concat_ds,
    original_concat_ds,
    desc="contrast_trial_start",
    keys=(
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
):
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

            # Replace the whole column to avoid dtype conflicts
            md[k] = ser

        win_ds.metadata = md.reset_index(drop=True)
        if hasattr(win_ds, "y"):
            y_np = win_ds.metadata["target"].astype(float).to_numpy()
            win_ds.y = y_np[:, None]  # (N, 1)

    return windows_concat_ds


def keep_only_recordings_with(desc, concat_ds):
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            logging.warning(
                f"Recording {ds.raw.filenames[0]} does not contain event '{desc}'"
            )
    return BaseConcatDataset(kept)
