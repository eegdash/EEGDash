# Possibles approaches to define the windows could be:
#     ______________|_________________|___________________|_____________
#                stimulus.         response.       feedback.
# Op1: *************
# Op2:                    ******************* (Using + 2.0)
# Op3:                 ******************** (Using + 0.5)
# Op4:               *****************
###################

# Alternative:
# categorize the response time.

import numpy as np
import pandas as pd
import mne
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import robust_scale

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import SPoC


from mne_bids import get_bids_path_from_fname

from eegdash.dataset import EEGChallengeDataset
from skorch.helper import SliceDataset, to_numpy
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    create_windows_from_events,
    Preprocessor,
    preprocess,
)

# Repro
rng = np.random.RandomState(0)

SFREQ = 100  # Hz (windows use this)
EPOCH_LEN_S = 2.0  # (kept for trial-start annotation duration only)
cache_dir = Path("~/eegdash_data/eeg_challenge_cache_100").expanduser()
cache_dir.mkdir(parents=True, exist_ok=True)


# extract evets:
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
    raw.set_annotations(new_ann)
    return raw


# %% ------------------------------------ Add stimulus/response anchor events
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
        raw.set_annotations(ann + aux)
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


# %% ------------------------------------ Utility: keep recordings that have a given event
def keep_only_recordings_with(desc, concat_ds):
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            print(
                f"[warn] Recording {ds.raw.filenames[0]} does not contain event '{desc}'"
            )
    return BaseConcatDataset(kept)


# %% ------------------------------------ Load dataset, write annotations & anchors
dataset = EEGChallengeDataset(
    query={"task": "contrastChangeDetection"},
    release="R5",
    cache_dir=cache_dir,
)


def pick_visual(raw):
    # Only channels from the occipital and parietal regions
    picks = mne.pick_channels_regexp(raw.info, regexp=r"^(O|PO|P)(z|[0-9]+)?$")
    if len(picks) >= 8:  # keep at least a small montage
        raw.pick(picks)
    return raw


def pick_strict_eeg(raw, min_var=1e-20):
    """Keep EEG only; also drop obvious REF/TRIG-like or near-constant channels."""
    # Step A: pick EEG via mne.pick_types on this raw's info
    picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        stim=False,
        eog=False,
        ecg=False,
        emg=False,
        misc=False,
        seeg=False,
        ecog=False,
        resp=False,
    )
    raw.pick(picks)

    # Step B: drop channels by name heuristics (sometimes mislabeled as EEG)
    bad_name_tokens = ("stim", "status", "trigger", "trig", "ref", "masto")
    drop_by_name = [
        ch for ch in raw.ch_names if any(tok in ch.lower() for tok in bad_name_tokens)
    ]
    if drop_by_name:
        raw.drop_channels(drop_by_name)
    return raw


factor = 1e6
preprocessors = []
# add early in your preprocessors, before annotate/windowing
preprocessors = [
    Preprocessor(pick_strict_eeg, apply_on_array=False),
    # Preprocessor("set_eeg_reference", ref_channels='average'),
    # Preprocessor(pick_visual, apply_on_array=False),
    Preprocessor(lambda data: np.multiply(data, factor)),  # Convert from V to uV
    Preprocessor(
        annotate_trials_with_target,
        apply_on_array=False,
        target_field="rt_from_stimulus",
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
    ),
    Preprocessor(add_aux_anchors, apply_on_array=False),
    Preprocessor(robust_scale, channel_wise=True),
]
preprocess(dataset, preprocessors, n_jobs=1)

# ANCHOR = "stimulus_anchor"
ANCHOR = "stimulus_anchor"
MAX_RT = 1.5  # seconds (cap)
POST_RESP = 2  # tail after expected response

TMIN_S = 0.0
TMAX_S = MAX_RT + POST_RESP

# Keep only recordings that actually contain stimulus anchors
dataset = keep_only_recordings_with(ANCHOR, dataset)

# Create single-interval windows (stim-locked, long enough to include the response)
single_windows = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(TMIN_S * SFREQ),
    window_size_samples=int((2.0 * SFREQ)),
    window_stride_samples=int((1.0 * SFREQ)),
    trial_stop_offset_samples=int(TMAX_S * SFREQ),
    preload=True,
)

single_windows = add_extras_columns(
    single_windows,
    dataset,
    desc=ANCHOR,
    keys=(
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
)

pick_preprocessor = []

# picking eeg channels
single_windows = preprocess(single_windows, pick_preprocessor, n_jobs=-1)


# ============================ Subject-wise split ============================
def concat_from_subjects(by_subject, subjects):
    """Flatten per-subject BaseConcatDataset objects into one BaseConcatDataset."""
    parts = []
    for sid in subjects:
        ds = by_subject.get(sid)
        if ds is None:
            # skip missing ids; you could raise instead
            continue
        parts.extend(ds.datasets)  # flatten the inner datasets
    return BaseConcatDataset(parts)


def make_splits(by_subject, train_ids, valid_ids, test_ids):
    # sanity checks
    s_train, s_valid, s_test = map(set, (train_ids, valid_ids, test_ids))
    overlap = (s_train & s_valid) | (s_train & s_test) | (s_valid & s_test)
    if overlap:
        raise ValueError(f"Subject IDs appear in multiple splits: {sorted(overlap)}")

    known = set(by_subject.keys())
    missing = (s_train | s_valid | s_test) - known
    if missing:
        print(
            f"[warn] {len(missing)} subject(s) not found and will be skipped: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}"
        )

    train_set = concat_from_subjects(by_subject, train_ids)
    valid_set = concat_from_subjects(by_subject, valid_ids)
    test_set = concat_from_subjects(by_subject, test_ids)
    return train_set, valid_set, test_set


def split_by_subject(single_windows, valid_frac=0.1, test_frac=0.1, seed=42):
    subjects = single_windows.get_metadata()["subject"].unique()

    train_subj, holdout_subj = train_test_split(
        subjects, test_size=(valid_frac + test_frac), random_state=seed, shuffle=True
    )
    valid_size = valid_frac / (valid_frac + test_frac)
    valid_subj, test_subj = train_test_split(
        holdout_subj, test_size=(1.0 - valid_size), random_state=seed + 1, shuffle=True
    )
    windows_per_subject = single_windows.split("subject")

    return make_splits(windows_per_subject, train_subj, valid_subj, test_subj)


spl = split_by_subject(single_windows, valid_frac=0.1, test_frac=0.1, seed=42)
train_set, valid_set, test_set = spl

tot = len(single_windows)
print(
    {
        k: f"{len(v)}/{tot} = {len(v) / tot:.1%}"
        for k, v in {"train": train_set, "valid": valid_set, "test": test_set}.items()
    }
)

# ============================ Masks, arrays, baselines ============================


def arrays_from_set(bc_set):
    X = to_numpy(SliceDataset(bc_set, 0))
    y = to_numpy(SliceDataset(bc_set, 1))

    return X, y


X_tr, y_tr = arrays_from_set(train_set)
X_va, y_va = arrays_from_set(valid_set)
X_te, y_te = arrays_from_set(test_set)

X_tr = X_tr[:, :-1, :]
X_va = X_va[:, :-1, :]
X_te = X_te[:, :-1, :]

# Baseline = predict train mean
print(
    f"Baseline RMSE (predict train mean): {rmse(y_va, np.full_like(y_va, y_tr.mean())):.4f}"
)

# Choose a safe number of filters relative to #channels
n_channels = X_tr.shape[1]
n_times = X_tr.shape[2]
n_filters = max(1, min(6, n_channels - 1))
print(
    f"Using n_filters={n_filters} (n_channels={n_channels}) with {n_times} time points"
)
alpha = 1.0


class DemeanTime(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # center each channel per window across time
        return X - X.mean(axis=-1, keepdims=True)


class SafeCov(Covariances):
    """Covariances + tiny diagonal loading to keep SPD."""

    def __init__(self, estimator="oas", eps=1e-6):
        super().__init__(estimator=estimator)
        self.eps = eps

    def transform(self, X, y=None):
        C = super().transform(X)  # (n_trials, n_chan, n_chan)
        I = np.eye(C.shape[-1], dtype=C.dtype)
        return C + self.eps * I[None, ...]


class SafeLog(BaseEstimator, TransformerMixin):
    """Log with floor to avoid -inf."""

    def __init__(self, eps=1e-12):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, self.eps, None)  # floor strictly positive
        X = np.log(X)
        X[~np.isfinite(X)] = 0.0  # paranoia
        return X


# keep your choices
n_channels = X_tr.shape[1]
n_filters = max(1, min(6, n_channels - 1))
alpha = 1.0

pipe = make_pipeline(
    DemeanTime(),
    SafeCov(estimator="oas", eps=1e-6),  # <- stabilize covs
    SPoC(nfilter=n_filters, log=True),  # <- no internal log
    StandardScaler(),
    Ridge(alpha=alpha, random_state=0),
)

# optional sanity check before fitting
assert np.all(np.isfinite(X_tr)), "X_tr has NaN/Inf before pipeline"
pipe.fit(X_tr, y_tr.ravel())

y_hat_tr = pipe.predict(X_tr)
y_hat_va = pipe.predict(X_va)
y_hat_te = pipe.predict(X_te)

print(f"Fixed SPoC (nfilter={n_filters}) + Ridge(alpha={alpha})")
print(f"Train RMSE: {rmse(y_tr, y_hat_tr):.4f} | R^2: {r2_score(y_tr, y_hat_tr):.4f}")
print(f"Valid RMSE: {rmse(y_va, y_hat_va):.4f} | R^2: {r2_score(y_va, y_hat_va):.4f}")
print(f"Test  RMSE: {rmse(y_te, y_hat_te):.4f} | R^2: {r2_score(y_te, y_hat_te):.4f}")
