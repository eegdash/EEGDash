"""How do I train a leakage-safe baseline classifier on EEG?
=============================================================

A model that scores 0.78 on held-out windows is only useful when you
also know what 0.50 (chance) and 0.55 (a transparent linear baseline)
look like on the same split. This tutorial trains that linear
baseline on three subjects of OpenNeuro ``ds002718`` (Wakeman & Henson
2015), reachable through `NEMAR <https://nemar.org>`_ (Delorme et al.
2022). Four bands of log power per channel feed
:class:`sklearn.linear_model.LogisticRegression` (Pedregosa et al.
2011); a 3-fold cross-subject loop with
:class:`sklearn.model_selection.GroupKFold` keeps every subject in
exactly one fold. The deliverable is a single three-panel figure that
answers three questions on one screen: do the features separate the
classes, how does the accuracy vary across held-out subjects, and
which trials does the model confuse?

.. sphinx_gallery_thumbnail_path = '_static/thumbs/plot_12_train_a_baseline.png'
"""

# %% [markdown]
# Learning objectives
# -------------------
# - Compute log band-power features in four canonical bands (theta, alpha, beta, gamma) per channel from event-locked windows of a real BIDS dataset.
# - Run a 3-fold cross-subject loop with :class:`~sklearn.model_selection.GroupKFold` so a subject never appears in both train and test.
# - Fit a :class:`~sklearn.linear_model.LogisticRegression` baseline and read per-fold accuracy, mean +/- std, and a row-normalized :func:`~sklearn.metrics.confusion_matrix` from the same call.
# - Compare those numbers against ``majority_baseline`` chance level on the same split.
# - Produce a three-panel diagnostic that lets a reader judge the baseline at a glance.

# %% [markdown]
# Requirements
# ------------
# - About 90 s on CPU on first run; under 30 s once cached.
# - Network on first call (~30 MB into ``cache_dir``); offline thereafter.
# - Prerequisites: :doc:`plot_11_leakage_safe_split` (cross-subject
#   splits), :doc:`plot_10_preprocess_and_window` (event windowing).
# - Concept: :doc:`/concepts/features_vs_deep_learning`.

# %% [markdown]
# Why a baseline before a deep net?
# ---------------------------------
# A baseline is a number you can defend in code review. Logistic
# regression on band-power features has three properties a black-box
# net does not: every coefficient maps to one (channel, band) pair, the
# whole pipeline fits in 200 lines, and the runtime stays inside a
# CPU-only budget. Cisotto & Chicco 2024 frame this as Tip 5: a
# classifier you understand at 0.62 is more useful for benchmark
# bookkeeping than a classifier you do not understand at 0.71. The
# linear baseline is also the gating fence: a deep network that fails
# to clear the linear bar usually has a leakage or labelling bug, not
# a capacity gap (Schirrmeister et al. 2017).
#
# Three reasons we wire the cross-subject loop *first*, before any
# feature engineering:
#
# - **Subject as a confound.** EEG amplitude differs more across
#   subjects than across conditions. A within-subject split
#   double-counts that variance and inflates accuracy.
# - **The unit of generalization is the subject.** The benchmark
#   question is "does this generalize to a new person?", not "does this
#   memorize this person?".
# - **The split fixes the chance level.** ``majority_baseline`` is
#   computed on the held-out test set, so the chance number you report
#   tracks the actual class balance of the test fold, not a notional
#   50 / 50 prior.

# %%
# Setup. ``random_state=42`` on every estimator and splitter is what
# keeps the printed accuracy byte-stable across runs (E3.21).
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eegdash
from eegdash import EEGDashDataset
from eegdash.splits import majority_baseline
from eegdash.viz import use_eegdash_style

use_eegdash_style()
mne.set_log_level("ERROR")
warnings.simplefilter("ignore", category=RuntimeWarning)
SEED = 42
CACHE_DIR = Path(os.environ.get("EEGDASH_CACHE", Path.home() / ".eegdash_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"eegdash {eegdash.__version__} | cache_dir={CACHE_DIR}")

# %% [markdown]
# Step 1: Pull three subjects of ds002718
# ---------------------------------------
# **Predict.** ``ds002718`` is a face-perception study with three
# conditions: ``famous``, ``unfamiliar``, and ``scrambled`` (Wakeman &
# Henson 2015). We pit ``famous`` against ``scrambled`` so the classes
# stay roughly balanced and the chance level lands at 0.50; the
# ``unfamiliar`` events are kept out of the training data to avoid a
# 2:1 imbalance that would inflate raw-accuracy chance to 0.67. Three
# subjects (``002``, ``003``, ``004``) keep the runtime inside budget
# while leaving enough subjects for a 3-fold cross-subject split.

# %%
DATASET = "ds002718"
SUBJECTS = ["002", "003", "004"]
TASK = "FaceRecognition"
dataset = EEGDashDataset(
    cache_dir=CACHE_DIR, dataset=DATASET, subject=SUBJECTS, task=TASK
)
records_summary = pd.Series(
    {
        "n_recordings": len(dataset.datasets),
        "subjects": ", ".join(SUBJECTS),
        "raw n_channels": dataset.datasets[0].raw.info["nchan"],
        "raw sfreq (Hz)": float(dataset.datasets[0].raw.info["sfreq"]),
    },
    name="value",
).to_frame()
records_summary

# %% [markdown]
# Annotation discovery: which event names are actually in the file?
# -----------------------------------------------------------------
# **Run.** Before mapping events to integers, count what is in
# :attr:`mne.io.Raw.annotations`. Hard-coding ``mapping={'face': 0,
# 'scrambled': 1}`` against an assumed schema is the most common cause
# of a silent zero-window dataset.

# %%
descriptions: list[str] = []
for record in dataset.datasets:
    descriptions.extend(record.raw.annotations.description.tolist())
event_counts = (
    pd.Series(descriptions, name="description")
    .value_counts()
    .rename_axis("description")
    .to_frame(name="count")
)
event_counts.head(12)

# %% [markdown]
# **Investigate.** The trial-type column carries fine-grained labels:
# ``famous_new``, ``famous_second_early``, ``famous_second_late``,
# ``unfamiliar_new``, ``scrambled_new``, ``scrambled_second_late``,
# plus ``left_nonsym`` / ``right_sym`` button-press markers. To keep
# the chance level at the canonical 0.50, we collapse the three
# **famous-face** patterns into class ``0`` and the three
# **scrambled-face** patterns into class ``1``; the ``unfamiliar_*``
# events are dropped (they would skew the balance to 2:1). Button
# presses and ``boundary`` markers are ignored. Famous vs scrambled is
# the canonical Wakeman & Henson contrast.

# %% [markdown]
# Step 2: Two safe preprocessors and event windowing
# --------------------------------------------------
# Two preprocessors keep the runtime predictable: drop non-EEG channels
# and resample to 100 Hz. The event-locked windowing is one call to
# :func:`braindecode.preprocessing.create_windows_from_events`. Each
# window covers 1 s after stimulus onset (``trial_start_offset_samples
# = 0``, ``trial_stop_offset_samples = sfreq``) which spans the early
# visual response while keeping the window count manageable.

# %%
TARGET_SFREQ = 100  # Hz
WINDOW_SECONDS = 1.0
preprocess(
    dataset,
    [
        Preprocessor("pick_types", eeg=True, eog=False, misc=False),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
    ],
)
window_size_samples = int(WINDOW_SECONDS * TARGET_SFREQ)
EVENT_MAPPING = {
    "famous_new": 0,
    "famous_second_early": 0,
    "famous_second_late": 0,
    "scrambled_new": 1,
    "scrambled_second_early": 1,
    "scrambled_second_late": 1,
}
CLASS_NAMES = ("famous", "scrambled")
windows = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=window_size_samples,
    preload=True,
    drop_bad_windows=True,
    mapping=EVENT_MAPPING,
)
ch_names = list(windows.datasets[0].windows.info["ch_names"])
n_channels = len(ch_names)
sfreq = float(windows.datasets[0].windows.info["sfreq"])
print(
    f"n_windows={len(windows)} | n_channels={n_channels} | "
    f"sfreq={sfreq:.0f} Hz | window_size_samples={window_size_samples}"
)

# %% [markdown]
# Step 3: Materialize windows + per-window subject id
# ---------------------------------------------------
# The cross-subject splitter needs a ``groups`` array shaped
# ``(n_windows,)`` that holds the subject id of every window. The
# easiest route is iterating windows and reading
# :attr:`braindecode.datasets.WindowsDataset.description` once per
# per-record subdataset.

# %%
X_list: list[np.ndarray] = []
y_list: list[int] = []
groups_list: list[str] = []
for sub_ds in windows.datasets:
    subj = str(sub_ds.description.get("subject"))
    for k in range(len(sub_ds)):
        x_k, y_k, _ = sub_ds[k]
        X_list.append(np.asarray(x_k, dtype=np.float32))
        y_list.append(int(y_k))
        groups_list.append(subj)
X = np.stack(X_list)
y = np.asarray(y_list, dtype=int)
groups = np.asarray(groups_list)

# %% [markdown]
# **Predict.** ``X.shape`` should be
# ``(n_windows, n_channels, window_size_samples)``. The class counts on
# ``np.bincount(y)`` should sit close to 1:1 because the three famous
# mappings and the three scrambled mappings carry roughly the same
# trial counts in ds002718.

# %%
shape_summary = pd.Series(
    {
        "X.shape": str(X.shape),
        "X.dtype": str(X.dtype),
        "n_famous windows": int((y == 0).sum()),
        "n_scrambled windows": int((y == 1).sum()),
        "subjects in groups": ", ".join(sorted(set(groups))),
    },
    name="value",
).to_frame()
shape_summary

# %% [markdown]
# Step 4: Compute log band-power features
# ---------------------------------------
# For each window the feature vector is one log-power value per EEG
# channel and per band: theta (4-8 Hz), alpha (8-13 Hz), beta
# (13-30 Hz), gamma (30-45 Hz). The feature shape is
# ``(n_windows, n_bands * n_channels)``. Computing power as
# ``log(mean(|FFT|**2))`` over a band (Chambon et al. 2018;
# Schirrmeister et al. 2017) is the cheapest band-power feature that
# survives a review. Four bands give the linear classifier enough
# spectral signal to clear chance on this contrast; staying with
# only theta + alpha lands on a flat coin-toss.

# %%
BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),  # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0),  # beta
    (30.0, 45.0),  # gamma
)
BAND_NAMES = ("theta", "alpha", "beta", "gamma")


def log_band_power(
    X_t: np.ndarray, sfreq: float, bands: tuple[tuple[float, float], ...]
) -> np.ndarray:
    """Stack log-band-power features per channel for several bands.

    Parameters
    ----------
    X_t : numpy.ndarray
        ``(n_windows, n_channels, n_times)`` real-valued window tensor.
    sfreq : float
        Sampling rate in Hz.
    bands : tuple of (fmin, fmax) tuples
        Band edges in Hz.

    Returns
    -------
    numpy.ndarray
        ``(n_windows, len(bands) * n_channels)`` log-power features.

    """
    spec = np.fft.rfft(X_t, axis=-1)
    power = (np.abs(spec) ** 2) / X_t.shape[-1]
    freqs = np.fft.rfftfreq(X_t.shape[-1], d=1.0 / sfreq)
    feats = []
    for fmin, fmax in bands:
        band_mask = (freqs >= fmin) & (freqs < fmax)
        # Add a tiny floor so log(0) does not appear when a band is empty.
        feats.append(np.log(power[..., band_mask].mean(axis=-1) + 1e-12))
    return np.concatenate(feats, axis=-1).astype(np.float32)


F = log_band_power(X, sfreq, BANDS)
print(
    f"feature matrix={F.shape} (log-power per channel for "
    f"{', '.join(BAND_NAMES)}) | dtype={F.dtype}"
)

# %% [markdown]
# Discovery: feature distributions
# --------------------------------
# A quick descriptive table on the feature matrix is the easiest way to
# spot a dead channel (variance ~ 0) or a saturated band (variance much
# larger than its peers). We inspect the first eight features to keep
# the table short.

# %%
feature_names = [f"{band}_{ch}" for band in BAND_NAMES for ch in ch_names]
features_df = pd.DataFrame(F, columns=feature_names)
features_df.iloc[:, :8].describe().round(3)

# %% [markdown]
# Step 5: 3-fold cross-subject CV with GroupKFold
# -----------------------------------------------
# **Predict.** With three subjects and ``GroupKFold(n_splits=3)``, each
# fold trains on two subjects and tests on the third. The held-out
# subject id is the same as the group id of the test windows.
# **Run.** :class:`~sklearn.model_selection.GroupKFold` walks every
# possible held-out group; we store accuracy and confusion-matrix counts
# per fold.

# %%
N_FOLDS = 3
splitter = GroupKFold(n_splits=N_FOLDS)

fold_accuracies: list[float] = []
fold_held_out: list[str] = []
fold_chance: list[float] = []
fold_assignment = np.full(len(y), -1, dtype=int)
pooled_y_true: list[np.ndarray] = []
pooled_y_pred: list[np.ndarray] = []

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(F, y, groups=groups)):
    held_out = sorted(set(groups[test_idx].tolist()))
    fold_held_out.append(held_out[0])
    fold_assignment[test_idx] = fold_idx

    # Standardize then fit. Per-fold scaling fitted on the train slice
    # only keeps the test fold leakage-safe; logistic regression on
    # untransformed log-power features fails to converge.
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=SEED, max_iter=2000)),
        ]
    )
    clf.fit(F[train_idx], y[train_idx])
    y_pred = clf.predict(F[test_idx])
    acc = float(accuracy_score(y[test_idx], y_pred))
    fold_accuracies.append(acc)
    fold_chance.append(
        float(majority_baseline(y[train_idx], y[test_idx])["chance_level"])
    )
    pooled_y_true.append(np.asarray(y[test_idx]))
    pooled_y_pred.append(np.asarray(y_pred))

y_true_pooled = np.concatenate(pooled_y_true)
y_pred_pooled = np.concatenate(pooled_y_pred)

mean_acc = float(np.mean(fold_accuracies))
std_acc = float(np.std(fold_accuracies, ddof=0))
chance_overall = float(np.mean(fold_chance))
print(
    f"cross-subject CV: mean={mean_acc:.3f} +/- {std_acc:.3f} | "
    f"chance={chance_overall:.3f} | folds={N_FOLDS}"
)

# %% [markdown]
# Result table: per-fold accuracy vs chance
# -----------------------------------------
# One row per fold so the chance disclosure (E5.43) and the model number
# sit on the same screen. The held-out column is the subject id that
# was *not* in the training fold.

# %%
results_df = pd.DataFrame(
    {
        "fold": np.arange(1, N_FOLDS + 1),
        "held-out subject": [f"sub-{sid}" for sid in fold_held_out],
        "accuracy": np.round(fold_accuracies, 3),
        "chance": np.round(fold_chance, 3),
        "lift": np.round(np.asarray(fold_accuracies) - np.asarray(fold_chance), 3),
    }
).set_index("fold")
results_df

# %% [markdown]
# **Investigate.** A famous-vs-scrambled split on band-power features
# typically lands in the 0.53-0.60 range with three subjects (chance
# = 0.50, balanced classes). Anything above 0.85 on this minimal
# feature set is a leakage smell: re-check that ``groups`` is the
# subject id, not the trial id, and that the event mapping has not
# collapsed both classes into one. A number near 0.50 is the honest
# floor; deep models should beat it before anyone reports them.

# %% [markdown]
# Common mistake: training on the held-out fold by accident
# ---------------------------------------------------------
# **Run.** A subtle slip when wiring a cross-validation loop is fitting
# the model on the test slice. The fix is mechanical, but the symptom
# is a deceptively high accuracy that the figure below would otherwise
# rubber-stamp. We trigger the slip on purpose and recover.

# %%
try:
    sneaky_train_idx, sneaky_test_idx = next(splitter.split(F, y, groups=groups))
    # Wrong on purpose: fitting on the *test* fold then scoring on it.
    sneaky_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=SEED, max_iter=2000)),
        ]
    )
    sneaky_clf.fit(F[sneaky_test_idx], y[sneaky_test_idx])
    sneaky_acc = float(
        accuracy_score(y[sneaky_test_idx], sneaky_clf.predict(F[sneaky_test_idx]))
    )
    # Recovery: re-fit on the train slice; the gap is the bug.
    recovered_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=SEED, max_iter=2000)),
        ]
    )
    recovered_clf.fit(F[sneaky_train_idx], y[sneaky_train_idx])
    recovered_acc = float(
        accuracy_score(y[sneaky_test_idx], recovered_clf.predict(F[sneaky_test_idx]))
    )
    print(
        f"Train-on-test (wrong) acc={sneaky_acc:.2f} | "
        f"train-on-train (correct) acc={recovered_acc:.2f} | "
        f"gap={sneaky_acc - recovered_acc:+.2f}"
    )
except ValueError as exc:
    print(f"Caught ValueError: {str(exc)[:100]}")

# %% [markdown]
# Three-panel diagnostic figure
# -----------------------------
# Three numbers on a line are easy to misread. The figure below carries
# the same story across three panels: the PCA scatter shows whether the
# band-power features separate the classes; the bar chart shows the
# spread of held-out-subject accuracy around the mean and against the
# chance line; the row-normalized
# :func:`~sklearn.metrics.confusion_matrix` shows which class the model
# confuses on the held-out fold. The drawing helpers live in a sibling
# ``_baseline_diagnostic`` module so the rendering plumbing stays out of
# this tutorial; the call below is the only line that matters.

# %%
from _baseline_diagnostic import draw_baseline_diagnostic

fig = draw_baseline_diagnostic(
    X_features=F,
    y_classes=y,
    fold_assignment=fold_assignment,
    fold_accuracies=fold_accuracies,
    y_true_pooled=y_true_pooled,
    y_pred_pooled=y_pred_pooled,
    class_names=CLASS_NAMES,
    subjects=SUBJECTS,
    held_out_subjects=fold_held_out,
    chance_level=chance_overall,
    plot_id="plot_12",
)
plt.show()

# %% [markdown]
# **Investigate.** Read the three panels in order.
#
# 1. PCA scatter: do famous and scrambled markers form even partly
#    separable clouds, or do they overlap completely? Real linear
#    separability on band-power features is rare; even partial
#    separation is enough for a logistic regression to clear chance.
# 2. Per-fold bars: is every fold above the chance line, or is one
#    held-out subject pulling the mean down? Big across-fold variance
#    is the honest signature of cross-subject EEG.
# 3. Confusion matrix: a row-normalized matrix with a clean diagonal in
#    deep blue is the win condition; an off-diagonal stripe means the
#    model has collapsed onto the majority class.

# %% [markdown]
# Modify
# ------
# **Your turn.** Replace ``BANDS`` with a single broad band
# ``((4.0, 45.0),)`` and rerun Step 4 + Step 5. The feature matrix
# shrinks from ``4 * n_channels`` to ``n_channels``. Predict before
# running: does the mean cross-subject accuracy hold, drop, or rise?
# Wider bands smear the spectral signature so the linear separator has
# less to work with; keep an eye on the mean +/- std band.

# %% [markdown]
# Make
# ----
# **Mini-project.** Replace
# :class:`~sklearn.linear_model.LogisticRegression` with a
# :class:`~sklearn.ensemble.RandomForestClassifier` (default
# hyperparameters, ``random_state=42``) and rerun Step 5. Add a second
# row to ``results_df`` and explain in one sentence why the random
# forest beats or matches the linear baseline on band-power features.
# If it does not beat the linear baseline, that is the most useful
# finding the project will produce: linear features deserve a linear
# model.

# %% [markdown]
# Wrap-up
# -------
# We loaded three subjects of ``ds002718``, picked famous-vs-scrambled
# events, computed four log-band-power features per channel, and ran a
# 3-fold cross-subject loop with
# :class:`~sklearn.model_selection.GroupKFold`. The
# :class:`~sklearn.linear_model.LogisticRegression` baseline was
# anchored against ``majority_baseline`` chance level on the same split,
# and the three-panel figure showed where the features separate, where
# the per-fold variance lives, and which class the model confuses.
# The cross-subject mean +/- std is the only number worth quoting in a
# benchmark submission; the per-fold table shows whether that mean is
# stable.

# %% [markdown]
# Try it yourself
# ---------------
# - Add a fourth subject (``005``) and rerun. Predict whether the
#   variance band tightens (more groups in the leave-one-out loop) or
#   widens (the new subject is harder to generalize to).
# - Replace ``GroupKFold`` with
#   :class:`~sklearn.model_selection.LeaveOneGroupOut`. With three
#   subjects the two splitters agree, but on a 12-subject sweep the
#   leave-one-out loop returns 12 folds and a tighter standard error.
# - Drop the gamma band (30-45 Hz) from ``BANDS``. Does the PCA panel
#   show less separation? Does the held-out mean drop below chance, or
#   does theta-alpha-beta carry most of the lift on its own?

# %% [markdown]
# References
# ----------
# - Wakeman & Henson 2015, A multi-subject, multi-modal human neuroimaging dataset, *Scientific Data* 2:150001. https://doi.org/10.1038/sdata.2015.1
# - Delorme et al. 2022, NEMAR, an open access data, tools and compute resource operating on neuroelectromagnetic data, *Database* baac096. https://doi.org/10.1093/database/baac096
# - Pedregosa et al. 2011, Scikit-learn: Machine Learning in Python, *Journal of Machine Learning Research* 12. https://www.jmlr.org/papers/v12/pedregosa11a.html
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Chambon et al. 2018, A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series, *IEEE TNSRE* 26(4). https://doi.org/10.1109/TNSRE.2018.2813138
# - Schirrmeister et al. 2017, Deep learning with convolutional neural networks for EEG decoding and visualization, *Human Brain Mapping* 38(11). https://doi.org/10.1002/hbm.23730
# - Pernet et al. 2019, EEG-BIDS, *Scientific Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - Concept page: :doc:`/concepts/features_vs_deep_learning`.
