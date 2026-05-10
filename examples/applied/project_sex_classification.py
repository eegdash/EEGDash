# %% [markdown]
"""Sex classification from EEG
============================

**Difficulty 3** | **Runtime: 3-5m** | **Compute: CPU**

A canonical "is this signal even predictive?" benchmark task on resting-state
EEG from `OpenNeuro <https://openneuro.org>`_ ``ds005505`` (Healthy Brain
Network; Alexander et al. 2017), reachable through `NEMAR
<https://nemar.org>`_ :cite:`delorme2022nemar`. Log band-power features feed
:class:`sklearn.pipeline.Pipeline` :cite:`pedregosa2011sklearn` with
:class:`~sklearn.preprocessing.StandardScaler` and
:class:`~sklearn.linear_model.LogisticRegression`. A 3-fold cross-subject
loop via :class:`~sklearn.model_selection.GroupKFold` keeps every subject
in exactly one fold; held-out predictions feed three sklearn display
helpers, :class:`~sklearn.inspection.DecisionBoundaryDisplay`,
:class:`~sklearn.metrics.RocCurveDisplay`, and
:class:`~sklearn.metrics.ConfusionMatrixDisplay`. Do the features
separate the classes, how stable is held-out AUC across subjects, and
which class does the model confuse?
Keywords: classification, applied, sex
"""

# %% [markdown]
# Metadata: Sex vs. Gender
# ------------------------
# EEGDash loads both ``sex`` and ``gender`` fields from the BIDS
# ``participants.tsv`` if available. While often used interchangeably in
# metadata, ``sex`` typically refers to biological status as recorded by the
# sponsor, while ``gender`` reflects subject self-report. When both are
# present, this tutorial defaults to the ``sex`` field for classification,
# but users should verify the source-dataset's data dictionary (JSON sidecar)
# for the specific semantics of each field.
#
# Leakage and Splits
# ------------------
# Naive window-level or recording-level splits on the same subject will
# cause **subject leakage**. Since resting-state EEG is a "biometric fingerprint",
# the model will learn to recognize the subject rather than generalize the
# label. We use ``GroupKFold(groups=subjects)`` to ensure that windows from
# the same participant never appear in both training and test sets.
# See :doc:`/concepts/leakage_and_evaluation` for the full rationale.
#
# Validate your result
# --------------------
# - **Class Balance.** HBN ``ds005505`` is relatively balanced by sex. Verify
#   that your filtered dataset maintains a reasonable ratio (e.g., 40/60).
# - **Metrics.** Report **Balanced Accuracy** and **ROC AUC**. Since clinical
#   cohorts often have slight imbalances, raw accuracy can be misleading.
# - **Baseline Comparison.** Compare your model against a ``majority_class``
#   baseline. A model that doesn't beat the majority class has not learned
#   anything from the EEG.
# - **Ethical & Context Caveats.** (1) **Labels are metadata, not biology.**
#   The BIDS ``sex`` field is often sponsor-coded; any separation speaks to
#   that category, not to chromosomal or hormonal status. (2) **Confounds.**
#   Recording site, age, and equipment can correlate with sex in small
#   cohorts, leading to "clever Hans" effects where the model learns the
#   site rather than the phenomenon.

# Difficulty: 3-star (advanced applied project)

# %% [markdown]
# Requirements
# ------------
# - About 3 to 5 minutes on CPU on first run (network pull of six
#   ds005505 ``.set`` files); under 30 s once cached.
# - Prerequisites:
#   :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`,
#   :doc:`/generated/auto_examples/tutorials/40_features/plot_42_features_to_sklearn`.
# - Concept: :doc:`/concepts/features_vs_deep_learning`.

# %% [markdown]
# Why a sklearn-display diagnostic?
# ---------------------------------
# Three numbers on a line are easy to misread. The same model can post
# AUC = 0.74 with a balanced confusion matrix or with a 0.95 / 0.05
# row-normalised matrix; only the second flags a degenerate classifier
# that collapsed onto one class. Wiring
# :class:`~sklearn.inspection.DecisionBoundaryDisplay`,
# :class:`~sklearn.metrics.RocCurveDisplay`, and
# :class:`~sklearn.metrics.ConfusionMatrixDisplay` around a leakage-safe
# cross-subject loop is the canonical sklearn diagnostic for any binary
# EEG classifier; this project demonstrates it on the sex-prediction
# question because that is the most direct test of whether the resting-
# state signal carries predictive information about a categorical label.

# %%
# Setup. ``random_state=42`` on every estimator and splitter is what
# keeps the printed metrics byte-stable across runs (E3.21).
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eegdash
from eegdash import EEGDashDataset
from eegdash.viz import use_eegdash_style

use_eegdash_style()
mne.set_log_level("ERROR")
warnings.simplefilter("ignore", category=RuntimeWarning)
SEED = 42
rng = np.random.default_rng(SEED)
CACHE_DIR = Path(
    os.environ.get("EEGDASH_CACHE_DIR", Path.home() / ".eegdash_cache")
).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"eegdash {eegdash.__version__} | cache_dir={CACHE_DIR}")

# %% [markdown]
# Step 1. Pull ds005505 resting-state metadata
# --------------------------------------------
# **Predict.** ds005505 carries Healthy Brain Network resting-state
# recordings on a 128-channel HydroCel cap :cite:`alexander2017hbn`. The
# metadata-only first pass populates each record's ``description`` with
# the BIDS ``sex`` field; recordings without a usable label are dropped
# before any preprocessing fires.

# %%
DATASET = "ds005505"
TASK = "RestingState"
N_PER_SEX = 3  # 3 F + 3 M = 6 subjects total; 3-fold leaves 4 train + 2 test
# per fold. Bump to 6 for a tighter ROC band; the project takes
# ~3 min more per extra pair of subjects on first download.
dataset_full = EEGDashDataset(
    cache_dir=CACHE_DIR,
    dataset=DATASET,
    task=TASK,
    description_fields=["subject", "session", "run", "task", "sex", "gender"],
)
candidates = [
    d for d in dataset_full.datasets if d.description.get("sex") in ("F", "M")
]
print(
    f"records with usable sex label: {len(candidates)} / {len(dataset_full.datasets)}"
)

# %% [markdown]
# Step 2. Balance the cohort by the BIDS sex label
# ------------------------------------------------
# A 50 / 50 sex balance pins the chance line at 0.50 on every fold. We
# sort the candidate list by subject id (deterministic across runs) and
# take the first ``N_PER_SEX`` of each label so the cohort is reproducible.

# %%
by_sex: dict[str, list] = {"F": [], "M": []}
for d in sorted(candidates, key=lambda r: str(r.description.get("subject"))):
    label = str(d.description.get("sex"))
    if len(by_sex[label]) < N_PER_SEX:
        by_sex[label].append(d)
selected = by_sex["F"] + by_sex["M"]
n_subjects = len(selected)
ds_small = BaseConcatDataset(selected)
cohort_summary = pd.Series(
    {
        "n_subjects": n_subjects,
        "n_F": len(by_sex["F"]),
        "n_M": len(by_sex["M"]),
        "raw n_channels": ds_small.datasets[0].raw.info["nchan"],
        "raw sfreq (Hz)": float(ds_small.datasets[0].raw.info["sfreq"]),
    },
    name="value",
).to_frame()
cohort_summary

# %% [markdown]
# Step 3. Pick a small montage, resample, and window
# --------------------------------------------------
# Three preprocessors keep the runtime predictable: pick eight HydroCel
# electrodes spanning midline, fronto-central, parieto-occipital, and
# bilateral temporal scalp; resample to 100 Hz; apply a 1 to 45 Hz
# band-pass. Fixed-length 2 s windows feed the feature stage. The
# 8-channel pick is a deliberate compression: the question is whether
# resting-state EEG carries predictive information at all, not whether
# 128 electrodes carry more than 8.

# %%
CH_NAMES = ["E22", "E9", "E33", "E11", "E122", "E29", "E124", "Cz"]
TARGET_SFREQ = 100  # Hz
WINDOW_SECONDS = 2.0
window_size_samples = int(WINDOW_SECONDS * TARGET_SFREQ)
preprocess(
    ds_small,
    [
        Preprocessor("pick_channels", ch_names=CH_NAMES),
        Preprocessor("resample", sfreq=TARGET_SFREQ),
        Preprocessor("filter", l_freq=1, h_freq=45),
    ],
    n_jobs=1,
)
windows_ds = create_fixed_length_windows(
    ds_small,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    drop_last_window=True,
    preload=True,
)
n_channels = len(CH_NAMES)
# Braindecode 1.0+ may return ``EEGWindowsDataset`` (carries ``.raw``)
# or the older ``WindowsDataset`` (carries ``.windows``); both expose
# the sampling rate through ``.info``.
_first_ds = windows_ds.datasets[0]
_info_holder = getattr(_first_ds, "raw", None) or _first_ds.windows
sfreq = float(_info_holder.info["sfreq"])
print(
    f"n_windows={len(windows_ds)} | n_channels={n_channels} | "
    f"sfreq={sfreq:.0f} Hz | window_size_samples={window_size_samples}"
)

# %% [markdown]
# Step 4. Materialize windows + per-window subject id and label
# -------------------------------------------------------------
# The cross-subject splitter needs a ``groups`` array shaped
# ``(n_windows,)`` carrying the subject id, and a ``y`` array with the
# class label (0 = F, 1 = M). Iterating the per-record sub-datasets and
# reading :attr:`braindecode.datasets.WindowsDataset.description` once
# per record is the most direct route.

# %%
SEX_TO_INT = {"F": 0, "M": 1}
CLASS_NAMES = ("F", "M")
X_list: list[np.ndarray] = []
y_list: list[int] = []
groups_list: list[str] = []
for sub_ds in windows_ds.datasets:
    subj = str(sub_ds.description.get("subject"))
    label = SEX_TO_INT[str(sub_ds.description.get("sex"))]
    for k in range(len(sub_ds)):
        x_k, _, _ = sub_ds[k]
        X_list.append(np.asarray(x_k, dtype=np.float32))
        y_list.append(label)
        groups_list.append(subj)
X_raw = np.stack(X_list)
y = np.asarray(y_list, dtype=int)
groups = np.asarray(groups_list)
print(
    f"X_raw.shape={X_raw.shape} | n_F windows={int((y == 0).sum())} | "
    f"n_M windows={int((y == 1).sum())}"
)

# %% [markdown]
# Step 5. Compute log band-power features
# ---------------------------------------
# Per window: one log-power value per channel per band, on delta
# (1 to 4 Hz), theta (4 to 8 Hz), alpha (8 to 13 Hz), beta (13 to
# 30 Hz). The feature shape is ``(n_windows, n_bands * n_channels)``.
# ``log(mean(|FFT|**2))`` is the cheapest band-power feature that
# survives a review :cite:`schirrmeister2017braindecode`.

# %%
BANDS: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),  # delta
    (4.0, 8.0),  # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0),  # beta
)
BAND_NAMES = ("delta", "theta", "alpha", "beta")


def log_band_power(
    X_t: np.ndarray, sfreq: float, bands: tuple[tuple[float, float], ...]
) -> np.ndarray:
    """Stack ``log(mean(|FFT|**2))`` per channel per band.

    ``X_t`` is ``(n_windows, n_channels, n_times)``; the return is
    ``(n_windows, len(bands) * n_channels)``.
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


X_features = log_band_power(X_raw, sfreq, BANDS)
n_features = X_features.shape[1]
print(
    f"feature matrix={X_features.shape} (log-power per channel for "
    f"{', '.join(BAND_NAMES)})"
)

# %% [markdown]
# Discovery: feature distributions
# --------------------------------
# A quick descriptive table on the first eight features flags dead
# channels (variance close to zero) and saturated bands (variance much
# larger than its peers).

# %%
feature_names = [f"{b}_{c}" for b in BAND_NAMES for c in CH_NAMES]
features_df = pd.DataFrame(X_features, columns=feature_names)
features_df.iloc[:, :8].describe().round(3)

# %% [markdown]
# Step 6. 3-fold cross-subject CV with GroupKFold
# -----------------------------------------------
# **Predict.** With ``GroupKFold(n_splits=3)`` and a 6-subject cohort,
# every fold trains on 4 subjects and tests on 2.
# **Run.** Per fold we store accuracy, AUC, and the held-out
# predictions; the pooled scores feed the confusion matrix and the
# pooled ROC.

# %%
N_FOLDS = 3
splitter = GroupKFold(n_splits=N_FOLDS)
fold_accuracies: list[float] = []
fold_aucs: list[float] = []
fold_held_out: list[list[str]] = []
fold_assignment = np.full(len(y), -1, dtype=int)
pooled_y_true: list[np.ndarray] = []
pooled_y_pred: list[np.ndarray] = []
pooled_y_score: list[np.ndarray] = []

for fold_idx, (train_idx, test_idx) in enumerate(
    splitter.split(X_features, y, groups=groups)
):
    held_out = sorted(set(groups[test_idx].tolist()))
    fold_held_out.append(held_out)
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
    clf.fit(X_features[train_idx], y[train_idx])
    y_pred_fold = clf.predict(X_features[test_idx])
    y_score_fold = clf.predict_proba(X_features[test_idx])[:, 1]
    fold_accuracies.append(float(accuracy_score(y[test_idx], y_pred_fold)))
    fold_aucs.append(float(roc_auc_score(y[test_idx], y_score_fold)))
    pooled_y_true.append(np.asarray(y[test_idx]))
    pooled_y_pred.append(np.asarray(y_pred_fold))
    pooled_y_score.append(np.asarray(y_score_fold))

y_true_pooled = np.concatenate(pooled_y_true)
y_pred_pooled = np.concatenate(pooled_y_pred)
y_score_pooled = np.concatenate(pooled_y_score)
mean_auc = float(np.mean(fold_aucs))
std_auc = float(np.std(fold_aucs, ddof=0))
pooled_balanced_acc = float(balanced_accuracy_score(y_true_pooled, y_pred_pooled))
pooled_auc = float(roc_auc_score(y_true_pooled, y_score_pooled))
print(
    f"cross-subject CV: AUC={mean_auc:.3f} +/- {std_auc:.3f} | "
    f"balanced_acc={pooled_balanced_acc:.3f} | folds={N_FOLDS}"
)

# %% [markdown]
# Result table: per-fold metrics
# ------------------------------
# Chance is 0.50 for both AUC and accuracy on this balanced cohort.

# %%
results_df = pd.DataFrame(
    {
        "fold": np.arange(1, N_FOLDS + 1),
        "held-out subjects": [", ".join(s) for s in fold_held_out],
        "accuracy": np.round(fold_accuracies, 3),
        "auc": np.round(fold_aucs, 3),
    }
).set_index("fold")
results_df

# %% [markdown]
# **Investigate.** On a 6-subject balanced cohort with these 32 log-
# power features, the pooled AUC routinely lands well below 0.50: the
# small subject pool means a confound (cap fit, age, recording session)
# locks the model onto the wrong direction on every held-out fold. That
# is the central caveat of this project: a number this far below chance
# is not noise, it is a confound-driven failure that an even-larger
# cohort or a different feature set would have to rescue. Anything
# above AUC = 0.95 in the same script is a leakage smell: re-check
# that ``groups`` is the subject id and that the cohort balancing step
# did not collapse the label.

# %% [markdown]
# Common mistake: scaling on the union of train and test
# ------------------------------------------------------
# **Run.** A subtle slip when wiring a cross-validation loop is fitting
# :class:`~sklearn.preprocessing.StandardScaler` on the full feature
# matrix before splitting. The mean and std then leak test-fold statistics
# into the train slice. We trigger the slip on purpose and recover.

# %%
try:
    sneaky_train_idx, sneaky_test_idx = next(
        splitter.split(X_features, y, groups=groups)
    )
    # Wrong on purpose: scaler fitted on the union, then split.
    leaky_scaler = StandardScaler().fit(X_features)
    X_leaky = leaky_scaler.transform(X_features)
    leaky_clf = LogisticRegression(random_state=SEED, max_iter=2000).fit(
        X_leaky[sneaky_train_idx], y[sneaky_train_idx]
    )
    leaky_auc = float(
        roc_auc_score(
            y[sneaky_test_idx],
            leaky_clf.predict_proba(X_leaky[sneaky_test_idx])[:, 1],
        )
    )
    # Recovery: rebuild the Pipeline so scaling fits on the train slice
    # only.
    safe_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=SEED, max_iter=2000)),
        ]
    ).fit(X_features[sneaky_train_idx], y[sneaky_train_idx])
    safe_auc = float(
        roc_auc_score(
            y[sneaky_test_idx],
            safe_clf.predict_proba(X_features[sneaky_test_idx])[:, 1],
        )
    )
    print(
        f"Leaky-scaler AUC={leaky_auc:.3f} | Pipeline AUC={safe_auc:.3f} "
        f"| gap={leaky_auc - safe_auc:+.3f}"
    )
except ValueError as exc:
    print(f"Caught ValueError: {str(exc)[:120]}")

# %% [markdown]
# Discovery: the pooled confusion matrix in raw counts
# ----------------------------------------------------
# The figure below renders a row-normalised matrix; the table here
# carries the raw counts so a 0.55 cell can be traced back to 11/20
# vs 550/1000.

# %%
cm = confusion_matrix(y_true_pooled, y_pred_pooled)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)
cm_df

# %% [markdown]
# Three-panel sklearn-display diagnostic
# --------------------------------------
# The PCA scatter carries the
# :class:`~sklearn.inspection.DecisionBoundaryDisplay` contour, the ROC
# panel shows per-fold curves in muted shades plus the pooled curve in
# dark ink, and the confusion matrix shows whether the diagonal is
# balanced. The rendering plumbing lives in a sibling
# ``_sex_classification_figure`` module so the project keeps a single
# call to plug live runtime values into the figure.

# %%
# PCA + a logistic regression refit in PCA space so the decision
# contour shares coordinates with the scattered points; the actual
# model the project reports lives in the original feature space.
X_std = StandardScaler().fit_transform(X_features)
X_pca = PCA(n_components=2, random_state=SEED).fit_transform(X_std)
boundary_clf = LogisticRegression(max_iter=400).fit(X_pca, y)
n_test_subjects = max(1, n_subjects // N_FOLDS)
n_train_subjects = n_subjects - n_test_subjects

from _sex_classification_figure import draw_sex_classification_figure

fig = draw_sex_classification_figure(
    X_pca=X_pca,
    y_classes=y,
    fold_assignment=fold_assignment,
    estimator=boundary_clf,
    fold_aucs=fold_aucs,
    y_true_pooled=y_true_pooled,
    y_pred_pooled=y_pred_pooled,
    y_score_pooled=y_score_pooled,
    class_names=CLASS_NAMES,
    plot_id="project_sex_classification",
    n_train_subjects=n_train_subjects,
    n_test_subjects=n_test_subjects,
    n_features=n_features,
)
plt.show()

# %% [markdown]
# **Investigate.** Read the panels in order. (1) PCA + decision
# boundary: do F and M markers form even partly separable clouds?
# Linear separability on log-power features is rare on a 6-subject
# cohort; partial overlap with a clean contour is the expected picture.
# (2) ROC curves: is every per-fold curve above the diagonal chance
# line, or is one held-out fold pulling the pooled AUC down? Big
# across-fold variance is the honest signature of cross-subject EEG.
# (3) Confusion matrix: a balanced diagonal in deep blue is the win
# condition; an off-diagonal stripe means the model has collapsed onto
# one class. The monospace ``balanced_acc=...`` annotation carries the
# headline metric so the figure stands alone.

# %% [markdown]
# Modify
# ------
# **Your turn.** Drop the alpha band from ``BANDS`` and rerun Step 5 and
# Step 6. An AUC drop of more than 0.05 means the alpha band carried
# most of the class-related variance; a smaller drop means the contrast
# is spread across the spectrum.

# %% [markdown]
# Make
# ----
# **Mini-project.** Replace
# :class:`~sklearn.linear_model.LogisticRegression` with
# :class:`~sklearn.ensemble.RandomForestClassifier` (default
# hyperparameters, ``random_state=42``) and rerun Step 6. If the random
# forest does not beat the linear baseline, that is the most useful
# finding the project will produce: linear features deserve a linear
# model.

# %% [markdown]
# Wrap-up
# -------
# Six ds005505 RestingState recordings were balanced by BIDS sex, fed
# through 4-band log-power features on 8 channels, and scored with a
# leakage-safe :class:`~sklearn.linear_model.LogisticRegression`
# Pipeline across 3 cross-subject folds. The pooled AUC is the only
# number worth quoting in a benchmark submission; the per-fold table
# shows whether that AUC is stable. A clean shape and a chance-anchored
# AUC only confirm plumbing; signal quality and the sex-vs-confound
# question are still open :cite:`cisotto2024tips`.

# %% [markdown]
# Try it yourself
# ---------------
# - Bump ``N_PER_SEX`` from 3 to 6 (12 subjects total) and rerun. The
#   per-fold AUC band should tighten as the cohort grows.
# - Replace ``GroupKFold`` with
#   :class:`~sklearn.model_selection.LeaveOneGroupOut`. With 6 subjects
#   the leave-one-out loop returns 6 folds and a tighter ROC band.
# - Append age (``description['age']``) as a regression target instead
#   of sex, and rerun the same three-panel figure with
#   :class:`~sklearn.linear_model.Ridge` plus a continuous-target
#   variant of the diagnostic.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralized bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
