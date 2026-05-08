"""How well does an EEG decoder generalise to a never-seen subject?
==================================================================

Cross-subject generalisation is the gold standard for any decoding
claim. Train on N-1 subjects, test on the held-out one, repeat for
every subject: that is leave-one-subject-out cross-validation (LOSO),
the protocol behind the MOABB benchmark :cite:`aristimunha2023transferstructure` and
the de-facto evaluation in clinical-EEG decoding. Brookshire et al.
2024 surveyed 81 deep-learning EEG papers and found data leakage in
roughly half; on properly subject-held-out splits, the same
architectures dropped on average from 0.83 accuracy to 0.62.
Cisotto & Chicco 2024 (Tip 9) name leakage the single most common
reporting mistake. ``ds002718`` :cite:`wakeman2015`, reachable
through `NEMAR <https://nemar.org>`_ :cite:`delorme2022nemar`, is the
running example throughout the gallery.

Where ``plot_11`` proved a single split is leakage-free and ``plot_12``
trained one model on one cross-subject split, this tutorial steps up
to the actual evaluation: a LOSO loop that holds a different subject
out each time, a subject x subject transfer heatmap, and a pooled
confusion matrix over every held-out prediction. The deliverable is a
single three-panel figure.

.. sphinx_gallery_thumbnail_path = '_static/thumbs/plot_51_cross_subject_evaluation.png'

So how big is the across-subject spread once you run the loop?
"""

# %% [markdown]
# Learning objectives
# -------------------
#
# - Explain why cross-subject evaluation is the gold standard for EEG decoding generalisation.
# - Build a leave-one-subject-out loop with :class:`sklearn.model_selection.LeaveOneGroupOut` keyed on ``subject``.
# - Compute a subject x subject train-test transfer matrix and read which held-out subjects are systematically harder.
# - Quote ``mean +/- std`` of LOSO :func:`sklearn.metrics.balanced_accuracy_score` against a chance level computed on the test fold.
# - Aggregate predictions across folds into a single :class:`sklearn.metrics.ConfusionMatrixDisplay` to see which class the model confuses on held-out subjects.
#
# Requirements
# ------------
#
# - Prerequisites: :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split` (cross-subject splits) and ``plot_12_train_a_baseline`` (one model on one split).
# - About 30 s on CPU. No network: the cohort is built in-script.
# - Concept: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup. ``random_state=42`` on every estimator and splitter and
# ``np.random.seed`` keeps the printed accuracy byte-stable across
# runs (E3.21).
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from eegdash.splits import (
    assert_no_leakage,
    describe_split,
    get_splitter,
    k_fold,
    majority_baseline,
)
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# %% [markdown]
# Why LOSO and not a single 80/20 split?
# --------------------------------------
#
# A single cross-subject split returns one number; LOSO returns N
# numbers, one per held-out subject. The mean is what benchmark tables
# publish, but the *spread* tells you whether the model works for
# everyone or just for the subjects who happened to land on the easy
# side of the random fold. Aristimunha et al. 2023 wired the MOABB
# benchmark around exactly this protocol: every BCI paradigm (motor
# imagery, P300, SSVEP) is scored as ``mean +/- std`` over per-subject
# LOSO folds, so a method with low mean and low std is preferred over
# a method with the same mean and a long tail of failed subjects.
# Cisotto & Chicco 2024 frame the per-subject view as Tip 9: never
# quote a single accuracy without the across-subject standard deviation
# that produced it.
#
# The transfer matrix in panel 1 breaks this down further. Cell
# ``(i, j)`` is the balanced accuracy of a model trained on source
# subject ``i`` and evaluated on subject ``j``. A column with low values
# means subject ``j`` is hard regardless of who trained the model; a row
# with low values means subject ``i`` does not contribute useful signal.

# %% [markdown]
# Step 1. Build per-subject metadata for 8 subjects
# ---------------------------------------------------
#
# We materialise a synthetic table: 8 subjects, 60 windows each, with
# a 2-D feature carrying class signal plus a per-subject offset (the
# "subject fingerprint" that makes leakage so dangerous).
# :mod:`eegdash.splits` accepts a
# :class:`braindecode.datasets.WindowsDataset` or this DataFrame.


# %%
def make_cohort(sizes, *, prefix: str, rng):
    """Return ``(X, metadata)`` for a synthetic cross-subject toy task."""
    rows, X_list = [], []
    for s, n_w in enumerate(sizes):
        labels = rng.integers(0, 2, size=n_w)
        bias = 0.10 * s
        for w, lab in enumerate(labels):
            base = bias + rng.standard_normal(2) * 0.7
            X_list.append([float(lab) + base[0], -float(lab) + base[1]])
            rows.append(
                {
                    "sample_id": f"{prefix}-{s:02d}__w{w:03d}",
                    "subject": f"sub-{s:02d}",
                    "session": "ses-01",
                    "run": "run-01",
                    "dataset": f"ds-{prefix}",
                    "target": int(lab),
                }
            )
    return np.asarray(X_list, dtype=float), pd.DataFrame(rows)


N_SUBJECTS = 8
N_WINDOWS_PER_SUBJECT = 60
X, metadata = make_cohort([N_WINDOWS_PER_SUBJECT] * N_SUBJECTS, prefix="loso", rng=rng)
y = metadata["target"].to_numpy()
groups = metadata["subject"].to_numpy()
print(
    f"rows={len(metadata)} | subjects={metadata['subject'].nunique()} | "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# Step 2. Predict the LOSO fold count, then build the splits
# ------------------------------------------------------------
#
# **Predict.** Leave-one-subject-out with N subjects produces exactly
# N folds (one per held-out subject). Will the per-fold test set have
# 60 windows or 480? Pick one, then read the fold count below.
#
# **Run.** :class:`~sklearn.model_selection.LeaveOneGroupOut` with
# ``groups=metadata["subject"]`` is the canonical LOSO splitter. The
# :func:`eegdash.splits.get_splitter` registry returns the same object
# under the ``"cross_subject"`` engine when you ask for one fold per
# subject. We use sklearn directly here so the loop reads as plain
# scikit-learn; the manifest path mirrors the one ``plot_11``
# demonstrated.

# %%
n_loso_folds = LeaveOneGroupOut().get_n_splits(X, y, groups)
print(f"n_subjects={N_SUBJECTS} | n_loso_folds={n_loso_folds}")

splitter = get_splitter(
    "cross_subject",
    n_folds=N_SUBJECTS,
    n_splits=N_SUBJECTS,
    random_state=SEED,
)
folds = list(k_fold(metadata, splitter=splitter, target="target"))
overlap = assert_no_leakage(folds, metadata, by="subject")
assert overlap == 0, "cross-subject manifest leaked subjects"
print(
    f"splitter={type(splitter).__name__} | folds={len(folds)} | "
    f"max subject overlap={overlap}"
)

# %% [markdown]
# Step 3. Run the LOSO loop and pool the predictions
# ----------------------------------------------------
#
# **Run (#2).** For each fold: fit
# :class:`~sklearn.linear_model.LogisticRegression` on the N-1 subjects
# in the train mask, predict the one held-out subject, score with
# :func:`~sklearn.metrics.balanced_accuracy_score`. Append every
# (true, pred) pair into pooled arrays so the pooled confusion matrix
# in the headline figure carries every held-out window once.


# %%
def loso_loop(X, y, metadata, folds):
    """Return per-fold balanced accuracy plus pooled (true, pred)."""
    fold_acc, fold_chance, fold_subject = [], [], []
    pooled_true, pooled_pred = [], []
    for k in range(len(folds)):
        train_mask = folds[k][0]
        test_mask = folds[k][1]
        clf = LogisticRegression(random_state=SEED, max_iter=300)
        clf.fit(X[train_mask], y[train_mask])
        y_pred = clf.predict(X[test_mask])
        y_true = y[test_mask]
        fold_acc.append(float(balanced_accuracy_score(y_true, y_pred)))
        fold_chance.append(
            float(majority_baseline(y[train_mask], y[test_mask])["chance_level"])
        )
        held_out = sorted(metadata.loc[test_mask, "subject"].unique())
        fold_subject.append(held_out[0] if held_out else f"fold-{k}")
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
    return (
        np.asarray(fold_acc),
        np.asarray(fold_chance),
        fold_subject,
        np.concatenate(pooled_true),
        np.concatenate(pooled_pred),
    )


fold_acc, fold_chance, held_out_subjects, y_true_pooled, y_pred_pooled = loso_loop(
    X, y, metadata, folds
)
mean_loso = float(fold_acc.mean())
std_loso = float(fold_acc.std(ddof=0))
chance_overall = float(fold_chance.mean())
for k, (a, c, s) in enumerate(zip(fold_acc, fold_chance, held_out_subjects)):
    print(f"Fold {k}: held-out {s} | balanced_acc={a:.3f} | chance={c:.3f}")
print(
    f"LOSO summary: balanced_acc={mean_loso:.3f} +/- {std_loso:.3f} | "
    f"chance={chance_overall:.3f} | n_folds={n_loso_folds}"
)

# %% [markdown]
# Step 4. Each fold's test set has DIFFERENT subjects
# -----------------------------------------------------
#
# **Run (#3).** The cross-subject contract is that every held-out
# subject appears in exactly one test fold; the union across folds
# tiles the cohort. :func:`eegdash.splits.describe_split` returns the
# audit; the per-fold lookup below confirms the contract holds.

# %%
describe_split(folds, metadata, target="target", print_report=False)
test_subjects_by_fold = []
for _tr_mask, te_mask in folds:
    subs = sorted(metadata.loc[te_mask, "subject"].unique())
    test_subjects_by_fold.append(subs)
print(
    f"union across folds: {len(set().union(*test_subjects_by_fold))} | "
    f"cohort size: {N_SUBJECTS}"
)

# %% [markdown]
# Step 5. Build the subject x subject transfer matrix
# -----------------------------------------------------
#
# **Investigate.** A LOSO mean collapses N folds into one number. The
# transfer matrix keeps the resolution: cell ``(i, j)`` = balanced
# accuracy of a model trained on source subject ``i`` alone and
# evaluated on held-out subject ``j``. The diagonal ``(j, j)`` is the
# within-subject case and is masked because cross-subject
# generalisation is the point. A column with low values flags a test
# subject who is hard to decode regardless of who trained the model;
# a row with low values flags a source subject whose data does not
# transfer. Bouchard et al. and the MOABB benchmark report variants of
# this matrix as the diagnostic for *who* the cohort is hard for.


# %%
def transfer_matrix_pairwise(X, y, metadata, subject_ids):
    """Cell (i, j): train on source subject i alone, score on subject j."""
    n = len(subject_ids)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, src in enumerate(subject_ids):
        src_mask = (metadata["subject"] == src).to_numpy()
        if len(np.unique(y[src_mask])) < 2:
            continue
        clf = LogisticRegression(random_state=SEED, max_iter=300)
        clf.fit(X[src_mask], y[src_mask])
        for j, tgt in enumerate(subject_ids):
            if i == j:
                continue
            tgt_mask = (metadata["subject"] == tgt).to_numpy()
            matrix[i, j] = float(
                balanced_accuracy_score(y[tgt_mask], clf.predict(X[tgt_mask]))
            )
    return matrix


subject_ids = sorted(metadata["subject"].unique())
transfer_matrix = transfer_matrix_pairwise(X, y, metadata, subject_ids)
column_means = np.nanmean(transfer_matrix, axis=0)
hardest = subject_ids[int(column_means.argmin())]
easiest = subject_ids[int(column_means.argmax())]
print(
    f"transfer matrix: shape={transfer_matrix.shape} | "
    f"hardest test subject={hardest} | easiest test subject={easiest}"
)

# %% [markdown]
# Step 6. The per-subject accuracy distribution
# -----------------------------------------------
#
# A tiny ASCII histogram. *Spread* matters as much as the mean: a high
# mean with high std means the model works for some subjects and fails
# for others. The MOABB benchmark publishes both numbers for every BCI
# task; treat ``mean - std`` as the lower envelope of what a new
# subject can expect.

# %%
print("Per-subject balanced-accuracy histogram:")
edges = np.linspace(min(fold_acc) - 0.01, max(fold_acc) + 0.01, 6)
for low, high in zip(edges[:-1], edges[1:]):
    n = sum(low <= a < high for a in fold_acc)
    print(f"  [{low:.2f}, {high:.2f}): {'#' * n}")

# %% [markdown]
# Result: one number, one error bar, against chance (E5.43)
# -----------------------------------------------------------

# %%
print(
    f"LOSO balanced accuracy: {mean_loso:.3f} +/- {std_loso:.3f} | "
    f"chance level: {chance_overall:.3f} | metric: balanced_accuracy"
)

# %% [markdown]
# A common mistake, and how to recover
# --------------------------------------
#
# **Run.** The most common slip in a LOSO loop is asking for more folds
# than subjects (``n_folds=20`` on an 8-subject cohort).
# :class:`~sklearn.model_selection.GroupKFold` raises ``ValueError`` --
# catch it and clamp to N.

# %%
try:
    bad = get_splitter(
        "cross_subject",
        n_folds=20,
        n_splits=20,
        random_state=SEED,
    )
    list(k_fold(metadata, splitter=bad, target="target"))
except ValueError as exc:
    print(f"Caught ValueError: {str(exc)[:90]}")
    fixed = get_splitter(
        "cross_subject",
        n_folds=N_SUBJECTS,
        n_splits=N_SUBJECTS,
        random_state=SEED,
    )
    print(
        f"Recovery: clamp n_folds to n_subjects={N_SUBJECTS} -> {type(fixed).__name__}"
    )

# %% [markdown]
# Modify: compare 5-fold cross-subject vs LOSO variance
# -------------------------------------------------------
#
# **Modify.** Drop the fold count from N to 5. The same model, the
# same windows, fewer folds. The mean barely moves; the std almost
# always shrinks because each test fold pools two subjects, averaging
# out the per-subject noise. LOSO is the higher-fidelity variance
# estimate this cohort can give.

# %%
splitter5 = get_splitter(
    "cross_subject",
    n_folds=5,
    n_splits=5,
    random_state=SEED,
)
folds5 = list(k_fold(metadata, splitter=splitter5, target="target"))
assert_no_leakage(folds5, metadata, by="subject")
acc5, _, _, _, _ = loso_loop(X, y, metadata, folds5)
print(
    f"5-fold cross-subject: {acc5.mean():.3f} +/- {acc5.std(ddof=0):.3f} | "
    f"LOSO ({N_SUBJECTS} folds): {mean_loso:.3f} +/- {std_loso:.3f}"
)

# %% [markdown]
# Make: apply the loop to a cohort with imbalanced subjects
# -----------------------------------------------------------
#
# **Make.** Real cohorts rarely have equal trials per subject. Build a
# cohort where subjects contribute different counts, re-run LOSO. The
# contract holds (no subject leakage); the headline ``mean +/- std``
# tells you whether the imbalance hurts generalisation.

# %%
sizes_imb = [20, 30, 30, 40, 50, 50, 60, 80]
X_imb, meta_imb = make_cohort(
    sizes_imb, prefix="imb", rng=np.random.default_rng(SEED + 1)
)
y_imb = meta_imb["target"].to_numpy()
splitter_imb = get_splitter(
    "cross_subject",
    n_folds=len(sizes_imb),
    n_splits=len(sizes_imb),
    random_state=SEED,
)
folds_imb = list(k_fold(meta_imb, splitter=splitter_imb, target="target"))
assert_no_leakage(folds_imb, meta_imb, by="subject")
acc_imb, _, _, _, _ = loso_loop(X_imb, y_imb, meta_imb, folds_imb)
print(
    f"imbalanced LOSO: {acc_imb.mean():.3f} +/- {acc_imb.std(ddof=0):.3f} | "
    f"sizes={sizes_imb}"
)

# %% [markdown]
# Headline figure, transfer matrix, LOSO bars, pooled confusion
# ---------------------------------------------------------------
#
# Three panels read together: panel 1 is the subject x subject transfer
# matrix; panel 2 is the LOSO per-subject accuracy bars sorted worst
# to best with the chance reference line and the ``mean +/- std`` band;
# panel 3 is the pooled confusion matrix from
# :class:`~sklearn.metrics.ConfusionMatrixDisplay` over every held-out
# prediction. The drawing helpers live in a sibling
# ``_cross_subject_figure`` module so the matplotlib geometry stays out
# of this tutorial; the call below is the only line that matters.

# %%
from _cross_subject_figure import draw_cross_subject_figure

fig = draw_cross_subject_figure(
    transfer_matrix=transfer_matrix,
    subject_ids=subject_ids,
    fold_accuracies=fold_acc,
    y_true_pooled=y_true_pooled,
    y_pred_pooled=y_pred_pooled,
    class_names=("class 0", "class 1"),
    held_out_subjects=held_out_subjects,
    chance_level=chance_overall,
    plot_id="plot_51",
)
plt.show()

# %% [markdown]
# **Investigate.** Read the three panels in order.
#
# 1. Transfer matrix: scan column by column. A column that is uniformly
#    pale blue means the held-out subject is hard regardless of the
#    training fold; a column that is uniformly deep blue means an easy
#    subject. Row variation tells you whether one source subject
#    contributes more than the others.
# 2. LOSO bars: is every held-out subject above the chance line, or is
#    the worst subject pulling the mean down? Big across-subject
#    variance is the honest signature of cross-subject EEG.
# 3. Confusion matrix: a clean diagonal in deep blue is the win
#    condition; an off-diagonal stripe means the model has collapsed
#    onto one class on the held-out subjects. The annotation strip
#    below carries the pooled ``balanced_acc`` and the total number of
#    held-out windows.

# %% [markdown]
# Wrap-up
# -------
#
# We built per-subject metadata, asked
# :func:`eegdash.splits.get_splitter` for an N-fold cross-subject
# manifest, asserted zero subject leakage, ran a LOSO loop with
# :class:`~sklearn.linear_model.LogisticRegression`, and reported
# ``mean +/- std`` of :func:`~sklearn.metrics.balanced_accuracy_score`
# against a :func:`~eegdash.splits.majority_baseline` chance level.
# Disjoint test subjects across folds tile the cohort. The transfer
# matrix is the diagnostic a reviewer reaches for when the headline
# mean looks fine but the std is suspicious.

# %% [markdown]
# Try it yourself
# ---------------
#
# - Replace :class:`~sklearn.linear_model.LogisticRegression` with :class:`~sklearn.linear_model.LogisticRegressionCV` (still ``random_state=42``). Does the LOSO std shrink?
# - Reorder ``subject_ids`` in the transfer matrix to put the hardest test subject first. The figure becomes the diagnostic for which subject to investigate next.
# - Swap the synthetic cohort for the windows + manifest you saved in ``plot_11`` and re-run LOSO end-to-end.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
