"""Evaluate cross-subject generalisation with leave-subjects-out
==============================================================

How well does the model generalise to a never-seen-before subject?
That single question is the gold standard for any decoding claim and
the de-facto protocol of the MOABB benchmark (Chevallier, Aristimunha
et al. 2024, doi:10.48550/arXiv.2404.15319). Where ``plot_11`` proved a
single split is leakage-free and ``plot_12`` trained one model on one
split, this tutorial steps up to the actual evaluation: a 5-fold
cross-subject loop that holds a different group of subjects out each
time and reports ``mean +/- std`` across folds against a chance level
(Cisotto & Chicco 2024 Tip 9, doi:10.7717/peerj-cs.2256). One score
from one fold is a point estimate; five scores from five disjoint
subject groups is what benchmark tables actually publish. So how big
is the spread once we move from one held-out subject to a fold?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_51_cross_subject_evaluation.png'

# %% [markdown]
# Learning objectives
# -------------------
# - explain why cross-subject evaluation is the gold standard for generalisation.
# - build a 5-fold cross_subject split with ``get_splitter`` + ``assert_no_leakage``.
# - compute ``mean +/- std`` across folds against ``majority_baseline`` chance.
# - describe each fold's test cohort using ``describe_split``.
#
# Requirements
# ------------
# - Prereqs:
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`
#   and ``plot_12_train_a_baseline`` (one model on one split).
# - Theory: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup -- seed (E3.21) and imports.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    describe_split,
    get_splitter,
    majority_baseline,
    make_split_manifest,
)
from eegdash.viz import use_eegdash_style

use_eegdash_style()
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)


# %% [markdown]
# Step 1 -- Build per-subject metadata for >= 8 subjects
# ------------------------------------------------------
#
# We materialise a synthetic table for 10 subjects with a 2-D feature
# carrying class signal plus a per-subject offset ("subject fingerprint").
# ``eegdash.splits`` accepts a ``WindowsConcatDataset`` or this DataFrame.


# %%
def make_cohort(sizes, prefix, rng):
    """Return ``(X, metadata)`` for a synthetic cross-subject toy task."""
    rows, X_list = [], []
    for s, n_w in enumerate(sizes):
        labels = rng.integers(0, 2, size=n_w)
        bias = 0.05 * s
        for w, lab in enumerate(labels):
            base = bias + rng.standard_normal(2) * 0.7
            X_list.append([float(lab) + base[0], -float(lab) + base[1]])
            rows.append(
                {
                    "sample_id": f"{prefix}-{s:02d}__w{w:03d}",
                    "subject": f"{prefix}-sub-{s:02d}",
                    "session": "ses-01",
                    "run": "run-01",
                    "dataset": f"ds-{prefix}",
                    "target": int(lab),
                }
            )
    return np.asarray(X_list, dtype=float), pd.DataFrame(rows)


N_SUBJECTS = 10
X, metadata = make_cohort([30] * N_SUBJECTS, "bal", rng)
y = metadata["target"].to_numpy()
print(
    f"rows={len(metadata)} subjects={metadata['subject'].nunique()} "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# Step 2 -- Predict the subject overlap before vs after
# -----------------------------------------------------
#
# **Predict.** If we hold out 2 of the 10 subjects, fit on the 8 others,
# and score on the held-out 2, how many subject IDs will appear on both
# sides of the split -- 0, around 5, or all 10? Write your guess.
#
# **Run.** A single ``cross_subject`` fold and the before/after count.

# %%
splitter = get_splitter(
    "cross_subject", engine="sklearn", n_folds=5, n_splits=5, random_state=SEED
)
manifest = make_split_manifest(splitter, y, metadata, target="target")
fold0 = manifest["folds"][0]
train_subj = set(metadata[metadata["sample_id"].isin(fold0["train"])]["subject"])
test_subj = set(metadata[metadata["sample_id"].isin(fold0["test"])]["subject"])
print(
    f"subject_overlap before (window-shuffle baseline): {N_SUBJECTS}; "
    f"after cross_subject (fold 0): {len(train_subj & test_subj)}"
)

# %% [markdown]
# Step 3 -- Assert no leakage on every fold
# -----------------------------------------
#
# **Run (#2).** ``get_splitter`` here returns sklearn's ``GroupKFold``
# keyed on ``subject`` (MOABB's CrossSubjectSplitter is also available
# via ``engine="moabb"``). ``make_split_manifest`` freezes splitter +
# kwargs + library versions + per-fold sample IDs into JSON.
# ``assert_no_leakage(by="subject")`` walks every fold and emits the
# ``leakage_report`` line consumed by E5.42.

# %%
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "cross-subject manifest leaked subjects"
print(
    f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']} | "
    f"max subject overlap: {overlap}"
)

# %% [markdown]
# Step 4 -- Train one baseline per fold; report mean +/- std
# ----------------------------------------------------------
#
# Loop over folds, materialise each split via ``apply_split_manifest``,
# fit ``LogisticRegression(random_state=42)`` (plot_12's baseline),
# score with ``accuracy_score``. The headline is ``mean +/- std`` plus
# a ``majority_baseline`` chance level (E5.43 forbids naked points).


# %%
def cross_subject_loop(X, y, meta, manifest):
    """Return per-fold accuracies and chance levels."""
    accs, chances = [], []
    for k in range(manifest["n_folds"]):
        tr = apply_split_manifest(meta, manifest, fold=k, split="train")
        te = apply_split_manifest(meta, manifest, fold=k, split="test")
        clf = LogisticRegression(random_state=SEED, max_iter=200).fit(X[tr], y[tr])
        accs.append(float(accuracy_score(y[te], clf.predict(X[te]))))
        chances.append(float(majority_baseline(y[tr], y[te])["chance_level"]))
    return accs, chances


fold_acc, fold_chance = cross_subject_loop(X, y, metadata, manifest)
for k, (a, c) in enumerate(zip(fold_acc, fold_chance)):
    print(f"Fold {k}: acc={a:.3f} | chance={c:.3f}")
mean_acc = float(np.mean(fold_acc))
std_acc = float(np.std(fold_acc, ddof=1))
mean_chance = float(np.mean(fold_chance))

# %% [markdown]
# Step 5 -- ``describe_split``: each fold's test set has DIFFERENT subjects
# -------------------------------------------------------------------------
#
# **Run (#3).** Each fold's *test* subject set differs from every other
# fold's; together they tile the cohort. That is the cross-subject contract.

# %%
describe_split(manifest, metadata, target="target", print_report=False)
test_subjects_by_fold = []
for k, fold in enumerate(manifest["folds"]):
    subs = sorted(
        metadata[metadata["sample_id"].isin(fold["test"])]["subject"].unique()
    )
    test_subjects_by_fold.append(subs)
    print(f"Fold {k}: test_subjects={subs}")
print(
    f"Union across folds: {len(set().union(*test_subjects_by_fold))} "
    f"(cohort = {N_SUBJECTS})"
)

# %% [markdown]
# Step 6 -- Investigate the per-fold accuracy distribution
# --------------------------------------------------------
#
# **Investigate.** A tiny ASCII histogram. *Spread* matters as much as
# the mean: a high mean with high std means the model works for some
# subjects and fails for others -- the signal a cross-subject loop
# surfaces. This is the evaluation MOABB benchmarks publish.

# %%
print("Per-fold accuracy histogram:")
edges = np.linspace(min(fold_acc) - 0.01, max(fold_acc) + 0.01, 6)
for low, high in zip(edges[:-1], edges[1:]):
    n = sum(low <= a < high for a in fold_acc)
    print(f"  [{low:.2f}, {high:.2f}): {'#' * n}")

# %% [markdown]
# Result -- one number, one error bar, against chance (E5.43)
# -----------------------------------------------------------

# %%
print(
    f"Cross-subject 5-fold accuracy: {mean_acc:.3f} +/- {std_acc:.3f} | "
    f"chance level: {mean_chance:.3f} | metric: accuracy"
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
# **Run.** Asking for more folds than subjects (``n_folds=20`` on a
# 10-subject cohort) is the most common slip in a benchmark loop.
# ``GroupKFold`` raises ``ValueError`` -- catch it and clamp.

# %%
try:
    bad = get_splitter(
        "cross_subject", engine="sklearn", n_folds=20, n_splits=20, random_state=SEED
    )
    make_split_manifest(bad, y, metadata, target="target")
except ValueError as exc:
    print(f"Caught ValueError: {str(exc)[:90]}")
    fixed = get_splitter(
        "cross_subject",
        engine="sklearn",
        n_folds=N_SUBJECTS,
        n_splits=N_SUBJECTS,
        random_state=SEED,
    )
    print(
        f"Recovery: clamp n_folds to n_subjects={N_SUBJECTS} -> {type(fixed).__name__}"
    )

# %% [markdown]
# Modify -- try ``n_folds=10`` for finer-grained variance
# -------------------------------------------------------
#
# **Modify.** Bump folds from 5 to 10. With 10 subjects this becomes
# leave-one-subject-out: each test fold sees one subject, so the
# variance estimate is the highest fidelity this cohort can give. The
# mean usually moves only a little; the std almost always grows.

# %%
splitter10 = get_splitter(
    "cross_subject", engine="sklearn", n_folds=10, n_splits=10, random_state=SEED
)
manifest10 = make_split_manifest(splitter10, y, metadata, target="target")
assert_no_leakage(manifest10, metadata, by="subject")
acc10, _ = cross_subject_loop(X, y, metadata, manifest10)
print(
    f"n_folds=10: {np.mean(acc10):.3f} +/- {np.std(acc10, ddof=1):.3f} | "
    f"5-fold: {mean_acc:.3f} +/- {std_acc:.3f}"
)

# %% [markdown]
# Make -- apply the loop to a cohort with imbalanced subjects
# -----------------------------------------------------------
#
# **Make.** Real cohorts rarely have equal trials per subject. Build a
# cohort where subjects contribute 10 to 60 windows, re-run the same
# 5-fold loop. The contract still holds (no subject leakage); the
# ``mean +/- std`` tells you whether the imbalance hurts generalisation.

# %%
sizes_imb = [10, 10, 20, 20, 30, 30, 40, 60]
X2, meta2 = make_cohort(sizes_imb, "imb", rng)
y2 = meta2["target"].to_numpy()
splitter2 = get_splitter(
    "cross_subject", engine="sklearn", n_folds=5, n_splits=5, random_state=SEED
)
manifest2 = make_split_manifest(splitter2, y2, meta2, target="target")
assert_no_leakage(manifest2, meta2, by="subject")
acc2, _ = cross_subject_loop(X2, y2, meta2, manifest2)
print(
    f"imbalanced 5-fold: {np.mean(acc2):.3f} +/- {np.std(acc2, ddof=1):.3f} | "
    f"sizes={sizes_imb}"
)

# %% [markdown]
# Wrap-up
# -------
# We built per-subject metadata, asked ``get_splitter`` for a 5-fold
# cross-subject manifest, asserted zero subject leakage, trained a
# logistic baseline per fold, and reported ``mean +/- std`` against a
# ``majority_baseline`` chance level. Disjoint test subjects across
# folds tile the cohort -- the headline of any cross-subject paper.

# %% [markdown]
# Try it yourself
# ---------------
# - Replace ``LogisticRegression`` with ``LogisticRegressionCV`` (still ``random_state=42``).
# - Pass ``stratified=True`` to a ``GroupKFold`` fallback and re-check class balance.
# - Swap the synthetic cohort for the windows + manifest you saved in plot_11.

# %% [markdown]
# Links
# -----
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - API: ``eegdash.splits.get_splitter``, ``make_split_manifest``,
#   ``apply_split_manifest``, ``assert_no_leakage``, ``describe_split``,
#   ``majority_baseline``.
# - Chevallier, Aristimunha et al. 2024, MOABB benchmark
#   (https://doi.org/10.48550/arXiv.2404.15319).
# - Cisotto & Chicco 2024, Ten quick tips, *PeerJ Comp. Sci.*
#   (https://doi.org/10.7717/peerj-cs.2256) -- Tip 9.
