"""Within-subject decoding with k-fold splits
==============================================

When is staying inside a subject the right scientific question? Cross-subject
generalisation is the headline of plot_11, but plenty of EEG paradigms care
about a single brain at a time -- a calibration-style P300 speller, a clinical
seizure detector tuned to one patient, or a lab paradigm where inter-subject
variance dominates the signal you want to study. In those cases ``subject``
overlap between train and test is *intentional*, not a bug. We build a 5-fold
within-subject split, prove it is trial-disjoint, and quote per-subject
accuracy against ``majority_baseline`` chance level. Compare to
Chevallier, Aristimunha et al. 2024 (doi:10.48550/arXiv.2404.15319) and the
MOABB ``WithinSubjectSplitter`` API. So when is within-subject evaluation
the right call?
"""

# %% [markdown]
# ## Learning objectives
#
# - identify when within-subject evaluation is appropriate (calibration decoders, single-subject diagnostics, high inter-subject variance).
# - build a 5-fold within-subject manifest with ``get_splitter("within_subject", n_folds=5)``.
# - read ``describe_split`` and recognise that ``subject_overlap=1`` is the design, not a leak.
# - assert no trial overlap with ``assert_no_leakage`` and verify the JSON ``leakage_report`` line.
# - compare per-subject accuracy against ``majority_baseline`` chance level.
#
# ## Requirements
#
# - You finished plot_11_leakage_safe_split and plot_12_train_a_baseline.
# - CPU only, runtime ~3 minutes.

# %%
# Setup -- seed and imports.
import json
import warnings

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

warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)

# %% [markdown]
# ## Step 1 -- Build a per-subject windowed metadata table
#
# We mock 12 subjects x 2 sessions x 8 windows = 192 rows. Each row carries
# a ``trial`` identifier (one trial per window here) and a synthetic class
# label. ``eegdash.splits`` accepts either a Braindecode ``WindowsDataset``
# or a DataFrame with the same columns; sticking to a DataFrame keeps the
# split discipline isolated from any I/O.

# %%
rng = np.random.default_rng(42)
rows = []
for s in range(12):
    for ses in range(2):
        for w in range(8):
            sample_id = f"sub-{s:02d}__ses-{ses:02d}__w{w:03d}"
            rows.append(
                {
                    "subject": f"sub-{s:02d}",
                    "session": f"ses-{ses:02d}",
                    "run": "run-01",
                    "dataset": "ds-within-tutorial",
                    "sample_id": sample_id,
                    "trial": sample_id,  # one trial per window
                    "target": int((s + w) % 2),
                }
            )
metadata = pd.DataFrame(rows)
n_features = 8
features = rng.normal(size=(len(metadata), n_features))
# Subject-specific bias amplifies inter-subject variance, the failure mode that
# motivates within-subject evaluation in the first place.
subject_bias = rng.normal(size=(metadata["subject"].nunique(), n_features))
subject_index = metadata["subject"].astype("category").cat.codes.to_numpy()
features += subject_bias[subject_index]
# Class-conditional shift so a per-subject classifier can actually learn.
features[metadata["target"].to_numpy() == 1, 0] += 0.7
print(
    f"Windows metadata: rows={len(metadata)}, "
    f"subjects={metadata['subject'].nunique()}, "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# ## Step 2 -- Predict the right invariant
#
# **Predict.** In a *cross*-subject split (plot_11) we wanted
# ``subject_overlap == 0`` -- a subject must never sit on both sides. For
# *within-subject* evaluation, what should ``subject_overlap`` be in every
# fold: 0, 1, or all 12? Pick before scrolling.

# %% [markdown]
# Answer: ``1``. Every fold trains and tests on the *same* subject; we are
# answering "does this decoder calibrate to subject X?", not "does it
# generalise to a new person". The leak we still police is on ``trial`` --
# no individual window may appear in both train and test.

# %% [markdown]
# ## Step 3 -- Build the 5-fold within-subject manifest
#
# **Run.** ``get_splitter("within_subject", n_folds=5, random_state=42)``
# returns MOABB's ``WithinSubjectSplitter`` (or a sklearn ``GroupKFold``
# fallback keyed on ``subject``). It draws a fresh 5-fold split *inside each
# subject* -- so for 12 subjects you get 12 x 5 = 60 fold pairs.
# ``make_split_manifest`` freezes them to a JSON-serialisable dict.

# %%
splitter = get_splitter("within_subject", n_folds=5, random_state=42)
manifest = make_split_manifest(
    splitter, metadata["target"].to_numpy(), metadata, target="target"
)
print(
    f"Splitter: {manifest['splitter_class']} | "
    f"n_folds (subj x folds): {manifest['n_folds']}"
)

# %% [markdown]
# ## Step 4 -- Assert no trial leakage and read the audit
#
# ``assert_no_leakage(by="trial")`` walks every fold and intersects the
# ``trial`` values across train/test. It always emits the JSON line
# ``{"leakage_report": {"overlap": 0, "by": "trial"}}`` (E5.42). A
# *subject*-level leakage assertion would fail by design -- that is the
# whole point of this evaluation.

# %%
trial_overlap = assert_no_leakage(manifest, metadata, by="trial")
assert trial_overlap == 0, "Within-subject manifest reused a trial across folds!"
summary = describe_split(manifest, metadata, target="target", print_report=False)
fold0 = summary["per_fold"][0]
print(
    f"Fold 0: train={fold0['n_train']} ({fold0['subjects_train']} subj), "
    f"test={fold0['n_test']} ({fold0['subjects_test']} subj), "
    f"classes_test={fold0['class_balance_test']}"
)
# Confirm subject_overlap == 1 is the design.
overlapping_subjects = []
for fold_record in manifest["folds"]:
    train_sub = set(
        metadata[metadata["sample_id"].isin(fold_record["train"])]["subject"]
    )
    test_sub = set(metadata[metadata["sample_id"].isin(fold_record["test"])]["subject"])
    overlapping_subjects.append(len(train_sub & test_sub))
print(
    f"Subject overlap per fold (intentional): min={min(overlapping_subjects)}, "
    f"max={max(overlapping_subjects)}"
)

# %% [markdown]
# **Investigate.** Every fold has exactly one subject in train and the same
# subject in test, while the trial intersection is empty. ``describe_split``
# confirms balanced classes per fold (the synthetic table is 50/50 by
# construction; on real EEG you would inspect ``class_balance_test`` here).
# Cisotto & Chicco 2024 Tip 9 calls this the "calibration regime": valid as
# long as you state the intent and never extrapolate the score to a new
# subject.

# %% [markdown]
# ## Step 5 -- Train per-subject and quote chance level
#
# **Run.** For each fold we materialise train/test masks, fit a
# ``LogisticRegression(random_state=42)``, and aggregate accuracy across
# folds. ``majority_baseline`` returns the chance level from the *test*
# class proportions -- E5.43 requires every reported accuracy to ship with
# its chance line.

# %%
fold_scores = []
fold_chance = []
for fold_index in range(manifest["n_folds"]):
    train_mask = apply_split_manifest(
        metadata, manifest, fold=fold_index, split="train"
    )
    test_mask = apply_split_manifest(metadata, manifest, fold=fold_index, split="test")
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        continue
    y_train = metadata.loc[train_mask, "target"].to_numpy()
    y_test = metadata.loc[test_mask, "target"].to_numpy()
    if len(np.unique(y_train)) < 2:
        continue  # degenerate fold, skip
    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(features[train_mask], y_train)
    fold_scores.append(accuracy_score(y_test, clf.predict(features[test_mask])))
    fold_chance.append(majority_baseline(y_train, y_test)["chance_level"])
mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
mean_chance = float(np.mean(fold_chance)) if fold_chance else float("nan")

# %% [markdown]
# ## Result -- accuracy with chance level
#
# Within-subject mean accuracy across 12 subjects x 5 folds compared to the
# majority-class chance level (E5.43). The score is per-subject by
# construction; it does **not** generalise to a new participant -- use
# ``cross_subject`` from plot_11 for that question.

# %%
print(
    f"Within-subject result: mean accuracy={mean_score:.3f}, "
    f"chance={mean_chance:.3f}, n_evaluated_folds={len(fold_scores)}/{manifest['n_folds']}"
)
print(
    "Final invariants:",
    json.dumps(
        {
            "n_subjects": int(metadata["subject"].nunique()),
            "n_folds": int(manifest["n_folds"]),
            "trial_overlap": int(trial_overlap),
            "subject_overlap_per_fold": int(max(overlapping_subjects)),
            "mean_accuracy": round(mean_score, 3),
            "mean_chance_level": round(mean_chance, 3),
        }
    ),
)

# %% [markdown]
# ## Modify -- swap to within-session
#
# **Modify.** What if you suspect session-day effects (caffeine, electrode
# drift) dominate within a subject? Swap ``within_subject`` for
# ``within_session``: the splitter then iterates k-fold *inside each
# session*, so subject *and* session overlap by design while trials stay
# disjoint.

# %%
session_splitter = get_splitter("within_session", n_folds=4, random_state=42)
session_manifest = make_split_manifest(
    session_splitter, metadata["target"].to_numpy(), metadata, target="target"
)
session_overlap = assert_no_leakage(session_manifest, metadata, by="trial")
print(
    f"within_session manifest: n_folds={session_manifest['n_folds']}, "
    f"trial_overlap={session_overlap}"
)

# %% [markdown]
# ## Try it yourself -- Extensions
#
# - bump ``n_folds`` to 10 and watch per-fold test sizes shrink.
# - re-run with ``random_state=7`` and confirm trial disjointness still holds.
# - replace the synthetic features with windows from plot_10 + plot_40 and
#   plot per-subject accuracy as a bar chart, sorted by score.

# %% [markdown]
# ## Links
#
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - API: ``get_splitter``, ``majority_baseline``, ``assert_no_leakage``.
# - Chevallier, Aristimunha et al. 2024 (doi:10.48550/arXiv.2404.15319) --
#   MOABB benchmark, within- vs cross-subject evaluation.
# - Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256) -- ten quick tips for
#   clinical EEG (Tip 9 on evaluation).
# - MOABB plotting tutorials: ``score_plot``, ``paired_plot``,
#   ``meta_analysis_plot`` (moabb.neurotechx.com/docs/api.html).
