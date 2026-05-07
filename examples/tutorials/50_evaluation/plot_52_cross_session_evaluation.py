"""Evaluate cross-session drift within the same subjects
=====================================================

Why does a model that scores 90% on session 1 of subject ``sub-03`` slump
on session 2 of the *same* subject? Calibration drift: electrode
placement shifts between days, gel impedance drops, attention
fluctuates, and the time of day reshapes the alpha rhythm. The decoder
inherits all those nuisance factors and the score degrades.

We show that drift on 8 subjects x 3 sessions of mock data: score a
within-session model (train and test inside one session), then swap to
``get_splitter("cross_session")`` and re-score on the same data. The
drop -- per fold and per subject -- quantifies how much of the
within-session score was calibration memorisation rather than paradigm
decoding (Chevallier, Aristimunha et al. 2024,
doi:10.48550/arXiv.2404.15319; Cisotto & Chicco 2024, Tip 9,
doi:10.7717/peerj-cs.2256). So how does calibration drift across
sessions of the same subject?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_52_cross_session_evaluation.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - explain why decoders drift across sessions of the same subject.
# - build a cross-session split with ``get_splitter("cross_session")``.
# - assert that no session appears in both train and test for any
#   ``(subject, fold)`` pair while subjects remain shared.
# - compare within-session accuracy to cross-session accuracy and read the
#   drift delta per subject.
#
# Requirements
# ------------
#
# - You finished
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`.
# - Theory: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup -- seed, imports, mock-data sizes.
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from eegdash.splits import (
    assert_no_leakage,
    describe_split,
    get_splitter,
    majority_baseline,
    make_split_manifest,
)

warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)
N_SUBJECTS, N_SESSIONS, N_WINDOWS = 8, 3, 8

# %% [markdown]
# Step 1 -- Build per-subject per-session metadata
# ------------------------------------------------
#
# 8 subjects x 3 sessions x 8 windows = 192 windows. Features carry a
# subject-specific cluster (a "neural fingerprint" the within-session
# model rides) plus a per-session shift (the calibration drift). Labels
# are paradigm-driven, so a cross-session model must ignore the shift.

# %%
rng = np.random.default_rng(42)
subj_off = rng.normal(scale=2.0, size=(N_SUBJECTS, 4))
ses_drift = rng.normal(scale=1.5, size=(N_SUBJECTS, N_SESSIONS, 4))
rows, features = [], []
for s in range(N_SUBJECTS):
    for ses in range(N_SESSIONS):
        for w in range(N_WINDOWS):
            label = (s + w) % 2
            paradigm = np.array([label * 1.5, -label * 1.5, 0.0, 0.0])
            features.append(
                subj_off[s]
                + ses_drift[s, ses]
                + paradigm
                + rng.normal(scale=0.5, size=4)
            )
            rows.append(
                {
                    "subject": f"sub-{s:02d}",
                    "session": f"ses-{ses + 1:02d}",
                    "run": "run-01",
                    "dataset": "ds-mock-cross-session",
                    "sample_id": f"sub-{s:02d}__ses-{ses + 1:02d}__w{w:03d}",
                    "target": int(label),
                }
            )
metadata, X = pd.DataFrame(rows), np.asarray(features)
y = metadata["target"].to_numpy()
print(
    f"Windows: rows={len(metadata)}, subjects={metadata['subject'].nunique()}, "
    f"sessions/subject={metadata.groupby('subject')['session'].nunique().min()}, "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# Step 2 -- Predict, then build the cross-session split
# -----------------------------------------------------
#
# **Predict.** How should ``session_overlap`` differ from
# ``subject_overlap`` here, compared with ``plot_11``'s cross-subject
# split? ``plot_11`` forbade subject sharing. Here we forbid sharing a
# session but *want* the held-out subject in train (calibration is
# per-subject). MOABB loops leave-one-session-out *inside each subject*:
# 8 x 3 = 24 folds, one subject on both sides, only the held-out
# session in test.
#
# **Run.** ``make_split_manifest`` freezes the splitter output as IDs.

# %%
splitter = get_splitter("cross_session", n_folds=3, n_splits=3, random_state=42)
manifest = make_split_manifest(splitter, y, metadata, target="target")
print(f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']}")

# %% [markdown]
# Step 3 -- Assert no session leakage, confirm subjects are shared
# ----------------------------------------------------------------
#
# ``assert_no_leakage(..., by="session")`` walks every fold and prints the
# JSON ``leakage_report`` line consumed by runtime validator E5.42. We
# then assert the *opposite* invariant for ``subject``: every fold must
# reuse the same subject on both sides -- this is what makes the split
# "within-subject, across-session" rather than cross-subject.

# %%
session_overlap = assert_no_leakage(manifest, metadata, by="session")
assert session_overlap == 0, "cross_session manifest leaked a session!"
fold_subj_overlaps = []
for fold in manifest["folds"]:
    tr_s = set(metadata.loc[metadata["sample_id"].isin(fold["train"]), "subject"])
    te_s = set(metadata.loc[metadata["sample_id"].isin(fold["test"]), "subject"])
    fold_subj_overlaps.append(len(tr_s & te_s))
assert min(fold_subj_overlaps) == 1, "every fold must keep the same subject"
print(
    f"session_overlap per fold = 0 (asserted); "
    f"subject_overlap per fold = {fold_subj_overlaps[0]} across "
    f"{len(fold_subj_overlaps)} folds"
)

# %% [markdown]
# Step 4 -- Run within-session and cross-session on the SAME data
# ---------------------------------------------------------------
#
# **Run (#2).** Two evaluations on the identical ``(X, y, metadata)``:
# (1) within-session, 75/25 split per ``(subject, session)`` block
# (calibration ceiling); (2) cross-session, the 24 manifest folds.
#
# **Investigate.** The drop is calibration drift: the part of the
# within-session score that came from memorising the day-specific feature
# distribution rather than the paradigm.


# %%
def _fit_score(tr, te):
    clf = LogisticRegression(random_state=42, max_iter=200).fit(X[tr], y[tr])
    return accuracy_score(y[te], clf.predict(X[te]))


within_scores, cross_scores = [], []
drift: dict[str, dict] = {
    s: {"within": 0.0, "cross": []} for s in metadata["subject"].unique()
}
for subj, sub_md in metadata.groupby("subject"):
    sub_w = []
    for _, ses_md in sub_md.groupby("session"):
        idx = ses_md.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(0.75 * len(idx))
        sub_w.append(_fit_score(idx[:cut], idx[cut:]))
    drift[subj]["within"] = float(np.mean(sub_w))
    within_scores.extend(sub_w)

for fold in manifest["folds"]:
    tr = np.where(metadata["sample_id"].isin(fold["train"]).to_numpy())[0]
    te = np.where(metadata["sample_id"].isin(fold["test"]).to_numpy())[0]
    s = _fit_score(tr, te)
    cross_scores.append(s)
    drift[metadata.iloc[te[0]]["subject"]]["cross"].append(s)

within_acc, cross_acc = float(np.mean(within_scores)), float(np.mean(cross_scores))
chance = majority_baseline(y_train=y, y_test=y)["chance_level"]
print(
    f"within-session accuracy = {within_acc:.3f}, "
    f"cross-session accuracy = {cross_acc:.3f}, "
    f"chance = {chance:.3f}, drift_delta = {within_acc - cross_acc:+.3f}"
)

# %% [markdown]
# Step 5 -- describe_split shows the per (subject, session) audit
# ---------------------------------------------------------------
#
# **Run (#3).** ``describe_split`` reports per-fold sample/subject/session
# counts and class balance. We print the first three of 24 folds: each
# holds one subject in train (2 sessions) and the same subject in test
# (1 session), with balanced classes.

# %%
summary = describe_split(manifest, metadata, target="target", print_report=False)
for i, fold in enumerate(summary["per_fold"][:3]):
    bal = fold["class_balance_test"]
    ratio = max(bal.values()) / (sum(bal.values()) or 1)
    print(
        f"Fold {i}: train={fold['n_train']} ({fold['sessions_train']} ses), "
        f"test={fold['n_test']} ({fold['sessions_test']} ses), "
        f"class_balance_ratio={ratio:.2f}"
    )

# %% [markdown]
# Result -- calibration drift is real
# -----------------------------------
#
# Subject overlap is 1 per fold by design; session_overlap is 0 (E5.42
# reports it). Within-session > cross-session > chance, and the gap is
# drift you would see on real multi-session BIDS data too.

# %%
ranked = sorted(
    (
        (s, d["within"], float(np.mean(d["cross"])) if d["cross"] else float("nan"))
        for s, d in drift.items()
    ),
    key=lambda r: -(r[1] - r[2]),
)[:3]
print(f"Top 3 drift subjects (within, cross): {ranked}")

# %% [markdown]
# A common slip -- and how to recover
# -----------------------------------
# **Run.** Calling ``assert_no_leakage`` with the default ``by="subject"``
# on a cross-session manifest looks like a leak (subjects are shared on
# purpose). Recovery: pass ``by="session"``.

# %%
try:
    assert_no_leakage(manifest, metadata)  # default by="subject" => positive overlap
    raise AssertionError("expected a LeakageError")
except Exception as exc:
    print(f"Caught {type(exc).__name__}: retrying with by='session'.")
    print(f"Recovery overlap = {assert_no_leakage(manifest, metadata, by='session')}")

# %% [markdown]
# Modify -- try a 2-session subject
# ---------------------------------
#
# **Modify.** Drop one session from ``sub-00`` and re-run. MOABB
# contributes 2 folds for *that* subject (LOSO over 2 sessions) and 3
# folds for the other 7, giving 23 in total -- a reminder that fold count
# depends on each subject's session inventory.

# %%
keep = ~((metadata["subject"] == "sub-00") & (metadata["session"] == "ses-03"))
trimmed_md = metadata.loc[keep].reset_index(drop=True)
trimmed_y = trimmed_md["target"].to_numpy()
trimmed_man = make_split_manifest(
    get_splitter("cross_session", n_folds=3, random_state=42),
    trimmed_y,
    trimmed_md,
    target="target",
)
print(
    f"Trimmed: {trimmed_man['n_folds']} folds (was {manifest['n_folds']}), "
    f"min sessions/subject="
    f"{trimmed_md.groupby('subject')['session'].nunique().min()}"
)

# %% [markdown]
# Try it yourself
# ---------------
#
# - vary ``random_state`` and confirm session disjointness still holds.
# - increase per-session drift in the mock data and watch the delta widen.
# - swap to a real BIDS dataset with ``description['session']`` set.
#
# Links
# -----
#
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - API: ``get_splitter``, ``make_split_manifest``, ``assert_no_leakage``,
#   ``describe_split``, ``majority_baseline``.
# - Chevallier, Aristimunha et al. 2024 (doi:10.48550/arXiv.2404.15319) --
#   MOABB benchmark; cross-session evaluation and calibration drift.
# - Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256) -- Tip 9.
