"""How much does a within-session decoder drift across sessions of the same subject?
====================================================================================

A model that scores 90% on session 1 of ``sub-03`` can slump to 70% on
session 2 of the *same* subject. The cap and electrodes were placed a
millimetre off, the gel impedance dropped, the participant slept poorly,
or the recording fell at the wrong time of day. The decoder inherits
all of those nuisance factors and the score drops the next morning.
This is calibration drift: a covariate shift in the feature distribution
that the within-session score never had to handle (Jayaram & Barachant
2018, doi:10.1088/1741-2552/aabea9).

We measure that drift on 8 subjects x 3 sessions of the synthetic mock
cohort that ships with EEGDash. We score a within-session model
(train and test inside one session, the calibration ceiling), then swap
to ``get_splitter("cross_session")`` and re-score on the same data.
The drop, per fold and per subject, quantifies how much of the
within-session score was calibration memorisation rather than paradigm
decoding (Chevallier, Aristimunha et al. 2024,
doi:10.48550/arXiv.2404.15319; Cisotto & Chicco 2024 Tip 9,
doi:10.7717/peerj-cs.2256). EEGDash sources the same metadata as
NEMAR (Delorme et al. 2022, doi:10.1162/imag_a_00026), so the same
contract applies on real BIDS data with ``description['session']`` set.
Riemannian alignment of the per-session covariance matrices is one
documented recovery for severe drift; here we just measure it.

The polished version of this tutorial uses ``plot_52``'s sibling helper
``draw_cross_session_figure`` to render a 1 x 3 figure: a session x
session transfer matrix, paired within-vs-between bars per subject,
and a drift-magnitude histogram across subjects.

So how much does a within-session decoder drift across sessions of the
same subject?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_52_cross_session_evaluation.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - Explain why decoders drift across sessions of the same subject (electrode placement, impedance, attention, time of day).
# - Build a cross-session split with :func:`eegdash.splits.get_splitter` (``"cross_session"``) keyed on the BIDS ``session`` entity.
# - Assert that no session appears in both train and test for any ``(subject, fold)`` pair while subjects remain shared by design.
# - Compare within-session accuracy to cross-session accuracy and read the per-subject drift delta against :func:`eegdash.splits.majority_baseline`.
# - Plot the 1 x 3 ``draw_cross_session_figure`` on the live numbers (transfer matrix, paired bars, drift histogram).
#
# Requirements
# ------------
#
# - You finished :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`, :doc:`/auto_examples/tutorials/10_core_workflow/plot_12_train_a_baseline`, and :doc:`/auto_examples/tutorials/50_evaluation/plot_50_within_subject_evaluation`.
# - Theory: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup, seed (E3.21), imports, mock-cohort sizing.
import warnings

import matplotlib.pyplot as plt
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
from eegdash.viz import use_eegdash_style

from _cross_session_figure import draw_cross_session_figure

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
N_SUBJECTS, N_SESSIONS, N_WINDOWS = 8, 3, 8

# %% [markdown]
# Step 1. Build per-subject per-session metadata
# ------------------------------------------------
#
# 8 subjects x 3 sessions x 8 windows = 192 windows. Features carry a
# subject-specific cluster (a "neural fingerprint" the within-session
# model rides) plus a per-session shift (the calibration drift). Labels
# are paradigm-driven by construction, so a cross-session model must
# ignore the shift and key on the paradigm signal.

# %%
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
    f"Windows: rows={len(metadata)}, "
    f"subjects={metadata['subject'].nunique()}, "
    f"sessions/subject={metadata.groupby('subject')['session'].nunique().min()}, "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# Step 2. Predict, then build the cross-session split
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
splitter = get_splitter("cross_session", n_folds=3, n_splits=3, random_state=SEED)
manifest = make_split_manifest(splitter, y, metadata, target="target")
print(f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']}")

# %% [markdown]
# Step 3. Assert no session leakage; confirm subjects are shared
# ----------------------------------------------------------------
#
# ``assert_no_leakage(..., by="session")`` walks every fold and prints
# the JSON ``leakage_report`` line consumed by runtime validator E5.42.
# We then assert the *opposite* invariant for ``subject``: every fold
# must reuse the same subject on both sides. That keeps the split
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
# Step 4. Run within-session and cross-session on the SAME data
# ---------------------------------------------------------------
#
# **Run (#2).** Two evaluations on the identical ``(X, y, metadata)``:
# (1) within-session, 75/25 split per ``(subject, session)`` block
# (calibration ceiling); (2) cross-session, the 24 manifest folds.
#
# **Investigate.** The drop is calibration drift: the part of the
# within-session score that came from memorising the day-specific
# feature distribution rather than the paradigm. Jayaram & Barachant
# 2018 (doi:10.1088/1741-2552/aabea9) frame the same gap as covariate
# shift between calibration sessions; Riemannian alignment of the
# per-session covariance matrices is one documented recovery on
# motor-imagery cohorts. We just measure the gap here.


# %%
def _fit_score(tr_idx: np.ndarray, te_idx: np.ndarray) -> float:
    """Fit a logistic baseline on ``tr_idx`` and score on ``te_idx``."""
    clf = LogisticRegression(random_state=SEED, max_iter=200).fit(X[tr_idx], y[tr_idx])
    return float(accuracy_score(y[te_idx], clf.predict(X[te_idx])))


within_scores: list[float] = []
cross_scores: list[float] = []
within_per_subject: dict[str, list[float]] = {}
cross_per_subject: dict[str, list[float]] = {}
for subj, sub_md in metadata.groupby("subject"):
    sub_w: list[float] = []
    for _, ses_md in sub_md.groupby("session"):
        idx = ses_md.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(0.75 * len(idx))
        sub_w.append(_fit_score(idx[:cut], idx[cut:]))
    within_per_subject[str(subj)] = sub_w
    within_scores.extend(sub_w)

for fold in manifest["folds"]:
    tr = np.where(metadata["sample_id"].isin(fold["train"]).to_numpy())[0]
    te = np.where(metadata["sample_id"].isin(fold["test"]).to_numpy())[0]
    score = _fit_score(tr, te)
    cross_scores.append(score)
    test_subject = str(metadata.iloc[te[0]]["subject"])
    cross_per_subject.setdefault(test_subject, []).append(score)

within_acc = float(np.mean(within_scores))
cross_acc = float(np.mean(cross_scores))
chance = float(majority_baseline(y_train=y, y_test=y)["chance_level"])
print(
    f"within-session accuracy = {within_acc:.3f}, "
    f"cross-session accuracy = {cross_acc:.3f}, "
    f"chance = {chance:.3f}, drift_delta = {within_acc - cross_acc:+.3f}"
)

# %% [markdown]
# Step 5. ``describe_split`` shows the per-(subject, session) audit
# -------------------------------------------------------------------
#
# **Run (#3).** ``describe_split`` reports per-fold sample, subject,
# session counts and class balance. We print the first three of 24
# folds: each holds one subject in train (2 sessions) and the same
# subject in test (1 session), with balanced classes.

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
# Step 6. Build the session x session transfer matrix
# -----------------------------------------------------
#
# The aggregate numbers say drift is real; the next step is to read
# *which* session pair drifts the most. We score every
# ``(train_session, test_session)`` pair, averaged across subjects.
# The diagonal carries the within-session ceiling; the off-diagonal
# carries the cross-session score under drift.

# %%
session_ids = sorted(metadata["session"].unique())
session_matrix = np.zeros((N_SESSIONS, N_SESSIONS))
session_counts = np.zeros((N_SESSIONS, N_SESSIONS))
for r, train_ses in enumerate(session_ids):
    for c, test_ses in enumerate(session_ids):
        per_subject_scores = []
        for subj in metadata["subject"].unique():
            tr_mask = (metadata["subject"] == subj) & (metadata["session"] == train_ses)
            te_mask = (metadata["subject"] == subj) & (metadata["session"] == test_ses)
            tr_idx = np.where(tr_mask.to_numpy())[0]
            te_idx = np.where(te_mask.to_numpy())[0]
            if r == c:
                # Within-session cell: 75/25 inside the same block (avoid
                # the trivially perfect train-on-test cell that would
                # paint the diagonal at 1.00 by leakage).
                local = tr_idx.copy()
                rng.shuffle(local)
                cut = int(0.75 * len(local))
                per_subject_scores.append(_fit_score(local[:cut], local[cut:]))
            else:
                per_subject_scores.append(_fit_score(tr_idx, te_idx))
        session_matrix[r, c] = float(np.mean(per_subject_scores))
        session_counts[r, c] = len(per_subject_scores)
print(
    "session x session transfer matrix (mean accuracy across "
    f"{int(session_counts.min())} subjects per cell):"
)
print(
    pd.DataFrame(
        session_matrix.round(3),
        index=[f"train={s}" for s in session_ids],
        columns=[f"test={s}" for s in session_ids],
    )
)

# %% [markdown]
# Step 7. Render the cross-session drift figure
# -----------------------------------------------
#
# ``draw_cross_session_figure`` (sibling helper) takes the matrix, the
# per-subject within and between accuracies, and the chance level, and
# returns a 1 x 3 panel: matrix on the left, paired bars in the middle,
# drift histogram on the right. The subtitle reports
# ``n_subjects | n_sessions | mean_within | mean_between | drift`` from
# the live numbers above.

# %%
subject_ids = sorted(metadata["subject"].unique())
within_per_subject_arr = np.asarray(
    [float(np.mean(within_per_subject[s])) for s in subject_ids]
)
cross_per_subject_arr = np.asarray(
    [float(np.mean(cross_per_subject[s])) for s in subject_ids]
)
fig = draw_cross_session_figure(
    session_matrix=session_matrix,
    subject_ids=subject_ids,
    within_session_acc=within_per_subject_arr,
    between_session_acc=cross_per_subject_arr,
    chance=chance,
    plot_id="plot_52",
)
plt.show()

# %% [markdown]
# Result: calibration drift is real
# -----------------------------------
#
# Subject overlap is 1 per fold by design; ``session_overlap`` is 0
# (E5.42 reports it). Within-session > cross-session > chance, and the
# gap is drift you would see on real multi-session BIDS data too. The
# top three drifters in this synthetic cohort are listed below; severe
# drifters are typical candidates for Riemannian alignment of the
# per-session covariance matrices before the classifier.

# %%
ranked = sorted(
    [
        (
            subj,
            float(np.mean(within_per_subject[subj])),
            float(np.mean(cross_per_subject[subj]))
            if cross_per_subject.get(subj)
            else float("nan"),
        )
        for subj in subject_ids
    ],
    key=lambda r: -(r[1] - r[2]),
)[:3]
print("Top 3 drift subjects (subject, within, cross):")
for subj, w, c in ranked:
    print(f"  {subj}: within={w:.3f}, cross={c:.3f}, drift={w - c:+.3f}")

# %% [markdown]
# A common slip, and how to recover
# -----------------------------------
#
# **Run.** Calling ``assert_no_leakage`` with the default
# ``by="subject"`` on a cross-session manifest looks like a leak
# (subjects are shared on purpose). Recovery: pass ``by="session"``.

# %%
try:
    assert_no_leakage(manifest, metadata)  # default by="subject"
    raise AssertionError("expected a LeakageError")
except Exception as exc:
    print(f"Caught {type(exc).__name__}: retrying with by='session'.")
    print(f"Recovery overlap = {assert_no_leakage(manifest, metadata, by='session')}")

# %% [markdown]
# Modify: try a 2-session subject
# ---------------------------------
#
# **Modify.** Drop one session from ``sub-00`` and re-run. MOABB
# contributes 2 folds for *that* subject (LOSO over 2 sessions) and 3
# folds for the other 7, giving 23 in total. Fold count depends on
# each subject's session inventory, which is the same reason real
# BIDS cohorts produce uneven fold counts when sessions are missing.

# %%
keep = ~((metadata["subject"] == "sub-00") & (metadata["session"] == "ses-03"))
trimmed_md = metadata.loc[keep].reset_index(drop=True)
trimmed_y = trimmed_md["target"].to_numpy()
# Drop ``n_folds`` so MOABB falls back to its native LeaveOneGroupOut
# behaviour: subjects with 2 remaining sessions contribute 2 folds, the
# others contribute one fold per session as usual.
trimmed_man = make_split_manifest(
    get_splitter("cross_session", random_state=SEED),
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
# - inflate ``ses_drift`` (line 1 of step 1) by a factor of 2 and
#   re-render the figure: the right tail of panel 3 grows.
# - swap to a real BIDS dataset with ``description['session']`` set;
#   the same manifest contract applies on EEGDash + NEMAR data.
# - replace logistic regression with a Riemannian alignment pipeline
#   on per-session covariance matrices and watch the off-diagonal cells
#   move.
#
# Mini-project
# ------------
#
# **Mini-project.** Pick a real multi-session NEMAR dataset, materialise
# windows with ``apply_split_manifest`` per fold, fit the same baseline,
# and compare the drift histogram with the synthetic version. The shape
# of the histogram (mean drift, tail width) is the per-cohort
# generalisation report; benchmark publications increasingly include
# both panels (Chevallier, Aristimunha et al. 2024).
#
# Links
# -----
#
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - API: :func:`eegdash.splits.get_splitter`,
#   :func:`eegdash.splits.make_split_manifest`,
#   :func:`eegdash.splits.assert_no_leakage`,
#   :func:`eegdash.splits.describe_split`,
#   :func:`eegdash.splits.majority_baseline`,
#   :func:`eegdash.splits.apply_split_manifest`.
# - Sklearn: :class:`sklearn.linear_model.LogisticRegression`,
#   :func:`sklearn.metrics.accuracy_score`.
# - Chevallier, Aristimunha et al. 2024
#   (https://doi.org/10.48550/arXiv.2404.15319), MOABB benchmark,
#   cross-session evaluation, calibration drift.
# - Cisotto & Chicco 2024 (https://doi.org/10.7717/peerj-cs.2256) --
#   ten quick tips for clinical EEG, Tip 9.
# - Delorme et al. 2022, NEMAR (https://doi.org/10.1162/imag_a_00026)
#  , BIDS metadata source for EEGDash queries.
# - Jayaram & Barachant 2018
#   (https://doi.org/10.1088/1741-2552/aabea9), covariate shift and
#   transfer learning in BCI calibration.
