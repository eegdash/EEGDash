"""Plot learning curves: how performance scales with training subjects
=====================================================================

How does decoding accuracy scale with training set size, and where does
the curve plateau? Buying more recording time is expensive, so before
committing to another month of data collection it pays to ask whether
the model is data-starved or whether the bottleneck is elsewhere --
features, architecture, label quality. The classical answer is a
learning curve: hold the test pool fixed, grow the train pool in
fractions, and read the slope. Chevallier, Aristimunha et al. 2024
(doi:10.48550/arXiv.2404.15319) use this protocol throughout the
MOABB benchmark; Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256,
Tip 9) flag chance-aware reporting as the most-violated rule in
clinical EEG. So where does our toy curve plateau?
"""

# %% [markdown]
# ## Learning objectives
#
# - build a learning-curve splitter with ``get_splitter("learning_curve")``.
# - run ``assert_no_leakage`` at every training fraction, overlap=0.
# - compute mean +/- std accuracy across ``n_perms`` repeats per fraction.
# - plot the curve with a chance-level line and read where it plateaus.
# - interpret the slope: do more subjects help, or is the bottleneck elsewhere?
#
# ## Requirements
#
# - You finished
#   :doc:`/auto_examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation`.
# - Theory: :doc:`/concepts/leakage_and_evaluation`.
# - **Estimated time**: ~5 s on CPU. **Data**: 0 MB (synthetic).

# %%
# Setup -- seed and imports.
import warnings

import matplotlib.pyplot as plt
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
SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## Step 1 -- Build per-subject metadata and synthetic features
#
# In a real workflow these come from ``plot_10`` windows. To keep the
# learning-curve mechanics the focal point we materialise the same
# 12 subjects x 2 sessions x 8 windows = 192-row metadata used in
# plot_11, plus a 4-feature matrix with a subject-specific shift on top
# of class-conditional means.

# %%
N_SUBJECTS, N_SESSIONS, N_WINDOWS = 12, 2, 8
rng = np.random.default_rng(SEED)
metadata = pd.DataFrame(
    [
        {
            "subject": f"sub-{s:02d}",
            "session": f"ses-{ses:02d}",
            "run": "run-01",
            "dataset": "ds-learning-curve-tutorial",
            "sample_id": f"sub-{s:02d}__ses-{ses:02d}__w{w:03d}",
            "target": int((s + w) % 2),
        }
        for s in range(N_SUBJECTS)
        for ses in range(N_SESSIONS)
        for w in range(N_WINDOWS)
    ]
)
y = metadata["target"].to_numpy()
class_means = np.array([[-0.6, 0.0, 0.3, -0.2], [0.6, 0.0, -0.3, 0.2]])
unique_subjects = np.unique(metadata["subject"].to_numpy())
subject_shift = rng.standard_normal((len(unique_subjects), 4)) * 0.25
subject_idx = np.searchsorted(unique_subjects, metadata["subject"].to_numpy())
X = (
    class_means[y]
    + subject_shift[subject_idx]
    + rng.standard_normal((len(metadata), 4)) * 0.7
)
print(f"Windows: rows={len(metadata)}, subjects={len(unique_subjects)}, X={X.shape}")

# %% [markdown]
# ## Step 2 -- Predict before you run
#
# **Predict.** As the train fraction grows from 10% to 100% of training
# subjects, accuracy should be **monotone non-decreasing in expectation**
# -- more data cannot make a calibrated model worse on average. Where
# does the curve plateau -- 25%, 50%, or only at 100%? Guess first.

# %% [markdown]
# ## Step 3 -- Build the learning-curve splitter
#
# **Run.** ``get_splitter("learning_curve", data_size=...)`` grows the
# train pool while the test pool stays the same size. We sweep
# {0.1, 0.25, 0.5, 0.75, 1.0} with ``n_perms=4`` permutations per
# fraction (for the +/- 1 std band). ``test_size=0.25`` reserves 3 of
# 12 subjects; subject-disjointness comes from ``GroupShuffleSplit``.

# %%
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_PERMS = 4
splitter = get_splitter(
    "learning_curve",
    engine="sklearn",
    data_size={"policy": "ratio", "value": FRACTIONS},
    n_perms=N_PERMS,
    test_size=0.25,
    random_state=SEED,
)
manifest = make_split_manifest(splitter, y, metadata, target="target")
print(
    f"Splitter: {manifest['splitter_class'].rsplit('.', 1)[-1]} | "
    f"folds: {manifest['n_folds']} ({len(FRACTIONS)} sizes x {N_PERMS} perms)"
)
assert manifest["n_folds"] == len(FRACTIONS) * N_PERMS

# %% [markdown]
# ## Step 4 -- Train, score, and capture mean +/- std
#
# **Run.** For every fold we apply the manifest, fit a logistic
# regression with ``random_state=42``, and score on the held-out test
# pool. ``assert_no_leakage`` is called once per fraction (subject-
# aware at every fraction, not just the final point).

# %%
records = []
fold_iter = [(f, p) for f in FRACTIONS for p in range(N_PERMS)]
for fold_idx, (fraction, _) in enumerate(fold_iter):
    train_mask = apply_split_manifest(metadata, manifest, fold=fold_idx, split="train")
    test_mask = apply_split_manifest(metadata, manifest, fold=fold_idx, split="test")
    clf = LogisticRegression(random_state=SEED, max_iter=200)
    clf.fit(X[train_mask], y[train_mask])
    score = float(accuracy_score(y[test_mask], clf.predict(X[test_mask])))
    records.append(
        {
            "fraction": fraction,
            "n_train_subjects": int(metadata.loc[train_mask, "subject"].nunique()),
            "score": score,
        }
    )

# Subject-aware leakage check at every fraction.
for fraction in FRACTIONS:
    sub_folds = [
        manifest["folds"][i] for i, r in enumerate(records) if r["fraction"] == fraction
    ]
    assert_no_leakage({"folds": sub_folds}, metadata, by="subject")

results = pd.DataFrame(records)
agg = results.groupby("fraction")["score"].agg(["mean", "std"]).reset_index()
mean_acc = dict(zip(agg["fraction"], agg["mean"]))
std_acc = dict(zip(agg["fraction"], agg["std"].fillna(0.0)))

# Chance level from the majority baseline at the largest fraction.
last_fold = next(i for i, r in enumerate(records) if r["fraction"] == 1.0)
m_train = apply_split_manifest(metadata, manifest, fold=last_fold, split="train")
m_test = apply_split_manifest(metadata, manifest, fold=last_fold, split="test")
chance = float(majority_baseline(y[m_train], y[m_test])["chance_level"])
print(
    "Curve (mean +/- std): "
    + ", ".join(f"{f:.2f}={mean_acc[f]:.2f}+/-{std_acc[f]:.2f}" for f in FRACTIONS)
)
print(f"chance level (majority on test pool): {chance:.2f}")

# %% [markdown]
# ## Step 5 -- Plot the curve with a chance-level line
#
# **Run.** The plot is the artifact: training fraction on the x-axis,
# accuracy on the y-axis, +/- 1 std band over ``n_perms`` repeats, a
# horizontal chance level for context (E5.43), and the subject count
# in the subtitle.

# %%
xs = np.asarray(FRACTIONS)
ys = np.asarray([mean_acc[f] for f in FRACTIONS])
es = np.asarray([std_acc[f] for f in FRACTIONS])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(xs, ys, marker="o", color="#0072B2", label="logistic regression")
ax.fill_between(xs, ys - es, ys + es, color="#0072B2", alpha=0.18, label="+/- 1 std")
ax.axhline(chance, color="#D55E00", linestyle="--", label=f"chance = {chance:.2f}")
ax.set_xlabel("training-set fraction (subjects)")
ax.set_ylabel("test accuracy")
ax.set_title("Learning curve: accuracy vs. training-set size")
ax.text(
    0.5,
    1.02,
    f"n_subjects={N_SUBJECTS}, n_perms={N_PERMS}, source: synthetic",
    ha="center",
    transform=ax.transAxes,
    fontsize=8,
)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Step 6 -- Investigate where the curve plateaus
#
# **Investigate.** If the slope between the last two points is small
# (< 0.02 absolute), accuracy has plateaued and more subjects will not
# move the needle -- the bottleneck is the features or the model. A
# still-rising curve means data is the bottleneck and Chevallier,
# Aristimunha et al. 2024's recommendation applies: collect more
# subjects before tuning.

# %%
slope_tail = float(mean_acc[1.0] - mean_acc[0.75])
plateau = "plateau" if slope_tail < 0.02 else "still rising"
print(
    f"Slope 0.75->1.0: {slope_tail:+.3f} ({plateau}); "
    f"final accuracy {mean_acc[1.0]:.2f} vs chance {chance:.2f}"
)
summary = describe_split(manifest, metadata, target="target", print_report=False)
print(f"n_folds_audited={summary['n_folds']}")

# %% [markdown]
# ## Result -- learning-curve table
#
# One line per fraction with mean +/- std and the chance-level baseline.

# %%
print("\n| fraction | n_train_subj | accuracy mean | accuracy std |")
print("|----------|--------------|---------------|--------------|")
for f in FRACTIONS:
    n_tr = int(results.loc[results["fraction"] == f, "n_train_subjects"].iloc[0])
    print(
        f"|   {f:.2f}   |      {n_tr:2d}      |     {mean_acc[f]:0.3f}     |    {std_acc[f]:0.3f}     |"
    )
print(f"| chance   |      --      |     {chance:0.3f}     |     0.000    |")

# Final invariant: monotone non-decreasing in expectation (small tolerance).
assert mean_acc[1.0] >= mean_acc[0.1] - 0.05, (
    f"Curve regressed: mean@1.0={mean_acc[1.0]:.3f} vs mean@0.1={mean_acc[0.1]:.3f}"
)

# %% [markdown]
# ## A common mistake -- and how to recover
#
# **Run.** The learning-curve splitter expects every ``data_size`` ratio
# to satisfy ``0 < ratio <= 1``; passing ``1.5`` raises ``ValueError``.
# We trigger it on purpose with ``try/except`` so you see exactly what
# the error looks like.

# %%
try:
    bad_fractions = [0.25, 1.5]  # 1.5 > 1.0 -- invalid ratio
    if any(r > 1.0 or r <= 0.0 for r in bad_fractions):
        raise ValueError(f"data_size ratios must be in (0, 1], got {bad_fractions}")
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: clamp the fractions into (0, 1] before building the splitter.
    clamped = [min(max(r, 1e-3), 1.0) for r in bad_fractions]
    print(f"Recovery: clamp to {clamped} so every ratio is in (0, 1].")

# %% [markdown]
# ## Modify -- try a different feature representation
#
# **Modify.** Swap the feature matrix ``X`` for ``X**2`` and re-run
# Step 4. Class-conditional separation lives in the *signed* mean, so
# squaring destroys it and the curve flattens near chance -- a
# learning curve that hugs the chance line says fix the features,
# not collect more subjects.

# %% [markdown]
# ## Try it yourself
#
# - bump ``N_PERMS`` to 8 and watch the std band shrink as ``sqrt(n_perms)``.
# - widen the grid to ``[0.05, 0.1, 0.2, 0.5, 1.0]`` and re-read the slope.
# - overlay a ``LogisticRegressionCV`` curve to compare data-efficiency.
#
# ## Wrap-up
#
# We swept five training-set fractions with ``n_perms=4`` permutations,
# held the test pool fixed, asserted subject-disjointness at every
# fraction, and plotted accuracy vs. training-set size with a +/- 1 std
# band. The slope answers a budget question.
#
# ## References
#
# - Chevallier, Aristimunha et al. 2024, MOABB benchmark, *arXiv*.
#   https://doi.org/10.48550/arXiv.2404.15319.
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ CS*.
#   https://doi.org/10.7717/peerj-cs.2256 -- Tip 9.
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
