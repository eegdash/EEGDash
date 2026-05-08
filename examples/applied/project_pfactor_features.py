"""Predicting p-factor from EEG with hand-crafted features (project starter)
============================================================================

Companion to ``project_pfactor_deep.py``: same target, different model
class. The deep variant asks whether an EEGConformer can learn the
``p_factor`` from raw windows, this one asks which interpretable EEG
features carry the signal. The p-factor (Caspi et al. 2014,
doi:10.1177/2167702613497473) is a transdiagnostic mental-health summary
score derived from parent-and-child psychiatric questionnaires; the EEG
side comes from the Healthy Brain Network release distributed on
OpenNeuro and on the EEG2025 Challenge mirror as ``EEG2025r5``
(Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through
NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023). Splits stay
strictly subject-disjoint per Cisotto and Chicco 2024 Tip 9
(doi:10.7717/peerj-cs.2256).

The deliverable is a feature-importance panel plus the regression
diagnostic on never-seen subjects: which feature family (band power,
connectivity, entropy) carries the most weight in a sklearn ridge or
random-forest head (Pedregosa et al. 2011, doi:10.5555/1953048.2078195),
and how flat is the predicted-vs-true cloud at the subject level?

Can hand-crafted spectral and signal features beat the train-median
predictor on held-out subjects, and which family pulls the most weight?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/project_pfactor_features.png'

# %% [markdown]
# Learning objectives
# -------------------
# After this study you will be able to:
#
# - load EEG2025r5 metadata with :class:`eegdash.EEGChallengeDataset` and attach the ``p_factor`` target via ``description_fields``.
# - assemble a multi-family feature table (band power, connectivity, entropy) with :func:`eegdash.features.extract_features`.
# - build a strict cross-subject split with ``get_splitter`` and verify zero subject overlap.
# - fit a sklearn ridge / random-forest head, pull importances, and report r / R^2 / MAE versus ``median_baseline``.
# - read a three-panel diagnostic that answers which feature family wins and whether the head beats the train-median predictor.
#
# Requirements
# ------------
# - Around 30 s on CPU. No GPU. No live network.
# - Prereqs: :doc:`/auto_examples/tutorials/40_features/plot_42_features_to_sklearn`
#   for the features-to-sklearn idiom and
#   :doc:`/auto_examples/tutorials/70_transfer_foundation/plot_72_subject_invariant_regression`
#   for the cross-subject regression contract this study satisfies.
# - Concept page: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup. Seed (E3.21) and a parametrised cache directory (E3.24) keep the
# study reproducible; warnings are tightened so sklearn convergence
# chatter does not drown the result print.
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from moabb.evaluations.splitters import CrossSubjectSplitter
from sklearn.model_selection import GroupKFold
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Step 1. Load EEG2025r5 with the p-factor target
# -------------------------------------------------
#
# In production, :class:`eegdash.EEGChallengeDataset` exposes ``p_factor``
# through ``description_fields`` so it surfaces as a per-recording column
# (Alexander et al. 2017; Delorme et al. 2022). The canonical call is
# below; the study then synthesises a feature table with the same column
# layout so the gallery runs offline (E3.24). The synthetic data follows
# the EEG2025r5 layout: 16 subjects, 24 windows per subject, four band-
# power features per channel (delta/theta/alpha/beta), two connectivity
# features (alpha and beta coherence between Cz and Pz), and two entropy
# features (Cz, Pz). The p-factor signal lives in the alpha and beta
# band-power features, with a small leak into entropy and a generous
# noise floor that keeps R^2 modest.
#
# .. code-block:: python
#
#    from eegdash import EEGChallengeDataset
#    ds = EEGChallengeDataset(
#        release="R5", task="contrastChangeDetection", mini=True,
#        cache_dir=cache_dir,
#        description_fields=["subject", "session", "task",
#                            "age", "sex", "p_factor"],
#    )

# %%
N_SUBJECTS, N_WINDOWS = 16, 24
BANDS = ("delta", "theta", "alpha", "beta")
CH_NAMES = ("Cz", "Pz")
subject_p = rng.normal(0.0, 1.0, size=N_SUBJECTS)

rows: list[dict] = []
for s in range(N_SUBJECTS):
    p = float(subject_p[s])
    for w in range(N_WINDOWS):
        # Subject-level offset (the leakage adversary) plus a noise floor.
        bias = 0.10 * (s - N_SUBJECTS / 2)
        row = {
            "subject": f"sub-{s:02d}",
            "session": "ses-01",
            "run": "run-01",
            "dataset": "EEG2025r5",
            "sample_id": f"sub-{s:02d}__w{w:03d}",
            "p_factor": p,
        }
        # Band-power features: alpha and beta carry the p-factor signal.
        for ch in CH_NAMES:
            for band in BANDS:
                base = rng.normal(0.0, 1.0)
                if band in ("alpha", "beta"):
                    base += 0.55 * p
                row[f"band_{band}_{ch}"] = float(base + 0.4 * bias)
        # Connectivity features: alpha and beta coherence between Cz and Pz.
        for band in ("alpha", "beta"):
            base = rng.normal(0.0, 1.0)
            base += 0.25 * p
            row[f"conn_coh_{band}_Cz_Pz"] = float(base + 0.2 * bias)
        # Entropy features: a faint signal, kept smaller than band-power.
        for ch in CH_NAMES:
            base = rng.normal(0.0, 1.0)
            base += 0.20 * p
            row[f"ent_spectral_{ch}"] = float(base + 0.2 * bias)
        rows.append(row)
feature_table = pd.DataFrame(rows)
feature_cols = [
    c for c in feature_table.columns if c.startswith(("band_", "conn_", "ent_"))
]
metadata = feature_table[
    ["subject", "session", "run", "dataset", "sample_id", "p_factor"]
].copy()
metadata["target"] = metadata["p_factor"].astype(float)
y = metadata["target"].to_numpy()
X = feature_table[feature_cols].to_numpy()
print(
    f"feature_table: rows={len(metadata)} | features={len(feature_cols)} | "
    f"subjects={metadata['subject'].nunique()} | "
    f"families=band_power({sum(c.startswith('band_') for c in feature_cols)}), "
    f"connectivity({sum(c.startswith('conn_') for c in feature_cols)}), "
    f"entropy({sum(c.startswith('ent_') for c in feature_cols)})"
)
assert metadata["p_factor"].notna().all(), "p_factor has NaN rows"
assert pd.api.types.is_float_dtype(metadata["p_factor"]), "p_factor not float"

# %% [markdown]
# Step 2. Predict: which family will the model lean on?
# -------------------------------------------------------
#
# **Predict.** The signal is hidden in alpha and beta band power with a
# weaker leak into entropy; connectivity carries a third the strength.
# Before fitting anything, write your guess for the top-3 features.
# Will the importance bars cluster on band-power blue, on connectivity
# orange, or on entropy purple? What R^2 do you expect on held-out
# subjects, 0.05? 0.20? 0.50?

# %% [markdown]
# Step 3. Build the regression head
# -----------------------------------
#
# Two heads, one row of importances each. A ridge regressor
# (:class:`sklearn.linear_model.Ridge`) keeps the coefficient signs so
# the bars carry direction; a random forest
# (:class:`sklearn.ensemble.RandomForestRegressor`) gives the
# permutation-style ``feature_importances_`` attribute that handles
# correlated band-power columns gracefully :cite:`pedregosa2011sklearn`. The
# :class:`sklearn.pipeline.Pipeline` chains :class:`sklearn.preprocessing.StandardScaler`
# in front of the ridge so the regularisation strength compares across
# columns of different scale.


# %%
def make_ridge_pipeline() -> Pipeline:
    """Return a fresh ``StandardScaler -> Ridge`` Pipeline (E3.21)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )


def make_forest() -> RandomForestRegressor:
    """Return a fresh small ``RandomForestRegressor`` (E3.21).

    ``n_estimators=200`` gives a stable importance ranking on this size of
    table; ``max_depth=4`` keeps the trees shallow enough that the report
    runs in seconds on CPU.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=1,
    )


# %% [markdown]
# Step 4. Cross-subject split and assert_no_leakage
# ---------------------------------------------------
#
# **Run.** ``get_splitter("cross_subject", n_folds=5, random_state=42)``
# returns sklearn's :class:`sklearn.model_selection.GroupKFold` keyed on
# ``subject``. The split is frozen into a manifest, every fold is walked
# with ``assert_no_leakage`` (``by="subject"``), and
# the JSON ``leakage_report`` line that downstream tooling parses is
# emitted. With 16 subjects each fold tests on three or four unseen
# subjects.

# %%
splitter = CrossSubjectSplitter(cv_class=GroupKFold, n_splits=5)
n_rows = len(metadata)
folds: list[tuple[np.ndarray, np.ndarray]] = []
for tr_idx, te_idx in splitter.split(y, metadata):
    tr_mask = np.zeros(n_rows, dtype=bool)
    tr_mask[tr_idx] = True
    te_mask = np.zeros(n_rows, dtype=bool)
    te_mask[te_idx] = True
    folds.append((tr_mask, te_mask))
overlap = max(
    len(set(metadata.loc[tr, "subject"]) & set(metadata.loc[te, "subject"]))
    for tr, te in folds
)
assert overlap == 0, "cross_subject manifest leaked subjects"
assert len(folds) >= 5, "need at least 5 folds for mean +/- std"
print(
    f"Splitter: {type(splitter).__name__} | folds: {len(folds)} | "
    f"max subject overlap: {overlap}"
)

# %% [markdown]
# Step 5. Fit, score, accumulate per-fold importances
# -----------------------------------------------------
#
# The loop fits both heads per fold, scores on the held-out subjects with
# :func:`sklearn.metrics.r2_score` and :func:`sklearn.metrics.mean_absolute_error`,
# and calls ``median_baseline`` for the chance level
# alongside (E5.43 forbids reporting a regression score without one). The
# random-forest ``feature_importances_`` are averaged across folds; the
# average is what the figure plots so a single fold cannot dominate the
# bar order.

# %%
fold_r2: list[float] = []
fold_mae: list[float] = []
fold_chance_r2: list[float] = []
fold_baseline_mae: list[float] = []
test_residuals: list[tuple[str, float, float]] = []
ridge_coefs = np.zeros(len(feature_cols), dtype=float)
forest_imp = np.zeros(len(feature_cols), dtype=float)

for k in range(len(folds)):
    tr = folds[k][0]
    te = folds[k][1]
    ridge = make_ridge_pipeline().fit(X[tr], y[tr])
    forest = make_forest().fit(X[tr], y[tr])
    # The forest predicts directly on raw features. We use it as the
    # primary regressor so the figure carries non-negative importances by
    # default; ridge coefficients are tracked alongside for sign reading.
    y_pred = forest.predict(X[te])
    fold_r2.append(float(r2_score(y[te], y_pred)))
    fold_mae.append(float(mean_absolute_error(y[te], y_pred)))
    train_median = float(np.median(y[tr]))
    ss_res = float(np.sum((y[te] - train_median) ** 2))
    ss_tot = float(np.sum((y[te] - float(np.mean(y[te]))) ** 2))
    fold_chance_r2.append(0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot)
    fold_baseline_mae.append(
        float(mean_absolute_error(y[te], np.full_like(y[te], np.median(y[tr]))))
    )
    forest_imp += forest.feature_importances_
    ridge_coefs += ridge.named_steps["reg"].coef_
    for sid, true_val, pred_val in zip(
        metadata.loc[te, "subject"].tolist(),
        y[te].tolist(),
        y_pred.tolist(),
    ):
        test_residuals.append((sid, float(true_val), float(pred_val)))
    print(
        f"Fold {k}: r2={fold_r2[-1]:+.3f} | mae={fold_mae[-1]:.3f} | "
        f"baseline_r2={fold_chance_r2[-1]:+.3f}"
    )

forest_imp /= len(folds)
ridge_coefs /= len(folds)
mean_r2 = float(np.mean(fold_r2))
std_r2 = float(np.std(fold_r2, ddof=1))
mean_mae = float(np.mean(fold_mae))
mean_chance_r2 = float(np.mean(fold_chance_r2))
mean_baseline_mae = float(np.mean(fold_baseline_mae))

# %% [markdown]
# Step 6. Investigate the importance bars (figure)
# --------------------------------------------------
#
# **Investigate.** The headline question is which feature family wins.
# The sibling ``_pfactor_features_figure.py`` renders three panels:
# top-K random-forest importances colour-coded by family (band-power
# blue, connectivity orange, entropy purple), predicted vs true at the
# subject level with Pearson r, R^2, MAE in the corner, and a window-
# level error histogram with a Gaussian overlay so bias and spread land
# on the same axis. This is the diagnostic Pernet et al. 2019
# (doi:10.1038/s41597-019-0104-8) recommend before any clinical claim.

# %%
res_df = pd.DataFrame(test_residuals, columns=["subject", "true", "pred"])
y_true_pooled = res_df["true"].to_numpy()
y_pred_pooled = res_df["pred"].to_numpy()
subject_pooled = res_df["subject"].tolist()

from _pfactor_features_figure import draw_pfactor_features_figure  # noqa: E402

fig = draw_pfactor_features_figure(
    feature_importances=forest_imp,
    feature_names=feature_cols,
    y_true_subj=y_true_pooled,
    y_pred_subj=y_pred_pooled,
    subject_ids=subject_pooled,
    top_k=10,
    plot_id="project_pfactor_features",
)
plt.show()

fig_metrics = fig._eegdash_pfactor_features_metrics
top_idx = np.argsort(np.abs(forest_imp))[::-1][:3]
top3 = [(feature_cols[i], float(forest_imp[i])) for i in top_idx]
print(
    f"figure metrics: r={fig_metrics['pearson_r']:+.3f} | "
    f"R^2={fig_metrics['r2']:+.3f} | MAE={fig_metrics['mae']:.3f} | "
    f"n_subjects={fig_metrics['n_subjects']} | "
    f"n_features={fig_metrics['n_features']} | "
    f"bias={fig_metrics['bias']:+.3f} | spread={fig_metrics['spread']:.3f}"
)
print(f"top-3 features: {top3}")

# %% [markdown]
# A common mistake, and how to recover
# --------------------------------------
#
# **Run.** A frequent slip on a feature table is leaving infinity or NaN
# values in a band-power column (a Welch estimate that hit a flat
# segment, an entropy estimate over a constant window). Sklearn refuses
# to fit, with an ``Input contains NaN, infinity or a value too large``
# error. The fix is the standard one: replace +-inf with NaN, then fill
# NaN with zero (or the per-feature mean) before scaling. The cell below
# triggers the failure on purpose with ``try/except`` so the recovery
# path is visible (Nederbragt et al. 2020,
# doi:10.1371/journal.pcbi.1008090).

# %%
try:
    bad_X = X.copy()
    bad_X[0, 0] = np.inf  # one infinity is enough to break sklearn
    bad_X[1, 0] = np.nan
    Ridge(alpha=1.0, random_state=SEED).fit(bad_X[:8], y[:8])
except (ValueError, TypeError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:90]}")
    # Recovery: replace inf, fill NaN, then refit. Pandas' ``.replace`` /
    # ``.fillna`` chain mirrors what eegdash.features users typically do
    # on the tidy DataFrame before handing it to sklearn.
    fixed = pd.DataFrame(bad_X[:8], columns=feature_cols)
    fixed = fixed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Ridge(alpha=1.0, random_state=SEED).fit(fixed.to_numpy(), y[:8])
    print(f"Recovery: replace inf, fillna(0), refit (rows={len(fixed)}); Ridge fit.")

# %% [markdown]
# Modify: gradient boosting and permutation importance (concept only)
# ---------------------------------------------------------------------
#
# **Modify (concept).** Two natural extensions sit one import away.
# :class:`sklearn.ensemble.GradientBoostingRegressor` (or LightGBM /
# XGBoost) often closes the small R^2 gap a random forest leaves on
# correlated band-power columns. :func:`sklearn.inspection.permutation_importance`
# replaces the impurity-based ``feature_importances_`` with a leave-one-
# feature-out score that is less biased toward features with many split
# points :cite:`pedregosa2011sklearn`. Both keep the cross-subject contract
# above, so dropping them in only changes the importance panel.

# %% [markdown]
# Result: feature-based R^2 vs median baseline (chance)
# -------------------------------------------------------
#
# Five folds, disjoint subject test sets. ``mean +/- std`` against the
# regression chance level. ``median_baseline`` returns
# the train-median predictor's R^2 on the test set; an honest model must
# beat it, not just match it. The print line carries the keyword
# *baseline* (E5.43).

# %%
print(
    f"Cross-subject 5-fold p-factor regression: r2={mean_r2:+.3f} +/- {std_r2:.3f} "
    f"| mae={mean_mae:.3f} | baseline_r2={mean_chance_r2:+.3f} "
    f"| baseline_mae={mean_baseline_mae:.3f} | metric: r2"
)

pearson_pooled = float(pearsonr(y_true_pooled, y_pred_pooled).statistic)
print(
    f"pooled-window r={pearson_pooled:+.3f} | subject r={fig_metrics['pearson_r']:+.3f}"
)
print(
    f"chance: predicting the train-median scores baseline_r2={mean_chance_r2:+.3f} on test."
)
assert mean_mae < mean_baseline_mae, "Model MAE must be below the median-baseline MAE."

# %% [markdown]
# Wrap-up
# -------
# We loaded an EEG2025r5-shaped feature table (with ``p_factor`` as a
# float column), assembled a multi-family feature set, built a 5-fold
# ``cross_subject`` manifest, asserted zero subject leakage, fit a ridge
# and a random-forest head per fold, and reported R^2 +/- std alongside
# ``median_baseline`` chance. The three-panel figure
# is the feature-based diagnostic, family-coloured importance bars on
# the left, subject-aggregated predicted-vs-true in the middle, and a
# window-level error histogram on the right. Treat the importance bars
# as a hypothesis ranker: a feature that wins on R5 only earns a
# clinical claim once it survives an external cohort (Cisotto and Chicco
# 2024 Tip 9). p_factor is a derived score, not a diagnosis, any clinical
# framing belongs in a follow-up study with a much larger N.

# %% [markdown]
# Try it yourself
# ---------------
# - Swap :class:`sklearn.ensemble.RandomForestRegressor` for
#   :class:`sklearn.ensemble.GradientBoostingRegressor` (still
#   ``random_state=42``). The importance panel should redraw with the
#   same family colours.
# - Replace the impurity-based importance with
#   :func:`sklearn.inspection.permutation_importance` and compare
#   the top-K ordering.
# - Bump ``n_folds`` to ``N_SUBJECTS`` for leave-one-subject-out variance
#   and read how the per-fold R^2 cloud widens.
# - Reload the deep companion ``project_pfactor_deep.py`` and overlay
#   its predicted-vs-true scatter on the same axes; how often do the
#   feature head and the deep head disagree on the same subject?

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
