"""Age regression from EEG (applied case study)
==================================================

Can a feature-based regression head predict a child's age from a few
seconds of resting-state EEG, on subjects the model has never seen?
This applied case study takes the Healthy Brain Network release
``ds005505`` (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced
through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023), wires up
an :class:`eegdash.EEGDashDataset` query, builds a strict subject-aware
split (Cisotto and Chicco 2024, doi:10.7717/peerj-cs.2256), fits a
:class:`sklearn.linear_model.Ridge` head on band-power features, and
reports Pearson r, Spearman rho, R^2, and MAE against a median-baseline
predictor. EEG-based brain-age regression has a long line of prior work
(Zoubi et al. 2018, doi:10.3389/fnhum.2018.00461). The headline
question here is not whether we beat the published literature; the
honest one is, does the model beat the train-set median predictor on
never-seen subjects?
"""

# Difficulty: 3-star (advanced applied project)

# %% [markdown]
# Learning objectives
# -------------------
# After this case study you will be able to:
#
# - load HBN ``ds005505`` via :class:`eegdash.EEGDashDataset` with ``age`` and ``sex`` on ``description_fields``.
# - build a subject-aware split with :class:`sklearn.model_selection.GroupKFold` keyed on subject.
# - fit a :class:`sklearn.linear_model.Ridge` head on a feature table and report ``r2``, MAE, and a median-baseline reference.
# - read a three-panel diagnostic plate (predicted vs true, per-subject signed residual, error histogram) for an EEG regression.
#
# Requirements
# ------------
# - ~30 s on CPU when the gallery uses the offline feature path (default).
# - Real-data path needs network access to NEMAR plus ``braindecode[hub]``.
# - Prereqs: :doc:`/generated/auto_examples/tutorials/00_start_here/plot_02_dataset_to_dataloader`
#   and :doc:`/generated/auto_examples/tutorials/40_features/plot_42_features_to_sklearn`.
# - Concept: :doc:`/concepts/leakage_and_evaluation` for the subject-aware
#   evaluation rationale that the figure plate is built around.

# %%
# Step 1. Setup, seeds, parametrised cache
# ----------------------------------------
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Step 2. Predict, what do you expect?
# ------------------------------------
#
# **Predict.** Constant predictors set the chance level. A model that
# always returns the train-set median age scores ``r2 = 0`` against the
# test-set mean by definition. Brain-age regression on faint resting
# EEG is a hard inverse problem; what r2 do you expect a Ridge head on
# ~12 features to reach with only ~12 held-out subjects: 0.10? 0.40?
# 0.80? Write your guess before scrolling.

# %% [markdown]
# Step 3. Load HBN ds005505 (real path) or synthesise a feature table (gallery)
# ------------------------------------------------------------------------------
#
# In production, :class:`eegdash.EEGDashDataset` exposes ``age`` and
# ``sex`` through ``description_fields`` so they surface as per-recording
# columns. The canonical call is shown below; this case study then
# synthesises a feature table with the same column layout so the
# rendered gallery runs offline and the smoke test stays fast.
#
# .. code-block:: python
#
#    from eegdash import EEGDashDataset
#    from braindecode.preprocessing import (
#        Preprocessor, create_fixed_length_windows, preprocess,
#    )
#    from braindecode.features import extract_features
#
#    ds = EEGDashDataset(
#        dataset="ds005505",
#        cache_dir=cache_dir,
#        description_fields=["subject", "session", "task", "age", "sex"],
#    )
#    preprocess(ds, [
#        Preprocessor("pick_channels", ch_names=["Cz", "Pz", "Oz", "Fz"]),
#        Preprocessor("resample", sfreq=128),
#        Preprocessor("filter", l_freq=1.0, h_freq=55.0),
#    ], n_jobs=4)
#    windows = create_fixed_length_windows(
#        ds, window_size_samples=256, window_stride_samples=256,
#        drop_last_window=True, preload=False,
#    )
#    features = extract_features(windows, feature_groups=["spectral"])
#
# We mirror the resulting shape: 12 subjects, ~24 windows per subject,
# 12 features per window (band-power proxies + variance on Cz/Pz/Oz),
# plus a subject-level ``age`` drawn from the HBN child cohort range
# (roughly 6 to 18 years).

# %%
N_SUBJECTS, N_WINDOWS = 12, 24
BANDS = ("delta", "theta", "alpha", "beta")
CH_NAMES = ("Cz", "Pz", "Oz")
subject_age = rng.uniform(6.0, 18.0, size=N_SUBJECTS)
subject_sex = rng.choice(["F", "M"], size=N_SUBJECTS)
rows: list[dict] = []
for s in range(N_SUBJECTS):
    a = float(subject_age[s])
    sx = str(subject_sex[s])
    for w in range(N_WINDOWS):
        # Per-window features: a faint age signal in alpha / beta plus a
        # mild subject-level offset (the part the cross-subject loop must
        # not memorise) and a noise floor that keeps R^2 modest.
        bias = 0.04 * (s - N_SUBJECTS / 2)
        row = {
            "subject": f"sub-NDARAA{s:04d}",
            "session": "ses-01",
            "run": "run-01",
            "dataset": "ds005505",
            "sample_id": f"sub-NDARAA{s:04d}__w{w:03d}",
            "age": a,
            "sex": sx,
        }
        for ch in CH_NAMES:
            row[f"var_{ch}"] = float(rng.gamma(2.0, 1.0) + bias)
            for band in BANDS:
                base = rng.normal(0.0, 1.0)
                if band in ("alpha", "beta"):
                    # Age tracks alpha / beta in resting EEG (Zoubi 2018).
                    base += 0.18 * (a - 12.0)
                row[f"spec_{band}_{ch}"] = float(base + 0.4 * bias)
        rows.append(row)
feature_table = pd.DataFrame(rows)
feature_cols = [c for c in feature_table.columns if c.startswith(("var_", "spec_"))]
metadata = feature_table[
    ["subject", "session", "run", "dataset", "sample_id", "age", "sex"]
].copy()
metadata["target"] = metadata["age"].astype(float)
y = metadata["target"].to_numpy()
X = feature_table[feature_cols].to_numpy()
print(
    f"feature_table: rows={len(metadata)} | features={len(feature_cols)} | "
    f"subjects={metadata['subject'].nunique()} | "
    f"age_range=[{metadata['age'].min():.1f}, {metadata['age'].max():.1f}] yr"
)
assert metadata["age"].notna().all(), "age has NaN rows"
assert pd.api.types.is_float_dtype(metadata["age"]), "age is not float"

# %% [markdown]
# Step 4. Build a Ridge regression head on top of features
# --------------------------------------------------------
#
# A :class:`sklearn.preprocessing.StandardScaler` then
# :class:`sklearn.linear_model.Ridge` :class:`sklearn.pipeline.Pipeline`
# (Pedregosa et al. 2011, doi:10.5555/1953048.2078195) is the regression
# analogue of a logistic head. ``alpha=1.0`` is a defensible default for
# a 12-feature table, and ``random_state=42`` keeps the loop byte stable.


# %%
def make_regressor() -> Pipeline:
    """Return a fresh ``StandardScaler -> Ridge`` Pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )


# %% [markdown]
# Step 5. Cross-subject 5-fold split, no subject leakage
# ------------------------------------------------------
#
# **Run.** :class:`sklearn.model_selection.GroupKFold` keyed on
# ``subject`` is the contract: every subject appears in exactly one
# test fold. We assert zero subject overlap before scoring, and we
# pool the held-out predictions across folds so the diagnostic plate
# in step 7 sees every subject once. Subject leakage routinely
# produces optimistic accuracies (Cisotto and Chicco 2024 Tip 9), so
# the assertion below is non-negotiable.

# %%
groups = metadata["subject"].to_numpy()
unique_subjects = np.unique(groups)
n_folds = min(5, len(unique_subjects))
splitter = GroupKFold(n_splits=n_folds)

fold_r2: list[float] = []
fold_mae: list[float] = []
fold_baseline_mae: list[float] = []
held_records: list[tuple[str, str, float, float]] = []
for k, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
    train_subj = set(groups[tr].tolist())
    test_subj = set(groups[te].tolist())
    overlap = train_subj & test_subj
    assert not overlap, f"Fold {k} leaked subjects: {sorted(overlap)}"

    pipe = make_regressor().fit(X[tr], y[tr])
    y_pred = pipe.predict(X[te])
    fold_r2.append(float(r2_score(y[te], y_pred)))
    fold_mae.append(float(mean_absolute_error(y[te], y_pred)))
    median_pred = float(np.median(y[tr]))
    fold_baseline_mae.append(
        float(mean_absolute_error(y[te], np.full_like(y[te], median_pred)))
    )
    for sid, sx, true_val, pred_val in zip(
        metadata.loc[te, "subject"].tolist(),
        metadata.loc[te, "sex"].tolist(),
        y[te].tolist(),
        y_pred.tolist(),
    ):
        held_records.append((sid, sx, float(true_val), float(pred_val)))
    print(
        f"Fold {k}: r2={fold_r2[-1]:+.3f} | mae={fold_mae[-1]:.3f} yr | "
        f"baseline_mae={fold_baseline_mae[-1]:.3f} yr"
    )

mean_r2 = float(np.mean(fold_r2))
std_r2 = float(np.std(fold_r2, ddof=1))
mean_mae = float(np.mean(fold_mae))
mean_baseline_mae = float(np.mean(fold_baseline_mae))

# %% [markdown]
# Step 6. A common mistake, and how to recover
# --------------------------------------------
#
# **Run.** A frequent slip is wiring a non-numeric ``age`` column into
# :class:`sklearn.linear_model.Ridge`. ``age`` arrives as strings if a
# CSV was loaded without dtype hints (or if a NEMAR sidecar serialises
# it as text), and Ridge then refuses to fit. We trigger the failure
# on purpose with ``try / except`` so the error message and the
# recovery sit next to each other in the rendered gallery.

# %%
try:
    bad_y = metadata["age"].astype(str).to_numpy()
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], bad_y[:8])
except (ValueError, TypeError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:90]}")
    # Recovery: cast the target to float before fitting any regression head.
    fixed_y = pd.to_numeric(metadata["age"], errors="coerce").to_numpy()
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], fixed_y[:8])
    print(f"Recovery: cast age to float (dtype={fixed_y.dtype}); Ridge fit.")

# %% [markdown]
# Step 7. Investigate, three-panel diagnostic plate
# -------------------------------------------------
#
# **Investigate.** A single MAE number hides every failure mode that
# matters. The sibling ``_age_regression_figure.py`` renders three
# panels: predicted vs true age (one point per held-out subject, with
# Pearson r, Spearman rho, R^2, and MAE in a corner box, points colored
# by sex), per-subject signed residuals sorted by ``|residual|`` (orange
# for over-prediction, blue for under-prediction), and a histogram of
# window-level errors with a Gaussian density overlay so the bias
# (mean) and the spread (std) read at a glance. This is the kind of
# diagnostic Pernet et al. 2019 (doi:10.1038/s41597-019-0104-8)
# recommend before any clinical claim.

# %%
held_df = pd.DataFrame(held_records, columns=["subject", "sex", "true", "pred"])
y_true_pooled = held_df["true"].to_numpy()
y_pred_pooled = held_df["pred"].to_numpy()
subject_pooled = held_df["subject"].tolist()
sex_pooled = held_df["sex"].tolist()

from _age_regression_figure import draw_age_regression_figure

fig = draw_age_regression_figure(
    y_true_subj=y_true_pooled,
    y_pred_subj=y_pred_pooled,
    subject_ids=subject_pooled,
    sex_or_fold=sex_pooled,
    plot_id="project_age_regression",
)
plt.show()

# Pull figure-side metrics back into the script namespace so the
# wrap-up print line stays consistent with the corner box.
fig_metrics = fig._eegdash_age_metrics
print(
    f"figure metrics: r={fig_metrics['pearson_r']:+.3f} | "
    f"rho={fig_metrics['spearman_rho']:+.3f} | "
    f"R^2={fig_metrics['r2']:+.3f} | "
    f"MAE={fig_metrics['mae']:.3f} yr | "
    f"n_subjects={fig_metrics['n_subjects']}"
)

# %% [markdown]
# Modify, where to take this next (concept only)
# ----------------------------------------------
#
# **Modify (concept).** Two upgrades close most of the gap to a real
# scientific result. First, scale the cohort: HBN ships several
# thousand recordings, and reliable brain-age estimates need at least
# a few hundred subjects per fold. Second, swap Ridge for a deep
# encoder, :class:`braindecode.models.EEGConformer` (Song et al. 2023,
# doi:10.1109/TNSRE.2022.3230250) trained on raw windows, with the
# same cross-subject contract enforced at the dataset level. We do not
# run it here (gpu_required + multi-hour fit out of scope), but the
# subject-aware split contract above is the floor any such system must
# still satisfy.

# %% [markdown]
# Result, model vs median baseline
# --------------------------------
#
# Five folds, disjoint subject test sets. ``mean +/- std`` against the
# regression chance level. The median-baseline MAE is the floor the
# trained model must come in below. The pooled-window vs subject-level
# numbers split the disagreement: pooled noise vs subject-level rank
# reversal (Cisotto and Chicco 2024 Tip 7).

# %%
pearson_pooled = float(pearsonr(y_true_pooled, y_pred_pooled).statistic)
spearman_pooled = float(spearmanr(y_true_pooled, y_pred_pooled).statistic)
print(
    f"Cross-subject 5-fold age regression: r2={mean_r2:+.3f} +/- {std_r2:.3f} "
    f"| mae={mean_mae:.3f} yr | baseline_mae={mean_baseline_mae:.3f} yr "
    f"| metric: r2"
)
print(
    f"pooled-window r={pearson_pooled:+.3f} | rho={spearman_pooled:+.3f} | "
    f"subject r={fig_metrics['pearson_r']:+.3f} | "
    f"subject rho={fig_metrics['spearman_rho']:+.3f}"
)
print(
    f"chance: predicting the train-median age scores baseline_mae="
    f"{mean_baseline_mae:.3f} yr on test."
)
assert mean_mae < mean_baseline_mae, "Model MAE must be below the median-baseline MAE."

# %% [markdown]
# Wrap-up
# -------
# We loaded an HBN-shaped feature table with ``age`` as a float column,
# built a 5-fold ``GroupKFold`` split keyed on ``subject``, asserted
# zero subject leakage, fit a :class:`sklearn.linear_model.Ridge` head
# per fold, and reported R^2 plus MAE alongside the median-baseline
# predictor. The three-panel figure is the diagnostic that surfaces
# what a single MAE hides: a flat predicted-vs-true cloud means the
# model regresses toward the train-set mean, a one-sided residual bar
# chart means a single subject drags the error, and a wide Gaussian on
# the histogram means the per-window spread leaves little room for
# clinical claims. Age in HBN is a derived metadata column, not a
# diagnosis; any clinical framing belongs in a follow-up study with
# much larger N.

# %% [markdown]
# Try it yourself
# ---------------
# - Swap :class:`sklearn.linear_model.Ridge` for
#   :class:`sklearn.neural_network.MLPRegressor` (still ``random_state=42``).
# - Pre-train a :class:`braindecode.models.EEGConformer` on raw windows
#   and feed encoder activations into the same Ridge head.
# - Bump ``n_folds`` to ``N_SUBJECTS`` for leave-one-subject-out variance.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
