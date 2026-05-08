"""How do I submit a baseline to EEG2025 Challenge 2 (predict the p-factor)?
==============================================================================

The EEG2025 Foundation Challenge ships two regression tracks, and
Challenge 2 asks for a single number per subject, the **p-factor**,
from a short clip of resting-state EEG. The p-factor (Caspi et al.
2014, doi:10.1177/2167702613497473) is a general dimension of
psychopathology derived from the Child Behavior Checklist; the EEG
side comes from the Healthy Brain Network release distributed via
:class:`~eegdash.dataset.EEGChallengeDataset` (Alexander et al. 2017,
doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al.
2022, doi:10.1093/nargab/lqac023).

This starter kit walks through three steps before any fancy modelling:
load the challenge cohort with ``p_factor`` attached, fit a feature
ridge on a strict cross-subject split, and produce a leaderboard-style
result card next to the public top score (Cisotto & Chicco 2024,
doi:10.7717/peerj-cs.2256).

Learning objectives
-------------------
After this tutorial you will be able to:

- load EEG2025 Challenge 2 recordings via :class:`eegdash.EEGChallengeDataset`.
- fit a :class:`sklearn.linear_model.Ridge` head and report Pearson r, R^2, MAE.
- produce a leaderboard card placing the starter score next to chance and target.
- phrase the result in clinical-cautious language for the p-factor.

The headline question is not "can we win Challenge 2?", the p-factor
signal in EEG is faint, but the honest one: does this model beat the
train median predictor on never-seen subjects?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/tutorial_challenge_2.png'

# %% [markdown]
# .. _challenge-2:
# .. meta::
#    :html_theme.sidebar_secondary.remove: true
# .. contents:: This example covers:
#    :local:
#    :depth: 2

# %% [markdown]
# Requirements
# ------------
# - About 30 s on CPU; GPU optional. No live network in the gallery build.
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - Leaderboard contract: `eeg2025.github.io <https://eeg2025.github.io>`__.
#
# Open in Colab:
#
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_2.ipynb
#    :alt: Open In Colab

# %%
# Step 0. Setup and imports
# ---------------------------
# numpy seeding (E3.21), parametrised cache (E3.24), and the
# ``use_eegdash_style`` one-call rcParams setup so every figure inherits
# the EEGDash identity (Helvetica fallback, muted grid, Data Rail).
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    get_splitter,
    make_split_manifest,
    median_baseline,
)
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = Path(os.environ.get("EEGDASH_CACHE_DIR", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"device={device} | cache_dir={cache_dir}")

# %% [markdown]
# Step 1. Load the challenge cohort with ``p_factor`` attached
# --------------------------------------------------------------
#
# **Predict.** Before reading the snippet, write down: how many
# subjects do you expect on the R5 mini release? Mini releases ship 20
# subjects with the resting-state block plus the active cognitive
# tasks; ``p_factor`` is a subject-level score, so every recording from
# one subject carries the same value.
#
# **Run.** :class:`~eegdash.dataset.EEGChallengeDataset` exposes
# ``p_factor`` through ``description_fields``. The canonical call is
# below; the rendered gallery then synthesises a feature table with the
# same column layout so the build runs offline.
#
# .. code-block:: python
#
#    from eegdash import EEGChallengeDataset
#    from eegdash.paths import get_default_cache_dir
#
#    ds = EEGChallengeDataset(
#        release="R5",
#        task="contrastChangeDetection",
#        mini=True,
#        cache_dir=str(get_default_cache_dir()),
#        description_fields=[
#            "subject", "session", "run", "task",
#            "age", "sex", "p_factor",
#        ],
#    )
#    cohort = ds.description.copy()
#    print(cohort.shape, cohort["p_factor"].describe())

# %%
# Mirror the Challenge 2 schema: 20 subjects, a handful of windows per
# subject, eight features per window (band-power proxies plus variance
# at Cz / Pz), and a subject-level ``p_factor`` drawn from N(0, 1).
# The columns below match what
# :class:`~eegdash.dataset.EEGChallengeDataset` surfaces on a live
# call, so the rest of the notebook is identical online and offline.
N_SUBJECTS, N_WINDOWS = 20, 24
BANDS = ("delta", "theta", "alpha", "beta")
CH_NAMES = ("Cz", "Pz")
subject_p = rng.normal(0.0, 1.0, size=N_SUBJECTS)
rows: list[dict] = []
for s in range(N_SUBJECTS):
    p = float(subject_p[s])
    for w in range(N_WINDOWS):
        bias = 0.15 * (s - N_SUBJECTS / 2)
        row = {
            "subject": f"sub-{s:02d}",
            "session": "ses-01",
            "run": "run-01",
            "dataset": "EEG2025r5mini",
            "sample_id": f"sub-{s:02d}__w{w:03d}",
            "p_factor": p,
        }
        for ch in CH_NAMES:
            row[f"var_{ch}"] = float(rng.gamma(2.0, 1.0) + bias)
            for band in BANDS:
                base = rng.normal(0.0, 1.0)
                if band in ("alpha", "beta"):
                    base += 0.4 * p  # p-factor signal lives in faster rhythms
                row[f"spec_{band}_{ch}"] = float(base + 0.5 * bias)
        rows.append(row)
feature_table = pd.DataFrame(rows)
feature_cols = [c for c in feature_table.columns if c.startswith(("var_", "spec_"))]
metadata = feature_table[
    ["subject", "session", "run", "dataset", "sample_id", "p_factor"]
].copy()
metadata["target"] = metadata["p_factor"].astype(float)
y = metadata["target"].to_numpy()
X = feature_table[feature_cols].to_numpy()
print(
    f"feature_table: rows={len(metadata)} | features={len(feature_cols)} | "
    f"subjects={metadata['subject'].nunique()} | "
    f"p_factor_dtype={metadata['p_factor'].dtype}"
)
assert metadata["p_factor"].notna().all(), "p_factor column has NaN rows"
assert pd.api.types.is_float_dtype(metadata["p_factor"]), "p_factor is not float"

# %% [markdown]
# Step 2. Predict: what r should chance look like?
# --------------------------------------------------
#
# **Predict.** A constant predictor returning the train-set median has
# Pearson ``r = 0`` against the held-out subjects by definition;
# :func:`eegdash.splits.median_baseline` formalises this on the R^2
# side. What ``r`` do you expect a feature ridge to reach on faint EEG
# features, ``0.10``? ``0.30``? ``0.50``? Write your guess.

# %% [markdown]
# Step 3. Build a leakage-safe cross-subject split
# --------------------------------------------------
#
# **Run.** ``get_splitter("cross_subject", n_folds=5, random_state=42)``
# returns sklearn's :class:`sklearn.model_selection.GroupKFold` keyed
# on ``subject``. The split is frozen into a manifest, walked with
# :func:`eegdash.splits.assert_no_leakage` (``by="subject"``), and the
# ``leakage_report`` line is what the audit pipeline parses.

# %%
splitter = get_splitter("cross_subject", n_folds=5, n_splits=5, random_state=SEED)
manifest = make_split_manifest(splitter, y, metadata, target="target")
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "cross_subject manifest leaked subjects"
assert manifest["n_folds"] >= 5, "need at least 5 folds for mean +/- std"
print(
    f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']} | "
    f"max subject overlap: {overlap}"
)

# %% [markdown]
# Step 4. Fit a ridge head per fold and pool predictions
# --------------------------------------------------------
#
# A :class:`sklearn.preprocessing.StandardScaler` ``->``
# :class:`sklearn.linear_model.Ridge`
# :class:`sklearn.pipeline.Pipeline` (Pedregosa et al. 2011) is the
# regression analogue of the plot_42 classification baseline. ``alpha=1.0``
# is a defensible default for an 8-feature table; ``GroupKFold`` keys on
# subject so a held-out subject never appears in any train fold.


# %%
def make_regressor() -> Pipeline:
    """Return a fresh ``StandardScaler -> Ridge`` Pipeline (E3.21)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )


fold_r2: list[float] = []
fold_mae: list[float] = []
fold_chance: list[float] = []
fold_baseline_mae: list[float] = []
test_residuals: list[tuple[str, float, float]] = []
for k in range(manifest["n_folds"]):
    tr = apply_split_manifest(metadata, manifest, fold=k, split="train")
    te = apply_split_manifest(metadata, manifest, fold=k, split="test")
    pipe = make_regressor().fit(X[tr], y[tr])
    y_pred = pipe.predict(X[te])
    fold_r2.append(float(r2_score(y[te], y_pred)))
    fold_mae.append(float(mean_absolute_error(y[te], y_pred)))
    base = median_baseline(y[tr], y[te])
    fold_chance.append(float(base["baseline_score"]))
    fold_baseline_mae.append(
        float(mean_absolute_error(y[te], np.full_like(y[te], np.median(y[tr]))))
    )
    for sid, tv, pv in zip(
        metadata.loc[te, "subject"].tolist(), y[te].tolist(), y_pred.tolist()
    ):
        test_residuals.append((sid, float(tv), float(pv)))
    print(
        f"Fold {k}: r2={fold_r2[-1]:+.3f} | mae={fold_mae[-1]:.3f} | "
        f"baseline_r2={fold_chance[-1]:+.3f}"
    )

mean_r2 = float(np.mean(fold_r2))
mean_mae = float(np.mean(fold_mae))
mean_chance = float(np.mean(fold_chance))
mean_baseline_mae = float(np.mean(fold_baseline_mae))

# %% [markdown]
# Step 5. Investigate: pooled metrics on the held-out cohort
# ------------------------------------------------------------
#
# **Investigate.** Mean R^2 across folds is volatile when each fold
# tests on a handful of subjects. Pooling the held-out predictions and
# scoring once gives the leaderboard metric: Pearson ``r`` as the
# primary score, ``MAE`` as a secondary diagnostic (Cisotto & Chicco
# 2024 Tip 7).

# %%
res_df = pd.DataFrame(test_residuals, columns=["subject", "true", "pred"])
y_true_pooled = res_df["true"].to_numpy()
y_pred_pooled = res_df["pred"].to_numpy()
subject_pooled = res_df["subject"].tolist()

# Subject-level aggregation: every window of one subject has the same
# y_true, so the leaderboard score is computed on per-subject means.
subj_view = res_df.groupby("subject", as_index=False).mean(numeric_only=True)
y_true_subj_arr = subj_view["true"].to_numpy()
y_pred_subj_arr = subj_view["pred"].to_numpy()
subject_pearson = float(pearsonr(y_true_subj_arr, y_pred_subj_arr).statistic)
subject_r2 = float(r2_score(y_true_subj_arr, y_pred_subj_arr))
subject_mae = float(mean_absolute_error(y_true_subj_arr, y_pred_subj_arr))
print(
    f"subject-level: r={subject_pearson:+.3f} | "
    f"R^2={subject_r2:+.3f} | MAE={subject_mae:.3f} | "
    f"n_subjects={len(subj_view)}"
)

# %% [markdown]
# Step 6. Render the three-panel starter-kit figure
# ---------------------------------------------------
#
# The drawing helpers live in a sibling
# :mod:`_challenge_2_figure` module so the rendering plumbing stays
# out of this tutorial. Panel 1 is the train-cohort p-factor histogram;
# panel 2 is the predicted vs true scatter on held-out subjects;
# panel 3 is a leaderboard-style result card placing the starter-kit
# baseline next to chance and the public top score.

# %%
from _challenge_2_figure import draw_challenge_2_figure  # noqa: E402

# Train-cohort distribution: one row per subject (mirrors the
# per-subject p_factor column of the CSV).
p_factor_distribution = metadata.drop_duplicates("subject")["p_factor"].to_numpy()

# Leaderboard rows: chance, the pooled starter score, and a placeholder
# for the EEG2025 dashboard top score (``score=nan`` so the bar reads
# "n/a" until the organisers publish the live number).
leaderboard_rows = [
    {
        "team": "median baseline (chance)",
        "regime": "chance",
        "score": 0.0,
        "metric": "r",
    },
    {
        "team": "starter kit (this notebook)",
        "regime": "starter",
        "score": subject_pearson,
        "metric": "r",
    },
    {
        "team": "EEG2025 leaderboard top",
        "regime": "target",
        "score": float("nan"),  # placeholder until the dashboard is live
        "metric": "r",
    },
]

fig = draw_challenge_2_figure(
    p_factor_distribution=p_factor_distribution,
    y_true_subj=y_true_pooled,
    y_pred_subj=y_pred_pooled,
    leaderboard_rows=leaderboard_rows,
    subject_ids=subject_pooled,
    plot_id="tutorial_challenge_2",
)
plt.show()

# Pull figure-side metrics back into the tutorial namespace so the
# wrap-up print line stays consistent with the corner annotation.
fig_metrics = fig._eegdash_challenge_2_metrics
print(
    f"figure metrics: r={fig_metrics['pearson_r']:+.3f} | "
    f"R^2={fig_metrics['r2']:+.3f} | MAE={fig_metrics['mae']:.3f} | "
    f"n_subjects={fig_metrics['n_subjects']}"
)

# %% [markdown]
# A common mistake, and how to recover
# --------------------------------------
#
# **Run.** A frequent slip is wiring a non-numeric target column into
# :class:`sklearn.linear_model.Ridge`: ``p_factor`` arrives as strings
# if a CSV is loaded without dtype hints, and Ridge then refuses to
# solve. The cell below triggers it on purpose with ``try/except`` so
# the failure mode is visible (Nederbragt et al. 2020,
# doi:10.1371/journal.pcbi.1008090).

# %%
try:
    bad_y = metadata["p_factor"].astype(str).to_numpy()  # string p-factor
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], bad_y[:8])
except (ValueError, TypeError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:90]}")
    fixed_y = pd.to_numeric(metadata["p_factor"], errors="coerce").to_numpy()
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], fixed_y[:8])
    print(f"Recovery: cast p_factor to float (dtype={fixed_y.dtype}); Ridge fit.")

# %% [markdown]
# Modify: deep-learning baseline (concept only)
# -----------------------------------------------
#
# **Modify (concept).** A stronger Challenge 2 entry typically swaps
# the feature ridge for a Braindecode encoder fed raw 2-second windows,
# trained with :class:`torch.nn.functional.l1_loss` against the
# subject-level p-factor. The skeleton stays in a code block so the
# gallery build remains CPU-only; the cross-subject loop in Step 4 is
# the contract any deep model must still satisfy.
#
# .. code-block:: python
#
#    from braindecode.models import EEGNeX
#    from torch.utils.data import DataLoader
#    from torch.nn.functional import l1_loss
#    from torch import optim
#
#    model = EEGNeX(n_chans=129, n_outputs=1, n_times=2 * 100).to(device)
#    optimizer = optim.Adamax(model.parameters(), lr=2e-3)
#    loader = DataLoader(windows_ds, batch_size=128, shuffle=True)
#    for X_batch, y_batch, _, _ in loader:
#        optimizer.zero_grad()
#        X_batch = X_batch.to(dtype=torch.float32, device=device)
#        y_batch = y_batch.to(dtype=torch.float32, device=device).unsqueeze(1)
#        loss = l1_loss(model(X_batch), y_batch)
#        loss.backward()
#        optimizer.step()
#    torch.save(model.state_dict(), "weights_challenge_2.pt")

# %% [markdown]
# Result: starter-kit baseline vs median chance
# -----------------------------------------------
#
# Five folds, disjoint subject test sets; the print line carries the
# keyword *baseline* and the ``metric: r2`` tag (E5.43).
# :func:`eegdash.splits.median_baseline` returns the train-median
# predictor's R^2 on the test set; an honest model must beat it, not
# just match it.

# %%
print(
    f"Cross-subject 5-fold p-factor regression: r2={mean_r2:+.3f} "
    f"| mae={mean_mae:.3f} | baseline_r2={mean_chance:+.3f} "
    f"| baseline_mae={mean_baseline_mae:.3f} | metric: r2"
)
print(
    f"Subject-level leaderboard score: r={subject_pearson:+.3f} | "
    f"R^2={subject_r2:+.3f} | MAE={subject_mae:.3f}"
)
print(
    f"chance: predicting the train-median scores baseline_r2="
    f"{mean_chance:+.3f} on test."
)
assert mean_mae < mean_baseline_mae, "Model MAE must be below the median-baseline MAE."

# %% [markdown]
# Wrap-up
# -------
# We loaded a Challenge 2 cohort with ``p_factor`` attached, built a
# cross-subject 5-fold split, asserted zero subject leakage, fit a
# Ridge head per fold, and pooled predictions for the leaderboard
# score. The three-panel figure is the starter-kit submission card:
# the histogram pins the target distribution; the scatter shows whether
# the model tracks the diagonal or collapses onto the train mean; the
# result card places the starter baseline next to chance and the
# challenge top score. ``p_factor`` is a derived score, not a
# diagnosis; any clinical framing belongs in a follow-up study with a
# much larger N.

# %% [markdown]
# Try it yourself
# ---------------
# - Swap :class:`sklearn.linear_model.Ridge` for
#   :class:`sklearn.neural_network.MLPRegressor` (still
#   ``random_state=42``); compare pooled ``r`` against the figure.
# - Pre-train :class:`braindecode.models.ShallowFBCSPNet` on the
#   passive tasks and feed its activations as features.
# - Bump ``n_folds`` to ``N_SUBJECTS`` for leave-one-subject-out.

# %% [markdown]
# References
# ----------
# See :doc:`/references` for the centralised bibliography of papers
# cited above. Add or amend an entry once in
# :file:`docs/source/refs.bib`; every tutorial inherits the update.
