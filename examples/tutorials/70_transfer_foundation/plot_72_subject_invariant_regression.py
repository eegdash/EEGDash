"""Subject-invariant p-factor regression (EEG2025 Challenge 2)
================================================================

Challenge 2 of the EEG2025 Foundation Challenge asks whether we can
predict the p-factor (Caspi et al. 2014, doi:10.1177/2167702613497473 --
a "psychiatric general factor" derived from the Child Behavior Checklist)
from EEG recordings released through ``EEGChallengeDataset`` without
silently overfitting to subject identity, the way naive cross-validation
inadvertently does on most clinical EEG benchmarks today.

The setup is a strict out-of-distribution regression: the test cohort
never overlaps with the train cohort on subject. A model that secretly
memorises subjects scores well within-fold and collapses on a new
sitting. So we build the ``cross_subject`` loop from plot_51, fit a
feature-based regression head, and report ``r2`` against
``median_baseline`` -- the chance level for regression (Cisotto & Chicco
2024 Tip 9, doi:10.7717/peerj-cs.2256). The headline question is not
"can we win Challenge 2?" -- the p-factor signal in EEG is genuinely
faint -- but the more honest one: does our model actually beat the
train-set median predictor on never-seen-before subjects?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - load EEG2025 Challenge 2 recordings via ``EEGChallengeDataset``.
# - build a strict cross-subject 5-fold split with ``get_splitter``.
# - fit a Ridge head and report ``r2`` against ``median_baseline`` chance.
# - examine a per-subject residual histogram for invariance failures.
#
# ## Requirements
# - ~30 s on CPU; GPU optional. No live network.
# - Prereqs: ``plot_70_challenge_dataset_basics`` and ``plot_42_features_to_sklearn``.
# - Concept: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup -- numpy seeding (E3.21) plus a parametrised cache (E3.24).
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1 -- Load EEGChallengeDataset for the p-factor task
#
# In production, ``EEGChallengeDataset`` exposes ``p_factor`` through
# ``description_fields`` so it surfaces as a per-recording column. The
# canonical call is shown below; this tutorial then synthesises a feature
# table with the same column layout so the gallery runs offline (E3.24).
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
#
# We mirror Challenge 2's structure: 12 subjects, ~24 windows per subject,
# eight features per window (band-power proxies + variance), and a
# subject-level p_factor drawn from a roughly N(0, 1) distribution.

# %%
N_SUBJECTS, N_WINDOWS = 12, 24
BANDS = ("delta", "theta", "alpha", "beta")
CH_NAMES = ("Cz", "Pz")
subject_p = rng.normal(0.0, 1.0, size=N_SUBJECTS)
rows: list[dict] = []
for s in range(N_SUBJECTS):
    p = float(subject_p[s])
    for w in range(N_WINDOWS):
        # Per-window features: a faint p-factor signal in alpha/beta plus
        # a subject-level offset (the "subject identity" we must NOT memorise)
        # and a generous noise floor that keeps R^2 modest.
        bias = 0.15 * (s - N_SUBJECTS / 2)
        row = {
            "subject": f"sub-{s:02d}",
            "session": "ses-01",
            "run": "run-01",
            "dataset": "EEG2025r5",
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
assert metadata["p_factor"].notna().all(), "p_factor has NaN rows"
assert pd.api.types.is_float_dtype(metadata["p_factor"]), "p_factor not float"

# %% [markdown]
# ## Step 2 -- Predict: what r2 should chance look like?
#
# **Predict.** A constant predictor that always returns the train-set
# median has ``r2 = 0`` against the test-set mean by definition --
# ``median_baseline`` formalises this. What r2 do you expect Ridge on
# faint EEG features to reach -- 0.05? 0.20? 0.50? Write your guess.

# %% [markdown]
# ## Step 3 -- Build a regression head on top of features
#
# A ``StandardScaler -> Ridge`` Pipeline (Pedregosa et al. 2011,
# doi:10.5555/1953048.2078195) is the regression analogue of plot_42's
# logistic head. ``alpha=1.0`` is a defensible default for an 8-feature
# table; ``random_state=42`` keeps the loop byte-stable.


# %%
def make_regressor() -> Pipeline:
    """Return a fresh ``StandardScaler -> Ridge`` Pipeline (E3.21)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )


# %% [markdown]
# ## Step 4 -- Cross-subject split, assert_no_leakage, train per fold
#
# **Run.** ``get_splitter("cross_subject", n_folds=5, random_state=42)``
# returns sklearn's ``GroupKFold`` keyed on ``subject``. We freeze the
# split into a manifest, walk every fold with ``assert_no_leakage(by=
# "subject")``, and emit the JSON ``leakage_report`` line that E5.42
# parses. With 12 subjects each fold tests on ~2-3 unseen subjects.

# %%
splitter = get_splitter(
    "cross_subject", engine="sklearn", n_folds=5, n_splits=5, random_state=SEED
)
manifest = make_split_manifest(splitter, y, metadata, target="target")
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "cross_subject manifest leaked subjects"
assert manifest["n_folds"] >= 5, "need at least 5 folds for mean +/- std"
print(
    f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']} | "
    f"max subject overlap: {overlap}"
)

# %% [markdown]
# ## Step 5 -- Mean r2 +/- std across folds vs the median baseline
#
# Loop the manifest, fit the Pipeline on the train fold, score on the
# held-out subjects with ``r2_score`` and ``mean_absolute_error``, and
# call ``median_baseline(y_train, y_test)`` for the chance level
# alongside (E5.43 forbids reporting a regression score without one).

# %%
fold_r2: list[float] = []
fold_mae: list[float] = []
fold_chance: list[float] = []
fold_baseline_mae: list[float] = []
test_residuals: list[tuple[str, float]] = []
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
    for sid, residual in zip(
        metadata.loc[te, "subject"].tolist(), (y_pred - y[te]).tolist()
    ):
        test_residuals.append((sid, float(residual)))
    print(
        f"Fold {k}: r2={fold_r2[-1]:+.3f} | mae={fold_mae[-1]:.3f} | "
        f"baseline_r2={fold_chance[-1]:+.3f}"
    )

mean_r2 = float(np.mean(fold_r2))
std_r2 = float(np.std(fold_r2, ddof=1))
mean_mae = float(np.mean(fold_mae))
mean_chance = float(np.mean(fold_chance))
mean_baseline_mae = float(np.mean(fold_baseline_mae))

# %% [markdown]
# ## Step 6 -- Investigate per-subject residuals
#
# **Investigate.** Mean r2 hides the failure mode that matters here:
# subject-invariance. If residuals concentrate around zero across all
# held-out subjects the model generalises; if a few subjects monopolise
# the error we are still leaking subject-specific signal. Aggregate the
# absolute residual per held-out subject and print a tiny histogram --
# this is the diagnostic Pernet et al. 2019 (doi:10.1038/s41597-019-0104-8)
# recommend before any clinical claim.

# %%
res_df = pd.DataFrame(test_residuals, columns=["subject", "residual"])
per_subject = (
    res_df.groupby("subject")["residual"]
    .apply(lambda r: float(np.mean(np.abs(r))))
    .sort_values()
)
print("Per-subject |residual| (sorted):")
print(per_subject.to_string(float_format=lambda v: f"{v:.3f}"))
edges = np.linspace(0.0, max(per_subject.max() * 1.05, 1e-3), 6)
print("ASCII histogram of per-subject |residual|:")
for low, high in zip(edges[:-1], edges[1:]):
    n = int(((per_subject >= low) & (per_subject < high)).sum())
    print(f"  [{low:.3f}, {high:.3f}): {'#' * n}")

# %% [markdown]
# ## A common mistake -- and how to recover
#
# **Run.** A frequent slip is wiring a non-numeric target column into
# ``Ridge.fit`` -- ``p_factor`` arrives as strings if a CSV was loaded
# without dtype hints, and Ridge then refuses to solve. We trigger it on
# purpose with ``try/except`` so you see exactly what the error looks
# like (Nederbragt et al. 2020, doi:10.1371/journal.pcbi.1008090).

# %%
try:
    bad_y = metadata["p_factor"].astype(str).to_numpy()  # string p-factor
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], bad_y[:8])
except (ValueError, TypeError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:90]}")
    # Recovery: cast the target to float before fitting any regression head.
    fixed_y = pd.to_numeric(metadata["p_factor"], errors="coerce").to_numpy()
    Ridge(alpha=1.0, random_state=SEED).fit(X[:8], fixed_y[:8])
    print(f"Recovery: cast p_factor to float (dtype={fixed_y.dtype}); Ridge fit.")

# %% [markdown]
# ## Modify -- domain-adversarial training (concept only)
#
# **Modify (concept).** A more aggressive way to enforce subject
# invariance is *domain-adversarial training* (Ganin et al. 2016,
# doi:10.5555/2946645.2946704). One head predicts ``p_factor``; a second
# head, attached via a gradient-reversal layer, tries to recover the
# subject identity. Penalising the *adversary's* accuracy pushes the
# encoder to discard subject-specific features. We do not run it here
# (Difficulty 3, GPU + braindecode encoder out of scope), but the cross-
# subject loop above is the contract any such system must still satisfy.

# %% [markdown]
# ## Result -- model r2 vs median baseline (chance)
#
# Five folds, disjoint subject test sets. ``mean +/- std`` against the
# regression chance level. Note that ``median_baseline`` returns the
# train-median predictor's r2 on the test set; an honest model must beat
# it, not just match it. The print line carries the keyword *baseline*
# (E5.43).

# %%
print(
    f"Cross-subject 5-fold p-factor regression: r2={mean_r2:+.3f} +/- {std_r2:.3f} "
    f"| mae={mean_mae:.3f} | baseline_r2={mean_chance:+.3f} "
    f"| baseline_mae={mean_baseline_mae:.3f} | metric: r2"
)
print(
    f"chance: predicting the train-median scores baseline_r2={mean_chance:+.3f} on test."
)
assert mean_mae < mean_baseline_mae, "Model MAE must be below the median-baseline MAE."

# %% [markdown]
# ## Wrap-up
# We loaded a Challenge-2-shaped feature table (with ``p_factor`` as a
# float column), built a 5-fold ``cross_subject`` manifest, asserted zero
# subject leakage, fit a Ridge head per fold, and reported r2 +/- std
# alongside ``median_baseline`` chance. The per-subject residual
# histogram is the subject-invariant diagnostic. p_factor is a derived
# score, not a diagnosis -- any clinical framing belongs in a follow-up
# study with much larger N.

# %% [markdown]
# ## Try it yourself
# - Swap ``Ridge`` for ``MLPRegressor`` (still ``random_state=42``).
# - Pre-train a Braindecode ``ShallowFBCSPNet`` and feed activations in.
# - Bump ``n_folds`` to ``N_SUBJECTS`` for leave-one-subject-out variance.

# %% [markdown]
# ## References
# - Cisotto & Chicco 2024, *PeerJ CS* 10:e2256.
#   https://doi.org/10.7717/peerj-cs.2256
# - Schirrmeister et al. 2017, *Hum. Brain Mapp.* 38:5391.
#   https://doi.org/10.1002/hbm.23730
# - Pernet et al. 2019, *Sci. Data* 6:103.
#   https://doi.org/10.1038/s41597-019-0104-8
# - Caspi et al. 2014, *Clin. Psychol. Sci.* 2:119.
#   https://doi.org/10.1177/2167702613497473
# - Nederbragt et al. 2020, Ten simple rules for teaching coding,
#   *PLOS Comp. Biol.* https://doi.org/10.1371/journal.pcbi.1008090
