"""Train a scikit-learn baseline on EEGDash features
=====================================================

We have a tidy band-power + variance feature table from plot_40 -- one row
per window, columns shaped ``<feature>_<channel>``. Before reaching for a
deep net we want a transparent eyes-open vs eyes-closed accuracy on a
leakage-safe split, with the chance level on the same line.

Can a logistic-regression Pipeline on a handful of EEG features beat the
majority-class chance level on a held-out subject?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_42_features_to_sklearn.png'
# %% [markdown]
# Learning objectives
# -------------------
# After this tutorial you will be able to:
#
# - Build a small feature DataFrame compatible with scikit-learn.
# - Run a leakage-safe ``cross_subject`` split via ``eegdash.splits`` (E5.42).
# - Fit a ``StandardScaler -> LogisticRegression`` Pipeline without test peek.
# - Compare accuracy against ``majority_baseline`` chance level (E5.43).
# - Save the trained Pipeline with ``joblib`` for downstream reuse.
#
# Requirements
# ------------
# - ~3 s on CPU, no GPU, no network.
# - Prerequisites: ``plot_40_first_features``, ``plot_11_leakage_safe_split``.
# - Concept: :doc:`/concepts/features_vs_deep_learning`.

# %%
# Setup -- numpy seeding (E3.21) plus a parametrised cache (E3.24).
import json
import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    get_splitter,
    majority_baseline,
    make_split_manifest,
)

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Step 1 -- Extract a small feature table
# ---------------------------------------
# In production you would reload ``plot_40_features.parquet``; offline we
# synthesise the same column layout (per-channel variance + four-band
# power). Eyes-closed gets the textbook alpha bump (Berger 1929,
# doi:10.1007/BF01797193).

# %%
CH_NAMES = ["O1", "Oz", "O2", "Cz"]
BANDS = ("delta", "theta", "alpha", "beta")
N_SUBJECTS, N_PER_SUBJECT = 4, 24
rng = np.random.default_rng(SEED)
rows = []
for subj in range(N_SUBJECTS):
    for w in range(N_PER_SUBJECT):
        label = (subj + w) % 2
        row = {"subject": f"sub-{subj:02d}", "target": int(label)}
        for ch in CH_NAMES:
            row[f"var_{ch}"] = float(rng.gamma(2.0, 1e-12) + 1e-12)
            for band in BANDS:
                base = rng.lognormal(mean=-2.0, sigma=0.3)
                if band == "alpha" and label == 1:
                    base *= 4.0 if ch != "Cz" else 1.5
                row[f"spec_{band}_{ch}"] = float(base)
        rows.append(row)
feature_table = pd.DataFrame(rows)
feature_cols = [c for c in feature_table.columns if c not in {"subject", "target"}]
n_features = len(feature_cols)
print(
    f"feature table: rows={len(feature_table)} | features={n_features} | "
    f"subjects={feature_table['subject'].nunique()} | "
    f"classes={dict(feature_table['target'].value_counts())}"
)

# %% [markdown]
# Step 2 -- Predict before you fit
# --------------------------------
# **Predict.** With ~20 features and a built-in alpha contrast, what
# accuracy do you expect a regularised linear baseline to reach above
# chance (~0.5 with balanced classes)? The PRIMM cycle (Sentance et al.
# 2019, doi:10.1080/08993408.2019.1608781) turns the gap into learning.
#
# Step 3 -- Leakage-safe split
# ----------------------------
# **Run #1.** ``get_splitter("cross_subject", ...)`` returns a MOABB
# ``CrossSubjectSplitter`` (or a sklearn ``GroupKFold`` keyed on
# ``subject``). ``assert_no_leakage`` prints the JSON contract line E5.42
# parses.

# %%
metadata = feature_table[["subject", "target"]].copy()
metadata["session"] = "ses-01"
metadata["run"] = "run-01"
metadata["dataset"] = "ds-plot42-mock"
metadata["sample_id"] = [f"row_{i:04d}" for i in range(len(metadata))]
splitter = get_splitter(
    "cross_subject", n_folds=N_SUBJECTS, n_splits=N_SUBJECTS, random_state=SEED
)
manifest = make_split_manifest(
    splitter, metadata["target"].to_numpy(), metadata, target="target"
)
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "cross_subject manifest leaked!"
train_mask = apply_split_manifest(metadata, manifest, fold=0, split="train")
test_mask = apply_split_manifest(metadata, manifest, fold=0, split="test")
X_train = feature_table.loc[train_mask, feature_cols].to_numpy()
X_test = feature_table.loc[test_mask, feature_cols].to_numpy()
y_train = feature_table.loc[train_mask, "target"].to_numpy()
y_test = feature_table.loc[test_mask, "target"].to_numpy()
print(
    f"fold 0: train={len(y_train)} "
    f"(n_subj={feature_table.loc[train_mask, 'subject'].nunique()}) | "
    f"test={len(y_test)} "
    f"(n_subj={feature_table.loc[test_mask, 'subject'].nunique()})"
)

# %% [markdown]
# Step 4 -- StandardScaler -> LogisticRegression Pipeline
# -------------------------------------------------------
# A ``Pipeline`` (Pedregosa et al. 2011, doi:10.5555/1953048.2078195)
# fits ``StandardScaler`` on ``X_train`` only and reapplies ``transform``
# at predict time; ``random_state=42`` makes accuracy byte-stable.

# %%
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=SEED, max_iter=400)),
    ]
)
pipe.fit(X_train, y_train)

# %% [markdown]
# Step 5 -- Score the test fold against chance
# --------------------------------------------
# **Run #2.** ``majority_baseline`` returns the test-set frequency of the
# most common label -- the metric pitfall Cisotto & Chicco 2024 Tip 9
# (doi:10.7717/peerj-cs.2256) flags hardest.

# %%
logreg_acc = float(accuracy_score(y_test, pipe.predict(X_test)))
chance = float(majority_baseline(y_train, y_test)["chance_level"])
print(
    f"LogReg pipeline accuracy: {logreg_acc:.3f} | "
    f"chance level: {chance:.3f} | metric: accuracy"
)

# %% [markdown]
# Step 6 -- Investigate the confusion matrix
# ------------------------------------------
# **Investigate.** Diagonal dominance means the alpha contrast
# transferred to a held-out subject; off-diagonal mass flags class drift.

# %%
cm = confusion_matrix(y_test, pipe.predict(X_test))
print("confusion_matrix [rows=true, cols=pred]:")
print(
    pd.DataFrame(
        cm, index=["true_open", "true_closed"], columns=["pred_open", "pred_closed"]
    )
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
#
# **Run.** A common slip is calling ``.fit`` on a bare classifier with
# un-scaled features -- ``LogisticRegression`` then warns about
# convergence (or in stricter mode raises ``ConvergenceWarning`` as an
# error). We trigger it with ``try/except`` so you see exactly what the
# failure mode looks like.

# %%
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bare = LogisticRegression(random_state=SEED, max_iter=10)
        bare.fit(X_train * 1e9, y_train)  # un-scaled, tiny max_iter
    print("Caught nothing (already-scaled features?).")
except (Warning, ValueError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:80]}")
    # Recovery: wrap the classifier in a Pipeline that scales first.
    print("Recovery: wrap StandardScaler -> LogisticRegression in a Pipeline.")

# %% [markdown]
# Modify -- swap LogisticRegression for RidgeClassifier
# -----------------------------------------------------
# **Modify.** Same Pipeline, different head: ``RidgeClassifier`` is the
# closed-form L2 alternative.

# %%
ridge = Pipeline(
    [("scaler", StandardScaler()), ("clf", RidgeClassifier(random_state=SEED))]
)
ridge.fit(X_train, y_train)
ridge_acc = float(accuracy_score(y_test, ridge.predict(X_test)))
print(f"RidgeClassifier accuracy: {ridge_acc:.3f}")

# %% [markdown]
# Make -- try LightGBM, fall back to RandomForest
# -----------------------------------------------
# **Make.** ``lightgbm`` is optional; ``try/except ImportError`` falls
# back to ``RandomForestClassifier``.

# %%
try:
    from lightgbm import LGBMClassifier  # type: ignore

    boost = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(random_state=SEED, n_estimators=200, verbose=-1)),
        ]
    )
    boost_name = "LightGBM"
except ImportError:
    boost = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=SEED, n_estimators=200)),
        ]
    )
    boost_name = "RandomForest"
boost.fit(X_train, y_train)
boost_acc = float(accuracy_score(y_test, boost.predict(X_test)))
print(f"{boost_name} accuracy: {boost_acc:.3f}")

# %% [markdown]
# Result -- model vs chance table
# -------------------------------
# All three Pipelines are scored on the same held-out subject with chance
# alongside (E5.43); the Pipeline is saved for reuse.

# %%
result = pd.DataFrame(
    [
        {
            "model": "LogReg",
            "accuracy": logreg_acc,
            "chance": chance,
            "n_features": n_features,
        },
        {
            "model": "Ridge",
            "accuracy": ridge_acc,
            "chance": chance,
            "n_features": n_features,
        },
        {
            "model": boost_name,
            "accuracy": boost_acc,
            "chance": chance,
            "n_features": n_features,
        },
    ]
)
print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
pipeline_path = cache_dir / "plot_42_logreg_pipeline.joblib"
joblib.dump(pipe, pipeline_path)
print(
    json.dumps(
        {
            "model_accuracy_test": round(logreg_acc, 3),
            "chance_level": round(chance, 3),
            "n_features": n_features,
            "saved_pipeline": pipeline_path.name,
        }
    )
)

# %% [markdown]
# Try it yourself / Extensions
# ----------------------------
# - Bump ``n_per`` to 64 and re-run; the gap above chance widens.
# - Reload the parquet table from plot_40 and reuse this Pipeline.
# - Inspect ``pipe.named_steps['clf'].coef_`` and rank top-k features.
# - Loop ``apply_split_manifest`` over all folds and report mean +/- std.
#
# References
# ----------
# - Cisotto & Chicco 2024, *PeerJ CS* 10:e2256. https://doi.org/10.7717/peerj-cs.2256
# - Pernet et al. 2019, *Sci. Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - Pedregosa et al. 2011, *JMLR* 12:2825. https://doi.org/10.5555/1953048.2078195
# - HBN ds005514 (Release 9). https://doi.org/10.18112/openneuro.ds005514.v1.0.0
