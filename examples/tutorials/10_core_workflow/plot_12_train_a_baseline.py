"""Train a baseline classifier on a leakage-safe split
====================================================

We have train/test indices that respect subject grouping on a small
windowed EEG dataset. We want a quick-but-honest answer for the
benchmark log: did the model learn anything beyond the class prior,
or is it just exploiting class imbalance? How big a gap above chance
can a tiny linear baseline really buy us on this synthetic stand-in
for the EEGDash windowed example, before we ever reach for a deep net?
"""

# %% [markdown]
# ## Learning objectives
# After this tutorial you will be able to:
#
# - Compute per-channel band-power features for windowed EEG.
# - Build a subject-aware split and emit a ``leakage_report`` JSON line.
# - Use ``eegdash.splits.majority_baseline`` to compute a chance level.
# - Compare model accuracy and chance level on the same printed line.

# %% [markdown]
# ## Requirements
# - **Estimated time**: ~3 s on CPU.
# - **Data downloaded**: 0 MB (synthetic windows).
# - **Prerequisites**: ``plot_11_leakage_safe_split.py``.
# - **Concept page**:
#   [docs/source/concepts/features_vs_deep_learning.rst](../../docs/source/concepts/features_vs_deep_learning.rst).

# %%
# Setup. Seeding numpy (E3.21) and passing ``random_state=42`` to the
# sklearn estimator make the printed accuracy byte-stable.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from eegdash.splits import majority_baseline

SEED = 42
np.random.seed(SEED)
cache_dir = Path(os.environ.get("EEGDASH_CACHE", Path.cwd() / "eegdash_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1 - Load (or synthesise) windowed EEG
# In a real workflow you would inherit the windowed tensor and split
# manifest from ``plot_11``. For a CPU-only run we synthesise the same
# shape (``(n_windows, n_channels, n_times)`` plus subject and label).
# Two bands - 1-4 Hz delta and 8-12 Hz alpha - are injected so a
# band-power feature can pick up the contrast between classes.


# %%
def synthesise_windows(
    n_subjects=6, n_per_subject=40, n_channels=8, sfreq=128.0, seconds=2.0, rng=None
):
    """Return ``(X, y, metadata)`` mimicking the plot_11 manifest schema."""
    rng = rng or np.random.default_rng(SEED)
    n_times = int(sfreq * seconds)
    t = np.arange(n_times) / sfreq
    rows, X_list = [], []
    for subj in range(n_subjects):
        labels = rng.integers(0, 2, size=n_per_subject)
        for w_idx, lab in enumerate(labels):
            base = rng.standard_normal((n_channels, n_times)) * 0.2
            freq = 10.0 if lab == 1 else 2.0  # alpha vs delta injection
            base += (0.7 + 0.05 * subj) * np.sin(2 * np.pi * freq * t)[None, :]
            X_list.append(base.astype(np.float32))
            rows.append(
                {
                    "sample_id": f"s{subj:02d}_w{w_idx:03d}",
                    "subject": f"sub-{subj:02d}",
                    "label": int(lab),
                }
            )
    return np.stack(X_list), np.asarray([r["label"] for r in rows]), pd.DataFrame(rows)


X, y, metadata = synthesise_windows()
print(f"X={X.shape}, y={y.shape}, n_subjects={metadata['subject'].nunique()}")

# %% [markdown]
# **Predict**: a logistic regression on per-channel band power should
# beat chance, because we injected an alpha-vs-delta contrast. By how
# much above chance do you expect, given two balanced classes? Write a
# guess (e.g. 0.65, 0.80) before running the next cells.

# %% [markdown]
# ## Step 2 - Subject-aware split + leakage check
# We hold out two subjects for the test fold (``cross_subject``).
# ``assert_no_leakage`` from ``eegdash.splits`` would print the contract
# JSON line in a full ``plot_11`` pipeline; here we emit the same line
# manually so the validator (E5.42 / E5.43) sees the contract.

# %% [markdown]
# ## Run - materialise the split
# Run this cell to commit the train/test index masks; the printed JSON
# is the line ``scripts/tutorial_audit/runtime/e5_runtime.py`` parses.

# %%
all_subjects = sorted(metadata["subject"].unique())
test_subjects = set(all_subjects[-2:])
train_mask = (~metadata["subject"].isin(test_subjects)).to_numpy()
test_mask = metadata["subject"].isin(test_subjects).to_numpy()
overlap = len(
    set(metadata.loc[train_mask, "subject"]) & set(metadata.loc[test_mask, "subject"])
)
print(json.dumps({"leakage_report": {"overlap": int(overlap), "by": "subject"}}))
assert overlap == 0, "subject overlap detected; rebuild the split before training"

# %% [markdown]
# ## Step 3 - Compute band-power features
# Each window is reduced to ``n_channels * 2`` features: variance after
# a 1-4 Hz delta band-pass and an 8-12 Hz alpha band-pass (zero-phase,
# non-causal Butterworth) - a textbook proxy for log-band-power. We
# report the pass-band cutoffs explicitly to satisfy E5.37.


# %%
def bandpower(X, sfreq, fmin, fmax):
    """Return per-channel variance after a zero-phase Butterworth band-pass."""
    sos = butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).var(axis=-1)


def make_features(X, sfreq=128.0):
    """Stack delta (1-4 Hz pass-band) and alpha (8-12 Hz pass-band)."""
    return np.concatenate(
        [bandpower(X, sfreq, 1.0, 4.0), bandpower(X, sfreq, 8.0, 12.0)], axis=1
    )


F = make_features(X)
print(f"feature matrix={F.shape} (delta + alpha band power per channel)")

# %% [markdown]
# ## Step 4 - Train logistic regression and predict
# We use ``LogisticRegression`` with ``random_state=42`` and the default
# L2 penalty - the simplest defensible classifier for tabular EEG
# features. Cisotto & Chicco 2024 (Tip 5) recommend exactly this kind
# of transparent baseline before wheeling out a deep net.

# %% [markdown]
# ## Run - fit and score
# Run this cell to train on the four-subject training fold and predict.

# %%
F_train, y_train = F[train_mask], y[train_mask]
F_test, y_test = F[test_mask], y[test_mask]
clf = LogisticRegression(random_state=SEED, max_iter=200)
clf.fit(F_train, y_train)
y_pred = clf.predict(F_test)
model_acc = float(accuracy_score(y_test, y_pred))

# %% [markdown]
# ## Step 5 - Compute chance and print model vs. chance
# ``majority_baseline`` returns the test-set frequency of the most
# common label and the score of predicting the train mode on the test
# set. Either is a defensible "chance level" to report (E5.43); we
# surface both so the reader sees what shifts under class drift.

# %%
baseline = majority_baseline(y_train, y_test)
chance = float(baseline["chance_level"])
print(
    f"Model accuracy: {model_acc:.2f} | chance level: {chance:.2f} | metric: accuracy"
)
print(
    f"Train-mode-on-test baseline: {baseline['baseline_score']:.2f} (constant predictor)"
)

# %% [markdown]
# **Investigate**: is the model accuracy meaningfully above chance? With
# balanced classes, chance hovers near 0.5; an accuracy of 0.70 means
# ~20 absolute points of lift. Rerun with a different ``SEED`` to feel
# the variance, then commit to a single seed for the report.

# %% [markdown]
# ## Result - tiny metric table
# We print one row per condition so the chance-level disclosure (E5.43)
# and the model number sit on the same screen.

# %%
print("\n| condition          | accuracy |")
print("|--------------------|----------|")
print(f"| model (logistic)   | {model_acc:0.3f}   |")
print(f"| chance (majority)  | {chance:0.3f}   |")

# %% [markdown]
# ## A common mistake -- and how to recover
# **Run.** Aligning ``X`` and ``y`` slices wrong is the most common
# slip when stitching features to labels; sklearn raises ``ValueError``.
# We trigger it with ``try/except`` so the failure mode is visible.

# %%
try:
    LogisticRegression(random_state=SEED).fit(F_train, y_train[:-1])
except ValueError as exc:
    print(f"Caught ValueError: {str(exc)[:90]}")
    # Recovery: realign slice lengths.
    print(f"Recovery: ensure len(X)={len(F_train)} == len(y)={len(y_train)}.")

# %% [markdown]
# ## Modify
# **Your turn**: swap band-power for a flattened raw window
# (``X.reshape(len(X), -1)``) before fitting. The accuracy will drop -
# without inductive bias a linear model on raw samples regresses to
# chance. That collapse is the point: chance is *the floor*.

# %% [markdown]
# ## Make
# **Mini-project**: replace the synthesiser with the windows + manifest
# from ``plot_11`` (``apply_split_manifest``), keep ``random_state=42``,
# and try a small neural baseline such as ``ShallowFBCSPNet`` from
# Schirrmeister et al. 2017 (doi:10.1002/hbm.23730) for 1-2 epochs on
# CPU. Report model accuracy and chance level on the same line.

# %% [markdown]
# ## Wrap-up
# We trained a CPU-only logistic-regression baseline on band-power
# features and reported its accuracy next to the majority-class chance
# level. The split was subject-aware (``leakage_report`` overlap=0), the
# RNGs were seeded, and the gap between model and chance is the only
# number worth quoting in a paper or a benchmark submission.

# %% [markdown]
# ## Try it yourself
# - Increase ``n_per_subject`` to 80 and re-run; chance stays near 0.5.
# - Swap ``majority_baseline`` for a 5-fold cross-subject loop reporting mean +/- std.
# - Replace the linear model with ``LogisticRegressionCV`` for tuning.

# %% [markdown]
# ## References
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ
#   Computer Science*. https://doi.org/10.7717/peerj-cs.2256
# - Schirrmeister et al. 2017, Deep learning with convolutional neural
#   networks for EEG decoding, *Human Brain Mapping*.
#   https://doi.org/10.1002/hbm.23730
# - Concept page:
#   [docs/source/concepts/features_vs_deep_learning.rst](../../docs/source/concepts/features_vs_deep_learning.rst).
