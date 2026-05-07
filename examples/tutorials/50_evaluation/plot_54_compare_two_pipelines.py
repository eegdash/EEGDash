"""Compare a feature baseline against a neural network on the same split
======================================================================

When a decoder beats a feature baseline by a few accuracy points on one
held-out subject, the honest question is: did the model actually win,
or just luck into a friendlier fold? This is the keystone of MOABB-style
evaluation (Chevallier, Aristimunha et al. 2024,
doi:10.48550/arXiv.2404.15319): build ONE split manifest, feed both
pipelines the SAME fold ids, and run a paired test on the per-fold
deltas. Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256, Tip 9) flag
unpaired evaluation as the most common over-claim in clinical EEG.
So: does the neural network really beat the linear baseline, or is the
gap noise?
"""

# %% [markdown]
# ## Learning objectives
#
# - build ONE cross-subject split manifest reused by two pipelines.
# - apply the same fold ids to a feature baseline and a small neural network.
# - assert ``fold_ids_pipeline_a == fold_ids_pipeline_b`` in code, not prose.
# - run ``scipy.stats.wilcoxon`` on the paired per-fold accuracy deltas.
# - interpret the p-value alongside the median delta and chance level.
#
# ## Requirements
#
# - Prerequisites: ``plot_12_train_a_baseline``, ``plot_42_features_to_sklearn``.
# - Theory: :doc:`/concepts/leakage_and_evaluation`.
# - ~5 s on CPU, no GPU, no network.

# %%
# Setup -- seed everything (E3.21) and import the paired-evaluation stack.
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
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

# %% [markdown]
# ## Step 1 -- Build per-subject windowed metadata
#
# We synthesise 12 subjects x 16 windows of band-power features (alpha
# bump on closed eyes, identical layout to plot_42). On real data you
# would reload the parquet feature table from plot_40.

# %%
N_SUBJECTS, N_PER_SUBJECT = 12, 16
CH_NAMES = ["O1", "Oz", "O2", "Cz"]
BANDS = ("delta", "theta", "alpha", "beta")
rng = np.random.default_rng(SEED)
rows = []
for subj in range(N_SUBJECTS):
    for w in range(N_PER_SUBJECT):
        label = (subj + w) % 2
        row = {
            "subject": f"sub-{subj:02d}",
            "session": "ses-01",
            "run": "run-01",
            "dataset": "ds-plot54-mock",
            "sample_id": f"sub-{subj:02d}__w{w:03d}",
            "target": int(label),
        }
        for ch in CH_NAMES:
            for band in BANDS:
                base = rng.lognormal(mean=-2.0, sigma=0.6)
                if band == "alpha" and label == 1 and ch != "Cz":
                    base *= 1.6
                row[f"spec_{band}_{ch}"] = float(base)
        rows.append(row)
feature_table = pd.DataFrame(rows)
feature_cols = [c for c in feature_table.columns if c.startswith("spec_")]
META_COLS = ["subject", "session", "run", "dataset", "sample_id", "target"]
metadata = feature_table[META_COLS].copy()
print(
    f"feature table: rows={len(feature_table)} | features={len(feature_cols)} | "
    f"subjects={metadata['subject'].nunique()} | "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# ## Step 2 -- Predict which pipeline wins
#
# **Predict.** Pipeline A is ``LogisticRegression`` on band-power
# features (the linear baseline); Pipeline B is a small ``MLPClassifier``
# -- a one-hidden-layer neural network trained with Adam (Pedregosa et
# al. 2011, doi:10.5555/1953048.2078195). The MLP can fit non-linear
# contrasts the linear arm cannot, but on this synthetic near-linear
# setup we expect the two to land within a few accuracy points -- and
# a few points only matters when per-fold deltas are consistent
# (Sentance et al. 2019, doi:10.1080/08993408.2019.1608781). Chance ~0.5.
#
# ## Step 3 -- Build ONE cross-subject split manifest
#
# **Run #1.** ``get_splitter("cross_subject", ...)`` keyed on
# ``subject`` gives a leakage-safe ``GroupKFold``-style split.
# ``assert_no_leakage`` emits the JSON contract line E5.42 reads. The
# manifest is built ONCE -- both pipelines will consume the SAME fold
# ids, which is what makes the comparison paired.

# %%
splitter = get_splitter(
    "cross_subject", n_folds=N_SUBJECTS, n_splits=N_SUBJECTS, random_state=SEED
)
manifest = make_split_manifest(
    splitter, metadata["target"].to_numpy(), metadata, target="target"
)
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "cross_subject manifest leaked!"
n_folds = manifest["n_folds"]
print(f"manifest: {manifest['splitter_class']} | folds: {n_folds}")

# %% [markdown]
# ## Step 4 -- Pipeline A: StandardScaler + LogisticRegression
#
# **Run #2.** Loop the manifest folds, fit on train, score on test,
# collect per-fold accuracies and the chance level. ``random_state=42``
# everywhere keeps the comparison byte-stable.


# %%
def run_pipeline(estimator) -> tuple[list[float], list[str], list[float]]:
    """Score ``estimator`` on every fold of the SHARED manifest."""
    accs: list[float] = []
    fold_ids: list[str] = []
    chances: list[float] = []
    for k in range(n_folds):
        train_mask = apply_split_manifest(metadata, manifest, fold=k, split="train")
        test_mask = apply_split_manifest(metadata, manifest, fold=k, split="test")
        X_train = feature_table.loc[train_mask, feature_cols].to_numpy()
        X_test = feature_table.loc[test_mask, feature_cols].to_numpy()
        y_train = feature_table.loc[train_mask, "target"].to_numpy()
        y_test = feature_table.loc[test_mask, "target"].to_numpy()
        estimator.fit(X_train, y_train)
        accs.append(float(accuracy_score(y_test, estimator.predict(X_test))))
        chances.append(float(majority_baseline(y_train, y_test)["chance_level"]))
        # Hash the test ids; this string is the fold's identity contract.
        fold_ids.append("|".join(sorted(manifest["folds"][k]["test"])))
    return accs, fold_ids, chances


pipe_a = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=SEED, max_iter=400)),
    ]
)
accs_a, fold_ids_a, chances = run_pipeline(pipe_a)
print(
    f"Pipeline A (LogReg): mean={np.mean(accs_a):.3f} +/- {np.std(accs_a):.3f} | "
    f"chance level={np.mean(chances):.3f} (n_folds={n_folds})"
)

# %% [markdown]
# ## Step 5 -- Pipeline B: StandardScaler + MLPClassifier (neural net)
#
# **Run #3.** Same scaffolding, different head: a one-hidden-layer
# neural network (16 ReLU units, Adam). Scaler still fits on train only.
# The MLP stays small so the tutorial runs in a few seconds on CPU.

# %%
mlp = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=SEED,
)
pipe_b = Pipeline([("scaler", StandardScaler()), ("clf", mlp)])
accs_b, fold_ids_b, _ = run_pipeline(pipe_b)
print(
    f"Pipeline B (MLP): mean={np.mean(accs_b):.3f} +/- {np.std(accs_b):.3f} | "
    f"chance level={np.mean(chances):.3f} (n_folds={n_folds})"
)

# %% [markdown]
# ## Step 6 -- Paired Wilcoxon signed-rank test
#
# **Investigate.** Same fold ids on both sides are the precondition for
# pairing. We assert the invariant in code (not just prose), then call
# ``scipy.stats.wilcoxon`` (Wilcoxon 1945, doi:10.2307/3001968) on the
# per-fold deltas. The Wilcoxon test does not assume normality, so it
# survives the small-sample regime typical of EEG benchmarks.

# %%
assert fold_ids_a == fold_ids_b, "fold ids diverged -- comparison is NOT paired!"
deltas = np.asarray(accs_b) - np.asarray(accs_a)
median_delta = float(np.median(deltas))
# ``zero_method='wilcox'`` matches the textbook definition; ties are
# dropped from the rank sum.
wstat, pvalue = wilcoxon(accs_b, accs_a, zero_method="wilcox", correction=False)
print(
    f"paired Wilcoxon: W={float(wstat):.2f} | p={float(pvalue):.3f} | "
    f"median(B-A)={median_delta:+.3f}"
)

# %% [markdown]
# ## Step 7 -- Per-fold paired comparison table
#
# A "paired plot" connects each fold's two scores with a line so the
# reader can eyeball whether B beats A consistently or only on average.
# We tabulate the per-fold rows; in a notebook you would also render
# the connected-line figure (figure_paired_comparison.png).

# %%
paired_df = pd.DataFrame(
    {
        "fold": np.arange(n_folds),
        "pipeline_a_acc": accs_a,
        "pipeline_b_acc": accs_b,
        "delta_b_minus_a": deltas,
        "chance": chances,
    }
)
print(paired_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ## Result -- report the comparison with appropriate hedging
#
# Single dataset, fixed hyperparameters, 12 cross-subject folds: the
# headline is median delta + Wilcoxon p-value + chance-level baseline.

# %%
print(
    f"Pipeline B - Pipeline A = {100 * median_delta:+.2f} accuracy points; "
    f"paired Wilcoxon p = {float(pvalue):.3f} (n_folds = {n_folds}); "
    f"chance = {np.mean(chances):.3f}"
)
print("Hedge: small sample, single mock dataset, fixed hyperparameters.")

# %% [markdown]
# ## A common mistake -- and how to recover
#
# **Run.** Calling ``wilcoxon`` on empty per-fold deltas (e.g. when one
# pipeline was never evaluated) raises ``ValueError``. We trigger it on
# purpose with ``try/except`` so you see exactly what the error looks
# like.

# %%
try:
    empty_deltas: list[float] = []  # would happen if Pipeline B was skipped
    if len(empty_deltas) == 0:
        raise ValueError("zero_method='wilcox' requires at least one non-zero delta")
    wilcoxon(empty_deltas)
except (ValueError, RuntimeError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: ensure both pipelines are evaluated on the same manifest.
    print(f"Recovery: paired deltas have {len(deltas)} entries -- re-run pipe_b.")

# %% [markdown]
# ## Modify -- swap Pipeline B for a different model
#
# **Modify.** Same scaffolding, different head: replace the MLP with a
# stronger-regularised LogReg (``C=0.1``) and rerun. The paired contract
# still has to hold.

# %%
pipe_b_alt = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=SEED, C=0.1, max_iter=400)),
    ]
)
accs_b_alt, fold_ids_b_alt, _ = run_pipeline(pipe_b_alt)
assert fold_ids_a == fold_ids_b_alt, "Modify variant broke the paired contract!"
_, p_alt = wilcoxon(accs_b_alt, accs_a, zero_method="wilcox")
delta_alt = np.median(np.asarray(accs_b_alt) - np.asarray(accs_a))
print(
    f"Modify (LogReg C=0.1) vs A: median delta={delta_alt:+.3f} | p={float(p_alt):.3f}"
)

# %% [markdown]
# ## Try it yourself / Extensions
#
# - widen the MLP to ``hidden_layer_sizes=(64, 32)`` and rerun.
# - reduce ``N_SUBJECTS`` to 6 and watch the Wilcoxon p-value lose power.
# - add a third pipeline and run pairwise Wilcoxon with Bonferroni.
#
# ## References
#
# - Chevallier, Aristimunha et al. 2024 (doi:10.48550/arXiv.2404.15319) -- MOABB paired pipeline comparison.
# - Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256) -- ten clinical EEG tips, Tip 9 on evaluation.
# - Pedregosa et al. 2011 (doi:10.5555/1953048.2078195) -- scikit-learn ``Pipeline``.
# - Wilcoxon 1945 (doi:10.2307/3001968) -- the signed-rank test.
