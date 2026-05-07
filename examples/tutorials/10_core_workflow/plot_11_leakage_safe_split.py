"""Split EEG data without subject leakage
======================================

Why does randomly splitting EEG windows on the windows from plot_10
give you 99% accuracy that does not generalize to a new participant on a
held-out subject from the same EEGDash recording set?

Because every recording contributes hundreds of overlapping windows from
the same subject, so a uniform random split scatters each subject across
both train and test. The model memorises subject-specific neural
fingerprints and the score collapses on the next person. We show the
failure first, then build the leakage-safe alternative with the
``eegdash.splits`` helpers and a MOABB-backed cross-subject splitter
(Cisotto & Chicco 2024, Tip 9, doi:10.7717/peerj-cs.2256), persisted as
BIDS-style split metadata (Pernet et al. 2019,
doi:10.1038/s41597-019-0104-8). So why does a random window split look great
and yet fail in deployment?
"""

# sphinx_gallery_thumbnail_path = '_static/thumbs/plot_11_leakage_safe_split.png'

# %% [markdown]
# Learning objectives
# -------------------
#
# - identify subject leakage as the failure mode of naive random splits.
# - build a leakage-safe 5-fold split with ``get_splitter("cross_subject")``.
# - run ``assert_no_leakage`` and read the JSON ``leakage_report`` it emits.
# - compare a leaky window-level random split against a subject-aware split.
# - save a split manifest and read the ``describe_split`` audit.
#
# Requirements
# ------------
#
# - You finished
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`.
# - Theory: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup -- seed and imports.
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    describe_split,
    get_splitter,
    make_split_manifest,
    manifest_to_json,
)

warnings.simplefilter("ignore", category=FutureWarning)
np.random.seed(42)

# %% [markdown]
# Step 1 -- Load windows metadata for a handful of subjects
# ---------------------------------------------------------
#
# After ``plot_10`` you would reload windows from disk and read
# ``windows_ds.description`` for per-recording subject IDs. To keep the
# split discipline the only moving part we materialise the metadata table
# directly: ``eegdash.splits`` works the same way on a Braindecode
# ``WindowsDataset`` and on a DataFrame -- both expose a ``subject`` column.
# 12 subjects x 2 sessions x 8 windows = 192 rows.

# %%
rows = [
    {
        "subject": f"sub-{s:02d}",
        "session": f"ses-{ses:02d}",
        "run": "run-01",
        "dataset": "ds-windowed-tutorial",
        "sample_id": f"sub-{s:02d}__ses-{ses:02d}__w{w:03d}",
        "target": int((s + w) % 2),
    }
    for s in range(12)
    for ses in range(2)
    for w in range(8)
]
metadata = pd.DataFrame(rows)
print(
    f"Windows metadata: rows={len(metadata)}, "
    f"subjects={metadata['subject'].nunique()}, "
    f"classes={dict(metadata['target'].value_counts())}"
)

# %% [markdown]
# Step 2 -- Predict, then run the WRONG way
# -----------------------------------------
#
# **Predict.** If we shuffle these 192 windows uniformly and put 20% in a
# test fold, how many subjects will end up in both train and test -- 0,
# around 5, or all 12?
#
# **Run.** We do exactly that wrong split: a window-level random shuffle.

# %%
rng = np.random.default_rng(42)
shuffled = rng.permutation(len(metadata))
cut = int(0.8 * len(metadata))
naive_train = metadata.iloc[shuffled[:cut]]
naive_test = metadata.iloc[shuffled[cut:]]
leaked = sorted(set(naive_train["subject"]) & set(naive_test["subject"]))
naive_overlap = len(leaked)
print(
    f"Naive random split: train={len(naive_train)}, test={len(naive_test)}, "
    f"subject_overlap={naive_overlap}/{metadata['subject'].nunique()}"
)
assert naive_overlap > 0

# %% [markdown]
# **Investigate.** Almost every subject sits on both sides of the split. A
# classifier here can memorise the alpha-rhythm fingerprint of subject 03 and
# recognise it again on the test windows of subject 03 -- the accuracy is a
# subject-identification score, not a paradigm decoding score. Cisotto &
# Chicco 2024 (Tip 9) flag this as the most common evaluation pitfall in
# clinical EEG; Chevallier, Aristimunha et al. 2024 use the same cross-subject
# protocol throughout the MOABB benchmark.

# %% [markdown]
# Step 3 -- Build a leakage-safe 5-fold split manifest
# ----------------------------------------------------
#
# **Run (#2).** ``get_splitter("cross_subject", ...)`` returns a MOABB
# ``CrossSubjectSplitter`` (or a sklearn ``GroupKFold`` keyed on ``subject``
# when MOABB is unavailable). Either way no fold can put the same subject
# on both sides. ``make_split_manifest`` freezes the output into a JSON
# manifest with provenance: class + kwargs, library versions, target.

# %%
splitter = get_splitter("cross_subject", n_folds=5, n_splits=5, random_state=42)
manifest = make_split_manifest(
    splitter, metadata["target"].to_numpy(), metadata, target="target"
)
print(f"Splitter: {manifest['splitter_class']} | folds: {manifest['n_folds']}")

# %% [markdown]
# Step 4 -- Prove no subject leakage and read the audit
# -----------------------------------------------------
#
# ``assert_no_leakage`` walks every fold, intersects ``subject`` values
# across train/test, and always prints one JSON line --
# ``{"leakage_report": {"overlap": 0, "by": "subject"}}`` for a clean split,
# the same line with non-zero ``overlap`` (and a raised ``LeakageError``)
# when a fold leaks. Runtime validator E5.42 reads exactly that line.
# **Run (#3).** ``describe_split`` then prints a one-screen audit: per-fold
# sizes, distinct subjects on each side, and per-fold class balance.

# %%
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "Cross-subject manifest leaked!"
summary = describe_split(manifest, metadata, target="target")
fold0 = summary["per_fold"][0]
balance0 = fold0["class_balance_test"]
class_balance_ratio = max(balance0.values()) / (sum(balance0.values()) or 1)
print(
    f"Fold 0: subjects_train={fold0['subjects_train']}, "
    f"subjects_test={fold0['subjects_test']}, "
    f"class_balance_ratio={class_balance_ratio:.2f}"
)

# %% [markdown]
# Step 5 -- Materialise one fold and persist the manifest
# -------------------------------------------------------
#
# ``apply_split_manifest`` returns a boolean mask for any fold; the
# manifest serialises to plain JSON -- the BIDS "split metadata" Pernet
# et al. 2019 advocate for sharing alongside derivatives.

# %%
train_mask = apply_split_manifest(metadata, manifest, fold=0, split="train")
test_mask = apply_split_manifest(metadata, manifest, fold=0, split="test")
cache_dir = Path("./eegdash_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
manifest_path = cache_dir / "plot_11_split_manifest.json"
manifest_path.write_text(manifest_to_json(manifest), encoding="utf-8")

# %% [markdown]
# Result -- naive vs leakage-safe
# -------------------------------
#
# Naive split leaked subjects; safe split prints
# ``{"leakage_report": {"overlap": 0, "by": "subject"}}`` (E5.42), with 5
# folds across 12 subjects and balanced classes.

# %%
print(f"Fold 0: train={int(train_mask.sum())}, test={int(test_mask.sum())}")
print(
    "Final invariants:",
    json.dumps(
        {
            "n_subjects_total": int(metadata["subject"].nunique()),
            "n_folds": int(manifest["n_folds"]),
            "subject_overlap": int(overlap),
            "naive_random_split_overlap": int(naive_overlap),
            "class_balance_ratio_fold0": round(float(class_balance_ratio), 3),
        }
    ),
)

# %% [markdown]
# A common mistake -- and how to recover
# --------------------------------------
# **Run.** Mistyping the splitter name is the single most common slip
# in this API; ``get_splitter`` raises ``KeyError`` with the valid
# names. We trigger it with ``try/except`` so the error is visible.

# %%
try:
    _ = get_splitter("subject_split", n_folds=5, random_state=42)
except (KeyError, ValueError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    # Recovery: use the canonical name.
    fixed = get_splitter("cross_subject", n_folds=5, random_state=42)
    print(f"Recovery: get_splitter('cross_subject') -> {type(fixed).__name__}")

# %% [markdown]
# Modify -- try a session-aware split
# -----------------------------------
#
# **Modify.** Swap ``"cross_subject"`` for ``"cross_session"`` and re-run
# ``assert_no_leakage`` with ``by="session"``: the scaffolding stays put,
# only the invariant changes.

# %%
session_splitter = get_splitter("cross_session", n_folds=2, random_state=42)
session_manifest = make_split_manifest(
    session_splitter, metadata["target"].to_numpy(), metadata, target="target"
)
session_overlap = assert_no_leakage(session_manifest, metadata, by="session")
print(f"cross_session overlap: {session_overlap}")

# %% [markdown]
# Make -- apply it to your own windows
# ------------------------------------
#
# **Make.** Apply this flow to the windows you saved in plot_10: pass the
# ``WindowsConcatDataset`` to ``to_split_metadata``, then back into
# ``make_split_manifest`` and ``assert_no_leakage``.

# %% [markdown]
# Extensions
# ----------
#
# - change ``random_state`` and confirm folds shift but disjointness holds.
# - edit ``n_folds`` to 10 and see the per-fold subject count change.
# - swap to a real ``WindowsConcatDataset`` from plot_10 and re-run end-to-end.
# - add ``by="session"`` to ``assert_no_leakage`` and watch the report flip.

# %% [markdown]
# Links
# -----
#
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - API: ``get_splitter``, ``make_split_manifest``, ``assert_no_leakage``.
# - Cisotto & Chicco 2024 (doi:10.7717/peerj-cs.2256) -- Tip 9.
# - Pernet et al. 2019 (doi:10.1038/s41597-019-0104-8) -- EEG-BIDS.
# - Chevallier, Aristimunha et al. 2024 (doi:10.48550/arXiv.2404.15319) -- MOABB benchmark.
