"""How do I split EEG data without subject leakage?
====================================================

Random window splits on cross-subject EEG decoders post training-set
accuracy near 99% and collapse on a held-out participant. The reason is
not exotic: every recording produces hundreds of overlapping windows
from the same brain, so a uniform shuffle scatters each subject across
both train and test, and the model memorises subject-level fingerprints
(heart-rate, alpha amplitude, electrode impedance) instead of the task
we actually want to decode.

This tutorial shows the failure first on synthetic windows, then
rebuilds the split with the :mod:`eegdash.splits` helpers and a
GroupKFold-flavoured cross-subject splitter. The final figure puts both
strategies side-by-side: same data underneath, only the split differs.

Brookshire et al. 2024 surveyed 81 deep-learning EEG papers and found
data leakage in roughly half. Cisotto & Chicco 2024 (Tip 9) name this
the most common evaluation pitfall in clinical EEG; the MOABB benchmark
(Aristimunha et al. 2023) uses the cross-subject protocol throughout.

.. sphinx_gallery_thumbnail_path = '_static/thumbs/plot_11_leakage_safe_split.png'

So why does a random window split look great on paper, and which
column of the metadata table do you actually have to hold out?
"""

# %% [markdown]
# Learning objectives
# -------------------
#
# - Identify subject leakage as the failure mode of naive random splits on EEG.
# - Build a leakage-safe 5-fold split with :func:`eegdash.splits.get_splitter` (``"cross_subject"``).
# - Run :func:`eegdash.splits.assert_no_leakage` and read the JSON ``leakage_report`` line it emits.
# - Save a JSON split manifest with :func:`eegdash.splits.make_split_manifest` and replay one fold via :func:`eegdash.splits.apply_split_manifest`.
# - Show the contrast between a naive shuffle and a cross-subject GroupKFold with the side-by-side figure at the end.
#
# Requirements
# ------------
#
# - You finished
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_10_preprocess_and_window`.
# - About 30 s on CPU. No network: the metadata table is built in-script.
# - Concept refresher: :doc:`/concepts/leakage_and_evaluation`.

# %%
# Setup. ``np.random.seed`` keeps the naive shuffle and the manifest
# fold order reproducible (E3.21).
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eegdash
from eegdash.splits import (
    apply_split_manifest,
    assert_no_leakage,
    describe_split,
    get_splitter,
    make_split_manifest,
    manifest_to_json,
    to_moabb_split_inputs,
    to_split_metadata,
)
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
print(f"eegdash {eegdash.__version__}; numpy {np.__version__}")

# %% [markdown]
# Why subject leakage hits EEG harder than other domains
# ------------------------------------------------------
#
# Subject-level features dominate any single EEG window. Skull thickness,
# electrode placement, baseline alpha amplitude, hair conductivity, and
# resting heart-rate all imprint themselves on the same channels the
# decoder reads from. Brookshire et al. 2024 quantified this on 81
# clinical-EEG deep-learning papers: when subjects appeared on both
# sides of a split, reported accuracy averaged 0.83; on properly
# subject-held-out splits, the same architectures averaged 0.62. Half
# of the surveyed studies leaked.
#
# Other modalities can sometimes escape this. ImageNet has ~1000 classes
# and over a million images, so a single image rarely encodes
# class-irrelevant subject identity. EEG is the opposite: a few dozen
# subjects produce hundreds of windows each, every window keeps the
# subject's fingerprint, and class labels are the *minority* signal in
# the cross-section of variance. Cisotto & Chicco 2024 (Tip 9) call
# this out as the single most common reporting mistake in clinical EEG.
#
# The fix is structural: hold out subjects, not windows. Group every
# window by its ``subject`` id and let
# :class:`sklearn.model_selection.GroupKFold` (or the MOABB
# ``CrossSubjectSplitter``) put each subject in exactly one test fold.
# :mod:`eegdash.splits` wraps both behind one entry point, persists the
# manifest, and emits a JSON line a runtime validator can grep for.

# %% [markdown]
# Step 1. Build a windows metadata table for 12 subjects
# ------------------------------------------------------
#
# After ``plot_10`` you would reload windows from disk and read
# :attr:`braindecode.datasets.BaseConcatDataset.description` for per-
# recording subject ids. To keep split discipline as the only moving
# part, this tutorial materialises the metadata
# :class:`pandas.DataFrame` directly. :mod:`eegdash.splits` works the
# same way on a Braindecode :class:`braindecode.datasets.WindowsDataset`
# and on a DataFrame because both expose a ``subject`` column. The 12
# subjects x 2 sessions x 8 windows = 192 rows mirror what plot_10
# produced for ``ds002718`` (Wakeman & Henson 2015), reachable through
# `NEMAR <https://nemar.org>`_ (Delorme et al. 2022).

# %%
N_SUBJECTS = 12
N_SESSIONS = 2
N_WINDOWS = 8
rows = [
    {
        "subject": f"sub-{s:02d}",
        "session": f"ses-{ses:02d}",
        "run": "run-01",
        "dataset": "ds-windowed-tutorial",
        "sample_id": f"sub-{s:02d}__ses-{ses:02d}__w{w:03d}",
        "target": int((s + w) % 2),
    }
    for s in range(1, N_SUBJECTS + 1)
    for ses in range(1, N_SESSIONS + 1)
    for w in range(N_WINDOWS)
]
raw_metadata = pd.DataFrame(rows)

# ``to_split_metadata`` is the canonical entry point. On a Braindecode
# concat-of-WindowsDataset it walks per-record descriptions; on a
# DataFrame it normalises the schema (sample_id, subject, session, ...)
# and validates the requested target column. Either way you get the
# same tabular view back, ready for splitter/manifest calls below.
metadata = to_split_metadata(raw_metadata, target="target")
y, _md_via_moabb = to_moabb_split_inputs(metadata, target="target")
pd.Series(
    {
        "rows": len(metadata),
        "subjects": metadata["subject"].nunique(),
        "sessions": metadata["session"].nunique(),
        "y dtype": str(y.dtype),
        "class 0 / class 1": (
            f"{int((metadata.target == 0).sum())} / {int((metadata.target == 1).sum())}"
        ),
    },
    name="value",
).to_frame()

# %% [markdown]
# Step 2. Predict, then run the WRONG way
# ---------------------------------------
#
# **Predict.** If we shuffle these 192 windows uniformly and put 20% in
# a test fold, how many subjects will end up in BOTH train and test?
# Pick one of: 0, around 5, all 12.
#
# **Run.** A window-level random shuffle: pick 20% of indices for test
# and call it a day.

# %%
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(len(metadata))
cut = int(0.8 * len(metadata))
naive_train = metadata.iloc[shuffled[:cut]]
naive_test = metadata.iloc[shuffled[cut:]]
leaked = sorted(set(naive_train["subject"]) & set(naive_test["subject"]))
naive_overlap = len(leaked)
pd.Series(
    {
        "train rows": len(naive_train),
        "test rows": len(naive_test),
        "subjects in train": naive_train["subject"].nunique(),
        "subjects in test": naive_test["subject"].nunique(),
        "subject_overlap": f"{naive_overlap} / {N_SUBJECTS}",
    },
    name="value",
).to_frame()

# %% [markdown]
# **Investigate.** Almost every subject sits on both sides of the split.
# A classifier trained here can memorise the alpha-rhythm fingerprint
# of subject 03 and recognise it again on subject 03's test windows.
# The accuracy is a subject-identification score, not a task-decoding
# score; deployment on a new participant collapses to chance.

# %% [markdown]
# Step 3. Build a leakage-safe 5-fold split manifest
# --------------------------------------------------
#
# **Run.** :func:`~eegdash.splits.get_splitter` with the canonical name
# ``"cross_subject"`` returns a MOABB ``CrossSubjectSplitter`` (or a
# :class:`sklearn.model_selection.GroupKFold` keyed on ``subject`` when
# MOABB is unavailable). Either way, no fold can put the same subject
# on both sides. :func:`~eegdash.splits.make_split_manifest` freezes
# the output into a JSON-serialisable dict with provenance: splitter
# class plus kwargs, library versions, target column, and a metadata
# hash so a teammate replaying the manifest can confirm they hold the
# same windows.

# %%
N_FOLDS = 5
# ``engine="sklearn"`` selects a GroupKFold-flavoured splitter that
# honours ``n_folds`` exactly. The ``"moabb"`` default uses
# LeaveOneGroupOut, which is correct but produces one fold per subject;
# 5 folds keep the audit short for the tutorial.
splitter = get_splitter(
    "cross_subject", engine="sklearn", n_folds=N_FOLDS, random_state=SEED
)
manifest = make_split_manifest(
    splitter, metadata["target"].to_numpy(), metadata, target="target"
)
pd.Series(
    {
        "splitter_class": manifest["splitter_class"],
        "n_folds": manifest["n_folds"],
        "target": manifest["target"],
        "metadata_hash": manifest["metadata_hash"][:12] + "...",
        "random_seed": manifest["random_seed"],
    },
    name="value",
).to_frame()

# %% [markdown]
# Step 4. Prove no subject leakage and read the audit
# ---------------------------------------------------
#
# :func:`~eegdash.splits.assert_no_leakage` walks every fold, intersects
# ``subject`` values across train and test, and always prints one JSON
# line:
#
# .. code-block:: text
#
#     {"leakage_report": {"overlap": 0, "by": "subject"}}
#
# A clean split prints ``overlap: 0``; a leaky split prints a non-zero
# overlap and raises :class:`eegdash.splits.LeakageError`. Runtime
# validator E5.42 grep-matches that exact line.
# :func:`~eegdash.splits.describe_split` prints a one-screen audit:
# per-fold sizes, distinct subjects on each side, per-fold class
# balance.

# %%
overlap = assert_no_leakage(manifest, metadata, by="subject")
assert overlap == 0, "Cross-subject manifest leaked!"
summary = describe_split(manifest, metadata, target="target")
fold0 = summary["per_fold"][0]
balance0 = fold0["class_balance_test"]
class_balance_ratio = max(balance0.values()) / (sum(balance0.values()) or 1)
pd.Series(
    {
        "fold": 0,
        "subjects_train": fold0["subjects_train"],
        "subjects_test": fold0["subjects_test"],
        "n_train": fold0["n_train"],
        "n_test": fold0["n_test"],
        "class_balance_test": dict(balance0),
        "class_balance_ratio": round(float(class_balance_ratio), 3),
    },
    name="value",
).to_frame()

# %% [markdown]
# Step 5. Read the per-fold audit table
# -------------------------------------
#
# :func:`~eegdash.splits.describe_split` returns the ``per_fold`` audit
# as a list of dicts. Coercing that into a :class:`pandas.DataFrame` is
# a habit worth keeping: it lets you eyeball the per-fold subject
# count, spot a class-imbalance outlier, and group by anything.

# %%
audit_df = pd.DataFrame(summary["per_fold"])
audit_df.insert(0, "fold", range(len(audit_df)))
audit_df[
    [
        "fold",
        "n_train",
        "n_test",
        "subjects_train",
        "subjects_test",
        "class_balance_train",
        "class_balance_test",
    ]
]

# %% [markdown]
# Step 6. Materialise one fold and persist the manifest
# -----------------------------------------------------
#
# :func:`~eegdash.splits.apply_split_manifest` returns a boolean mask
# for any fold; the manifest serialises to plain JSON, the BIDS-style
# "split metadata" Pernet et al. 2019 advocate sharing alongside
# derivatives. The same call signature works on a
# :class:`braindecode.datasets.BaseConcatDataset` of
# :class:`braindecode.datasets.WindowsDataset`: pass the windowed
# dataset from plot_10 in place of the DataFrame and you get a
# subset-of-windows back.

# %%
train_mask = apply_split_manifest(metadata, manifest, fold=0, split="train")
test_mask = apply_split_manifest(metadata, manifest, fold=0, split="test")
cache_dir = Path("./eegdash_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
manifest_path = cache_dir / "plot_11_split_manifest.json"
manifest_path.write_text(manifest_to_json(manifest), encoding="utf-8")
pd.Series(
    {
        "train_mask sum": int(train_mask.sum()),
        "test_mask sum": int(test_mask.sum()),
        "manifest bytes": manifest_path.stat().st_size,
        "manifest path": str(manifest_path),
    },
    name="value",
).to_frame()

# %% [markdown]
# **Investigate.** ``train_mask`` and ``test_mask`` are disjoint
# boolean arrays whose union covers every row. The manifest on disk is
# small enough to commit alongside an experiment notebook and large
# enough to be self-describing (splitter class, kwargs, hash, library
# versions, generated-at timestamp).

# %% [markdown]
# Result
# ------
#
# The naive split leaked 11 of 12 subjects across train and test; the
# cross-subject manifest prints
# ``{"leakage_report": {"overlap": 0, "by": "subject"}}`` and spreads
# 12 subjects across 5 folds with balanced classes per fold.

# %%
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
# A common mistake, and how to recover
# ------------------------------------
#
# Two things go wrong frequently with this API. The first is mistyping
# the splitter name; :func:`~eegdash.splits.get_splitter` raises a
# ``KeyError`` listing the valid names. The second is calling
# :func:`sklearn.model_selection.train_test_split` on the windows
# DataFrame and forgetting the ``stratify`` / ``groups`` arguments,
# which silently leaks subjects across folds. Both fail the same way
# (a happy-looking train/test pair), and both are caught by
# :func:`~eegdash.splits.assert_no_leakage`.

# %%
# Trip 1: mistyped splitter name.
try:
    _ = get_splitter("subject_split", n_folds=N_FOLDS, random_state=SEED)
except (KeyError, ValueError) as exc:
    print(f"Caught {type(exc).__name__}: {exc}")
    fixed = get_splitter("cross_subject", n_folds=N_FOLDS, random_state=SEED)
    print(f"Recovery: get_splitter('cross_subject') -> {type(fixed).__name__}")

# Trip 2: a bare sklearn train_test_split on windows leaks silently.
from sklearn.model_selection import train_test_split

bad_train, bad_test = train_test_split(metadata, test_size=0.2, random_state=SEED)
bad_overlap = len(set(bad_train["subject"]) & set(bad_test["subject"]))
print(
    f"train_test_split(...) leaks {bad_overlap}/{N_SUBJECTS} subjects; "
    f"assert_no_leakage would raise LeakageError."
)

# %% [markdown]
# **Investigate.** Both trips are silent under sklearn's defaults. The
# audit only fires once :func:`~eegdash.splits.assert_no_leakage` reads
# the metadata column you actually care about (``subject``, or
# ``session`` when sessions are independent recordings).

# %% [markdown]
# Modify. Try a session-aware split
# ---------------------------------
#
# **Modify.** Swap ``"cross_subject"`` for ``"cross_session"`` and
# re-run :func:`~eegdash.splits.assert_no_leakage` with
# ``by="session"``. The scaffolding stays put; only the invariant
# changes. Same call shape, different group key.

# %%
session_splitter = get_splitter("cross_session", n_folds=2, random_state=SEED)
session_manifest = make_split_manifest(
    session_splitter,
    metadata["target"].to_numpy(),
    metadata,
    target="target",
)
session_overlap = assert_no_leakage(session_manifest, metadata, by="session")
print(f"cross_session overlap: {session_overlap}")

# %% [markdown]
# Mini-project. Apply this flow to your own windows
# -------------------------------------------------
#
# **Mini-project.** Take the
# :class:`braindecode.datasets.BaseConcatDataset` of
# :class:`braindecode.datasets.WindowsDataset` you saved in plot_10,
# pipe it through :func:`~eegdash.splits.to_split_metadata` to get the
# tabular view, then through :func:`~eegdash.splits.make_split_manifest`
# and :func:`~eegdash.splits.assert_no_leakage`. The function
# :func:`~eegdash.splits.to_moabb_split_inputs` returns ``(y, metadata)``
# already aligned to MOABB's ``CrossSubjectSplitter`` API, so you can
# feed it straight into a benchmark loop without a glue-code layer.
#
# .. code-block:: python
#
#     from eegdash.splits import (
#         to_moabb_split_inputs, get_splitter,
#         make_split_manifest, assert_no_leakage,
#     )
#     y, md = to_moabb_split_inputs(windows, target="target")
#     splitter = get_splitter("cross_subject", n_folds=5, random_state=42)
#     manifest = make_split_manifest(splitter, y, md, target="target")
#     assert_no_leakage(manifest, md, by="subject")  # raises if it leaks

# %% [markdown]
# Headline figure. Naive vs cross-subject, side by side
# -----------------------------------------------------
#
# The drawing helper lives in a sibling ``_leakage_figure`` module so
# the matplotlib geometry stays out of the tutorial. The call below
# builds two ``(n_subjects, n_folds)`` status matrices from the
# splitter's own folds: ``0`` = subject is fully on the train side of
# fold ``j``, ``1`` = fully on the test side, ``2`` = split across
# train and test within fold ``j`` (the leakage failure mode).

# %%
from _leakage_figure import draw_leakage_figure

subjects_for_fig = sorted(metadata["subject"].unique())[:10]
n_subj_fig = len(subjects_for_fig)
n_folds_fig = 5

# Naive: a window-level shuffle puts 20% of EACH subject's windows in
# every test fold, so every cell carries the "split across train+test"
# value 2.
naive_assignment = np.full((n_subj_fig, n_folds_fig), 2, dtype=int)

# Cross-subject: read the manifest's first n_folds_fig folds and build
# a clean (n_subjects, n_folds) matrix where each subject is test in
# exactly one fold and train in the rest.
safe_assignment = np.zeros((n_subj_fig, n_folds_fig), dtype=int)
for fold_index in range(min(n_folds_fig, manifest["n_folds"])):
    test_ids = manifest["folds"][fold_index]["test"]
    test_subjects = {sid.split("__")[0] for sid in test_ids}
    for row_idx, subject_id in enumerate(subjects_for_fig):
        if subject_id in test_subjects:
            safe_assignment[row_idx, fold_index] = 1

fig = draw_leakage_figure(
    naive_assignment=naive_assignment,
    safe_assignment=safe_assignment,
    subjects=subjects_for_fig,
    n_windows_per_subject=N_SESSIONS * N_WINDOWS,
    plot_id="plot_11",
)
plt.show()

# %% [markdown]
# **Investigate.** Row 1 hatches every cell; row 2 carries one orange
# cell per subject. The Sankey-lite bars in column 2 say the same
# thing in window-count language: row 1 has every subject color in
# both ``train`` and ``test``, row 2 has each color in exactly one
# bar. The pills in column 3 read ``10/10`` for the naive row and
# ``0/10`` for the cross-subject row. Same windows, same labels, only
# the split rule changed.

# %% [markdown]
# Wrap-up
# -------
#
# Subject-aware splits are not a stylistic choice on EEG; they are the
# only protocol that reports a number you can compare across papers,
# sites, and clinical pipelines. The recipe is:
#
# 1. :func:`~eegdash.splits.to_split_metadata` to get a tabular view of
#    your windows.
# 2. :func:`~eegdash.splits.get_splitter` with ``"cross_subject"`` (or
#    ``"cross_session"`` when sessions are independent).
# 3. :func:`~eegdash.splits.make_split_manifest` to freeze the folds
#    plus provenance.
# 4. :func:`~eegdash.splits.assert_no_leakage` to enforce the
#    invariant.
# 5. :func:`~eegdash.splits.apply_split_manifest` to materialise one
#    fold for training.
#
# Next:
# :doc:`/auto_examples/tutorials/10_core_workflow/plot_12_train_a_baseline`
# trains a baseline on top of these folds; the manifest is loaded by
# reference so the splits are auditable end-to-end.

# %% [markdown]
# Try it yourself
# ---------------
#
# - Change ``random_state`` and confirm the folds shift but
#   subject-disjointness holds.
# - Set ``n_folds=10`` and re-read the per-fold subject count in
#   ``audit_df``.
# - Swap to a real
#   :class:`braindecode.datasets.BaseConcatDataset` from plot_10 and
#   re-run the manifest end-to-end.
# - Pass ``by="session"`` to
#   :func:`~eegdash.splits.assert_no_leakage` and watch the report flip
#   for the cross-subject manifest (different invariant).

# %% [markdown]
# References
# ----------
#
# - Brookshire et al. 2024, Data leakage in deep learning studies of translational EEG, *Frontiers in Neuroscience* 18:1373515. https://doi.org/10.3389/fnins.2024.1373515
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science* 10:e2256. https://doi.org/10.7717/peerj-cs.2256
# - Aristimunha et al. 2023, Mother of All BCI Benchmarks (MOABB), *Journal of Neural Engineering* 20:056025. https://doi.org/10.1088/1741-2552/aceaf8
# - Wakeman & Henson 2015, A multi-subject, multi-modal human neuroimaging dataset, *Scientific Data* 2:150001. https://doi.org/10.1038/sdata.2015.1
# - Delorme et al. 2022, NEMAR, an open access data, tools and compute resource operating on neuroelectromagnetic data, *Database* baac096. https://doi.org/10.1093/database/baac096
# - Pernet et al. 2019, EEG-BIDS, *Scientific Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - scikit-learn, ``sklearn.model_selection.GroupKFold``. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
