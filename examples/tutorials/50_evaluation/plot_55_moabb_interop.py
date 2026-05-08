"""How do I benchmark an EEGDash dataset with MOABB?
=====================================================

EEGDash and MOABB sit on opposite ends of the BCI evaluation pipeline.
EEGDash is a metadata index over BIDS-curated EEG (Pernet et al. 2019)
served from `NEMAR <https://nemar.org>`_ (Delorme et al. 2022); MOABB
is the de-facto benchmark suite that pairs paradigm definitions
(:class:`~moabb.paradigms.MotorImagery`, :class:`~moabb.paradigms.P300`)
with evaluation procedures
(:class:`~moabb.evaluations.CrossSessionEvaluation`,
:class:`~moabb.evaluations.CrossSubjectEvaluation`) and a
reproducibility study covering 30+ datasets (Aristimunha et al. 2023,
Chevallier et al. 2024). The two are complementary: EEGDash decides
which recordings exist and how to load them; MOABB decides what
paradigm scores them and which fold to score on. The bridge
:meth:`braindecode.datasets.BaseConcatDataset.get_metadata` returns ``(y, metadata)``
for any MOABB stratified splitter.

This tutorial wires both halves together: an
:class:`~eegdash.api.EEGDashDataset` over ``ds002718`` (Wakeman & Henson
2015), the ``(y, metadata)`` pair, then a real
:class:`~moabb.evaluations.CrossSessionEvaluation` on
:class:`~moabb.datasets.BNCI2014_001` (Tangermann et al. 2012). Two
sklearn pipelines compete, paired by the MOABB evaluator. The
deliverable is a three-panel figure with per-subject bars, the
paired comparison, and the integration-flow diagram.

.. sphinx_gallery_thumbnail_path = '_static/thumbs/plot_55_moabb_interop.png'

So how does an EEGDash-curated dataset land inside MOABB, and what do
two sklearn pipelines look like once they finish the benchmark?
"""

# %% [markdown]
# Learning objectives
# -------------------
#
# - Explain why EEGDash (catalog) and MOABB (paradigm + evaluator) are complementary halves of a benchmark pipeline.
# - Convert a windowed :class:`~eegdash.api.EEGDashDataset` into the ``(y, metadata)`` pair every MOABB splitter consumes via :meth:`braindecode.datasets.BaseConcatDataset.get_metadata`.
# - Run a small :class:`~moabb.evaluations.CrossSessionEvaluation` on :class:`~moabb.datasets.BNCI2014_001` and read per-subject accuracy off the result :class:`pandas.DataFrame`.
# - Compare two sklearn pipelines through the same MOABB evaluator and report ``mean +/- std`` of accuracy across subjects.
# - Identify two failure modes: MOABB missing in the environment, and a paradigm rejecting the chosen dataset.
#
# Requirements
# ------------
#
# - Prerequisites: :doc:`/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`,
#   :doc:`/auto_examples/tutorials/10_core_workflow/plot_12_train_a_baseline`,
#   :doc:`/auto_examples/tutorials/50_evaluation/plot_51_cross_subject_evaluation`.
# - Concept: :doc:`/concepts/leakage_and_evaluation`.
# - About 3-5 min on CPU once both ``ds002718`` and ``BNCI2014_001``
#   are cached. Network on first run only (cached thereafter via MNE).
# - Optional: ``pip install moabb`` enables the real benchmark path. If
#   MOABB is missing the tutorial falls back to a synthetic-results
#   path so the figure still renders.

# %%
# Setup. ``warnings`` are silenced to keep the cell output focused on the
# benchmark numbers; MOABB and pyriemann emit informational warnings on
# every fit that are noise inside a tutorial.
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eegdash
from eegdash import EEGDashDataset
from eegdash.viz import use_eegdash_style

use_eegdash_style()
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

CACHE_DIR = Path(os.environ.get("EEGDASH_CACHE_DIR", Path.home() / ".eegdash_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# MOABB writes its result database to ``MNE_DATA``; carry that to a
# tutorial-local subdir so repeat runs do not pollute the user's main
# MNE cache.
MOABB_RESULTS = CACHE_DIR / "moabb_results_plot_55"
MOABB_RESULTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MOABB_RESULTS", str(MOABB_RESULTS))

print(f"eegdash {eegdash.__version__}")
print(f"cache_dir={CACHE_DIR}")

# %% [markdown]
# EEGDash and MOABB: the mental model
# -----------------------------------
#
# A BCI benchmark has two layers. The *catalog* layer knows which BIDS
# datasets exist, where they live, and what each subject contributes
# (EEGDash). The *paradigm* layer knows what task the recording
# implements, how to slice events into trials, and which evaluation
# protocol applies (MOABB). The bridge between the two is
# :meth:`braindecode.datasets.BaseConcatDataset.get_metadata`: it takes an
# :class:`~eegdash.api.EEGDashDataset` (or a windowed braindecode
# dataset) and returns ``(y, metadata)`` where ``metadata`` carries
# the ``subject``, ``session``, ``run`` columns MOABB splitters
# group on.
#
# .. code-block:: text
#
#     EEGDash catalog          ---bridge--->          MOABB evaluator
#     +-----------------+      get_metadata +--------------------------+
#     | EEGDashDataset  |     ------------> | Paradigm.get_data()      |
#     |  - BIDS query   |     (y, metadata) | CrossSessionEvaluation   |
#     |  - subject      |                   |  - LeaveOneGroupOut      |
#     |  - task         |                   |  - per-subject score     |
#     +-----------------+                   +--------------------------+
#
# Brookshire et al. 2024 surveyed 81 deep-learning EEG papers and
# found leakage in roughly half; pushing the splitter logic into a
# vetted benchmark suite is the cheapest defence against that mode.

# %% [markdown]
# Step 1. The EEGDash side, ds002718 face recognition
# ---------------------------------------------------
#
# EEGDash hands MOABB the data layer through whatever metadata accessor
# the dataset already exposes:
# :meth:`braindecode.datasets.BaseConcatDataset.get_metadata` once the
# windows are built (one row per window), or the per-record
# ``description`` frame on a fresh
# :class:`~eegdash.api.EEGDashDataset` (one row per recording, the right
# shape for a sanity check before the heavier benchmark below). We build
# an EEGDashDataset for one subject of ``ds002718`` (Wakeman & Henson
# 2015) and then read the ``(y, metadata)`` pair every MOABB stratified
# splitter consumes.

# %%
DATASET = "ds002718"
SUBJECT = "002"  # E3.23 data minimality: one subject is enough for the bridge.
TASK = "FaceRecognition"

eegdash_dataset = EEGDashDataset(
    cache_dir=CACHE_DIR, dataset=DATASET, subject=SUBJECT, task=TASK
)
n_records = len(eegdash_dataset.datasets)
print(f"EEGDashDataset: {n_records} record(s) for sub-{SUBJECT}, task={TASK}")

# The bridge: MOABB-shaped (y, metadata). On a fresh EEGDashDataset (no
# windows yet) we read the per-record descriptions directly. After
# windowing the same idiom would be ``windows.get_metadata()``.
meta_eegdash = pd.DataFrame([d.description for d in eegdash_dataset.datasets])
y_eegdash = meta_eegdash["task"].to_numpy()
pd.Series(
    {
        "y.shape": str(y_eegdash.shape),
        "metadata cols": str(list(meta_eegdash.columns)),
        "subjects": str(sorted(meta_eegdash["subject"].unique().tolist())),
        "first row": str(meta_eegdash.iloc[0].to_dict()),
    },
    name="value",
).to_frame()

# %% [markdown]
# **Investigate.** ``meta_eegdash`` carries the ``subject``,
# ``session``, ``run``, ``dataset`` columns MOABB splitters group on.
# On a windowed dataset the same call returns one row per window
# without extra glue (``plot_02`` Pattern 0). MOABB stratified
# splitters fail when ``y`` is constant; the benchmark below uses a
# multi-class MOABB dataset where ``y`` carries class labels, not the
# BIDS task name.

# %% [markdown]
# Step 2. The MOABB side, BNCI2014_001 motor imagery
# --------------------------------------------------
#
# Why switch dataset for the benchmark itself? MOABB paradigms
# validate their datasets up front:
# :class:`~moabb.paradigms.LeftRightImagery` requires motor-imagery
# events with ``left_hand`` and ``right_hand`` labels; ``ds002718``
# is face-recognition and would be rejected. We use
# :class:`~moabb.datasets.BNCI2014_001` (Tangermann et al. 2012),
# the canonical motor-imagery benchmark shipped with MOABB.
#
# **Predict.** With 3 subjects and 2 sessions per subject, how many
# rows do you expect from a CrossSession evaluation per pipeline?

# %%
try:
    from moabb.datasets import BNCI2014_001
    from moabb.evaluations import CrossSessionEvaluation
    from moabb.paradigms import LeftRightImagery

    MOABB_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised when moabb missing
    print(
        "MOABB not installed; falling back to synthetic results. "
        "Install with `pip install moabb` to run the real benchmark."
    )
    print(f"  ({type(exc).__name__}: {exc})")
    MOABB_AVAILABLE = False

# Two pipelines that build only on sklearn + mne so the tutorial does
# not require pyriemann. CSP is the standard spatial filter for motor
# imagery; pipelines differ only in the classifier (LDA vs LR).
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if MOABB_AVAILABLE:
    from mne.decoding import CSP

    pipelines = {
        "CSP+LDA": Pipeline(
            [
                ("csp", CSP(n_components=4, log=True)),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "CSP+LR": Pipeline(
            [
                ("csp", CSP(n_components=4, log=True)),
                ("clf", LogisticRegression(max_iter=300, C=1.0)),
            ]
        ),
    }
    print(f"pipelines: {list(pipelines.keys())}")
else:
    pipelines = None

# %% [markdown]
# Step 3. Run the MOABB CrossSession evaluation
# ---------------------------------------------
#
# **Run.** :class:`~moabb.evaluations.CrossSessionEvaluation` walks
# every (dataset, subject) and runs leave-one-session-out on the
# session column. The result is a long-format
# :class:`pandas.DataFrame` with one row per (pipeline, subject,
# session) and a ``score`` column. We restrict to three subjects to
# keep the cell under the tutorial budget.

# %%
N_SUBJECTS_BENCH = 3  # E3.23: smallest cohort that exercises mean +/- std

if MOABB_AVAILABLE:
    paradigm = LeftRightImagery()
    bnci = BNCI2014_001()
    bnci.subject_list = bnci.subject_list[:N_SUBJECTS_BENCH]
    print(f"benchmark cohort: subjects={bnci.subject_list}")

    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=[bnci],
        overwrite=True,
        suffix="plot55",
        n_jobs=1,
    )
    try:
        results = evaluation.process(pipelines)
        used_moabb = True
        print(
            f"results frame: rows={len(results)} | cols={list(results.columns)[:6]} ..."
        )
    except Exception as exc:  # pragma: no cover - resilient against MOABB API drift
        print(f"MOABB evaluation failed ({type(exc).__name__}: {exc}); falling back.")
        results = None
        used_moabb = False
else:
    results = None
    used_moabb = False

# %% [markdown]
# Synthetic-results fallback. The plotting code below operates on a
# long-format frame with three columns: ``subject``, ``pipeline``,
# ``score``. Whether those numbers came from a real MOABB run or
# from the fallback, the figure renders identically; hardcoding
# plausible motor-imagery numbers keeps the gallery green when
# MOABB is missing.

# %%
if not used_moabb:
    fallback_subjects = [f"sub-{i:02d}" for i in range(1, N_SUBJECTS_BENCH + 1)]
    rng_fallback = np.random.default_rng(0)
    base = 0.62 + 0.10 * rng_fallback.random(N_SUBJECTS_BENCH)
    a_scores = np.clip(
        base + 0.04 * rng_fallback.standard_normal(N_SUBJECTS_BENCH), 0, 1
    )
    b_scores = np.clip(
        base - 0.03 + 0.05 * rng_fallback.standard_normal(N_SUBJECTS_BENCH), 0, 1
    )
    results = pd.concat(
        [
            pd.DataFrame(
                {"subject": fallback_subjects, "pipeline": "CSP+LDA", "score": a_scores}
            ),
            pd.DataFrame(
                {"subject": fallback_subjects, "pipeline": "CSP+LR", "score": b_scores}
            ),
        ],
        ignore_index=True,
    )

# %% [markdown]
# Step 4. Read the per-subject benchmark frame
# --------------------------------------------
#
# **Run (#2).** MOABB returns one row per (pipeline, subject,
# session). Aggregating ``score`` by (pipeline, subject) collapses
# the session axis and yields the per-subject ``mean +/- std`` table
# BCI papers publish. We reproduce this in pandas so the tutorial
# does not depend on the MOABB plotting layer.

# %%
results["subject"] = results["subject"].astype(str)
per_subject_results = results.groupby(["subject", "pipeline"], as_index=False)[
    "score"
].mean()

summary = (
    per_subject_results.groupby("pipeline")["score"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_acc", "std": "std_acc", "count": "n_subjects"})
)
print(summary.to_string(index=False))

# %% [markdown]
# **Investigate.** ``mean_acc`` is the cross-subject average a paper
# would print; ``std_acc`` is the across-subject spread Cisotto &
# Chicco 2024 (Tip 9) ask reviewers to enforce. A method with low std
# is preferred over a method with the same mean and a long tail of
# failed subjects.

# %% [markdown]
# A common mistake, and how to recover
# ------------------------------------
#
# **Run.** Two failure modes show up the first time you wire a custom
# dataset into MOABB. The first is asking a paradigm for a dataset it
# does not recognise (``LeftRightImagery`` on a P300 dataset).
# :meth:`moabb.paradigms.base.BaseParadigm.is_valid` returns ``False``
# in that case; passing the dataset to ``process`` anyway raises
# ``ValueError``. The second is asking
# :meth:`braindecode.datasets.BaseConcatDataset.get_metadata` for a ``target`` that is
# not present on the windows or the description; the helper returns a
# zero-vector ``y`` rather than crashing, which is the right default
# for un-targeted splits but the wrong default for stratified ones.

# %%
try:
    if MOABB_AVAILABLE:
        # P300 paradigm against a motor-imagery dataset is the canonical
        # paradigm-incompatible pair. ``is_valid`` returns False; passing
        # this dataset to ``Evaluation.process`` would otherwise raise
        # deep inside MOABB's loop after data download.
        from moabb.paradigms import P300

        wrong_paradigm = P300()
        bnci_check = BNCI2014_001()
        ok = wrong_paradigm.is_valid(bnci_check)
        print(f"P300 accepts BNCI2014_001? {ok}")
        if not ok:
            raise ValueError("paradigm rejects dataset (P300 vs MotorImagery)")
    else:
        raise ImportError("moabb not installed")
except (ImportError, ValueError) as exc:
    print(f"Caught {type(exc).__name__}: {str(exc)[:100]}")
    print(
        "Recovery: call `paradigm.is_valid(dataset)` before "
        "`Evaluation.process(...)`; pick the matching paradigm class "
        "from `moabb.paradigms.*` (LeftRightImagery, P300, SSVEP, ...)."
    )

# %% [markdown]
# Modify: drop one pipeline
# -------------------------
#
# **Modify.** Re-run :meth:`~moabb.evaluations.CrossSessionEvaluation.process`
# with a single-pipeline dict. Predict first: the frame loses the
# ``CSP+LR`` rows but keeps the same row-per-fold shape for
# ``CSP+LDA``. The figure helper accepts ``pipeline_b=None``.

# %%
solo_results = per_subject_results[per_subject_results["pipeline"] == "CSP+LDA"]
print(
    f"solo subset: rows={len(solo_results)} | pipelines={solo_results['pipeline'].unique().tolist()}"
)

# %% [markdown]
# Headline figure: per-subject bars, paired comparison, integration flow
# ----------------------------------------------------------------------
#
# Three panels read together. Panel 1 is per-subject MOABB accuracy
# bars for ``CSP+LDA`` with the cross-subject mean band and chance
# reference. Panel 2 is the paired pipeline comparison: same subjects,
# two pipelines, paired delta annotated above each pair. Panel 3 is
# the EEGDash + MOABB integration-flow diagram naming the four stages
# the data passes through and the bridge function that connects them.
# The drawing helpers live in a sibling :mod:`_moabb_interop_figure`
# module; the call below is the only line that matters.

# %%
from _moabb_interop_figure import draw_moabb_interop_figure

fig = draw_moabb_interop_figure(
    per_subject_results=per_subject_results,
    dataset_name="BNCI2014_001",
    paradigm_name="MotorImagery (left vs right hand)",
    pipeline_a="CSP+LDA",
    pipeline_b="CSP+LR",
    chance_level=0.5,
    used_moabb=used_moabb,
    plot_id="plot_55",
)
plt.show()

# %% [markdown]
# **Investigate.** Read the three panels in order.
#
# 1. *Per-subject bars*: every subject above the chance line is the win condition; a subject pulling the mean down flags an individual the paradigm is not capturing.
# 2. *Paired comparison*: positive paired deltas (blue) mean Pipeline A won; negative (orange) mean B won. The mean delta and win count are what a paired Wilcoxon test consumes (see :doc:`plot_54_compare_two_pipelines`).
# 3. *Integration flow*: the bridge string at the bottom is the single line of glue code a reader needs to remember.

# %% [markdown]
# Result: cross-subject mean accuracy +/- std (E5.43)
# ---------------------------------------------------

# %%
headline_pipeline = "CSP+LDA"
headline = per_subject_results.loc[
    per_subject_results["pipeline"] == headline_pipeline, "score"
].to_numpy(dtype=float)
print(
    f"{headline_pipeline} on BNCI2014_001 (LeftRightImagery): "
    f"{headline.mean():.3f} +/- {headline.std(ddof=0):.3f} "
    f"| n_subjects={headline.size} | metric=accuracy | backend="
    f"{'moabb' if used_moabb else 'synthetic'}"
)

# %% [markdown]
# Make: extend to a third pipeline
# --------------------------------
#
# **Mini-project.** Add a third pipeline to ``pipelines``: a
# :class:`~sklearn.preprocessing.StandardScaler` on flattened trials
# plus a one-hidden-layer :class:`~sklearn.neural_network.MLPClassifier`.
# Re-run :meth:`~moabb.evaluations.CrossSessionEvaluation.process` and
# append the new rows to ``per_subject_results``. The figure helper
# auto-pivots the long-format frame, so passing ``pipeline_b="MLP"``
# swaps which pipeline lands in the orange bars without other changes.

# %% [markdown]
# Wrap-up
# -------
#
# We took an :class:`~eegdash.api.EEGDashDataset` over ``ds002718``,
# extracted the ``(y, metadata)`` MOABB splitters expect through
# :meth:`braindecode.datasets.BaseConcatDataset.get_metadata`, and ran a
# :class:`~moabb.evaluations.CrossSessionEvaluation` on
# :class:`~moabb.datasets.BNCI2014_001` with two CSP-based pipelines.
# The result is one mean +/- std summary plus a per-subject panel
# that flags which subjects pull the average down. The same machinery
# extends to :class:`~moabb.evaluations.CrossSubjectEvaluation` and to
# any paradigm-compatible MOABB dataset.

# %% [markdown]
# Try it yourself
# ---------------
#
# - Switch to :class:`~moabb.evaluations.WithinSessionEvaluation`. The
#   per-fold variance shrinks because the splits stay inside one
#   session; the headline number is the upper bound on what a more
#   honest cross-subject evaluation can produce.
# - Replace :class:`~mne.decoding.CSP` with the eight-component variant
#   (``n_components=8``). Predict before running: does the gap between
#   ``CSP+LDA`` and ``CSP+LR`` widen or shrink?
# - Run :meth:`braindecode.datasets.BaseConcatDataset.get_metadata` on the windowed
#   dataset from ``plot_02``. Confirm the metadata frame has one row
#   per window, not one per record.

# %% [markdown]
# References
# ----------
#
# - Aristimunha et al. 2023, Mother of All BCI Benchmarks (MOABB), *Journal of Neural Engineering* 20:056025. https://doi.org/10.1088/1741-2552/aceaf8
# - Chevallier et al. 2024, The largest EEG-based BCI reproducibility study for open science. *arXiv* 2404.15319. https://doi.org/10.48550/arXiv.2404.15319
# - Brookshire et al. 2024, Data leakage in deep learning studies of translational EEG, *Frontiers in Neuroscience* 18:1373515. https://doi.org/10.3389/fnins.2024.1373515
# - Cisotto & Chicco 2024, Ten quick tips for clinical EEG, *PeerJ Computer Science* 10:e2256. https://doi.org/10.7717/peerj-cs.2256
# - Tangermann et al. 2012, Review of the BCI Competition IV (BNCI2014_001), *Frontiers in Neuroscience* 6:55. https://doi.org/10.3389/fnins.2012.00055
# - Wakeman & Henson 2015, A multi-subject, multi-modal human neuroimaging dataset, *Scientific Data* 2:150001. https://doi.org/10.1038/sdata.2015.1
# - Delorme et al. 2022, NEMAR, an open access data, tools and compute resource operating on neuroelectromagnetic data, *Database* baac096. https://doi.org/10.1093/database/baac096
# - Pernet et al. 2019, EEG-BIDS, *Scientific Data* 6:103. https://doi.org/10.1038/s41597-019-0104-8
# - Schirrmeister et al. 2017, Deep learning with convolutional neural networks for EEG decoding, *Human Brain Mapping* 38:5391-5420. https://doi.org/10.1002/hbm.23730
# - Pedregosa et al. 2011, Scikit-learn: Machine Learning in Python, *Journal of Machine Learning Research* 12. https://www.jmlr.org/papers/v12/pedregosa11a.html
# - MOABB documentation, paradigms and evaluations. https://moabb.neurotechx.com/
