.. _tutorial_evidence:

==========================
Tutorial Evidence Dashboard
==========================

EEGDash tutorials ship with a machine-verifiable audit trail. Every tutorial
has a YAML *spec* under ``docs/tutorials/_spec/`` declaring its rubric
expectations *before* any code is written: difficulty tier, runtime budget,
required PRIMM blocks, asserted invariants, expected figures, and the rubric
rules it must satisfy. The Continuous Integration workflow
``.github/workflows/tutorial-audit.yml`` then runs the rubric against every
pull request and produces an *evidence dossier* under
``docs/evidence/tutorials/<tutorial_id>/``. The dossier captures
``evidence.json`` (the full Findings list, deterministic), ``report.md``
(human-readable scorecard), the rendered figures, the cell-level timing, and
the human reviewer's ``reviewer_score.json``.

This page is the public-facing index for those dossiers. It is the place to
look when you want to know: which tutorials pass the rubric, where the
gaps are, how the current state compares to the
``docs/evidence/tutorials/_baseline_2026-05-06/`` snapshot, and which rules
each tutorial addresses with provenance back to the literature-anchored
rubric.

The system itself is described in two source documents that live at the
top of the ``docs/`` tree (they are Markdown, not part of the Sphinx
source tree):

- ``docs/tutorial_restructure_plan.md`` -- the 13 tutorials, file
  layout, and quality bar.
- ``docs/tutorial_implementation_strategy.md`` -- the 49-rule rubric,
  the 12-dimension scorecard, the validator implementations, and the
  three-role CI pipeline.

The reviewer-only rubric (8 rules) is documented in ``CONTRIBUTING.md``,
under "Tutorial Review Rubric (Reviewer-Only Rules)".

Aggregate report
================

The audit pipeline writes ``docs/evidence/tutorials/_aggregate.md`` whenever
it runs ``report.py --aggregate``. The block below pulls that file in as
literal text. If the aggregate report has not been generated yet (for
example on a fresh clone) the directive will emit a Sphinx warning and the
section will be empty until a run produces the file.

.. literalinclude:: ../../../docs/evidence/tutorials/_aggregate.md
   :language: markdown
   :linenos:

.. note::

   When the aggregate file is missing, Sphinx emits a build warning and
   skips the literalinclude. Run ``python -m
   scripts.tutorial_audit.report --aggregate`` (or ``make -f tutorials.mk
   tutorial-phase-report PHASE=<n>`` for a phase-scoped report) to
   generate it.

Per-tutorial dossiers
=====================

Each tutorial dossier renders to ``docs/evidence/tutorials/<id>/report.md``.
The toctree below is intentionally short during bootstrap; the audit
pipeline will extend it with one entry per tutorial as dossiers land.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial dossiers
   :glob:

   tutorial_evidence/*

The thirteen tutorials in scope
===============================

The following table is the binding scope for Releases 1 and 2. Filenames,
categories, and ordering match
``docs/tutorial_restructure_plan.md`` ("Recommended Initial Tutorial Set"
and "Proposed File Layout").

.. list-table:: Release 1 -- core learning path
   :header-rows: 1
   :widths: 6 50 12

   * - Order
     - File
     - Category
   * - 0
     - ``examples/tutorials/00_start_here/plot_00_first_search.py``
     - A start-here
   * - 1
     - ``examples/tutorials/00_start_here/plot_01_first_recording.py``
     - A start-here
   * - 2
     - ``examples/tutorials/00_start_here/plot_02_dataset_to_dataloader.py``
     - A start-here
   * - 3
     - ``examples/tutorials/10_core_workflow/plot_10_preprocess_and_window.py``
     - B core-workflow
   * - 4
     - ``examples/tutorials/10_core_workflow/plot_11_leakage_safe_split.py``
     - B core-workflow
   * - 5
     - ``examples/tutorials/10_core_workflow/plot_12_train_a_baseline.py``
     - B core-workflow
   * - 6
     - ``examples/tutorials/10_core_workflow/plot_13_save_and_reuse_prepared_data.py``
     - B core-workflow
   * - 7
     - ``examples/tutorials/40_features/plot_40_first_features.py``
     - E features

.. list-table:: Release 2 -- topical extensions
   :header-rows: 1
   :widths: 50 12

   * - File
     - Category
   * - ``examples/tutorials/20_event_related/plot_20_visual_p300_oddball.py``
     - C event-related
   * - ``examples/tutorials/20_event_related/plot_21_auditory_oddball.py``
     - C event-related
   * - ``examples/tutorials/30_resting_state/plot_30_eyes_open_closed.py``
     - D resting-state
   * - ``examples/tutorials/40_features/plot_41_feature_trees.py``
     - E features
   * - ``examples/tutorials/40_features/plot_42_features_to_sklearn.py``
     - E features

How a tutorial reaches the dashboard
====================================

1. Author claims a tutorial via ``make -f tutorials.mk tutorial-claim
   TUTORIAL=plot_NN_xxx BY=<handle>``. The spec advances to
   ``state: drafted``.
2. Author writes ``examples/tutorials/<cat>/plot_NN_xxx.py`` following the
   spec.
3. Local self-audit: ``make -f tutorials.mk tutorial-audit
   TUTORIAL=plot_NN_xxx``. Static stage runs in seconds.
4. Pull request opens. The ``tutorial-audit`` workflow runs the static
   rubric, lints the source via ``nbqa ruff``, aggregates the evidence, and
   uploads the dossier.
5. Reviewer files ``reviewer_score.json`` per the rubric in
   ``CONTRIBUTING.md`` ("Tutorial Review Rubric (Reviewer-Only Rules)").
6. ``make -f tutorials.mk tutorial-release TUTORIAL=plot_NN_xxx`` advances
   ``state: reviewed`` to ``state: merged``.

Reading the dossier
===================

A typical ``evidence.json`` carries deterministic ``totals`` (errors,
warns, infos), one ``rule_results`` entry per rubric rule, and a
``scorecard`` mapping the 12 operational dimensions defined in
``new_tutorials/validation_documentation.md`` ("Operational checklist").
Findings cite the rubric (``compass_artifact.md#E5.42``) and the plan
(``tutorial_restructure_plan.md#L902-L920``) so that any failing
observation traces back to a peer-reviewed pedagogical principle or to a
prescriptive line of the migration plan.

The runtime stages (clean-kernel execution, budgets, leakage check,
visual identity) are off by default at Day-0 and turn on once the first
three Release-1 tutorials are drafted. See
``docs/tutorial_implementation_strategy.md`` -- "Bootstrap (Day-0)" for
the full sequencing.
