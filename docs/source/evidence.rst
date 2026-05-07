.. _tutorial_evidence:

===========================
Tutorial Evidence Dashboard
===========================

EEGDash tutorials ship with a machine-verifiable audit trail. Every tutorial
declares its expectations *before* any code is written in a YAML spec under
``docs/tutorials/_spec/``: difficulty tier, runtime budget, required PRIMM
blocks, asserted invariants, expected figures, and which rubric rules it must
satisfy. The CI workflow ``.github/workflows/tutorial-audit.yml`` then runs
the rubric on every pull request and writes an evidence dossier under
``docs/evidence/tutorials/<id>/``.

This page is the public-facing index for those dossiers. The system itself is
described in two top-level Markdown documents — ``docs/tutorial_restructure_plan.md``
(the 13 tutorials, file layout, and quality bar) and
``docs/tutorial_implementation_strategy.md`` (the 49-rule rubric and the
three-role CI pipeline). The reviewer-only rubric (8 rules) lives in
``CONTRIBUTING.md`` under "Tutorial Review Rubric (Reviewer-Only Rules)".

Documentation map
=================

EEGDash documentation follows the `Diataxis framework <https://diataxis.fr>`_:
four quadrants, each answering a different question. The dashboard below covers
the two bottom rows — the runnable tutorials and recipes — but every
contributing tutorial cites the concept it depends on.

.. raw:: html

   <svg class="diataxis-quadrant" viewBox="0 0 800 280"
        xmlns="http://www.w3.org/2000/svg" role="img"
        aria-label="Diataxis four-quadrant documentation map for EEGDash">
     <defs>
       <linearGradient id="dxRail" x1="0" x2="1" y1="0" y2="0">
         <stop offset="0" stop-color="#F7941D"/>
         <stop offset="0.18" stop-color="#F7941D"/>
         <stop offset="0.22" stop-color="#006CA3"/>
         <stop offset="1" stop-color="#006CA3"/>
       </linearGradient>
     </defs>
     <rect x="0" y="0" width="800" height="6" fill="url(#dxRail)"/>
     <g font-family="Helvetica, Arial, sans-serif" fill="#102A43">
       <!-- Tutorials -->
       <g transform="translate(12,30)">
         <rect width="180" height="220" rx="10" ry="10"
               fill="#F7FBFE" stroke="#006CA3" stroke-width="1.5"/>
         <text x="90" y="34" text-anchor="middle"
               font-size="15" font-weight="700" fill="#004A76">Tutorials</text>
         <text x="90" y="56" text-anchor="middle"
               font-size="11" fill="#64748B">learning-oriented</text>
         <text x="90" y="118" text-anchor="middle"
               font-size="44" font-weight="700" fill="#006CA3">22</text>
         <text x="90" y="142" text-anchor="middle"
               font-size="11" fill="#64748B">guided lessons</text>
         <text x="90" y="190" text-anchor="middle"
               font-size="11" fill="#102A43">plot_00 → plot_73</text>
         <text x="90" y="208" text-anchor="middle"
               font-size="10" fill="#64748B">8 categories, A–H</text>
       </g>
       <!-- How-Tos -->
       <g transform="translate(204,30)">
         <rect width="180" height="220" rx="10" ry="10"
               fill="#F7FBFE" stroke="#006CA3" stroke-width="1.5"/>
         <text x="90" y="34" text-anchor="middle"
               font-size="15" font-weight="700" fill="#004A76">How-Tos</text>
         <text x="90" y="56" text-anchor="middle"
               font-size="11" fill="#64748B">task-oriented</text>
         <text x="90" y="118" text-anchor="middle"
               font-size="44" font-weight="700" fill="#006CA3">5</text>
         <text x="90" y="142" text-anchor="middle"
               font-size="11" fill="#64748B">recipes</text>
         <text x="90" y="190" text-anchor="middle"
               font-size="11" fill="#102A43">offline cache, SLURM</text>
         <text x="90" y="208" text-anchor="middle"
               font-size="10" fill="#64748B">scaling and HPC focus</text>
       </g>
       <!-- Concepts -->
       <g transform="translate(396,30)">
         <rect width="180" height="220" rx="10" ry="10"
               fill="#F7FBFE" stroke="#006CA3" stroke-width="1.5"/>
         <text x="90" y="34" text-anchor="middle"
               font-size="15" font-weight="700" fill="#004A76">Concepts</text>
         <text x="90" y="56" text-anchor="middle"
               font-size="11" fill="#64748B">understanding-oriented</text>
         <text x="90" y="118" text-anchor="middle"
               font-size="44" font-weight="700" fill="#006CA3">6</text>
         <text x="90" y="142" text-anchor="middle"
               font-size="11" fill="#64748B">explanations</text>
         <text x="90" y="190" text-anchor="middle"
               font-size="11" fill="#102A43">leakage, BIDS, features</text>
         <text x="90" y="208" text-anchor="middle"
               font-size="10" fill="#64748B">why, not how</text>
       </g>
       <!-- Reference -->
       <g transform="translate(588,30)">
         <rect width="200" height="220" rx="10" ry="10"
               fill="#F7FBFE" stroke="#006CA3" stroke-width="1.5"/>
         <text x="100" y="34" text-anchor="middle"
               font-size="15" font-weight="700" fill="#004A76">Reference</text>
         <text x="100" y="56" text-anchor="middle"
               font-size="11" fill="#64748B">information-oriented</text>
         <text x="100" y="118" text-anchor="middle"
               font-size="36" font-weight="700" fill="#006CA3">auto-doc</text>
         <text x="100" y="146" text-anchor="middle"
               font-size="11" fill="#64748B">API surface</text>
         <text x="100" y="190" text-anchor="middle"
               font-size="11" fill="#102A43">eegdash.* modules</text>
         <text x="100" y="208" text-anchor="middle"
               font-size="10" fill="#64748B">generated from source</text>
       </g>
     </g>
   </svg>


Aggregate status
================

The table below is generated at build time by the ``eegdash-evidence-dashboard``
directive, which reads each ``docs/evidence/tutorials/<id>/evidence.json`` and
the optional ``reviewer_score.json``. If a tutorial has no dossier yet (fresh
clone, audit not run) the row falls back to ``n/a``. Run
``python -m scripts.tutorial_audit.report --aggregate`` (or
``make -f tutorials.mk tutorial-phase-report PHASE=<n>``) to refresh the dossiers.

.. eegdash-evidence-dashboard::


Categories
==========

Tutorials are grouped by Diataxis-aware category. Each card links to the
rendered Sphinx-Gallery section and lists the tutorials currently in scope.

.. grid:: 2 2 3 4
   :gutter: 3
   :class-container: tutorial-evidence-cards

   .. grid-item-card:: A — Start here
      :link: /generated/auto_examples/tutorials/00_start_here/index
      :link-type: url
      :columns: 12 6 4 3

      **3 tutorials · proposed**

      First search, first recording, dataset to dataloader. The on-ramp for
      a brand-new EEGDash user.

   .. grid-item-card:: B — Core workflow
      :link: /generated/auto_examples/tutorials/10_core_workflow/index
      :link-type: url
      :columns: 12 6 4 3

      **4 tutorials · proposed**

      Preprocess and window, leakage-safe split, train a baseline, save and
      reuse prepared data.

   .. grid-item-card:: C — Event-related
      :link: /generated/auto_examples/tutorials/20_event_related/index
      :link-type: url
      :columns: 12 6 4 3

      **2 tutorials · proposed**

      Visual P300 oddball and the auditory-oddball contrast — both at
      difficulty 2.

   .. grid-item-card:: D — Resting state
      :link: /generated/auto_examples/tutorials/30_resting_state/index
      :link-type: url
      :columns: 12 6 4 3

      **1 tutorial · proposed**

      Eyes-open versus eyes-closed decoding, the canonical resting-state
      contrast.

   .. grid-item-card:: E — Features
      :link: /generated/auto_examples/tutorials/40_features/index
      :link-type: url
      :columns: 12 6 4 3

      **3 tutorials · proposed**

      First feature table, feature trees for spectral reuse, features into
      a scikit-learn pipeline.

   .. grid-item-card:: F — Evaluation
      :link: /generated/auto_examples/tutorials/50_evaluation/index
      :link-type: url
      :columns: 12 6 4 3

      **5 tutorials · proposed**

      Within-subject k-fold, cross-subject LSO, cross-session drift,
      learning curves, and pipeline comparisons.

   .. grid-item-card:: H — Transfer & foundations
      :link: /generated/auto_examples/tutorials/70_transfer_foundation/index
      :link-type: url
      :columns: 12 6 4 3

      **4 tutorials · proposed**

      EEG2025 challenge dataset basics, cross-task transfer, subject-invariant
      regression, fine-tuning a Braindecode foundation model.

   .. grid-item-card:: I — Scaling & HPC (how-tos)
      :link: /generated/auto_examples/how_to/index
      :link-type: url
      :columns: 12 6 4 3

      **5 how-tos · proposed**

      Download in advance, parallel feature extraction, HPC cache layout,
      SLURM templates, fully offline workflows.


How a tutorial reaches the dashboard
====================================

1. Author claims a slot with ``make -f tutorials.mk tutorial-claim TUTORIAL=plot_NN_xxx BY=<handle>`` (spec advances to ``state: drafted``).
2. Author writes ``examples/tutorials/<cat>/plot_NN_xxx.py`` against the spec.
3. Local self-audit: ``make -f tutorials.mk tutorial-audit TUTORIAL=plot_NN_xxx`` (static stage runs in seconds).
4. Pull request opens — the ``tutorial-audit`` workflow runs the static rubric, lints with ``nbqa ruff``, and uploads the dossier.
5. Reviewer files ``reviewer_score.json`` per the rubric in ``CONTRIBUTING.md``.
6. ``make -f tutorials.mk tutorial-release TUTORIAL=plot_NN_xxx`` advances ``state: reviewed`` to ``state: merged``.


Reading a dossier
=================

- ``evidence.json`` carries deterministic ``totals`` (errors, warns, infos), one ``rule_results`` entry per rubric rule, and a ``scorecard`` mapping the 12 operational dimensions defined in ``new_tutorials/validation_documentation.md``.
- Findings cite the rubric (``compass_artifact.md#E5.42``) and the plan (``tutorial_restructure_plan.md#L902-L920``), so any failing observation traces back to a peer-reviewed pedagogical principle or a prescriptive line of the migration plan.
- Runtime stages (clean-kernel execution, budgets, leakage check, visual identity) are off by default at Day-0 and turn on once the first three Release-1 tutorials are drafted; see ``docs/tutorial_implementation_strategy.md`` — "Bootstrap (Day-0)".
