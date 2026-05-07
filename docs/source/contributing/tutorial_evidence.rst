:orphan:

===========================
Tutorial Evidence Dashboard
===========================

The contributing-level evidence index now lives at the top of the docs tree:

- :doc:`/evidence` — the rendered dashboard (Diataxis map, per-tutorial table,
  category cards, and dossier links).

The page is generated at build time by the in-tree
``eegdash-evidence-dashboard`` directive, which reads each
``docs/evidence/tutorials/<id>/evidence.json`` (and the optional
``reviewer_score.json``) and emits a styled HTML table with state badges,
difficulty stars, totals, reviewer ``min_score``, and a link to the
GitHub-hosted dossier.

To regenerate the dossiers themselves, run::

    python -m scripts.tutorial_audit.report --aggregate

or, scoped to a single phase::

    make -f tutorials.mk tutorial-phase-report PHASE=<n>
