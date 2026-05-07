Applied Research Projects
=========================

Project-style examples that target a concrete scientific question --
age regression, p-factor prediction, sex classification, P300 transfer,
clinical-catalog summary -- with realistic data sizes, runtimes, and
limitations. Difficulty 2-3; assumes the *Start Here* trio and the core
workflow track.

These are not first-week tutorials. They are scaffolds for your own
analyses: each script frames a research question, picks an appropriate
evaluation regime, runs a defensible baseline, and surfaces the
limitations honestly so you know what would have to change before any
result here could be cited. Compared with tutorials, these projects
emphasise labels, splits, baselines, and reporting rather than the
individual EEGDash API calls. Sourced from
``docs/tutorial_restructure_plan.md`` Category G (lines 1052-1100,
"Applied Examples To Keep But Reframe").

What you will learn:

- How to frame an EEG-from-population study (age, sex, p-factor) as a
  single regression or classification problem with an honest baseline.
- How to choose between a feature pipeline and a deep model based on
  data size and the question being asked.
- How to apply transfer learning across paradigms (P300 transfer
  across subjects and sessions) without leaking labels.
- How to summarise a clinical catalogue (subjects, sessions,
  conditions, hours of recording) for inclusion in a paper.
- How to write up the limitations section that an EEG paper actually
  needs (Cisotto and Chicco 2024; Pernet et al. 2019 for BIDS).

Treat each script as a starting point for your own work, not a
prescriptive recipe.
