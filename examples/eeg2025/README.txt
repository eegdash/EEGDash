EEG 2025 Foundation Challenge
=============================

End-to-end pipelines for the two tracks of the EEG 2025 Foundation
Challenge: cross-task transfer learning, and subject-invariant
representations for clinical-factor prediction. Difficulty 3; assumes
the transfer and foundation-model track.

These tutorials are the runnable companions to the Foundation Challenge
manuscript. Each script ships with pre-trained weights so the build
reproduces the headline number without retraining from scratch, and
each frames the problem the way the Challenge protocol requires: the
training set, the held-out evaluation, and the metric. Pairs with
``docs/tutorial_restructure_plan.md`` Category H (lines 458-470) and
follows the transfer principles in Schirrmeister et al. (2017).

What you will learn:

- Challenge 1 (cross-task transfer): how to pretrain on a passive EEG
  task and transfer the learned representation to an active task,
  evaluated across held-out subjects.
- Challenge 2 (subject-invariant representation): how to predict
  clinical factors (here: p-factor) with representations that
  generalise across subjects, evaluated against a per-subject baseline.
- How to load the pre-trained weights that ship with this gallery and
  reproduce the published numbers without retraining.
- How to extend either pipeline to a third task or a different
  clinical factor.

Run the tutorials:

1. ``tutorial_challenge_1.py`` -- cross-task transfer learning.
2. ``tutorial_challenge_2.py`` -- subject-invariant representation for
   clinical-factor prediction.
