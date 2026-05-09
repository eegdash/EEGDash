# EEG2025 Foundation Challenge

End-to-end pipelines for the two tracks of the EEG2025 Foundation
Challenge: cross-task transfer learning, and subject-invariant
representations for clinical-factor prediction. Difficulty 3; assumes
the transfer and foundation-model track.

These tutorials are the runnable companions to the Foundation Challenge
manuscript. Each script ships with pre-trained weights so the build
reproduces the headline number without retraining from scratch, and
each frames the problem the way the Challenge protocol requires: the
training set, the held-out evaluation, and the metric. This follows
the transfer principles in Schirrmeister et al. (2017).

What you will learn:

- Challenge 1 (cross-task transfer): how to pretrain on a passive EEG
  task and transfer the learned representation to an active task,
  evaluated across held-out subjects.
- Challenge 2 (subject-invariant representation): how to predict
  clinical factors (here: p-factor) with representations that
  generalize across subjects, evaluated against a per-subject baseline.
- How to load the pre-trained weights that ship with this gallery and
  reproduce the published numbers without retraining.
- How to extend either pipeline to a third task or a different
  clinical factor.

Run the tutorials:

1. `tutorial_challenge_1.py` – cross-task transfer learning.
2. `tutorial_challenge_2.py` – subject-invariant representation for
   clinical-factor prediction.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 3-6m | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">EEG2025 Challenge 1 Baseline (CCD)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU (GPU Optional)">  <div class="sphx-glr-thumbnail-title">EEG2025 Challenge 2 Baseline (p-factor)</div>
</div>
<!-- thumbnail-parent-div-close --></div>
