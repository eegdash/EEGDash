# EEG 2025 Foundation Challenge

End-to-end pipelines for the two tracks of the EEG 2025 Foundation
Challenge: cross-task transfer learning, and subject-invariant
representations for clinical-factor prediction. Difficulty 3; assumes
the transfer and foundation-model track.

These tutorials are the runnable companions to the Foundation Challenge
manuscript. Each script ships with pre-trained weights so the build
reproduces the headline number without retraining from scratch, and
each frames the problem the way the Challenge protocol requires: the
training set, the held-out evaluation, and the metric. Pairs with
`docs/tutorial_restructure_plan.md` Category H (lines 458-470) and
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

1. `tutorial_challenge_1.py` – cross-task transfer learning.
2. `tutorial_challenge_2.py` – subject-invariant representation for
   clinical-factor prediction.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Challenge 1 of the EEG2025 Foundation Challenge asks you to decode a trial-level cognitive decision from EEG: in the contrastChangeDetection (CCD) task subjects watch two flickering striped discs, one disc&#x27;s contrast slowly ramps up, and the subject presses left or right to report which one. The data come from the Healthy Brain Network release (HBN; Alexander et al. 2017) served through NEMAR delorme2022nemar and shipped via EEGChallengeDataset as 100 Hz BDFs (downsampled, 0.5-50 Hz pass-band; Cisotto &amp; Chicco 2024). This starter kit walks through the four steps every Challenge 1 entry has to clear: load the CCD recordings, carve out a stimulus-locked window, train a small Braindecode CNN baseline schirrmeister2017braindecode, and ship one figure that ties the trial structure, the windowed signal, and the per-fold accuracy together (Aristimunha et al. 2025, doi:10.48550/arXiv.2506.19141). The deliverable is one (n_channels, n_samples) = (129, 200) window contract and one three-panel figure ready to drop into your submission.">  <div class="sphx-glr-thumbnail-title">How do I get my first baseline running for EEG2025 Challenge 1 (CCD)?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The EEG2025 Foundation Challenge ships two regression tracks, and Challenge 2 asks for a single number per subject, the p-factor, from a short clip of resting-state EEG. The p-factor (Caspi et al. 2014, doi:10.1177/2167702613497473) is a general dimension of psychopathology derived from the Child Behavior Checklist; the EEG side comes from the Healthy Brain Network release distributed via EEGChallengeDataset (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023).">  <div class="sphx-glr-thumbnail-title">How do I submit a baseline to EEG2025 Challenge 2 (predict the p-factor)?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
