# Transfer, Foundation Models, and EEG2025

Four advanced lessons on transfer learning and foundation-model
fine-tuning, framed around the EEG2025 Foundation Challenge.
Difficulty 3; assumes the core workflow, features, and evaluation
tracks.

Transfer is where the EEG decoding field is moving fastest, and it is
also where most of the unprincipled choices accumulate: tasks selected
to make the transfer score look good, evaluation that does not respect
subject boundaries, fine-tuning learning rates pulled from thin air.
These lessons follow Schirrmeister et al. (2017) for the architecture
and training principles, and use the EEG2025 Challenge tasks as the
concrete, reproducible benchmark.

What you will learn:

- How `EEGChallengeDataset` differs from `EEGDashDataset` and when
  to reach for which.
- How to set up a cross-task transfer experiment (Challenge 1):
  resting-state pretraining transferred to contrast-change detection.
- How to run subject-invariant regression for clinical-factor
  prediction (Challenge 2): predict p-factor across held-out subjects.
- How to fine-tune a Braindecode pretrained model on a downstream
  task with sane hyperparameter choices.
- How to read a transfer result critically: what scores actually mean
  when the source and target tasks share subjects.

Run the lessons in order:

1. `plot_70_challenge_dataset_basics.py` – `EEGChallengeDataset`
   basics.
2. `plot_71_cross_task_transfer.py` – EEG2025 Challenge 1.
3. `plot_72_subject_invariant_regression.py` – EEG2025 Challenge 2.
4. `plot_73_finetune_pretrained_model.py` – fine-tune a Braindecode
   pretrained model.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 10s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">How do I get started with the EEG2025 Foundation Challenge dataset?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 6m | Compute: GPU Preferred">  <div class="sphx-glr-thumbnail-title">Pretrain on resting-state, fine-tune on contrast-change detection (Simulated Data)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">Subject-invariant p-factor regression (EEG2025 Challenge 2)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 45s | Compute: GPU Recommended">  <div class="sphx-glr-thumbnail-title">How do I adapt a pretrained EEG model to a new task?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 20s | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">How do I plug EEGDash into the Meta NeuroAI ecosystem?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
