# Transfer, Foundation Models, and EEG2025

Four advanced lessons on transfer learning and foundation-model
fine-tuning, framed around the EEG 2025 Foundation Challenge.
Difficulty 3; assumes the core workflow, features, and evaluation
tracks.

Transfer is where the EEG decoding field is moving fastest, and it is
also where most of the unprincipled choices accumulate: tasks selected
to make the transfer score look good, evaluation that does not respect
subject boundaries, fine-tuning learning rates pulled from thin air.
These lessons follow Schirrmeister et al. (2017) for the architecture
and training principles, and use the EEG 2025 Challenge tasks as the
concrete, reproducible benchmark. Sourced from
`docs/tutorial_restructure_plan.md` Category H (lines 458-470).

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
2. `plot_71_cross_task_transfer.py` – EEG 2025 Challenge 1.
3. `plot_72_subject_invariant_regression.py` – EEG 2025 Challenge 2.
4. `plot_73_finetune_pretrained_model.py` – fine-tune a Braindecode
   pretrained model.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="The EEG 2025 Foundation Challenge ships its own loader, EEGChallengeDataset, on top of the same eegdash infrastructure that powers EEGDashDataset. The data pool is the Healthy Brain Network release (HBN; Alexander et al. 2017), reachable through NEMAR delorme2022nemar: every recording is downsampled from 500 Hz to 100 Hz, band-pass filtered 0.5-50 Hz, and shipped via a fixed S3 bucket so the leaderboard contract stays reproducible cisotto2024tips. This tutorial walks through the loader, surfaces the task taxonomy, the participant-level metadata, and the official catalog row a single recording carries. The deliverable is one pandas.DataFrame with records-per-task counts and one three-panel figure rendered from the live metadata.">  <div class="sphx-glr-thumbnail-title">How do I get started with the EEG 2025 Foundation Challenge dataset?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Can a small EEG encoder pretrained on passive resting-state windows help a downstream model decode contrastChangeDetection (CCD) that it never saw, on the same subjects drawn from the EEG2025 Challenge 1 mini release? In vision and language the answer is &quot;yes by a wide margin&quot;. For EEG the literature is younger but converging on the same shape: self-supervised or auxiliary-task pretraining tends to lift downstream accuracy when labels are scarce (Banville et al. 2021, doi:10.1109/TNSRE.2020.3040290; Defossez et al. 2023, doi:10.1038/s42256-023-00714-5). This tutorial wires the two halves of EEG2025 Challenge 1 together, passive source and active target on the same subject pool (Aristimunha et al. 2025, doi:10.48550/arXiv.2506.19141), and asks how big the gap between a fine-tuned encoder and a from-scratch baseline really is. When the encoder transfers, by how much does it beat chance?">  <div class="sphx-glr-thumbnail-title">Pretrain on resting-state, fine-tune on contrast-change detection</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Challenge 2 of the EEG2025 Foundation Challenge asks whether a model can predict the p-factor from EEG without secretly memorising subject identity. The p-factor (Caspi et al. 2014, doi:10.1177/2167702613497473) is a general dimension of psychopathology derived from the Child Behavior Checklist; the EEG side comes from the Healthy Brain Network release distributed via EEGChallengeDataset (Alexander et al. 2017, doi:10.1038/sdata.2017.181), surfaced through NEMAR (Delorme et al. 2022, doi:10.1093/nargab/lqac023). The setup is a strict out-of-distribution regression: the test cohort never overlaps with the train cohort on subject. A model that secretly memorises subjects scores well within-fold and collapses on a new sitting, so we build the cross_subject loop from plot_51, fit a feature-based regression head, and report r2 against median_baseline, the chance level for regression (Cisotto &amp; Chicco 2024 Tip 9, doi:10.7717/peerj-cs.2256). The headline question is not &quot;can we win Challenge 2?&quot;; the p-factor signal in EEG is genuinely faint. The honest one is: does this model beat the train-set median predictor on never-seen-before subjects?">  <div class="sphx-glr-thumbnail-title">Subject-invariant p-factor regression (EEG2025 Challenge 2)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="A pretrained EEG encoder packs hundreds of hours of recordings into a weight matrix. Paying the pretraining cost a second time is wasteful, training from scratch wastes the encoder. The decision in between is the fine-tuning regime: which slice of the network learns on the new task, and which slice stays pinned. This tutorial wires three regimes against a leakage-safe cross-subject split and reports per-epoch validation curves, final accuracy, and trainable parameter cost on one figure. The data come through a synthetic pretrain/target pair sized to mirror an OpenNeuro recording cataloged on NEMAR (Delorme et al. 2022, doi:10.1038/s41597-022-01795-4); the recipe transfers to any EEGDash windowed dataset by swapping synth_windows for an EEGDashDataset query. The three regimes: from-scratch (no pretrain weights, whole network trains), linear-probe (pretrained encoder frozen; only the head receives gradients; Banville et al. 2021, doi:10.1088/1741-2552/abca18), and full-finetune (encoder loaded, head reset, every parameter trains; Defossez et al. 2023, doi:10.1038/s42256-023-00714-5). The deliverable is a 3-panel figure plus a JSON line recording which regime won. So which one wins?">  <div class="sphx-glr-thumbnail-title">How do I adapt a pretrained EEG model to a new task?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Meta Research ships four projects under the NeuroAI umbrella (https://facebookresearch.github.io/neuroai/): NeuralFetch for unified dataset discovery across 12 catalogues, NeuralSet for turning neural and stimulus data into AI-ready tensors, NeuralTrain for deep-learning training on the resulting PyTorch dataset, and NeuralBench for a unified benchmark across brain foundation models. EEGDash is one of NeuralFetch&#x27;s 12 supported backends, alongside OpenNeuro (BIDS), DANDI (NWB), HuggingFace, Zenodo, Figshare, PhysioNet, Dryad, Donders, DataLad, Synapse, and OSF. This tutorial walks the four-stage path EEGDashDataset -&gt; NeuralFetch.Study -&gt; NeuralSet.Segmenter -&gt; torch.utils.data.DataLoader so a recording cataloged on NEMAR delorme2022nemar flows through the rest of the NeuroAI stack with the bytes intact cisotto2024tips. The recipe composes with self-supervised pretraining on EEG (Banville et al. 2021) and cross-task brain decoding defossez2023brain. The deliverable is a 3-panel figure: stage diagram, per-stage shape sanity check, integration matrix. So which projects already share the events DataFrame, and which ones live downstream of it?">  <div class="sphx-glr-thumbnail-title">How do I plug EEGDash into the Meta NeuroAI ecosystem?</div>
</div>
<!-- thumbnail-parent-div-close --></div>
