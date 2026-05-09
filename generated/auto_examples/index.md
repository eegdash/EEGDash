<a id="sphx-glr-generated-auto-examples"></a>

# Examples gallery

The EEGDash gallery is the runnable, narrative half of the docs: the **Concepts** chapter explains *why* a decision matters, the API reference enumerates every public symbol, and the gallery you’re reading shows the choices in motion against real BIDS-curated EEG records. Every script under `examples/` is a sphinx-gallery tutorial – meaning it executes top to bottom on every documentation build, and the captured first figure is the thumbnail you see below.

The intended path: read the curated **Tutorials** in order, dip into **How-to recipes** when you have a specific question, then scale up using the **Applied research projects**, the **EEG2025 Foundation Challenge** pipelines, and the **High-performance computing** track.

## Tutorials (curated learning path)

Seven categories, ordered the way we would teach them: install, load, decode events, decode state, engineer features, evaluate rigorously, then scale to transfer and foundation models.

### Choose your path

| Your goal                      | Start with                                                                            | Then read                                                       |
|--------------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| **Load my first dataset**      | [Start Here](tutorials/00_start_here/index.md)                                        | [Core Decoding Workflow](tutorials/10_core_workflow/index.md)   |
| **Train a classifier safely**  | [Core Decoding Workflow](tutorials/10_core_workflow/index.md)                         | [Evaluation and Benchmarking](tutorials/50_evaluation/index.md) |
| **Extract classical features** | [Feature Engineering](tutorials/40_features/index.md)                                 | [How-To Guides](how_to/index.md)                                |
| **Run on a cluster**           | [How-To Guides](how_to/index.md)                                                      | [HPC tutorials](hpc/index.md)                                   |
| **Join EEG2025**               | [Transfer, Foundation Models, and EEG2025](tutorials/70_transfer_foundation/index.md) | [EEG2025 Foundation Challenge](eeg2025/index.md)                |

Start with the absolute beginner tutorials.

Dive into real-world research case studies.

Move from local scripts to cluster-wide jobs.

Enter the official Foundation Challenge.

### Start Here

Difficulty 1. Three short lessons that take you from a fresh install to a working PyTorch `DataLoader` over real EEG records: find datasets and records, load one recording and inspect it, then turn an `EEGDashDataset` into windows and a dataloader. CPU-only, each runs in under a few minutes.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: &lt;2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Find datasets with the EEGDash API</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Load one EEG recording</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">EEG recording to PyTorch DataLoader</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Core Decoding Workflow

Difficulty 1-2. The canonical EEG decoding pipeline in four lessons: preprocess and window, split without subject leakage, train a baseline against chance, and persist prepared data for reuse. The leakage-safe split lesson is the rubric anchor for E3.27 invariants and Cisotto and Chicco 2024’s evaluation guidance.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Preprocess EEG and create windows</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Split EEG without subject leakage</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Train a leakage-safe baseline</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 5s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Save and reload prepared data</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Event-Related Decoding

Difficulty 2. Two lessons that decode labels coming from events and annotations rather than continuous state: a P3 target-versus-standard classifier on a visual oddball paradigm, then the auditory oddball framed as a contrast with the visual case.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Visual P300 oddball decoding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 3m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Auditory P300 oddball decoding</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Resting-State and State Decoding

Difficulty 1. The canonical beginner decoding lesson: eyes-open versus eyes-closed classification on resting-state EEG, decoded from alpha-rhythm differences with a band-power baseline.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: 5m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Decode eyes open vs. eyes closed</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Feature Engineering

Difficulty 1-2. EEGDash’s feature extraction package as a first-class option, not an afterthought to deep learning. Three lessons cover feature tables from windows, preprocessor and dependency trees that avoid recomputation, and a scikit-learn / LightGBM baseline straight from the feature table.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Extract band-power features</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 5s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Compose EEG markers from Welch PSD</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 10s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">EEGDash features to scikit-learn</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Evaluation and Benchmarking

Difficulty 2-3. Five lessons that treat decoding evaluation as a core skill, drawing on MOABB (Chevallier, Aristimunha et al. 2024). Builds from a single split toward benchmark-grade pipeline comparison: within-subject, cross-subject, cross-session, learning curves, and a paired Wilcoxon comparison of two pipelines.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Within-subject decoding evaluation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2-3 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Cross-subject decoding evaluation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2-3 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Cross-session decoding evaluation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2-3 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Decoding accuracy learning curves</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2-3 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Compare two decoding pipelines</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2-3 | Runtime: 2m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Benchmark EEGDash with MOABB</div>
</div>
<!-- thumbnail-parent-div-close --></div>

### Transfer, Foundation Models, and EEG2025

Difficulty 3. Four advanced lessons on transfer learning and foundation-model fine-tuning, framed around the EEG2025 Foundation Challenge: `EEGChallengeDataset` basics, cross-task transfer (Challenge 1), subject-invariant p-factor regression (Challenge 2), and fine-tuning a Braindecode pretrained model. Builds on Schirrmeister et al. 2017.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1-2 | Runtime: 10s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">How do I get started with the EEG2025 Foundation Challenge dataset?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 6m | Compute: GPU Preferred">  <div class="sphx-glr-thumbnail-title">Pretrain on resting-state, fine-tune on contrast-change detection (Simulated Data)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">Subject-invariant p-factor regression (EEG2025 Challenge 2)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 45s | Compute: GPU Recommended">  <div class="sphx-glr-thumbnail-title">How do I adapt a pretrained EEG model to a new task?</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 20s | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">How do I plug EEGDash into the Meta NeuroAI ecosystem?</div>
</div>
<!-- thumbnail-parent-div-close --></div>

## How-to recipes

Task-focused snippets that assume you already know the basics: how to download a dataset, run preprocessing on SLURM, parallelize feature extraction, use the HPC cache, and work offline. Each guide answers a single question; cross-link with the HPC track when relevant.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: 1m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Download an EEGDash dataset in advance and validate the local cache</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 30s | Compute: CPU (Multi-core)">  <div class="sphx-glr-thumbnail-title">Parallelize EEGDash feature extraction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 20s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Place the EEGDash cache on shared or local cluster storage</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 4m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">How-to: work offline against a populated EEGDash cache</div>
</div>
<!-- thumbnail-parent-div-close --></div>

## Applied research projects

Project-style examples that target a concrete scientific question – age regression, p-factor prediction, sex classification, P300 transfer, clinical-catalog summary – with realistic data sizes, runtimes, and limitations. Treat them as starting points, not prescriptive recipes.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Age regression from EEG</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 1 | Runtime: 20s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Survey clinical EEG datasets</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 2m | Compute: GPU Recommended">  <div class="sphx-glr-thumbnail-title">P300 transfer with AS-MMD</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 25s | Compute: GPU Recommended">  <div class="sphx-glr-thumbnail-title">Predict p-factor with deep learning</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Predict p-factor from EEG features</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 3-5m | Compute: CPU">  <div class="sphx-glr-thumbnail-title">Sex classification from EEG</div>
</div>
<!-- thumbnail-parent-div-close --></div>

## EEG2025 Foundation Challenge

End-to-end pipelines for the two EEG2025 Foundation Challenge tracks: cross-task transfer learning (passive to active), and subject-invariant representations for clinical factor prediction. Pre-trained weights ship alongside each tutorial.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 3-6m | Compute: CPU (GPU Recommended)">  <div class="sphx-glr-thumbnail-title">EEG2025 Challenge 1 Baseline (CCD)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 3 | Runtime: 30s | Compute: CPU (GPU Optional)">  <div class="sphx-glr-thumbnail-title">EEG2025 Challenge 2 Baseline (p-factor)</div>
</div>
<!-- thumbnail-parent-div-close --></div>

## High-performance computing

Reference setup for running EEGDash on shared HPC clusters: SLURM submission scripts (CPU and GPU), a Dockerfile, and a tutorial showing how to combine the on-disk cache with batch scheduling for an eyes-open / eyes-closed run.

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Difficulty 2 | Runtime: 5-10m | Compute: HPC / Cluster">  <div class="sphx-glr-thumbnail-title">Eyes Open vs. Closed Classification (HPC)</div>
</div>
<!-- thumbnail-parent-div-close --></div>
