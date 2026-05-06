# EEGDash Tutorial Restructure Plan

Date: 2026-05-04

This document audits the current EEGDash documentation and proposes a new tutorial
architecture for users doing machine learning on EEG, MEG, fNIRS, EMG, and iEEG
datasets. The plan follows the Diataxis documentation model and draws practical
inspiration from Neuromatch Academy, Braindecode, and MOABB.

## Sources Consulted

- [Diataxis](https://diataxis.fr/): documentation should be organized around four
  user needs: tutorials, how-to guides, reference, and explanation. Tutorials are
  learning-oriented lessons; how-to guides are goal-oriented recipes; reference is
  descriptive; explanation is understanding-oriented.
- [Neuromatch Academy courses](https://neuromatch.io/courses/): course material is
  organized as mini-modules with short conceptual introductions, narrative
  explanation, and interactive Python notebooks, with projects as a separate
  research track.
- [Neuromatch Computational Neuroscience tutorials](https://compneuro.neuromatch.io/tutorials/intro.html)
  and [Deep Learning tutorials](https://deeplearning.neuromatch.io/tutorials/intro.html):
  the curriculum uses prerequisite material, named days/modules, multiple small
  tutorials per concept, and separate project guidance.
- [Braindecode examples](https://braindecode.org/stable/auto_examples/index.html):
  examples are grouped by user task, including basic model building, data loading,
  advanced training strategies, and applied real-world datasets.
- [MOABB examples](https://moabb.neurotechx.com/docs/auto_examples/index.html):
  examples separate getting-started tutorials, paradigm-specific evaluations,
  data management, benchmarking, advanced examples, and learning-curve analyses.
- [MOABB splitters](https://moabb.neurotechx.com/docs/generated/moabb.evaluations.WithinSessionSplitter.html)
  and related evaluation APIs: MOABB exposes reusable splitter classes for
  within-session, within-subject, cross-session, cross-subject, cross-dataset,
  and learning-curve evaluation. These work on labels plus metadata and can be
  reused without adopting MOABB's full dataset/paradigm stack.
- [MOABB benchmark function](https://moabb.neurotechx.com/docs/generated/moabb.benchmark.html):
  benchmark orchestration already supports WithinSession, CrossSession, and
  CrossSubject evaluations over scikit-learn pipelines and compatible MOABB
  paradigms.
- [MOABB within-session splitter tutorial](https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_within_session_splitter.html):
  shows why naive splits can be misleading and demonstrates explicit use of
  `WithinSessionSplitter.split(y, metadata)` with `subject`, `session`, and
  `run` metadata.
- [MOABB plotting APIs](https://moabb.neurotechx.com/docs/api.html#statistics-visualization-and-utilities):
  expose `score_plot`, `paired_plot`, `summary_plot`,
  `meta_analysis_plot`, and `dataset_bubble_plot`, making benchmark plots part
  of the public evaluation workflow.
- [MOABB dataset bubble plot tutorial](https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_dataset_bubbles.html):
  shows a compact visual grammar for dataset size and structure: one bubble per
  subject, bubble size for trials or duration, color for paradigm, and alpha for
  sessions.
- [Current EEGDash tutorial gallery](https://eegdash.org/generated/auto_examples/index.html)
  and [current EEGDash user guide](https://eegdash.org/user_guide.html).
- [Hugging Face model cards](https://huggingface.co/docs/hub/en/model-cards)
  and [dataset cards](https://huggingface.co/docs/datasets/dataset_card):
  model/dataset metadata improves discoverability, responsible use, and
  reproducibility.
- [GitHub repository topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics):
  public topics help users discover repositories by purpose, community, and
  technology.
- [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html)
  and [paper format](https://joss.readthedocs.io/en/latest/paper.html):
  software papers reward installability, tests, documentation, open-source
  practices, research impact, and clear software design.
- [NeurIPS 2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers)
  and [NeurIPS 2026 Evaluations & Datasets announcement](https://blog.neurips.cc/category/2026-conference/):
  NeurIPS explicitly includes neuroscience, data-centric AI, SysML
  infrastructure, generalization, and an Evaluations & Datasets track where
  evaluation itself is treated as a scientific object.
- [NeurIPS 2025 Datasets & Benchmarks call](https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks):
  required dataset/benchmark code, accessible hosting, machine-readable
  metadata, and benchmark methodology are now central expectations.
- [ICLR 2026 call and author guide](https://iclr.cc/Conferences/2026/CallForPapers):
  ICLR lists representation learning, transfer, self-supervision, datasets,
  benchmarks, software infrastructure, neuroscience applications, ethics, and
  reproducibility as relevant concerns.
- [ICML 2026 call and author instructions](https://icml.cc/Conferences/2026/CallForPapers):
  ICML emphasizes original rigorous ML contributions with reproducible
  experiments, sound claims, code/data supplements, and clear relation to the
  broader ML literature.
- [Hugging Face MCP server documentation](https://huggingface.co/docs/hub/agents-mcp):
  the Hub MCP server exposes paper search, repository search, repository details,
  documentation search, and Space tools for AI assistants.
- Hugging Face MCP paper search results for EEG foundation models, EEG decoding,
  arbitrary electrode montage, cross-subject/cross-task benchmarks, clinical EEG
  benchmarks, and multimodal EEG decoding.
- [Braindecode foundation-model tutorial](https://braindecode.org/dev/auto_examples/model_building/plot_load_pretrained_models.html):
  Braindecode already supports Hub `from_pretrained`, `push_to_hub`,
  `return_features`, `reset_head`, `get_config`, `from_config`, and curated EEG
  foundation checkpoints.
- [Braindecode Signal-JEPA fine-tuning tutorial](https://braindecode.org/dev/auto_examples/advanced_training/plot_finetune_foundation_model.html):
  fine-tuning, channel preparation, event windows, freezing, and `EEGClassifier`
  are already covered in Braindecode.
- [Braindecode Hub dataset integration](https://braindecode.org/dev/auto_examples/datasets_io/plot_hub_integration.html):
  Braindecode already uploads/downloads `WindowsDataset`, `EEGWindowsDataset`,
  and `RawDataset` through Hugging Face Hub using Zarr and
  `BaseConcatDataset.pull_from_hub`.
- [OpenEEG-Bench Arena](https://huggingface.co/spaces/braindecode/OpenEEGBench):
  an existing Braindecode Hugging Face Space for open, reproducible EEG model
  comparison, with benchmark registry, leaderboard fields, and `eval.yaml`
  generation.
- Local source tree, especially `docs/source`, `examples/tutorials`,
  `examples/core`, `examples/eeg2025`, `examples/hpc`, and the public API in
  `eegdash`.

## Executive Summary

EEGDash has the pieces for a useful learning ecosystem: a metadata client,
dataset loaders compatible with braindecode/PyTorch, BIDS-aware local/offline
loading, feature extraction tools, competition datasets, and several decoding
examples. The current documentation does not yet teach those pieces in a
structured way.

The most important change is to stop treating every runnable script as a
"tutorial". A tutorial should be a reliable lesson that gives a new user early
success and gradually builds a mental model. Many current scripts are better
classified as how-to guides, applied examples, benchmarks, competition lessons,
or developer notes.

The proposed structure:

1. Keep a short "Start here" learning path with 6-8 true tutorials.
2. Move task-specific recipes into "How-to guides".
3. Add explanation pages for EEGDash concepts and EEG decoding pitfalls.
4. Expand the reference around stable public APIs and metadata schema.
5. Keep applied examples and EEG2025 material, but label them as advanced/project
   material rather than core onboarding.

The resulting documentation should answer one core user question:

> How do I go from "I need EEG data for a decoding experiment" to a reproducible,
> leakage-safe, inspectable ML workflow using EEGDash?

## What We Have Today

### Documentation Top Level

Current Sphinx navigation is roughly:

- `Datasets`: generated dataset catalog and summary visualizations.
- `User Guide`: single broad page covering `EEGDash`, `EEGDashDataset`, filters,
  local mode, and API configuration.
- `Install`: PyPI and source install.
- `Examples`: Sphinx-Gallery generated from `examples`.
- `Docs`: API reference.

This is serviceable as a site skeleton, but the learning path is shallow. The
user guide is compact and useful for quick orientation, but it tries to cover
metadata search, dataset loading, local mode, and API configuration in one page.
It does not yet teach EEG decoding workflow decisions.

### Current Sphinx-Gallery Inputs

The gallery is generated from `examples` using Sphinx-Gallery. The current
folders are:

- `examples/core`
- `examples/dev_scripts`
- `examples/tutorials`
- `examples/eeg2025`
- `examples/hpc`

The current public gallery groups them as:

- "EEG Dash"
- "Dev scripts for EEGDash"
- "General tutorials"
- "EEG 2025 Foundation Challenge"
- "HPC tutorials"

This grouping reflects repository history more than user learning needs.

### Quantitative Snapshot

The public gallery currently draws from 20 Python files and about 6,700 lines of
example code:

| Folder | Files | Approximate lines | Current character |
| --- | ---: | ---: | --- |
| `examples/core` | 4 | 1,582 | Mixed beginner tutorials and advanced transfer examples. |
| `examples/tutorials` | 11 | 3,531 | Long applied workflows, many marked `noplot_`. |
| `examples/eeg2025` | 3 | 1,085 | Competition-specific lessons and offline usage. |
| `examples/hpc` | 1 | 389 | Cluster-oriented duplicate of an EO/EC workflow. |
| `examples/dev_scripts` | 1 | 118 | Developer/debug material exposed in the public gallery. |

The generated `sg_execution_times.rst` records 20 gallery files and a historical
full execution time of more than 40 minutes. That is acceptable for an advanced
example gallery, but too fragile for the primary beginner learning path. The
first tutorials should be shorter, less network-dependent, and smoke-testable.

### Current Tutorial Inventory

| File | Current role | Keep, split, move, or retire |
| --- | --- | --- |
| `examples/core/tutorial_minimal.py` | Minimal EEGChallengeDataset plus braindecode model | Replace with a safer first tutorial. It currently notes obvious leakage and takes a long time in gallery execution. |
| `examples/core/tutorial_eoec.py` | Eyes open/closed classification | Keep as a core learning tutorial after shortening and making the split/evaluation story explicit. |
| `examples/core/tutorial_feature_extractor_open_close_eye.py` | EO/EC plus feature extraction | Merge concepts into a feature-engineering tutorial; avoid duplicating the EO/EC decoding tutorial. |
| `examples/core/p300_transfer_learning.py` | P300 transfer learning with AS-MMD | Move to advanced examples or transfer learning track. |
| `examples/tutorials/tutorial_api.py` | Metadata query tutorial | Keep, but promote earlier as "Find datasets and records". Add richer query patterns and expected outputs. |
| `examples/tutorials/noplot_tutorial_age_prediction.py` | Age regression with Conformer | Move to applied clinical/regression track. Make it a project-style example, not onboarding. |
| `examples/tutorials/noplot_tutorial_audi_oddball.py` | Auditory oddball classification | Keep as event-related decoding how-to or applied tutorial. |
| `examples/tutorials/noplot_tutorial_p3_oddball.py` | Visual P3 oddball classification | Keep one P3/oddball tutorial; either merge with auditory oddball comparison or turn one into a how-to. |
| `examples/tutorials/noplot_tutorial_feature_extraction.py` | Feature extraction plus sex classification | Split into "Extract features" and "Train classical baseline". Remove hidden TODOs. |
| `examples/tutorials/noplot_tutorial_features_eoec.py` | Long feature extraction walkthrough | Use as source material for a better feature extractor lesson; currently too long for a beginner tutorial. |
| `examples/tutorials/noplot_tutorial_pfactor_features.py` | p-factor feature regression | Move to applied clinical/regression examples. |
| `examples/tutorials/noplot_tutorial_pfactor_regression.py` | p-factor deep regression | Move to applied clinical/regression examples. Avoid duplicating p-factor feature tutorial. |
| `examples/tutorials/noplot_tutorial_sex_classification_cnn.py` | Sex classification CNN | Move to applied examples; keep only if framed carefully around labels, leakage, and limitations. |
| `examples/tutorials/plot_clinical_summary.py` | Dataset summary visualization | Move to "How to explore the catalog" or "Dataset catalog examples". |
| `examples/tutorials/tutorial_transfer_learning.py` | Simulated transfer learning | Retire or rewrite with real EEGDash data. A simulated example does not teach EEGDash enough. |
| `examples/eeg2025/tutorial_challenge_1.py` | EEG2025 challenge tutorial | Keep under a separate "Competition and foundation challenge" track. |
| `examples/eeg2025/tutorial_challenge_2.py` | EEG2025 challenge tutorial | Keep under separate challenge track. |
| `examples/eeg2025/tutorial_eegdash_offline.py` | Offline workflow | Move to how-to guide; also cross-link from HPC docs. |
| `examples/hpc/tutorial_eoec.py` | HPC version of EO/EC | Move to "Scaling and HPC" how-to. It should not appear as a duplicate beginner tutorial. |
| `examples/dev_scripts/debug_pybids_braindecode.py` | Developer/debug script | Remove from public tutorial gallery; move to developer notes if needed. |

### Strengths To Preserve

- EEGDash already uses familiar community tools: MNE, BIDS, braindecode,
  PyTorch, scikit-learn, and LightGBM.
- The examples demonstrate real EEG tasks, not toy arrays only.
- There are examples for metadata search, remote download, local/offline mode,
  preprocessing, windowing, feature extraction, deep learning, transfer, HPC,
  and competition use.
- The dataset catalog is a strong entry point. It should become part of the
  learning flow rather than a separate destination.

### Current Problems

- The gallery title "Tutorials!" and README text are superficial.
- Tutorials, how-to guides, applied examples, dev scripts, competition material,
  and HPC examples are mixed together.
- Many notebooks are too long for learning. Several are 300-580 lines and do
  discovery, cleaning, preprocessing, windowing, modeling, evaluation, and
  plotting in one pass.
- Some current examples teach unsafe habits, especially window-level splitting
  where subject/session leakage is possible.
- Several topics are duplicated: EO/EC, p-factor, sex classification, oddball.
- Some examples hide execution with `noplot_`, which makes gallery pages less
  trustworthy as tutorials.
- Several examples rely on live API/network state. True tutorials should use
  stable mini datasets or cached fixtures wherever possible.
- There is no clear answer to "Which tutorial should I do first?"
- There is no explicit distinction between `EEGDash`, `EEGDashDataset`, and
  `EEGChallengeDataset` as user-facing concepts across the tutorial set.
- The current user guide references broad behavior, but not the EEG ML decisions
  users actually need: leakage-safe splits, target labels, channel mismatch,
  sampling rates, event mapping, class imbalance, baselines, and reporting.

## Diataxis Target Structure

### Tutorials: Learning-Oriented Lessons

Tutorials should be the shortest path to competence. They should be sequential,
reliable, concrete, and opinionated. They should avoid optional branches and deep
explanations. Each tutorial should have visible results early and often.

For EEGDash, true tutorials should teach:

- How to find a dataset.
- How to create a dataset object.
- How lazy loading and caching work.
- How to inspect raw EEG metadata and a signal.
- How to preprocess and window safely.
- How to train a simple baseline.
- How to split by subject/session to avoid leakage.
- How to save/reuse prepared data.
- How to move from raw signals to features or neural networks.

### How-To Guides: Goal-Oriented Recipes

How-to guides should answer specific tasks:

- How do I download all recordings for a dataset?
- How do I work offline on an HPC cluster?
- How do I filter by subject, session, task, modality, sample rate, or channels?
- How do I make P300 windows?
- How do I use MOABB-style evaluation?
- How do I extract spectral features?
- How do I train a braindecode model with my own PyTorch loop?
- How do I use EEGDash with local BIDS data?
- How do I handle bad records with `on_error`, `drop_bad`, and `drop_short`?

These should be searchable and standalone, but not presented as the learning
path.

### Reference: Descriptive API And Data Contracts

Reference should be complete and stable:

- Public classes: `EEGDash`, `EEGDashDataset`, `EEGChallengeDataset`.
- Queryable fields and allowed operators.
- Record and dataset metadata schema.
- Cache layout and offline assumptions.
- Feature extractor API.
- Error classes and troubleshooting reference.
- REST API endpoints.
- Glossary of BIDS/EEGDash terms.

### Explanation: Understanding-Oriented Pages

Explanation should help users reason:

- Why subject leakage invalidates EEG decoding results.
- Why window-level samples are not independent.
- How BIDS entities map to EEGDash metadata.
- What lazy loading means and when data is downloaded.
- What EEG preprocessing choices change.
- When to use raw time-series models vs. handcrafted features.
- How cross-subject, cross-session, and cross-dataset evaluation differ.
- How EEGDash relates to MNE, braindecode, MOABB, OpenNeuro, and NEMAR.
- What "foundation challenge" data is and why `EEGChallengeDataset` differs
  from `EEGDashDataset`.

## Proposed Site Information Architecture

Recommended top-level navigation:

1. `Start`
   - Install
   - Quickstart
   - First dataset in 10 minutes
2. `Tutorials`
   - Core learning path
   - Decoding mini-course
   - Feature engineering mini-course
   - Evaluation mini-course
3. `How-to guides`
   - Data discovery
   - Loading and caching
   - Preprocessing and windows
   - Modeling
   - Scaling and offline
   - Local BIDS and contribution workflows
4. `Concepts`
   - EEGDash concepts
   - EEG decoding concepts
   - Interoperability
5. `Examples`
   - Applied tasks
   - Clinical/regression
   - Transfer/foundation models
   - EEG2025 challenge
   - HPC
6. `Datasets`
   - Catalog
   - Dataset detail pages
   - Dataset summary visualizations
7. `Reference`
   - Python API
   - REST API
   - Metadata schema
   - Feature bank
   - Troubleshooting

The current `Examples` link should not be the main learning entry point. It
should remain available, but a curated `Tutorials` landing page should guide
new users.

## Proposed Tutorial Categories

### Category A: Start Here

Purpose: give a new user a successful first EEGDash session in less than 20
minutes, ideally without GPU and without large downloads.

Tutorials:

1. `tutorial_00_first_search.py`: Find EEG datasets and records.
2. `tutorial_01_first_recording.py`: Load one recording and inspect it.
3. `tutorial_02_dataset_to_dataloader.py`: Convert an EEGDash dataset into
   windows and a PyTorch `DataLoader`.

### Category B: Core EEG Decoding Workflow

Purpose: teach the canonical pipeline for EEG ML.

Tutorials:

1. `tutorial_10_preprocess_and_window.py`: Apply MNE/braindecode preprocessing
   and create fixed-length/event windows.
2. `tutorial_11_leakage_safe_split.py`: Use MOABB-backed subject/session splits
   and explain why window-level random splits are unsafe.
3. `tutorial_12_train_a_baseline.py`: Train a simple baseline and report
   metrics.
4. `tutorial_13_save_and_reuse_prepared_data.py`: Save windows/features and
   reload them.

### Category C: Event-Related Decoding

Purpose: teach tasks where labels come from events and annotations.

Tutorials:

1. `tutorial_20_visual_p300_oddball.py`: P3 target vs. standard.
2. `tutorial_21_auditory_oddball.py`: Auditory oddball, ideally framed as a
   contrast with the visual P300 tutorial rather than a duplicate.
3. `tutorial_22_event_mapping_reference_example.py`: A compact how-to style
   guide for remapping annotations across datasets.

### Category D: Resting-State And State Decoding

Purpose: teach fixed windows and state labels.

Tutorials:

1. `tutorial_30_eyes_open_closed.py`: EO/EC classification, core beginner
   decoding tutorial.
2. `tutorial_31_resting_state_features.py`: Extract spectral and signal
   features from resting-state windows.

### Category E: Feature Engineering

Purpose: teach EEGDash's feature extraction package as a first-class option,
not just deep learning.

Tutorials:

1. `tutorial_40_first_features.py`: Extract a small feature table from windows.
2. `tutorial_41_feature_trees.py`: Use preprocessors and feature dependencies
   to avoid repeated computation.
3. `tutorial_42_features_to_sklearn.py`: Train a simple scikit-learn or
   LightGBM baseline from feature tables.
4. `tutorial_43_custom_feature.py`: Write a custom univariate or multivariate
   feature with decorators.

### Category F: Evaluation And Benchmarking

Purpose: teach EEG decoding evaluation as a core skill, taking inspiration from
MOABB.

Tutorials:

1. `tutorial_50_within_subject_evaluation.py`: When and how to do
   within-subject evaluation.
2. `tutorial_51_cross_subject_evaluation.py`: Generalization to unseen
   subjects.
3. `tutorial_52_cross_session_evaluation.py`: Generalization to unseen
   sessions/runs.
4. `tutorial_53_learning_curves.py`: Performance as a function of subjects,
   trials, or windows.
5. `tutorial_54_compare_two_pipelines.py`: Compare a feature baseline against a
   neural network under the same split.

### Category G: Applied Research Projects

Purpose: provide Neuromatch-style project starters. These are not first-week
tutorials; they are realistic examples users can adapt.

Tutorials/examples:

1. `project_age_regression.py`: Predict age from EEG.
2. `project_pfactor_regression_features.py`: Predict p-factor with features.
3. `project_pfactor_regression_deep.py`: Predict p-factor with a neural model.
4. `project_sex_classification.py`: Classification from resting-state data,
   with careful framing around labels, confounds, and evaluation.
5. `project_clinical_dataset_summary.py`: Explore clinical coverage in the
   catalog.

### Category H: Transfer, Foundation Models, And EEG2025

Purpose: keep challenge material coherent and separate.

Tutorials/examples:

1. `tutorial_70_challenge_dataset_basics.py`: Use `EEGChallengeDataset` and
   understand how it differs from `EEGDashDataset`.
2. `tutorial_71_cross_task_transfer.py`: EEG2025 Challenge 1.
3. `tutorial_72_subject_invariant_regression.py`: EEG2025 Challenge 2.
4. `tutorial_73_finetune_pretrained_model.py`: Fine-tune a pretrained
   braindecode/foundation model when a stable example is available.

### Category I: Scaling, Offline, And HPC

Purpose: serve users running EEGDash on clusters.

How-to guides:

1. `how_to_download_a_dataset.py`: Download all files in advance.
2. `how_to_work_offline.py`: Use `download=False`.
3. `how_to_use_hpc_cache.py`: Put cache on shared/local storage.
4. `how_to_run_preprocessing_on_slurm.md`: SLURM job template.
5. `how_to_parallelize_feature_extraction.py`: `n_jobs`, batch sizes, and
   persistence.

## Recommended Initial Tutorial Set

The first release should be smaller than the full plan. A high-quality first
set is better than many fragile notebooks.

### Release 1: Core Learning Path

| Order | Tutorial | Source material | Expected result |
| --- | --- | --- | --- |
| 0 | First EEGDash search | `tutorial_api.py`, user guide | User can query records, inspect fields, and find candidate datasets. |
| 1 | Load one recording | user guide, `EEGDashDataset` API | User can instantiate `EEGDashDataset`, access one `raw`, inspect channels/sfreq/duration, and plot a short segment. |
| 2 | Cache and offline basics | `tutorial_eegdash_offline.py` | User understands when downloads happen, where files go, and how to reuse cached data. |
| 3 | Preprocess and create windows | EO/EC examples | User can resample/filter/select channels and create windows. |
| 4 | Leakage-safe split | current sex/age/p-factor examples | User can split by subject and verify no subject appears in both sets. |
| 5 | Train a baseline | EO/EC or P300 examples | User trains a simple classifier and reports train/test metrics. |
| 6 | Extract features | feature examples | User extracts signal/spectral features and trains a classical baseline. |
| 7 | Event-related decoding | P3 oddball example | User maps events, epochs data, trains a target/standard classifier. |

### Release 2: Applied And Benchmarking Tracks

| Track | Tutorials |
| --- | --- |
| Evaluation | within-subject, cross-subject, cross-session, learning curves, compare pipelines |
| Clinical/regression | age regression, p-factor with features, p-factor deep model |
| Transfer | P300 transfer, EEG2025 challenge 1, EEG2025 challenge 2 |
| Scaling | HPC, offline cache, parallel preprocessing/features |
| Local data | load local BIDS, validate records, add a custom/local dataset |

## Tutorial Design Template

Every true tutorial should follow this structure:

1. Title: action-oriented and concrete.
2. Opening: "In this tutorial, we will ..." with the concrete artifact the user
   will create.
3. Requirements:
   - Estimated time.
   - Data size.
   - CPU/GPU expectations.
   - Network requirement.
   - Prior tutorials.
4. Setup:
   - Imports.
   - Cache directory.
   - Random seed.
   - Dataset ID.
5. Step-by-step lesson:
   - One new concept per step.
   - Run code.
   - Show expected output.
   - Say what to notice.
6. Checkpoints:
   - `assert len(dataset) > 0`
   - channel count
   - sampling frequency
   - class balance
   - no subject leakage
   - window shape
7. Result:
   - plot, printed summary, or small metric table.
8. Minimal wrap-up:
   - What was built.
   - Where to go next.
9. Links:
   - Detailed explanation page.
   - API reference.
   - Related how-to guides.

Avoid in true tutorials:

- Long theoretical sections.
- Multiple optional branches.
- Hidden TODOs.
- Random split of windows unless the tutorial is explicitly showing why it is
  wrong.
- Large live downloads for first-contact tutorials.
- Simulated data unless the purpose is to isolate modeling mechanics.
- Unexplained manual mutation of `.datasets` lists.
- Examples that depend on private local files.

## Proposed File Layout

Sphinx-Gallery can still be used, but the file system should represent the
learning architecture.

```text
docs/source/
  tutorials/
    index.rst
  how_to/
    index.rst
    data_discovery.rst
    caching_offline.rst
    preprocessing_windows.rst
    modeling.rst
    scaling_hpc.rst
  concepts/
    index.rst
    eegdash_objects.rst
    metadata_and_bids.rst
    leakage_and_evaluation.rst
    preprocessing_decisions.rst
    features_vs_deep_learning.rst
  reference/
    metadata_schema.rst
    troubleshooting.rst
examples/
  tutorials/
    00_start_here/
      README.txt
      tutorial_00_first_search.py
      tutorial_01_first_recording.py
      tutorial_02_dataset_to_dataloader.py
    10_core_workflow/
      README.txt
      tutorial_10_preprocess_and_window.py
      tutorial_11_leakage_safe_split.py
      tutorial_12_train_a_baseline.py
      tutorial_13_save_and_reuse_prepared_data.py
    20_event_related/
      README.txt
      tutorial_20_visual_p300_oddball.py
      tutorial_21_auditory_oddball.py
    30_resting_state/
      README.txt
      tutorial_30_eyes_open_closed.py
    40_features/
      README.txt
      tutorial_40_first_features.py
      tutorial_41_feature_trees.py
      tutorial_42_features_to_sklearn.py
  how_to/
    README.txt
    how_to_download_all.py
    how_to_work_offline.py
    how_to_handle_bad_records.py
    how_to_load_local_bids.py
  applied/
    README.txt
    project_age_regression.py
    project_pfactor_features.py
    project_pfactor_deep.py
    project_sex_classification.py
  eeg2025/
    README.txt
    tutorial_challenge_1.py
    tutorial_challenge_2.py
  hpc/
    README.txt
    tutorial_hpc_cache_and_slurm.py
```

Implementation notes:

- Create a curated `docs/source/tutorials/index.rst` instead of sending users
  directly to the raw Sphinx-Gallery index.
- Use Sphinx-Gallery sections only for runnable code pages.
- Use RST pages for conceptual explanations and non-code guidance.
- Remove `examples/dev_scripts` from public gallery configuration.
- Use explicit ordering rather than filename sorting alone.
- Add gallery thumbnails that show real outputs: signal snippet, event windows,
  feature heatmap, metric table, learning curve.
- Keep execution time visible. First-path tutorials should run quickly in CI or
  have small stable cached data.

## New User Questions To Answer

### Installation And Setup

- Which Python versions are supported?
- Should I use pip, uv, conda, or source install?
- What optional extras do I need for tutorials, features, docs, and tests?
- Can I run tutorials on CPU?
- Which tutorials require a GPU?
- What do I do if MNE, PyTorch, or braindecode installation fails?

### Dataset Discovery

- What datasets exist for my task?
- How do I find datasets by modality, task, subject count, clinical group, or
  sampling rate?
- How do I inspect dataset-level metadata before downloading files?
- How do I inspect record-level metadata?
- What is the difference between dataset documents and record documents?
- Which fields can I query with keyword arguments?
- When should I use MongoDB-style queries?
- How do I avoid accidentally querying too much?

### Loading And Caching

- What is the difference between `EEGDash`, `EEGDashDataset`, and
  `EEGChallengeDataset`?
- When is data downloaded?
- Where is data cached?
- How do I change the cache directory?
- How do I download all files ahead of time?
- How do I use EEGDash offline?
- How do I know whether my cache is complete?
- How do I load only one subject/task/session/run?
- How do I handle records that fail to load?
- What does `on_error="warn"` or `on_error="skip"` do?
- When should I call `drop_bad` or `drop_short`?

### EEG Data Inspection

- How do I print channel names, channel types, sample rate, and duration?
- How do I plot a short signal segment?
- How do I inspect annotations/events?
- How do I see BIDS entities?
- How do I detect missing or inconsistent metadata?
- How do I verify that labels are present and sane?

### Preprocessing

- Should I use MNE directly or braindecode preprocessors?
- How do I select channels?
- How do I resample?
- How do I filter?
- How do I set references?
- How do I handle artifacts?
- How do I ensure preprocessing is applied consistently across train/test?
- Which preprocessing steps can leak information if fit on all data?
- What should be saved after preprocessing?

### Windowing And Labels

- How do I create fixed-length windows?
- How do I create event-based windows?
- How do I map event annotations to labels?
- How do I attach metadata labels such as age, sex, or p-factor?
- What is the shape of one sample?
- What is the difference between recording-level labels and window-level labels?
- Why are windows from the same subject not independent?

### Evaluation

- What split should I use for my scientific question?
- How do I split by subject?
- How do I split by session?
- How do I do cross-dataset evaluation?
- How do I stratify without leakage?
- What metrics should I report for classification and regression?
- What is a meaningful baseline?
- How do I compute confidence intervals or chance-level checks?
- How do I compare pipelines fairly?

### Modeling

- How do I train a scikit-learn baseline?
- How do I train a braindecode model?
- How do I use a plain PyTorch loop?
- Which models are good first baselines?
- When should I use raw signals, spectral features, CSP, or deep learning?
- How do I normalize data?
- How do I avoid normalizing using validation/test information?
- How do I save and load models?

### Feature Extraction

- What features are included in `eegdash.features`?
- How do I list available features?
- How do I extract signal, spectral, complexity, connectivity, and CSP features?
- How do feature preprocessors work?
- How do I avoid recomputing Welch spectra repeatedly?
- How do I write a custom feature?
- How do I save and reload feature datasets?

### Scaling And Reproducibility

- How do I run this on a cluster?
- How do I choose `n_jobs`, batch sizes, and cache location?
- How do I make a reproducible experiment directory?
- How do I pin dataset versions?
- How do I log query, preprocessing, split, model, and metrics?
- How do I cite EEGDash and the source datasets?

### Interoperability

- How does EEGDash relate to MNE?
- How does EEGDash relate to braindecode?
- How does EEGDash relate to MOABB?
- Can I convert EEGDash datasets into MOABB-style evaluations?
- Can I use local BIDS datasets?
- Can I export features or windows for other tools?

## API Simplicity Themes Surfaced By The Tutorial Plan

The tutorials should use a small stable public surface. If a tutorial requires
manual mutation of internals, that is a signal to improve the API or create a
documented helper.

### Public Objects To Teach

- `EEGDash`: metadata discovery and database interaction.
- `EEGDashDataset`: ML-ready dataset loading from query, records, remote cache,
  or local BIDS layout.
- `EEGChallengeDataset`: competition/foundation challenge datasets.
- `eegdash.features`: feature extraction, feature banks, serialization.
- `eegdash.paths.get_default_cache_dir`: cache conventions.
- `eegdash.hbn.preprocessing` and `eegdash.hbn.windows`: HBN-specific helpers
  when the tutorial uses HBN data.

### Possible API Improvements To Consider

These are not required before restructuring, but they would make tutorials
simpler and safer:

- `EEGDashDataset.filter(...)`: return a filtered dataset without manual
  `BaseConcatDataset` reconstruction.
- `EEGDashDataset.split(by="subject", stratify=..., test_size=...)`: a
  leakage-safe helper backed by `eegdash.splits` adapters around MOABB and
  scikit-learn splitters.
- `EEGDashDataset.summary()`: print records, subjects, tasks, sessions, channels,
  sampling rates, and estimated duration.
- `EEGDashDataset.preview(index=0)`: load one recording and return a compact
  MNE/raw summary.
- `EEGDashDataset.estimate_download_size()`: help users plan storage.
- `EEGDashDataset.ensure_downloaded(n_jobs=-1)`: public alias around
  `download_all` if the intended public name is not already fixed.
- `EEGDashDataset.validate_local_cache()`: check offline completeness.
- `EEGDash.find(..., fields=[...])`: project only selected metadata fields if the
  backend supports it.
- `EEGDash.search_datasets(...)`: user-friendly dataset search by modality, task,
  keywords, and clinical tags.
- `make_split_manifest(splitter, y, metadata, ids=...)`: standalone helper if
  adding a method to `EEGDashDataset` is not desired.
- `task.make_windows(engine="braindecode", ...)`: a task-level convenience method
  that delegates to Braindecode windowing and records the exact function/kwargs
  used.

Tutorials should not wait for all of these. They should, however, clearly mark
any workaround as temporary and link to the preferred API once available.

## Detailed Tutorial Proposals

### `tutorial_00_first_search.py`

Goal: find records and datasets without downloading data.

Audience: new user who wants to know what EEGDash contains.

Steps:

1. Import `EEGDash`.
2. Create client.
3. Query a known small dataset with `limit`.
4. Print record count and keys.
5. Query by task/subject.
6. Compute simple metadata statistics: subjects, tasks, sampling rates, total
   duration.
7. Link to dataset catalog and metadata schema reference.

Include:

- Expected output.
- Warning that this transfers metadata only.
- A small explanation of dataset vs. record.

Move from:

- `examples/tutorials/tutorial_api.py`.

### `tutorial_01_first_recording.py`

Goal: load one recording and inspect it.

Steps:

1. Instantiate `EEGDashDataset(cache_dir=..., dataset=..., subject=..., task=...)`.
2. Print `len(dataset)` and `dataset.description`.
3. Access `dataset[0]`.
4. Load `.raw`.
5. Print `raw.info["sfreq"]`, channel count, duration, annotations.
6. Plot a short segment or simple PSD.

Include:

- Explanation of lazy loading.
- Cache path display.
- Sanity checks.

### `tutorial_02_dataset_to_dataloader.py`

Goal: create a minimal PyTorch-ready dataset without teaching a full model yet.

Steps:

1. Load a small dataset.
2. Apply one or two safe preprocessors.
3. Create fixed-length windows.
4. Inspect one `(X, y)` item.
5. Create `DataLoader`.
6. Print batch shape.

Include:

- No model training.
- Clear sample shape expectations.

### `tutorial_10_preprocess_and_window.py`

Goal: teach preprocessing and windowing as a reusable unit.

Steps:

1. Select a dataset with stable data.
2. Pick channels.
3. Resample.
4. Filter.
5. Create fixed-length or event-based windows.
6. Verify duration/window count.
7. Save windows to disk.

Include:

- Why preprocessing is separate from modeling.
- Link to deeper preprocessing explanation.

### `tutorial_11_leakage_safe_split.py`

Goal: teach split discipline.

Steps:

1. Load or synthesize a windows dataset from previous tutorial.
2. Show subject IDs in metadata.
3. Create a wrong window random split only as a warning, not as the final method.
4. Create a MOABB-backed cross-subject split manifest.
5. Assert disjoint subjects from the manifest and metadata.
6. Optionally stratify by label.
7. Print split summary.

Include:

- Short statement: windows from the same subject are correlated.
- Link to explanation page.

### `tutorial_12_train_a_baseline.py`

Goal: train a simple model after a leakage-safe split.

Steps:

1. Reuse windows and split from tutorial 11.
2. Compute majority/median baseline.
3. Train a small model:
   - for classification: logistic regression, shallow CNN, or ShallowFBCSPNet.
   - for regression: ridge/LightGBM or compact CNN.
4. Evaluate on held-out subjects.
5. Print metric table.

Include:

- Baseline before neural net.
- Minimal training loop.
- Reproducible seed.

### `tutorial_20_visual_p300_oddball.py`

Goal: decode event-related target vs. standard responses.

Steps:

1. Query P3 dataset.
2. Inspect annotations.
3. Map target/standard events.
4. Create event windows.
5. Check class balance.
6. Split safely.
7. Train baseline.
8. Plot average ERP or sample trials.

Include:

- Screenshot/thumbnail should show event-related waveform or class counts.

### `tutorial_30_eyes_open_closed.py`

Goal: teach resting-state state classification.

Steps:

1. Load HBN EO/EC data.
2. Reannotate eyes-open/eyes-closed intervals.
3. Create balanced windows.
4. Inspect one signal.
5. Split and train a simple classifier.

Move from:

- `examples/core/tutorial_eoec.py`
- `examples/tutorials/noplot_tutorial_features_eoec.py`

### `tutorial_40_first_features.py`

Goal: introduce `eegdash.features`.

Steps:

1. Load windows from EO/EC tutorial.
2. Extract `signal_variance`, RMS, spectral power bands.
3. Display feature table head.
4. Join labels/metadata.
5. Save feature dataset.

Include:

- Feature names should include channel names.
- Link to feature bank reference.

### `tutorial_41_feature_trees.py`

Goal: teach feature extractor trees and predecessor/preprocessor reuse.

Steps:

1. Show repeated spectral computation problem.
2. Add spectral preprocessor.
3. Extract multiple downstream spectral features.
4. Compare feature names and shape.

Move from:

- `examples/tutorials/noplot_tutorial_features_eoec.py`.

### `tutorial_42_features_to_sklearn.py`

Goal: train a classical ML model from EEGDash features.

Steps:

1. Load saved features.
2. Split by subject.
3. Normalize using train only.
4. Train logistic regression/random forest/LightGBM.
5. Report metrics and feature importance.

### `how_to_work_offline.py`

Goal: serve users on clusters or air-gapped machines.

Steps:

1. Populate cache online.
2. Instantiate with `download=False`.
3. Filter by subject/task offline.
4. Compare online/offline descriptions or shapes.
5. Troubleshoot missing files.

Move from:

- `examples/eeg2025/tutorial_eegdash_offline.py`.

### `how_to_handle_bad_records.py`

Goal: teach robust dataset construction when some files fail.

Steps:

1. Instantiate with `on_error="warn"` or `on_error="skip"`.
2. Iterate/load raw data.
3. Call `drop_bad`.
4. Call `drop_short(min_samples=...)`.
5. Report removed records.

This is important for large public archives where occasional files are broken,
private, or malformed.

## Applied Examples To Keep But Reframe

Applied examples should begin with a clear "This is not the first tutorial"
notice and link back to prerequisites.

### Age Regression

Keep as project material. It teaches a realistic continuous target. It must
emphasize:

- age label availability and cleaning;
- subject-level split;
- median/mean baseline;
- regression metrics;
- small-data limitations.

### p-factor Regression

Keep one feature-based and one deep-learning version only if they answer
different questions. Otherwise merge into one project with two pipelines.

Required framing:

- what p-factor is and where labels come from;
- privacy/clinical interpretation cautions;
- baseline metrics;
- cross-subject split;
- reporting uncertainty.

### Sex Classification

Keep only with careful language. It is easy for users to overinterpret this
task or reproduce confounded results.

Required framing:

- classification labels are metadata, not biological ground truth guarantees;
- channel/site/dataset confounds can dominate;
- use this as a workflow example, not a scientific claim.

### Clinical Dataset Summary

Move out of the model-training tutorial path. This is a catalog exploration
how-to or applied visualization example.

### Simulated Transfer Learning

Retire unless rewritten with real EEGDash data. It does not teach the library
well enough to occupy a public tutorial slot.

## Documentation Pages To Add

### Concepts

1. `concepts/eegdash_objects.rst`
   - `EEGDash` vs. `EEGDashDataset` vs. `EEGChallengeDataset`.
   - Records vs. datasets.
   - Remote vs. local mode.
2. `concepts/metadata_and_bids.rst`
   - BIDS entities.
   - `dataset`, `subject`, `session`, `task`, `run`.
   - Participants metadata and `description_fields`.
3. `concepts/lazy_loading_and_cache.rst`
   - When files download.
   - Cache layout.
   - Offline assumptions.
4. `concepts/leakage_and_evaluation.rst`
   - Window leakage.
   - Subject/session/dataset splits.
   - Metrics and baselines.
5. `concepts/preprocessing_decisions.rst`
   - Filtering, resampling, referencing, channels, artifacts.
6. `concepts/features_vs_deep_learning.rst`
   - When to choose feature baselines, CSP, or neural networks.
7. `concepts/interoperability.rst`
   - MNE, BIDS, braindecode, MOABB, OpenNeuro, NEMAR.

### How-To Guides

1. `how_to/find_datasets.rst`
2. `how_to/query_records.rst`
3. `how_to/download_all.rst`
4. `how_to/work_offline.rst`
5. `how_to/load_local_bids.rst`
6. `how_to/filter_by_metadata.rst`
7. `how_to/create_fixed_windows.rst`
8. `how_to/create_event_windows.rst`
9. `how_to/use_moabb_splitters.rst`
10. `how_to/extract_features.rst`
11. `how_to/train_braindecode_model.rst`
12. `how_to/run_on_slurm.rst`
13. `how_to/troubleshoot_loading_errors.rst`

### Reference

1. `reference/query_fields.rst`
   - Current allowed keyword fields include `data_name`, `dataset`, `subject`,
     `task`, `session`, `run`, `modality`, `sampling_frequency`, `nchans`,
     and `ntimes`.
2. `reference/metadata_schema.rst`
   - Dataset and record field definitions.
3. `reference/cache_layout.rst`
4. `reference/feature_bank.rst`
5. `reference/errors.rst`
6. `reference/glossary.rst`

## Quality Bar For New Tutorials

Each tutorial should be reviewed against this checklist:

- The first meaningful result appears within the first few cells.
- The tutorial has one clear learning goal.
- The code can run from a clean environment.
- Data size and runtime are stated.
- The tutorial does not rely on private local files.
- Network dependence is stated and minimized.
- Output checks are included.
- Splits avoid subject/session leakage unless explicitly teaching the pitfall.
- The tutorial reports a baseline.
- The tutorial has a stable thumbnail.
- The tutorial links to concepts and API reference instead of embedding long
  explanations.
- The tutorial is shorter than roughly 150-220 lines unless it is an advanced
  applied project.
- CI either runs it or deliberately verifies it through a smoke-test path.
- There is no hidden TODO.

## Migration Plan

### Phase 1: Audit And Triage

1. Mark every current gallery file as one of:
   - true tutorial;
   - how-to guide;
   - applied example;
   - challenge material;
   - developer-only;
   - retire.
2. Remove dev scripts from public gallery.
3. Add a curated tutorials landing page.
4. Add a "Learning path" card section to the homepage and user guide.
5. Fix naming:
   - no `noplot_tutorial_*` as public names;
   - consistent title casing;
   - consistent `braindecode` spelling.

### Phase 2: Build The First Learning Path

1. Rewrite `tutorial_api.py` as `tutorial_00_first_search.py`.
2. Write `tutorial_01_first_recording.py`.
3. Rewrite cache/offline basics from EEG2025 material into a general how-to.
4. Rewrite EO/EC as a concise decoding tutorial.
5. Write `tutorial_11_leakage_safe_split.py`.
6. Split feature extraction into first features and feature trees.
7. Add concept pages for objects, cache, and leakage.

### Phase 3: Reclassify Existing Long Examples

1. Move age, p-factor, sex classification to `examples/applied`.
2. Merge or clearly differentiate duplicate p-factor examples.
3. Keep one canonical P3 oddball tutorial; make auditory oddball a related
   applied example or comparison.
4. Move HPC tutorial to `examples/hpc` and link from how-to guide.
5. Retire or rewrite simulated transfer learning.

### Phase 4: Add Benchmarking And Community Alignment

1. Add MOABB-inspired evaluation tutorials:
   - within-subject;
   - cross-subject;
   - cross-session;
   - learning curves;
   - pipeline comparison.
2. Add Braindecode-inspired model tutorials:
   - simple training on windows;
   - cropped/event decoding if applicable;
   - data augmentation only as advanced material;
   - fine-tuning pretrained/foundation models when stable.
3. Add Neuromatch-inspired project pages:
   - project template;
   - research question prompts;
   - dataset selection guide;
   - reproducibility checklist.

### Phase 5: Maintenance And Governance

1. Add a tutorial review template to `CONTRIBUTING.md`.
2. Add a docs CI matrix:
   - fast smoke build;
   - selected tutorial execution;
   - full gallery optional/nightly.
3. Track tutorial runtime and data size.
4. Keep public tutorial dependencies within `eegdash[docs]`.
5. Require each new tutorial to declare:
   - learning goal;
   - audience;
   - runtime;
   - data;
   - expected outputs;
   - whether GPU/network is required.

## Suggested Homepage And Landing Page Copy

### Homepage Tutorial Entry

Title: `Learn EEGDash by building a decoding pipeline`

Body:

> Start with metadata search, load one recording, create windows, split by
> subject, and train a baseline. Then move to event-related decoding, feature
> extraction, and large-scale evaluation.

Buttons:

- `Start the first tutorial`
- `Browse how-to guides`
- `Read EEG decoding concepts`

## Promotion And Product Strategy

The public story should be stronger than "a Python library for 700+ datasets."
That framing is true, but it sounds like a data access layer. The more
compelling positioning is:

> EEGDash turns hundreds of messy public EEG/BIDS datasets into ML-ready
> PyTorch and braindecode workflows, with reproducible splits, benchmarks, and
> feature extraction built in.

In other words: sell the outcome, not the plumbing.

### Make The First Demo Irresistible

The first demo should be short enough to fit in a README, talk slide, tweet, or
conference poster:

```python
from eegdash import EEGDashDataset

ds = EEGDashDataset(
    dataset="ds005514",
    task="RestingState",
    subject=["NDARDB033FW5"],
)

print(ds.summary())
```

Then the aspirational next step:

```python
from eegdash.splits import (
    apply_split_manifest,
    get_splitter,
    make_split_manifest,
    to_moabb_split_inputs,
)

windows = ds.to_windows(window_size="2s", stride="2s")
y, metadata = to_moabb_split_inputs(windows, target="target")
manifest = make_split_manifest(get_splitter("cross_subject"), y, metadata)
train, test = apply_split_manifest(windows, manifest, fold=0)
```

The exact API does not exist yet. That is a useful design signal: the most
promotable version of EEGDash needs a small high-level task/workflow layer on
top of the current flexible API.

### Promote Tasks, Not Records

Users do not primarily want "records." They want task-ready workflows:

- P300 decoding.
- eyes-open vs. eyes-closed.
- age regression.
- p-factor prediction.
- sleep staging.
- motor imagery.
- cross-subject generalization.
- benchmark my model on public EEG.

The public API and documentation should therefore expose curated task entry
points:

```python
from eegdash.tasks import P300

task = P300(n_subjects=20)
manifest = task.make_split_manifest(strategy="cross_subject")
train, test = task.apply_split_manifest(manifest, fold=0)
```

This would make EEGDash feel closer to `torchvision.datasets`, Hugging Face
Datasets, or MOABB benchmarks than to a database wrapper.

### Create Benchmark Cards And OpenEEG-Bench Export

The highest promotional leverage is a benchmark surface, but EEGDash should not
compete with OpenEEG-Bench. EEGDash should publish task cards, split manifests,
and reproducible result artifacts that can feed OpenEEG-Bench and be rendered in
the docs.

Initial benchmark cards:

| Task | Dataset(s) | Split | Metric | Baselines |
| --- | --- | --- | --- | --- |
| EO/EC | HBN | cross-subject | balanced accuracy | CSP, EEGNet, ShallowFBCSP |
| P300 | visual/auditory oddball | cross-subject | AUROC | xDAWN, EEGNet |
| Age | HBN | subject split | MAE | ridge, LightGBM, Conformer |
| p-factor | EEG2025 | official split | RMSE | challenge baselines |

The first surface should be static and reproducible: task manifests, split
manifests, JSON result files, and OpenEEG-Bench export metadata. Community model
comparison should point to OpenEEG-Bench rather than a separate EEGDash
leaderboard.

### Make The Dataset Catalog Feel Like An EEG Atlas

The current catalog is useful. It should become more visually and practically
memorable:

- task cards for P300, resting-state, sleep, clinical, motor imagery, and
  multimodal datasets;
- "copy Python query" buttons on every filtered view;
- dataset pages with signal preview, montage/channel summary, event summary, and
  sample metadata;
- "ML readiness" badges for labels, events, participants metadata, MNE load
  status, benchmark eligibility, and license clarity;
- one-click links from dataset pages to tutorials and benchmark tasks.

The ideal flow is:

> Search in the UI -> filter datasets -> click "Use in Python" -> paste working
> code.

### Add A Visual Identity And Figure System

MOABB's strongest visual lesson is that benchmark documentation should look like
scientific evidence, not like marketing decoration. The plots and cards are
compact, source-aware, and tied to evaluation questions:

- score plots show one score per subject/dataset/pipeline and can overlay chance
  levels and adjusted significance thresholds;
- distribution plots combine density and individual samples, making variability
  visible instead of only reporting a mean;
- paired plots compare two algorithms directly with a diagonal reference and
  chance-level context;
- significance and meta-analysis plots make statistical comparison a first-class
  object;
- dataset bubble plots encode dataset shape at a glance: subjects, sessions,
  trials or duration, and paradigm;
- splitter diagrams make within-session, cross-session, and cross-subject
  evaluation strategies visually concrete;
- dataset pages use snapshot cards, visual summary blocks, HED/event summaries,
  channel summaries, benchmark highlights, citation/impact cards, and responsive
  layouts.

EEGDash should add this as a distinct documentation dimension: every tutorial
and task page should include at least one small visual artifact that answers a
user question.

Recommended EEGDash visual artifacts:

- dataset atlas card: subjects, sessions, runs, tasks, modalities, duration,
  channels, license, and ML-readiness badges;
- task card: target, metric, split policy, baseline score, required labels,
  preprocessing assumptions, and citations;
- split audit plot: train/test subjects, sessions, recordings, class balance,
  site balance, and leakage warnings;
- score plot: per-subject or per-session points with chance level, baseline
  line, and confidence intervals when appropriate;
- paired baseline plot: compare a user's model against CSP, ridge/logistic,
  EEGNet, or another Braindecode recipe under the same split;
- learning-curve plot: score versus subjects, windows, trials, or minutes of
  EEG;
- dataset overlap plot: show whether a pretraining corpus overlaps with an
  evaluation task by subject, dataset, session, or record ID;
- channel coverage plot: show common channels, missing channels, and montage
  mismatch for multi-dataset tasks;
- export card: indicate whether the result has a task manifest, split manifest,
  preprocessing record, model SHA, license, and OpenEEG-Bench export metadata.

Suggested visual style direction for EEGDash:

- evidence-first: white or very light surfaces, restrained grid lines, strong
  axis labels, visible units, and source/provenance text on every exported plot;
- colorblind-aware palette with separate roles for dataset/task categories,
  warnings, baselines, and selected model results;
- avoid purely decorative graphics; every chart should encode task, dataset,
  split, score, uncertainty, or provenance;
- use the same visual grammar in docs, README figures, social preview cards, and
  generated benchmark artifacts;
- keep plots exportable as SVG/PDF for papers and PNG for README/Hugging Face
  cards;
- prefer small multiples and compact cards over large hero-style illustrations
  for technical pages.

Initial plotting helpers to consider:

```text
eegdash.viz.dataset_card(...)
eegdash.viz.task_card(...)
eegdash.viz.split_audit_plot(...)
eegdash.viz.score_plot(...)
eegdash.viz.learning_curve_plot(...)
eegdash.viz.pretrain_eval_overlap_plot(...)
eegdash.viz.channel_coverage_plot(...)
```

These helpers should not duplicate MOABB plotting. For MOABB-compatible
benchmark results, prefer MOABB plots or a thin adapter. EEGDash-specific plots
should focus on dataset discovery, task readiness, split provenance, leakage,
and foundation-model contamination checks.

### Own "Leakage-Safe EEG ML"

Many EEG ML examples in the wild are scientifically weak because they split
windows randomly or report optimistic results. EEGDash can gain trust by making
safe evaluation visible and easy.

Promote:

- subject-safe splits;
- session-safe splits;
- task transfer splits;
- dataset transfer splits;
- automatic leakage checks;
- required baseline reporting.

Example aspirational API:

```python
from eegdash.tasks import load_task
from eegdash.splits import assert_no_leakage

task = load_task("eoec-hbn-mini")
manifest = task.make_split_manifest(strategy="cross_subject")
metadata = task.get_split_metadata()
assert_no_leakage(manifest, metadata, by="subject")
```

This is both a scientific feature and a marketing feature.

### Ship Polished Interactive Learning Assets

The tutorial restructure should produce public artifacts that are easy to share:

- "Train your first EEG decoder in 10 minutes."
- "Find EEG datasets for your research question."
- "Build a leakage-safe cross-subject benchmark."
- "Extract spectral features from 100 subjects."
- "Run EEGDash on an HPC cluster."

Each should have Colab/Binder links, a small stable dataset path, and a visible
result within the first few cells.

### Integrate With Hugging Face

Hugging Face is a useful discovery surface for modern ML users. EEGDash should
use it for:

- dataset cards for curated EEGDash task datasets;
- model cards for pretrained baselines;
- benchmark result metadata;
- demo Spaces for dataset search or model inference;
- links between models, datasets, licenses, papers, and evaluation results.

The goal is not to move the whole project to Hugging Face. The goal is to meet
ML users where they already search.

### Ship Baseline Recipes

People adopt libraries faster when they start from a working baseline. Add a
small set of baseline recipes that call scikit-learn and Braindecode rather than
reimplementing models:

```python
from eegdash.tasks import load_task

task = load_task("p300-visual-mini")
manifest = task.make_split_manifest(strategy="cross_subject")
train, test = task.apply_split_manifest(manifest, fold=0)
result = task.run_recipe("braindecode-eegnet", train=train, test=test)
```

The recipes can be simple. Their job is to make EEGDash feel complete, produce
reproducible artifacts, and anchor benchmark results while leaving model
implementation and training mechanics to Braindecode.

### Publish A Strong Software Paper

A future software paper should frame EEGDash as infrastructure for reproducible
EEG machine learning, not merely as a downloader:

> EEGDash: a BIDS-first data and benchmarking layer for large-scale EEG machine
> learning.

The paper should emphasize:

- state of the field and gap against MNE, braindecode, MOABB, OpenNeuro, and
  NEMAR;
- software design choices;
- dataset scale and metadata coverage;
- leakage-safe benchmark workflows;
- tutorials and reproducible materials;
- research impact and community readiness.

### README And Homepage Message

The README and homepage should lead with the promise:

```markdown
# EEGDash

Train EEG models on hundreds of public BIDS datasets with one Python API.

- Search 700+ EEG/MEG/iEEG/fNIRS datasets
- Load directly into PyTorch and braindecode
- Build leakage-safe subject/session splits
- Run standard EEG decoding benchmarks
- Extract spectral, connectivity, and signal features
```

Then show a 10-line demo, screenshots, benchmark badges, and tutorial links.

### What Not To Promote Too Hard

Do not lead with clinical prediction claims. Age, sex, and p-factor examples are
useful, but they can look scientifically overclaimed if not framed carefully.
The safer and stronger message is reproducible EEG ML infrastructure.

### Tutorials Landing Page Sections

1. `Start here`
   - First search
   - First recording
   - Dataset to DataLoader
2. `Core workflow`
   - Preprocess and window
   - Split safely
   - Train a baseline
3. `Decode EEG tasks`
   - Eyes open/closed
   - P300 oddball
   - Auditory oddball
4. `Feature engineering`
   - First features
   - Feature trees
   - Features to scikit-learn
5. `Evaluate and benchmark`
   - Within subject
   - Cross subject
   - Cross session
   - Learning curves
6. `Applied projects`
   - Age regression
   - p-factor regression
   - clinical catalog summary
7. `Scale up`
   - Offline mode
   - HPC
   - parallel features

## Hugging Face Hub Snapshot And Implementation Opportunities

Snapshot date: 2026-05-04.

This snapshot uses the public Hugging Face Hub API and Hub pages for repository
recency. The later paper-scan section uses the Hugging Face MCP server directly
over its streamable HTTP endpoint.

- recent EEG models:
  https://huggingface.co/api/models?search=eeg&sort=lastModified&direction=-1&limit=20
- recent EEG datasets:
  https://huggingface.co/api/datasets?search=eeg&sort=lastModified&direction=-1&limit=20
- recent EEG Spaces:
  https://huggingface.co/api/spaces?search=eeg&sort=lastModified&direction=-1&limit=20
- Braindecode organization page:
  https://huggingface.co/braindecode
- EEGDash catalog Space:
  https://huggingface.co/spaces/EEGDash/catalog

The Hub is currently a strong signal for where the EEG ML community is moving.
The important pattern is not a single repository. The important pattern is that
EEG on the Hub is becoming a discovery, model sharing, and demo surface.
EEGDash should therefore stop presenting itself only as a data access package
and become the infrastructure that makes Hub-visible EEG ML reproducible.

### What Is Recent On The Hub

Recent model activity includes:

- EEG-specific fine-tuning and application repos such as
  `Fetrat/adhd-eeg-gemma3-qlora`, updated 2026-05-01.
- New or updated representation learning repos such as `fbdeme/eeg-wm-jepa`,
  updated 2026-04-30, and `delta-tj/EEGFoundation`, updated 2026-04-28.
- Braindecode architecture/model cards updated around 2026-04-25, including
  `EEGTCNet`, `EEGSym`, `EEGSimpleConv`, `EEGPT`, `EEGNetv4`, `EEGNeX`,
  `EEGNet`, `USleep`, `TSception`, `TIDNet`, `SignalJEPA_*`, `ShallowFBCSPNet`,
  `SSTDPN`, `SPARCNet`, `REVE`, and others.
- Braindecode pages that frame models as reusable Hub artifacts, including
  model cards and a model explorer Space.

Recent dataset activity includes:

- `NeuroBench/thor_eeg_mi`, updated 2026-04-28.
- `hezy18/EEG-SVRec`, updated 2026-04-27.
- `opsecsystems/EEGWORLD`, updated 2026-04-27, with thousands of monthly
  downloads in the Hub API snapshot.
- Multimodal and novel interface datasets, including an EEG plus fNIRS
  handwriting trajectory dataset and an EEG-to-text sentence dataset.
- Multiple `EEGDash/*` pointer repositories, updated 2026-04-21, tagged with
  `eegdash`, `eeg`, `neuroscience`, `brain-computer-interface`, `pytorch`, and
  `mlcroissant`.

Recent Space activity includes:

- Several EEG demo Spaces for seizure detection, sleep staging, user
  identification, visual decoding, and cognitive analysis.
- `EEGDash/catalog`, updated 2026-04-22, already positioned as an EEG, MEG,
  neuroscience, BCI, Braindecode, PyTorch, and datasets browser.
- Braindecode's OpenEEG-Bench and model explorer pages, which show that public
  interactive benchmark and model exploration surfaces are becoming expected.

### Critical Read Of The Opportunity

The Hub is full of promising EEG assets, but most are isolated:

- A dataset card rarely tells users which canonical task it supports.
- A model card rarely states the exact dataset split, preprocessing, leakage
  constraints, and metrics used.
- A Space often demonstrates a result but is not enough to reproduce the
  training/evaluation protocol.
- Foundation-model and self-supervised EEG repos need credible downstream
  evaluation surfaces.
- Dataset repos are discoverable, but users still need a task layer that turns
  "this dataset exists" into "run this benchmark correctly."

This is the attractive product role for EEGDash:

> EEGDash is the bridge between public EEG datasets, Hub model cards, and
> reproducible benchmark results.

In practical terms, EEGDash should make a Hub user think:

```text
I found an EEG model. EEGDash tells me where to evaluate it.
I found an EEG dataset. EEGDash tells me the tasks and safe splits.
I found a benchmark score. EEGDash tells me the exact code to reproduce it.
```

That is more compelling to the NeurIPS, ICLR, and ICML audience than "another
dataset downloader." It speaks to foundation models, benchmark quality,
reproducibility, data governance, and model comparison.

### What To Implement To Highlight This

#### 1. Hub-Ready Task Repositories

Create a small set of curated EEGDash task repos on Hugging Face. These should
not be generic dataset mirrors. They should be task-first artifacts with
machine-readable manifests.

Initial target repos:

- `EEGDash/eoec-hbn-mini`
- `EEGDash/p300-visual-mini`
- `EEGDash/auditory-oddball-mini`
- `EEGDash/eeg2025-pfactor-mini`
- `EEGDash/eegdash-benchmark-splits`

Each task repo should include:

- `README.md` card with task definition, labels, intended use, limitations, and
  citation.
- `task.json` with dataset IDs, selection filters, label field, task type,
  preprocessing assumptions, metric names, and known caveats.
- `splits/*.json` with subject/session/group split IDs.
- `baseline_results.json` with model name, metric, confidence interval or seed
  distribution, code version, and data version.
- `load.py` or a Python snippet that calls the EEGDash public API.
- Dataset card metadata tags: `eegdash`, `eeg`, `braindecode`, `pytorch`,
  `benchmark`, `brain-computer-interface`, `neuroscience`, `mlcroissant`.

The important design choice is that the task card should be useful even before
the user installs EEGDash. It should explain what problem the repo solves and
show the exact install-and-run path.

#### 2. A Task API Inside EEGDash

The Hub task repos need a local API counterpart.

Proposed user-facing API:

```python
from eegdash.tasks import load_task

task = load_task("eoec-hbn-mini", cache_dir="./data")
ds = task.get_dataset()
manifest = task.make_split_manifest(strategy="cross_subject")
train, test = task.apply_split_manifest(ds, manifest, fold=0)

print(task)
print(task.metrics)
print(task.baselines)
```

Minimal implementation:

- Add `eegdash/tasks/`.
- Define `Task` dataclass with `name`, `description`, `dataset_query`,
  `label_field`, `target_transform`, `metrics`, `split_policy`,
  `preprocessing`, and `references`.
- Add a manifest loader that can read local manifests and, later, HF-hosted
  manifests.
- Add `load_task(name, source="local" | "hf")`.
- Add `Task.get_dataset()`, `Task.make_split_manifest()`,
  `Task.apply_split_manifest()`, `Task.describe()`,
  `Task.to_model_card_context()`.
- Start with one well-tested task: eyes open versus eyes closed.

This makes EEGDash feel like a modern ML library: users load tasks, not just
files.

#### 3. Split Manifests And Leakage Checks

For Hub credibility, EEGDash should publish and consume split manifests.

Implement:

- `eegdash.splits.to_moabb_split_inputs(ds, target=...)`
- `eegdash.splits.get_splitter("cross_subject", engine="moabb", **kwargs)`
- `eegdash.splits.make_split_manifest(splitter, y, metadata, sample_ids=...)`
- `eegdash.splits.load_split_manifest(path_or_repo_id)`
- `eegdash.splits.apply_split_manifest(ds, manifest, split="train")`
- `eegdash.splits.assert_no_leakage(manifest, metadata, by="subject")`
- `eegdash.splits.describe_split(ds, split_manifest)`

The manifest should store:

- dataset IDs;
- stable sample IDs;
- subject/session/run IDs;
- fold and split assignment;
- splitter class path, `cv_class`, and splitter kwargs;
- random seed;
- group key used for leakage prevention;
- stratification field when used;
- exclusions and quality filters;
- EEGDash version, MOABB version when used, metadata hash, and generation date.

This can become one of the strongest selling points of the project: EEGDash
does not merely give access to public data; it gives users reproducible and
auditable evaluation protocols.

#### 4. Baseline Model Cards

Publish EEGDash baseline model cards that are intentionally simple and
reproducible.

Initial baseline repos:

- `EEGDash/logistic-spectral-eoec-baseline`
- `EEGDash/shallowfbcsp-eoec-baseline`
- `EEGDash/eegnet-p300-baseline`
- `EEGDash/lightgbm-pfactor-features-baseline`

Each model card should include:

- exact task repo and split manifest;
- preprocessing pipeline;
- model class and hyperparameters;
- training command;
- evaluation command;
- metric table across seeds;
- limitations and leakage checks;
- links back to docs tutorials.

The goal is not to claim state of the art. The goal is to define credible,
copyable starting points that a model paper can beat.

#### 5. Benchmark Result Format

Add a small benchmark result schema to the repo.

Candidate path:

```text
benchmarks/
  schemas/
    result.schema.json
  results/
    eoec-hbn-mini/
      logistic-spectral.json
      shallowfbcsp.json
```

Each result should contain:

- `task_id`
- `dataset_ids`
- `split_id`
- `model_id`
- `model_repo`
- `library_versions`
- `metric_values`
- `seed_values`
- `preprocessing`
- `hardware`
- `runtime`
- `command`
- `commit`
- `notes`

This creates a stable path from code to documentation to Hub model cards to
leaderboard.

#### 6. Upgrade The EEGDash Catalog Space

The current `EEGDash/catalog` Space should become the public product surface,
not only a dataset browser.

Add tabs:

- `Datasets`: existing searchable catalog.
- `Tasks`: curated tasks with number of subjects, recordings, label definition,
  and links to task repos.
- `Benchmarks`: model/task/split table with metrics and reproduce commands.
- `Models`: baseline model cards plus external compatible Braindecode models.
- `Getting started`: a short executable snippet for loading a task.

Useful interactions:

- Copy `pip install eegdash` plus task load snippet.
- Copy `eegdash-benchmark run ...` command.
- Filter tasks by paradigm, modality, population, license, size, and supported
  split policy.
- Show "safe split available" badges.
- Show "baseline available" badges.

This is one of the most direct ways to make the project feel alive and
shareable. The Space becomes a living demo for the library.

#### 7. Hugging Face Trend Monitor

Add a small script so EEGDash can track what is happening around EEG on the Hub
without manual browsing.

Candidate path:

```text
scripts/huggingface_trends.py
```

Outputs:

- `docs/source/_static/hf_trends/recent_eeg_models.json`
- `docs/source/_static/hf_trends/recent_eeg_datasets.json`
- `docs/source/_static/hf_trends/recent_eeg_spaces.json`
- optional markdown summary for maintainers.

The script should query:

- `models?search=eeg`
- `models?search=biosignal`
- `models?search=brain`
- `datasets?search=eeg`
- `spaces?search=eeg`
- `models?author=braindecode`
- `datasets?author=EEGDash`

For each result, store:

- repo ID;
- last modified date;
- tags;
- downloads or likes when available;
- license;
- pipeline or task tags;
- whether a README/model card exists;
- whether the card links to data, code, or a benchmark.

Use cases:

- identify external models worth testing on EEGDash tasks;
- find missing metadata in EEGDash cards;
- keep homepage examples fresh;
- generate a short "recent community activity" block for maintainers.

This should be a maintainer script first, not a user-facing dependency.

#### 8. Card Generators

Create templates that generate consistent Hugging Face cards from EEGDash
metadata.

Candidate paths:

```text
eegdash/hub/cards.py
docs/templates/huggingface/dataset_card.md.j2
docs/templates/huggingface/task_card.md.j2
docs/templates/huggingface/model_card.md.j2
```

Functions:

- `render_dataset_card(summary_row)`
- `render_task_card(task_manifest)`
- `render_model_card(result_json)`
- `validate_card_metadata(card_text)`

This reduces manual maintenance and makes the project look professionally
curated across many repos.

#### 9. Pull-From-Hub Tutorial

Add a tutorial that explicitly connects EEGDash and Hugging Face:

```python
from braindecode.datasets import BaseConcatDataset

ds = BaseConcatDataset.pull_from_hub("EEGDash/ds004194")
```

Then show the alternative EEGDash path:

```python
from eegdash import EEGDashDataset

ds = EEGDashDataset(dataset="ds004194", cache_dir="./cache")
```

Explain when to use each:

- use `pull_from_hub` for mirrored, ready-to-consume Braindecode layouts;
- use `EEGDashDataset` for canonical-source streaming, metadata queries, and
  broader dataset coverage;
- use `load_task()` for benchmark-ready supervised workflows.

This tutorial should be short and very practical. It belongs in the `How-to`
section, not in the conceptual tutorial track.

#### 10. Evaluate Hub Models On EEGDash Tasks

Add a tutorial and API helper for evaluating a Hub-hosted model on an EEGDash
task.

Proposed API:

```python
from eegdash.tasks import load_task
from eegdash.evaluation import evaluate_hf_model

task = load_task("eoec-hbn-mini")
result = evaluate_hf_model(
    "braindecode/EEGNet",
    task=task,
    split="cross_subject",
)
print(result.metrics)
```

The first implementation can support only a narrow class of Braindecode model
cards. Even a constrained workflow is useful if it is honest and reproducible.

This is the feature that makes EEGDash relevant to people publishing models:
they can say, "my model runs on an EEGDash benchmark."

### Recommended First Milestone

Do not start with every task and every model. Ship one complete vertical slice:

1. `eegdash.tasks.load_task("eoec-hbn-mini")`
2. subject-level split manifest and leakage assertion
3. one feature baseline and one neural baseline
4. benchmark result JSON schema
5. task card on Hugging Face
6. baseline model card on Hugging Face
7. `EEGDash/catalog` benchmark tab
8. docs tutorial: "Run your first EEGDash benchmark"
9. docs how-to: "Use EEGDash datasets from Hugging Face"

This slice is enough for demos, README screenshots, conference conversations,
and external model authors.

### Tutorial Changes Implied By The Hub Strategy

Add these tutorial/how-to pages to the documentation plan:

- `tutorials/first_benchmark.py`: load a task, inspect splits, train a baseline,
  and report metrics.
- `how_to/use_huggingface_datasets.py`: compare `BaseConcatDataset.pull_from_hub`
  and `EEGDashDataset`.
- `how_to/publish_a_task_card.py`: create a task manifest and generate a Hub
  card.
- `how_to/publish_a_baseline_model_card.py`: turn a result JSON into a model
  card.
- `how_to/evaluate_a_hub_model.py`: evaluate a compatible Braindecode or HF
  model on an EEGDash task.
- `explanation/benchmark_splits_and_leakage.md`: why split manifests matter for
  EEG.
- `reference/task_manifest_schema.md`: exact manifest fields.
- `reference/benchmark_result_schema.md`: exact result fields.

### What This Lets EEGDash Claim

With the above implemented, EEGDash can make stronger public claims:

- "Load public EEG datasets, benchmark tasks, and safe splits with one API."
- "Evaluate EEG models from Hugging Face on reproducible EEGDash tasks."
- "Publish EEG benchmark results with machine-readable task, split, and result
  cards."
- "Bridge OpenNeuro/NEMAR datasets, Braindecode models, and Hugging Face model
  sharing."

These claims are more attractive than generic data access because they describe
an ecosystem role.

## Hugging Face MCP Paper Scan: What EEG Decoding Research Needs

Snapshot date: 2026-05-04.

This section uses the Hugging Face MCP server described at
https://huggingface.co/docs/hub/agents-mcp. The server was reached directly from
the shell through the streamable HTTP MCP endpoint at `https://huggingface.co/mcp`.
The available tools included `paper_search`, `hub_repo_search`,
`hub_repo_details`, `space_search`, and Hugging Face documentation search.

The most useful MCP paper search queries were:

- `EEG foundation model decoding cross subject cross task benchmark`
- `EEG decoding foundation models arbitrary electrode montage benchmark`
- `EEG self supervised representation learning foundation model cross dataset
  transfer`
- `EEG to text visual decoding multimodal foundation model`
- `EEG benchmark foundation models clinical applications`
- `large brainwave foundation models spurious correlations EEG benchmark`

### High-Signal Papers And Themes

The paper scan points to a clear research direction: EEG foundation models are
now competing on generalization, not only on within-dataset accuracy. The papers
most relevant to EEGDash include:

- [EEG Foundation Models: Progresses, Benchmarking, and Open Problems](https://hf.co/papers/2601.17883):
  compares open-source EEG foundation models and specialist baselines, with
  emphasis on cross-subject generalization, few-shot calibration, linear probing,
  fine-tuning, model scale, and inconsistent evaluation protocols.
- [EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding](https://arxiv.org/abs/2506.19141):
  frames the exact problem EEGDash should support: zero-shot transfer to new
  tasks and subjects, plus psychopathology factor prediction on large,
  high-density EEG.
- [BIOT: Biosignal Transformer for Cross-data Learning in the Wild](https://papers.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html):
  shows why flexible tokenization across mismatched channels, variable sequence
  lengths, and missing values matters for biosignal foundation models.
- [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://arxiv.org/abs/2405.18765):
  shows the push toward large-scale EEG pretraining over many datasets and tasks,
  and highlights channel patches, neural tokenizers, and masked prediction.
- [EEGPT / BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training](https://arxiv.org/abs/2410.19779):
  highlights electrode-wise modeling, autoregressive prediction, scaling, and
  multi-task transfer as foundation-model directions.
- [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://hf.co/papers/2412.07236):
  emphasizes separate spatial and temporal attention and adaptation to diverse
  downstream BCI formats.
- [S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention](https://hf.co/papers/2403.11772):
  connects self-supervised JEPA-style training to downstream EEG paradigms such
  as motor imagery, ERP, and SSVEP.
- [REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects](https://hf.co/papers/2510.21585):
  pushes arbitrary setup handling through positional encoding and large-scale
  pretraining across many datasets and subjects.
- [NeurIPT: Foundation Model for Neural Interfaces](https://openreview.net/forum?id=D4hcJPkJ3y):
  focuses on inter-subject, inter-task, inter-condition variability and diverse
  electrode configurations.
- [EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications](https://arxiv.org/abs/2512.08959):
  shows demand for unified clinical EEG benchmarks and reports that simple
  models can remain competitive under distribution shift.
- [Assessing the Capabilities of Large Brainwave Foundation Models](https://iclr.cc/virtual/2025/35099):
  stresses flawed setups, spurious correlations, artifacts, and subject-
  independent validation.
- [Test-Time Adaptation for EEG Foundation Models](https://hf.co/papers/2604.16926):
  points to real-world distribution shift and adaptation as a likely next
  benchmark axis.
- [Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation](https://hf.co/papers/2505.17099),
  [EEG2TEXT](https://hf.co/papers/2405.02165), and
  [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://hf.co/papers/2403.07721):
  show the appeal of multimodal decoding, but these are high-risk areas for
  overclaiming unless evaluation and controls are careful.

### Critical Takeaways For EEGDash

The strongest opportunity is not to present EEGDash as "the library that trains
the next huge EEG foundation model." That would be expensive, fragile, and easy
to overclaim.

The stronger opportunity is:

> EEGDash is the reproducible data, split, task, and evaluation layer for EEG
> foundation models.

The papers repeatedly expose the same unresolved infrastructure problems:

- EEG datasets vary in channel count, channel names, montage, sampling rate,
  duration, task design, labels, and artifacts.
- Model papers often use incompatible preprocessing and evaluation protocols.
- It is difficult to know whether pretraining and downstream evaluation overlap.
- Cross-subject, cross-session, cross-task, and cross-dataset claims are hard to
  audit.
- Linear probing, fine-tuning, few-shot calibration, and test-time adaptation are
  often mixed together without a common result schema.
- Simple baselines sometimes remain competitive, especially under clinical
  distribution shift.
- EEG-to-text, EEG-to-image, and thought-to-text language can attract attention,
  but also creates reputational risk if EEGDash appears to endorse speculative
  claims.

EEGDash can become more attractive to NeurIPS, ICLR, and ICML researchers by
making these problems easy to test.

### What To Implement From The Paper Scan: Reuse First

After checking Braindecode and OpenEEG-Bench more carefully, the implementation
plan should be stricter: EEGDash should not recreate model loading, model
registries, preprocessing engines, windowing engines, training loops, Hub dataset
serialization, or a separate foundation-model leaderboard. Those already exist,
or are actively being built, in Braindecode and OpenEEG-Bench.

The complementary role is:

> EEGDash should define datasets, tasks, provenance, leakage-safe splits, and
> export bridges. Braindecode should execute preprocessing/windowing/modeling.
> OpenEEG-Bench should display and compare benchmark results.

#### Existing Braindecode Capabilities To Reuse

Use these directly rather than wrapping them with parallel implementations:

- `BaseConcatDataset`, `RawDataset`, `WindowsDataset`, and `EEGWindowsDataset`
  as the data containers.
- `create_fixed_length_windows`, `create_windows_from_events`, and
  `create_windows_from_target_channels` for window creation.
- `Preprocessor`, `preprocess`, `EEGPrep`, and Braindecode/MNE preprocessing
  helpers for filtering, resampling, channel picking, interpolation, rereference,
  montage setting, and artifact-related operations.
- `EEGClassifier`, `EEGRegressor`, cropped losses, scoring callbacks, samplers,
  and augmentation utilities for training/evaluation mechanics.
- `EEGModuleMixin` for `from_pretrained`, `push_to_hub`, `get_config`,
  `from_config`, `reset_head`, `get_output_shape`, and model statistics.
- Braindecode's model zoo and model table, including EEGNet, EEGNeX,
  EEGConformer, ShallowFBCSPNet, BENDR, BIOT, CBraMod, EEGPT, LaBraM, LUNA,
  REVE, Signal-JEPA, and related variants.
- Braindecode Hub dataset support: `push_to_hub` and
  `BaseConcatDataset.pull_from_hub`, including Zarr-backed storage and generated
  dataset cards.
- Braindecode's Pydantic/Exca experiment configuration examples when we need
  typed, cached experiment configs.

This means EEGDash tutorials should show Braindecode code explicitly instead of
hiding it. A successful user should understand that EEGDash provides the
large-scale dataset/task layer and Braindecode provides the modeling layer.

#### Existing OpenEEG-Bench Capabilities To Reuse

OpenEEG-Bench already exists as `braindecode/OpenEEGBench` on Hugging Face. Its
current Space metadata identifies it as an open, reproducible EEG model
comparison arena. The Space repo already contains:

- a Docker-based leaderboard application;
- backend endpoints for benchmarks, leaderboard, models, and votes;
- a benchmark registry with BCIC-2a, PhysioNet MI, ISRUC-SLEEP, TUAB, TUEV,
  CHB-MIT, FACED, and SEED-V;
- an `eval.yaml` generation script for Hugging Face decentralized evaluation;
- sample result fields such as model name, adapter, precision, model SHA,
  architecture, weight type, per-dataset accuracies, average score, parameter
  count, trainable parameter count, Hub availability, submission date, base
  model, Hub license, and Hub likes.

EEGDash should feed this ecosystem, not create a competing "EEGDash Foundation
Model Arena." The EEGDash catalog can show task/data provenance and link out to
OpenEEG-Bench for model comparison.

#### Existing MOABB Split And Evaluation Capabilities To Reuse

MOABB is strongest where EEGDash should be careful: benchmark split semantics.
The current MOABB splitter API already separates "make train/test indices" from
"run a full MOABB evaluation." That is the right boundary for EEGDash. We can
reuse the splitter classes directly for EEGDash and Braindecode datasets by
providing aligned labels and metadata.

Reusable splitter classes:

- `WithinSessionSplitter`: stratified k-fold splits inside each
  subject-session block. This is useful for quick sanity checks and tutorials,
  but should be clearly labeled as an easier, higher-leakage-risk evaluation
  than cross-subject generalization.
- `WithinSubjectSplitter`: stratified k-fold splits within each subject across
  all available sessions. This is useful for personalization and per-subject
  adaptation tutorials.
- `CrossSessionSplitter`: leave-one-session-out style evaluation within each
  subject by default, implemented with `LeaveOneGroupOut` over sessions. This is
  useful when users ask whether a model survives recording-day or session shift.
- `CrossSubjectSplitter`: leave-one-subject-out style evaluation by default,
  implemented with `LeaveOneGroupOut` over subjects. This should be the default
  recommendation for most EEGDash decoding tutorials.
- `CrossDatasetSplitter`: leave-one-group-out evaluation over a configurable
  metadata column, defaulting to `dataset`. This is directly relevant to
  foundation-model and domain-generalization claims.
- `LearningCurveSplitter`: repeated subsampled training sets with a fixed test
  split. This is useful for "how much EEG data do I need?" tutorials and for
  making small-data behavior visible.

The important implementation idea is not to wrap these with new algorithms. Add
an EEGDash translation layer:

```python
from eegdash.tasks import get_task
from moabb.evaluations import CrossSubjectSplitter

task = get_task("visual-p300")
windows = task.make_windows(engine="braindecode")
y, metadata = task.to_moabb_split_inputs(windows, target="is_target")

splitter = CrossSubjectSplitter()
manifest = task.make_split_manifest(splitter, y=y, metadata=metadata)
train_set, test_set = task.apply_split_manifest(windows, manifest, fold=0)
```

Recommended `eegdash.splits` responsibilities:

- normalize EEGDash, Braindecode, and feature-dataset metadata into a
  splitter-ready table with one row per raw recording, epoch, window, or feature
  sample;
- guarantee stable sample identifiers such as `sample_id`, `window_id`,
  `record_id`, `subject`, `session`, `run`, `dataset`, `task`, `site`,
  `target`, and optional clinical or demographic columns;
- expose `to_moabb_split_inputs(dataset, target)` returning `(y, metadata)` for
  MOABB splitters;
- expose `get_splitter(name, engine="moabb", **kwargs)` so tutorials can say
  `cross_subject`, `cross_session`, `within_subject`, `within_session`,
  `cross_dataset`, or `learning_curve` without requiring users to memorize class
  names immediately;
- serialize splitter output into a split manifest containing train/test indices,
  stable sample IDs, fold IDs, splitter class path, `cv_class`, `cv_kwargs`,
  random seed, target definition, EEGDash version, MOABB version, metadata hash,
  generation timestamp, and exclusion reasons;
- apply a manifest back to `EEGDashDataset`, Braindecode `BaseConcatDataset`,
  Braindecode windows datasets, and `FeaturesConcatDataset`;
- run leakage assertions over configurable columns, for example subject,
  session, family, recording, acquisition site, and source dataset;
- summarize folds with sample counts, subject counts, class balance, session
  balance, dataset balance, and warnings when a task is too small for the
  requested split.

Full MOABB evaluation classes should be reused only when the EEGDash task fits
MOABB's native abstraction: a MOABB `BaseDataset`, a MOABB paradigm, and
scikit-learn-compatible pipelines. This is a good path for MI, P300, SSVEP, CVEP,
and fixed-window BCI examples. It should not become a requirement for every
EEGDash task, because EEGDash will also cover broader datasets, foundation-model
pretraining splits, clinical/behavioral prediction, and Hugging Face/Braindecode
workflows that do not naturally map to a MOABB paradigm.

Possible future bridge:

```python
moabb_dataset = task.to_moabb_dataset()
paradigm = task.to_moabb_paradigm()
```

That bridge should be optional and narrow. The first implementation should focus
on splitter reuse and manifests, because that gives the EEGDash community the
most value without reimplementing MOABB.

Tutorials to add around this:

- "Choose the right EEG split": explain within-session, within-subject,
  cross-session, cross-subject, and cross-dataset splits with concrete failure
  modes.
- "Use MOABB splitters with EEGDash tasks": create windows with Braindecode,
  convert to `(y, metadata)`, run a MOABB splitter, and train a small model.
- "Audit a split for leakage": intentionally create a bad split, detect overlap
  by subject/session/recording, then fix it.
- "Learning curves for EEG": use `LearningCurveSplitter` to show how accuracy
  changes as the number of trials or subjects increases.
- "When to use MOABB directly": show the path from an EEGDash-compatible BCI
  task to MOABB evaluation and benchmark output, while keeping the default
  EEGDash path simple.

#### Existing MOABB Visual And Plotting Capabilities To Reuse

MOABB also provides a useful visual model for EEGDash. Its plotting API treats
visualization as part of the evaluation contract rather than as optional polish.
Useful patterns:

- `score_plot`: strip plots by dataset and pipeline with chance-level context.
- `paired_plot`: direct two-algorithm comparisons with a diagonal reference.
- `summary_plot`: significance matrices for comparing pipelines.
- `meta_analysis_plot`: effect-size summaries with confidence intervals across
  datasets.
- `dataset_bubble_plot`: dataset-scale summaries where subjects, sessions,
  trials/duration, and paradigm are encoded visually.
- `plot_datasets_grid` and `plot_datasets_cluster`: atlas-style views over many
  datasets.

Design cues worth adapting:

- publication-style Matplotlib/Seaborn defaults;
- colorblind-aware categorical colors;
- a restrained palette built around navy, teal, sky blue, purple, amber, and
  coral, with muted grid lines and dark blue-gray text;
- muted grids and minimal spines;
- clear title/subtitle/source attribution;
- chance-level and significance annotations built into plots;
- compact dataset cards and visual summary blocks on API pages;
- social preview cards and benchmark/citation calls to action.

The local MOABB plotting style currently uses colors such as navy `#2F3E5C`,
teal `#1B9E77`, sky `#56B4E9`, purple `#7E63B8`, amber `#E69F00`, coral
`#D55E5E`, dark text `#1b3a57`, and muted grid lines `#758D99`. EEGDash can use
this as a reference for contrast and accessibility, but should define its own
palette so the projects remain visually distinct.

EEGDash should reuse MOABB plots when the data is a MOABB evaluation result. For
EEGDash-specific plots, the differentiating visual layer should cover:

- split manifests and leakage checks;
- task manifests and ML-readiness;
- pretraining/evaluation contamination;
- multi-dataset channel and metadata coverage;
- Hugging Face/OpenEEG-Bench export completeness.

#### Reimplementation Guardrails

Do not implement:

- a new EEG model zoo;
- new foundation-model `from_pretrained` wrappers;
- a custom training framework when Braindecode/skorch already handles it;
- a custom windowing engine;
- a custom Hub dataset storage format;
- a custom EEG cross-validation splitter suite that duplicates MOABB splitters;
- duplicate MOABB benchmark plots for MOABB-compatible result frames;
- a separate leaderboard that duplicates OpenEEG-Bench;
- a second experiment-configuration system if Braindecode's Pydantic/Exca path
  is sufficient;
- a generic preprocessing pipeline that shadows MNE/Braindecode preprocessing.

Only add thin EEGDash helpers when they encode EEGDash-specific knowledge:
dataset discovery, task definitions, metadata normalization, provenance,
licenses, splitter adapters, split manifests, benchmark export, and
pretraining/evaluation overlap checks.

#### 1. Task And Split Manifest Layer

This is EEGDash's most important implementation target. It should answer:

```text
Which records form this task?
What is the target label?
Which subject/session/run IDs belong to train, validation, and test?
Can this be converted into a Braindecode dataset or OpenEEG-Bench eval config?
```

Candidate API:

```python
from eegdash.tasks import load_task

task = load_task("eoec-hbn-mini")
raw_ds = task.get_dataset(cache_dir="./data")          # EEGDashDataset/BaseConcatDataset
windows = task.make_windows(engine="braindecode")      # calls braindecode windowing
manifest = task.make_split_manifest(windows, strategy="cross_subject")
```

Task manifests should include:

- dataset query and record filters;
- target field and label mapping;
- modality and paradigm;
- preprocessing assumptions, expressed as Braindecode/MNE calls or config;
- windowing recipe, expressed as arguments to Braindecode functions;
- split definitions by subject/session/run;
- metrics;
- license and citation requirements;
- known caveats and exclusions;
- export metadata for Hugging Face cards and OpenEEG-Bench.

#### 2. Pretraining Corpus Manifest

Foundation-model papers need scale, but scale without provenance is dangerous.
Implement an exportable manifest for pretraining corpora. This should be an
EEGDash metadata/provenance artifact, not a model-training system.

Candidate API:

```python
from eegdash.foundation import build_pretraining_manifest

manifest = build_pretraining_manifest(
    modality="eeg",
    min_duration_minutes=5,
    require_license=True,
    exclude_datasets=["ds004194"],
)
manifest.to_json("pretrain_manifest.json")
```

Manifest fields:

- dataset ID and canonical source;
- subject/session/run identifiers;
- task/paradigm tags;
- channel names and count;
- montage or electrode coordinates availability;
- sampling frequency;
- duration;
- license;
- known clinical/sensitive labels;
- quality/exclusion flags;
- whether labels are included or stripped;
- hash or stable record identifier;
- train/evaluation exclusion group.

Add a corresponding contamination check:

```python
from eegdash.foundation import assert_no_pretrain_eval_overlap

test_manifest = task.make_split_manifest(strategy="cross_subject").select("test")
assert_no_pretrain_eval_overlap(pretrain_manifest, test_manifest)
```

This is one of the most ML-conference-relevant features EEGDash can provide.

#### 3. Thin Model Requirement Metadata

Do not build a model registry that loads models. Braindecode already handles
model classes and Hub weights. EEGDash can maintain or derive a small metadata
overlay that describes what a task must provide before Braindecode can run a
model.

Candidate API:

```python
from eegdash.foundation import get_model_requirements

req = get_model_requirements("braindecode/signal-jepa")
print(req.required_sfreq)
print(req.channel_policy)
print(req.window_seconds)
```

Where possible, derive this from:

- Braindecode model config via `from_pretrained`/`get_config`;
- Hugging Face model card metadata;
- Braindecode documentation/model table;
- explicit local overrides only when metadata is missing.

This layer should return metadata and warnings. It should not instantiate custom
model wrappers.

#### 4. Preparation Report That Delegates To Braindecode

Foundation-model papers repeatedly care about channel layouts and missing
electrodes. EEGDash should expose a preparation report, but the actual work
should call MNE/Braindecode primitives.

Candidate API:

```python
from eegdash.foundation import prepare_for_braindecode_model

prepared, report = prepare_for_braindecode_model(
    task,
    model="braindecode/signal-jepa",
    split="train",
    engine="braindecode",
)
```

The report should say:

- which Braindecode/MNE operations were called;
- channel names before and after normalization;
- channels picked, dropped, interpolated, or missing;
- montage/coordinate availability;
- original and target sampling frequency;
- windowing function and exact arguments;
- number of recordings/windows kept or excluded;
- split and leakage-check status.

This is useful because it makes preprocessing auditable. It should not hide the
Braindecode calls.

#### 5. OpenEEG-Bench Export Schema

Instead of creating a separate benchmark schema, define an EEGDash result schema
that can export to OpenEEG-Bench-compatible fields and preserve EEGDash-specific
provenance.

Additional fields:

- `encoder_id`
- `encoder_source`
- `encoder_checkpoint`
- `pretraining_manifest_id`
- `protocol`: `linear_probe`, `fine_tune`, `few_shot`, `zero_shot`,
  `test_time_adaptation`
- `frozen_layers`
- `calibration_trials`
- `channel_policy`
- `resampling_policy`
- `window_policy`
- `data_overlap_check`
- `artifact_control_check`
- `subject_leakage_check`
- `dataset_leakage_check`
- `baseline_family`: specialist CNN, shallow model, feature baseline,
  foundation model

Every result should be able to produce:

- a table row for docs;
- a Hugging Face model card block;
- a JSON record that can be consumed by OpenEEG-Bench;
- an `eval.yaml` block when the task is ready for Hugging Face evaluation;
- a reproducibility command.

#### 6. EEGDash Catalog Plus OpenEEG-Bench, Not A Duplicate Arena

The public surface should be split cleanly:

- EEGDash catalog: dataset/task discovery, metadata, task manifests, split
  manifests, licensing, citations, and "load this task" snippets.
- OpenEEG-Bench: model comparison, leaderboard tables, submissions, benchmark
  results, votes, and Hub evaluation integration.

The EEGDash catalog can include a `Benchmarks` tab, but it should link to or
embed OpenEEG-Bench records rather than maintaining a competing table.

Useful EEGDash badges:

- `braindecode-ready`
- `openeeg-bench-ready`
- `subject-safe-split`
- `pretrain-overlap-checkable`
- `hf-card-generated`
- `license-clear`

#### 7. Tutorials From The Paper Scan

Add a foundation-model track to the tutorial plan.

Tutorials:

- `tutorials/foundation/plot_first_foundation_model_eval.py`
  - load one EEGDash task;
  - convert it to Braindecode windows using Braindecode functions;
  - load one compatible Braindecode/HF encoder with `from_pretrained`;
  - run a Braindecode/skorch evaluation recipe;
  - report subject-safe metrics.
- `tutorials/foundation/plot_signal_jepa_linear_probe.py`
  - use `braindecode/signal-jepa` or `signal-jepa_without-chans`;
  - explain channel matching and window requirements;
  - compare pretrained versus random initialization if cheap enough.
- `tutorials/foundation/plot_cross_subject_vs_within_subject.py`
  - demonstrate how inflated scores appear when subject leakage is allowed;
  - show the leakage check.
- `tutorials/foundation/plot_montage_adaptation.py`
  - show channel subset matching, missing channels, coordinate-based reporting,
    and why model input preparation matters.
- `tutorials/foundation/plot_pretraining_manifest.py`
  - build a pretraining manifest from EEGDash metadata;
  - exclude downstream benchmark datasets;
  - export a manifest for a paper.

How-to guides:

- `how_to/evaluate_hf_foundation_model_with_braindecode.md`
- `how_to/export_result_to_openeeg_bench.md`
- `how_to/create_foundation_model_result_card.md`
- `how_to/check_pretraining_overlap.md`
- `how_to/export_pretraining_corpus_manifest.md`
- `how_to/add_model_requirement_metadata.md`

Explanation pages:

- `explanation/eeg_foundation_models.md`
- `explanation/linear_probe_vs_finetune.md`
- `explanation/cross_subject_cross_task_cross_dataset.md`
- `explanation/montage_and_channel_mismatch.md`
- `explanation/spurious_correlations_in_eeg.md`

Reference pages:

- `reference/foundation_model_spec.md`
- `reference/pretraining_manifest_schema.md`
- `reference/foundation_benchmark_result_schema.md`
- `reference/evaluation_protocols.md`

#### 8. CLI Commands For Paper Authors

Add CLI commands that make EEGDash easy to cite in papers.

Candidate commands:

```bash
eegdash task list
eegdash task describe eoec-hbn-mini
eegdash task materialize eoec-hbn-mini --cache ./data
eegdash split check --task eoec-hbn-mini --split cross_subject
eegdash foundation requirements braindecode/signal-jepa
eegdash foundation prepare-report --task eoec-hbn-mini --model braindecode/signal-jepa
eegdash manifest pretrain --query 'modality=eeg' --exclude-task eoec-hbn-mini
eegdash manifest check-overlap pretrain_manifest.json result.json
eegdash hub render-card result.json
eegdash openeegbench export-task eoec-hbn-mini
eegdash openeegbench export-result result.json
```

These commands directly answer what ML paper authors need: inspect data,
materialize splits, generate reproducibility artifacts, and publish evidence.
The actual model training/evaluation should remain Braindecode or
OpenEEG-Bench code.

#### 9. What Not To Build First

Do not start by training a giant EEG foundation model inside EEGDash. That would
turn EEGDash into a model project and distract from its strongest role.

Do not lead with thought-to-text or clinical diagnosis claims. These topics are
visible and exciting, but they require careful controls, clinical framing, and
ethical review. EEGDash can support reproducible evaluation for those papers
without marketing itself around speculative claims.

Do not publish a leaderboard without split manifests, preprocessing records, and
overlap checks. A leaderboard without auditability would weaken the project.

Do not duplicate OpenEEG-Bench. When the goal is model comparison, contribute
task metadata, result files, and UI links to OpenEEG-Bench.

### Recommended Foundation-Model Milestone

The first milestone should be a complete, modest vertical slice:

1. Audit OpenEEG-Bench's current result schema and `eval.yaml` expectations.
2. Add one EEGDash task manifest, preferably EO/EC or a small ERP/P300 task.
3. Add leakage-safe subject split manifest and overlap-check metadata.
4. Add `task.make_windows(engine="braindecode")` that calls Braindecode
   windowing and returns a Braindecode dataset.
5. Add `get_model_requirements("braindecode/signal-jepa")` as metadata only.
6. Add `prepare_for_braindecode_model(..., return_report=True)` as an auditable
   delegation layer over MNE/Braindecode.
7. Run one Braindecode tutorial-style baseline or foundation-model evaluation.
8. Export result JSON and `eval.yaml` compatible with OpenEEG-Bench.
9. Add one docs tutorial: "Evaluate an EEGDash task with Braindecode and export
   to OpenEEG-Bench."
10. Add an EEGDash catalog link/badge for the OpenEEG-Bench result.

This is feasible, demonstrable, and aligned with the papers while keeping
ownership boundaries clean.

## Implementation Backlog

This section translates the product strategy into concrete engineering work.

### Current Useful Building Blocks

These already exist and should be reused rather than rebuilt:

- `EEGDash.find()` and `EEGDash.find_datasets()` for metadata discovery.
- `EEGDashDataset` for query-based, record-based, remote, and offline loading.
- `EEGChallengeDataset` for EEG2025 competition releases.
- `EEGDashDataset.download_all()` for prefetching data.
- `EEGDashDataset.drop_bad()` and `EEGDashDataset.drop_short()` for robust
  handling of failed or too-short recordings.
- `description_fields` for carrying labels and metadata into dataset
  descriptions.
- `eegdash.features.extract_features()` and feature bank functions.
- `FeaturesConcatDataset.split()` and feature table utilities.
- The generated dataset catalog and dataset summary tables.

The missing layer is not raw capability. It is ergonomic, opinionated workflow
support.

### Workstream 1: A More Attractive Public API

Implement small helpers that make the README/demo code obvious.

| Item | Scope | Why it matters |
| --- | --- | --- |
| `EEGDashDataset.summary()` | Return/print subjects, tasks, sessions, runs, modalities, channel counts, sampling rates, durations, cache path, and estimated size when available. | Makes first-contact output rewarding and debuggable. |
| `EEGDashDataset.preview(index=0)` | Load one recording and return a compact object or dict with `raw`, metadata, signal snippet, annotations, and plot helper. | Gives users a safe way to inspect before building a pipeline. |
| `EEGDashDataset.filter(**kwargs)` | In-memory filtering over existing records/datasets without reconstructing `BaseConcatDataset` manually. | Removes common tutorial boilerplate and unsafe internal mutation. |
| `EEGDashDataset.split(by="subject", stratify=None, test_size=..., random_state=...)` | Convenience wrapper over `eegdash.splits`, MOABB splitters, and scikit-learn splitters. | Makes leakage-safe evaluation a default behavior without duplicating MOABB logic. |
| `assert_no_leakage(manifest, metadata, by="subject")` | Utility in `eegdash.splits`. | Turns scientific best practice into executable checks. |
| `EEGDashDataset.ensure_downloaded()` | Public alias/wrapper around `download_all()` with progress and return summary. | More discoverable name for tutorials and docs. |
| `EEGDashDataset.estimate_download_size()` | Sum known record sizes if available, fall back to dataset summary. | Helps users avoid surprise storage/network costs. |
| `EEGDash.search_datasets(...)` | Friendly search over dataset summary fields: modality, task, clinical tags, source, subjects, license. | Turns database knowledge into user-facing discovery. |

Implementation order:

1. `summary()`
2. `filter()`
3. `split()`/`make_split_manifest()` and `assert_no_leakage()`
4. `preview()`
5. `search_datasets()`
6. `estimate_download_size()`
7. `ensure_downloaded()`

### Workstream 2: Task API

Add `eegdash.tasks` as the high-level entry point for ML users.

Suggested structure:

```text
eegdash/tasks/
  __init__.py
  base.py
  eoec.py
  p300.py
  age.py
  pfactor.py
  registry.py
  manifests/
    eoec_hbn.yaml
    p300_visual.yaml
    p300_auditory.yaml
    eeg2025_pfactor.yaml
```

Core concepts:

- `EEGTask`: base class with metadata query, label definition, preprocessing
  recipe, windowing recipe, split definitions, metrics, and baseline metadata.
- `TaskManifest`: YAML/JSON file describing datasets, filters, labels, splits,
  metrics, citations, and licensing notes.
- `get_task(name)`: registry lookup.

Example target API:

```python
from eegdash.tasks import get_task

task = get_task("eyes-open-closed")
windows = task.get_windows(cache_dir="./data", n_subjects=20)
manifest = task.make_split_manifest(windows, strategy="cross_subject")
train, test = task.apply_split_manifest(windows, manifest, fold=0)
```

Initial tasks:

1. `eyes-open-closed`
2. `visual-p300`
3. `auditory-oddball`
4. `age-regression`
5. `eeg2025-pfactor`

Do not overgeneralize early. Hard-code a few excellent task manifests first,
then extract abstractions after two or three tasks are stable.

### Workstream 3: Leakage-Safe Evaluation

Create `eegdash.splits` as an adapter and manifest layer around MOABB splitters
and scikit-learn splitters. Do not implement a parallel EEG splitter suite.

Needed functions/classes:

- `to_split_metadata(dataset, target=None)`: normalize EEGDash, Braindecode, and
  feature datasets into one row per sample/window with stable identifiers.
- `to_moabb_split_inputs(dataset, target=None)`: return `(y, metadata)` aligned
  to the sample order expected by MOABB splitter classes.
- `get_splitter(name, engine="moabb", **kwargs)`: map friendly names such as
  `cross_subject`, `cross_session`, `within_subject`, `within_session`,
  `cross_dataset`, and `learning_curve` to MOABB splitter instances.
- `make_split_manifest(splitter, y, metadata, sample_ids=None)`: run the
  splitter and serialize folds, IDs, splitter config, library versions, random
  seed, target definition, and metadata hash.
- `apply_split_manifest(dataset, manifest, fold=0, split="train")`: select the
  corresponding subset from raw datasets, windows datasets, or feature datasets.
- `assert_no_leakage(manifest_or_splits, metadata, by)`: assert no overlap by
  subject, session, recording, family, site, or dataset.
- `describe_split(manifest, metadata, target=None)`: report fold counts, subject
  counts, class balance, site/session/dataset coverage, and warnings.
- `majority_baseline(y_train, y_test)`, `median_baseline(y_train, y_test)`, and
  metric helpers for classification/regression.

Design requirements:

- Splitting should work for `BaseConcatDataset`, `EEGDashDataset`, Braindecode
  windows datasets, and `FeaturesConcatDataset` where possible.
- It should use metadata/description fields and stable IDs, not array indices
  alone.
- It should preserve enough provenance for a paper or benchmark submission:
  dataset version, task manifest version, splitter class, splitter kwargs,
  random seed, excluded records, and generated split hashes.
- It should produce a small report suitable for tutorials and benchmark pages.
- Tutorials should teach MOABB split names and behavior explicitly, while
  EEGDash code handles the metadata conversion and manifest persistence.

### Workstream 4: Windowing Convenience Layer

The current tutorials use Braindecode windowing directly. Keep that ownership.
EEGDash should not introduce a second windowing engine. The useful layer is a
task-level convenience method that stores EEGDash-specific defaults and delegates
to Braindecode.

Possible API:

```python
task = get_task("eyes-open-closed")
windows, report = task.make_windows(
    engine="braindecode",
    kind="fixed",
    window_size="2s",
    stride="2s",
    return_report=True,
)
```

and:

```python
windows, report = task.make_windows(
    engine="braindecode",
    kind="events",
    event_id={"standard": 0, "target": 1},
    tmin=-0.2,
    tmax=0.8,
    return_report=True,
)
```

Implementation principles:

- Call `braindecode.preprocessing.create_fixed_length_windows` or
  `create_windows_from_events` internally.
- Preserve metadata and target fields.
- Print/return a window summary.
- Do not hide advanced braindecode options; expose `**kwargs`.
- Store the exact Braindecode function name, kwargs, and package version in the
  report.

This should live as task methods first. Only extract `eegdash.windows` if two or
three task implementations prove the same helper is needed.

### Workstream 5: Baseline Recipes And Pipelines

Add baseline recipes, not a parallel model package. Classical feature baselines
can use scikit-learn. Neural baselines should instantiate Braindecode models and
train with Braindecode/skorch utilities.

Initial baselines:

- `MajorityBaseline`
- `MedianBaseline`
- `RidgeRegressionBaseline`
- `LogisticRegressionBaseline`
- `LightGBMBaseline` if dependency handling is acceptable
- `braindecode-shallowfbcspnet`
- `braindecode-eegnet`

The baseline layer should prioritize reproducible pipelines over novelty.

Suggested API:

```python
from eegdash.tasks import load_task

task = load_task("visual-p300")
result = task.run_recipe("braindecode-eegnet", split="cross_subject")
```

Result schema:

- task name;
- dataset IDs;
- split strategy;
- model name;
- preprocessing/windowing summary;
- metric values;
- random seed;
- package versions;
- runtime;
- hardware if available.

### Workstream 6: Benchmark Export And OpenEEG-Bench Bridge

Do not build a competing leaderboard. Add a reproducible benchmark export layer
that can render static docs locally and feed OpenEEG-Bench when a task/result is
ready.

Suggested structure:

```text
eegdash/benchmarks/
  __init__.py
  export.py
  results.py
  schemas.py
  tasks.py
benchmarks/
  configs/
    eoec_baselines.yaml
    p300_baselines.yaml
  results/
    eoec/
      shallowfbcspnet_seed42.json
      logistic_features_seed42.json
docs/source/benchmarks/
  index.rst
  eoec.rst
  p300.rst
  age.rst
```

Minimum viable benchmark:

1. one task: EO/EC;
2. one split: cross-subject;
3. two recipes: feature logistic regression and Braindecode ShallowFBCSPNet;
4. one metric: balanced accuracy;
5. JSON result files rendered into docs;
6. OpenEEG-Bench-compatible export artifact when the result is public-ready.

Later:

- add P300;
- add p-factor;
- add learning curves;
- add OpenEEG-Bench submission/export template;
- add CI/nightly benchmark smoke runs.

### Workstream 7: Dataset Atlas Improvements

Use the current dataset summary machinery, but add user-facing ML affordances.

Implementation items:

- Add "Use in Python" snippets to dataset pages:

  ```python
  from eegdash import EEGDashDataset

  ds = EEGDashDataset(dataset="...", task="...", cache_dir="./data")
  ```

- Add task chips/badges on dataset pages.
- Add ML readiness badges:
  - loads with MNE;
  - has events;
  - has participant labels;
  - has channel metadata;
  - benchmark-ready;
  - license known.
- Add signal preview thumbnails where stable cached examples exist.
- Add event/annotation summaries.
- Add "Related tutorials" blocks on dataset pages.
- Add search filters for task, clinical category, source, modality, license,
  subject count, and sampling rate.

The first version can be static during docs build. It does not need a backend.

### Workstream 8: Hugging Face Distribution Layer

Create public Hugging Face assets for discoverability.

Initial artifacts:

- `eegdash/eoec-hbn-mini` dataset card.
- `eegdash/p300-visual-mini` dataset card.
- `eegdash/eegnet-p300-baseline` model card.
- `eegdash/eoec-baseline-results` static result card with OpenEEG-Bench export
  metadata.

Each card should include:

- task;
- source datasets;
- license and citation;
- preprocessing recipe;
- split definition;
- limitations;
- intended use;
- evaluation results if model card;
- links back to EEGDash docs.

This should be automated from task manifests and benchmark result JSON where
possible, so cards do not drift.

### Workstream 9: Website, README, And Launch Assets

Implementation items:

- Rewrite README first screen around the stronger value proposition.
- Add a 10-line quickstart that produces a visible result.
- Add screenshots or generated images:
  - dataset atlas;
  - signal preview;
  - benchmark table;
  - feature table.
- Add a short GIF or video of "search dataset -> copy code -> load data."
- Add GitHub topics:
  - `eeg`
  - `meg`
  - `bids`
  - `mne-python`
  - `braindecode`
  - `pytorch`
  - `machine-learning`
  - `neuroscience`
  - `brain-computer-interface`
  - `open-data`
  - `benchmark`
- Add social preview card focused on "EEG ML from 700+ BIDS datasets."
- Add an examples badge or benchmark badge if stable.

### Workstream 10: Visual Identity And Plot System

Create a small visual identity kit for docs, examples, README figures, task
cards, and generated benchmark artifacts.

Needed artifacts:

- `docs/design/data-viz-design.md` with plot principles, colors, typography,
  spacing, badges, and figure export rules;
- `eegdash.viz` helpers for EEGDash-specific visuals, especially split audits,
  task cards, dataset cards, learning curves, channel coverage, and
  pretraining/evaluation overlap;
- Sphinx gallery examples that generate canonical figures for the docs;
- SVG/PDF export path for papers and PNG/social-card export for README and
  Hugging Face;
- figure checklist: title, subtitle, axis units, chance/baseline line when
  relevant, sample count, split name, metric, source dataset, split manifest ID,
  and generation date.

Design requirements:

- Reuse MOABB plots directly for MOABB evaluation result frames.
- Keep EEGDash colors related to but distinct from MOABB so the projects feel
  interoperable, not confused.
- Make leakage, split strategy, task readiness, and provenance visible in the
  figure itself.
- Use colorblind-aware categorical colors and avoid relying on color alone for
  warnings.
- Prefer compact scientific plots and cards over decorative illustrations.
- Every task tutorial should have one visual summary before the training code
  and one evaluation visual after the result.

Initial figure set:

1. Dataset atlas card.
2. Task manifest card.
3. Split audit plot.
4. Leakage check report.
5. Baseline score plot with chance line.
6. Learning curve.
7. Channel coverage plot.
8. Pretraining/evaluation overlap plot.
9. OpenEEG-Bench export completeness card.

### Workstream 11: Software Paper Readiness

Prepare for a JOSS or similar software paper after the library has a stronger
workflow surface.

Needed artifacts:

- stable public release;
- meaningful changelog;
- clear install instructions;
- runnable tutorial path;
- tests and CI signals;
- contribution guide;
- governance/support expectations;
- software architecture explanation;
- benchmark or reproducible materials as impact evidence;
- citation guidance for EEGDash and source datasets.

Paper narrative:

- problem: public EEG data exists, but ML-ready, reproducible use remains hard;
- solution: BIDS-first discovery/loading, PyTorch/braindecode compatibility,
  feature extraction, task manifests, and leakage-safe benchmark workflows;
- impact: reusable benchmark tasks, tutorials, competition integration, and
  community dataset atlas.

### Priority Recommendation

Do not implement everything at once. The smallest high-impact sequence is:

1. `summary()`, `filter()`, and MOABB-backed split manifests.
2. A polished `tutorial_00_first_search.py` and `tutorial_01_first_recording.py`.
3. One task API prototype: EO/EC.
4. One static benchmark page: EO/EC with two baselines.
5. README/homepage refresh with the new story.
6. Dataset page "Use in Python" snippets and ML readiness badges.
7. Hugging Face cards for one task dataset and one baseline model.

This sequence creates a promotable product loop:

> discover dataset -> load it -> split safely -> train baseline -> compare on a
> benchmark -> share result.

## Immediate Action Items

1. Create `docs/source/tutorials/index.rst`.
2. Create `docs/source/how_to/index.rst`.
3. Create `docs/source/concepts/index.rst`.
4. Rename gallery groups and remove "Tutorials!".
5. Remove `examples/dev_scripts` from public gallery.
6. Rewrite `examples/tutorials/README.txt` into real section text.
7. Promote `tutorial_api.py` to the first tutorial after revision.
8. Add a new "load one recording" tutorial.
9. Rewrite `tutorial_minimal.py` or remove it from the beginner path because it
   currently demonstrates a leakage-prone split.
10. Add a leakage-safe splitting tutorial before any modeling tutorial.
11. Split long feature examples into smaller pages.
12. Move advanced clinical/regression examples out of the beginner gallery.

## Success Criteria

The restructure is successful when a new EEG decoding user can:

1. Install EEGDash and run a first metadata query.
2. Identify a dataset relevant to a task.
3. Load one recording and understand what was downloaded.
4. Create windows and inspect sample shapes.
5. Split data without subject/session leakage.
6. Train at least one baseline.
7. Extract features and train a classical model.
8. Understand where to look for recipes, concepts, and API details.
9. Reuse the same workflow offline or on a cluster.
10. Know which examples are educational tutorials and which are advanced
    research/project examples.

## Guiding Principle

EEGDash tutorials should not merely demonstrate that the library works. They
should teach the EEG decoding community how to use open electrophysiology data
carefully: discoverable metadata, reproducible loading, transparent
preprocessing, leakage-safe evaluation, simple baselines, and scalable workflows.
