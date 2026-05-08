How-To Guides
=============

Task-focused recipes for specific EEGDash workflows. Each guide assumes
you already know the basics and want a direct answer to a single
question -- "how do I work offline?", "how do I parallelise feature
extraction?", "how do I run preprocessing on SLURM?". Difficulty 1-2;
assumes the *Start Here* trio.

How-to guides sit in the recipe quadrant of the Diataxis framework: not
a curated learning path (those are the tutorials), not a deep
explanation (that is *Concepts*), not a complete API reference. They
are the answers to the operational questions that come up once you are
running EEGDash for real work. Sourced from
``docs/tutorial_restructure_plan.md`` Category I (lines 615-621 and
1021-1037). Cross-link with the HPC track when relevant.

What you will learn:

- How to download a dataset locally and pin it in the cache so reruns
  do not refetch.
- How to parallelise feature extraction across CPU cores using
  ``joblib`` and EEGDash's batch helpers.
- How to run preprocessing as a SLURM array job on a shared cluster
  (paired with the HPC tutorials).
- How to use the HPC cache layout so two jobs on the same cluster
  share preprocessed data.
- How to work fully offline: cache management, manifest export, and
  reloading without network access.

Each how-to is a single self-contained script or markdown file.
