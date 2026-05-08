# How-To Guides

Task-focused recipes for specific EEGDash workflows. Each guide assumes
you already know the basics and want a direct answer to a single
question – “how do I work offline?”, “how do I parallelise feature
extraction?”, “how do I run preprocessing on SLURM?”. Difficulty 1-2;
assumes the *Start Here* trio.

How-to guides sit in the recipe quadrant of the Diataxis framework: not
a curated learning path (those are the tutorials), not a deep
explanation (that is *Concepts*), not a complete API reference. They
are the answers to the operational questions that come up once you are
running EEGDash for real work. Sourced from
`docs/tutorial_restructure_plan.md` Category I (lines 615-621 and
1021-1037). Cross-link with the HPC track when relevant.

What you will learn:

- How to download a dataset locally and pin it in the cache so reruns
  do not refetch.
- How to parallelise feature extraction across CPU cores using
  `joblib` and EEGDash’s batch helpers.
- How to run preprocessing as a SLURM array job on a shared cluster
  (paired with the HPC tutorials).
- How to use the HPC cache layout so two jobs on the same cluster
  share preprocessed data.
- How to work fully offline: cache management, manifest export, and
  reloading without network access.

Each how-to is a single self-contained script or markdown file.

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Download all files for a dataset in advance, validate completeness, and inspect the cache.">  <div class="sphx-glr-thumbnail-title">Download an EEGDash dataset in advance and validate the local cache</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Goal: scale eegdash.features.extract_features across multiple cores on one node by tuning n_jobs and batch_size, then persist the result so re-runs are free.">  <div class="sphx-glr-thumbnail-title">Parallelise EEGDash feature extraction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="On HPC clusters, where you put the EEGDash cache is often the difference between a 30-minute and a 30-second epoch. Shared filesystems (Lustre, GPFS, NFS) survive job restarts but throttle under metadata-heavy access; node-local NVMe is fast but volatile; and $HOME is almost always too slow for training. This how-to shows how to point eegdash.paths.get_default_cache_dir at the right tier, stage data once, and verify the cache before training.">  <div class="sphx-glr-thumbnail-title">Place the EEGDash cache on shared or local cluster storage</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Goal: instantiate EEGChallengeDataset with download=False and load the same records as the online path, with no network calls.">  <div class="sphx-glr-thumbnail-title">How-to: work offline against a populated EEGDash cache</div>
</div>
<!-- thumbnail-parent-div-close --></div>
