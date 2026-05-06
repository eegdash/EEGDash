.. _concepts-eegdash-objects:

EEGDash objects: ``EEGDash``, ``EEGDashDataset``, ``EEGChallengeDataset``
========================================================================

The library exposes three top-level objects that look similar at first glance
but answer very different questions. Picking the wrong one is the most common
source of confusion for new users. This page explains what each one is,
what it gives you back, and why each exists.

In short:

- ``EEGDash`` is a **catalogue client**. It talks to the metadata service and
  returns *records* (dicts of metadata). Nothing is downloaded.
- ``EEGDashDataset`` is a **PyTorch-compatible dataset**. It turns a
  catalogue query into a list of recordings that can be loaded, preprocessed,
  windowed, and iterated over.
- ``EEGChallengeDataset`` is a **frozen, derivative dataset**, used for
  shared-benchmark contexts (currently the EEG 2025 Competition). It loads
  pre-resampled, pre-filtered, pre-cut data so that every participant is
  evaluated against an identical signal.

Records vs. datasets
--------------------

A *record* is a metadata document for one BIDS recording: which dataset it
belongs to, which subject, task, session, run, channel count, sampling
frequency, the path on S3, and so on. A record does **not** contain the
samples themselves. ``EEGDash.find()`` returns records.

A *dataset* is a Python object that lazily resolves records into actual EEG
recordings (typically wrapping ``mne.io.Raw`` via the ``EEGDashRaw`` adapter).
``EEGDashDataset`` returns datasets. The first time you access ``.raw`` on
one of its entries, the underlying file is downloaded into the local cache;
subsequent accesses are offline.

This split exists because metadata is small and cheap (you can search 700+
datasets in seconds), but raw EEG is large and slow (one HBN session is
hundreds of MB). You usually want to inspect metadata first, decide what to
keep, and only then trigger downloads.

Typical use of ``EEGDash``
--------------------------

Use ``EEGDash`` when you want to *browse* the catalogue without committing
to a download. The result is a list of dicts; you can filter it in pure
Python before doing anything heavyweight.

.. code-block:: python

   from eegdash import EEGDash

   client = EEGDash()

   # Discover datasets matching loose, human-friendly filters.
   datasets_df = client.search_datasets(modality="eeg", task="rest",
                                        n_subjects_min=20)
   print(datasets_df[["dataset", "n_subjects", "task"]].head())

   # Drill into one dataset and look at individual recordings.
   records = client.find({"dataset": "ds002718", "task": "FacePerception"})
   print(f"Found {len(records)} recordings.")
   print(records[0].keys())  # subject, session, run, sampling_frequency, ...

No EEG samples were downloaded by either of those calls. The catalogue is
the API surface; downloads are explicit and live one layer deeper.

Typical use of ``EEGDashDataset``
---------------------------------

Use ``EEGDashDataset`` when you want a real, indexable dataset that you can
hand to braindecode preprocessing or a PyTorch ``DataLoader``. It accepts
the same filter keywords as ``EEGDash.find`` plus a ``cache_dir`` and a
small set of dataset-construction options.

.. code-block:: python

   from eegdash import EEGDashDataset

   ds = EEGDashDataset(
       cache_dir="./eegdash_cache",
       dataset="ds002718",
       task="FacePerception",
       subject=["sub-002", "sub-003"],
       description_fields=["subject", "session", "task", "age"],
   )

   print(len(ds))                  # number of recordings
   print(ds.description.head())    # tidy metadata table
   raw = ds[0].raw                 # triggers the first download
   raw.filter(0.5, 40)             # mne.io.Raw operations

Lazy loading and caching
------------------------

The dataset is *lazy*. Construction merely resolves the metadata; the
``raw`` attribute on each entry is materialised on first access and then
held in memory for the lifetime of that object. The sample bytes are
written to ``cache_dir / <dataset_id>``, mirroring the BIDS layout, so a
second run with the same query is offline. If you set ``download=False``
and the cache already has the data, the catalogue is bypassed entirely
and ``EEGDashDataset`` reads the local BIDS tree directly. This makes it
straightforward to share a cache between machines or to work without
network access once the first run completes.

The lazy mode also lets you pass an ``on_error`` policy: in pipelines
that scan many recordings, ``on_error="skip"`` plus a follow-up
``ds.drop_bad()`` is a robust pattern when a few files in a release are
known to be corrupt.

When to use ``EEGChallengeDataset``
-----------------------------------

``EEGChallengeDataset`` is a thin wrapper over the same machinery, but
points at a frozen, preprocessed bucket: the data are downsampled to a
fixed rate, filtered with a fixed band, and cut into the canonical task
blocks used by the EEG 2025 Competition. If you are participating in the
competition, you **must** use ``EEGChallengeDataset``; otherwise your
local results are not comparable to the public leaderboard. If you are
not participating, ``EEGChallengeDataset`` is still useful when you want
a fully reproducible benchmark: every user sees identical bytes.

The library prints a notice when you try to load competition releases
through plain ``EEGDashDataset`` to nudge users away from this footgun.

Choosing among the three
------------------------

A practical decision tree:

- "I just want to know what is out there." → ``EEGDash``.
- "I want a PyTorch-style dataset for my own analysis." → ``EEGDashDataset``.
- "I am running EEG 2025 Competition code, or I want strictly identical
  preprocessing across users." → ``EEGChallengeDataset``.

Mixing modes inside a single experiment is fine: a common workflow uses
``EEGDash.search_datasets`` to find candidate datasets, then constructs
``EEGDashDataset`` instances for the few you actually want to model.

Related tutorials
-----------------

- :doc:`/generated/auto_examples/tutorials/00_start_here/plot_00_first_search`
  walks through metadata-only catalogue exploration with ``EEGDash``.
- :doc:`/generated/auto_examples/tutorials/00_start_here/plot_01_first_recording`
  contrasts the catalogue view with a first ``EEGDashDataset`` load.
- :doc:`/generated/auto_examples/tutorials/00_start_here/plot_02_dataset_to_dataloader`
  builds a PyTorch ``DataLoader`` on top of an ``EEGDashDataset``.
- :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_13_save_and_reuse_prepared_data`
  shows how to persist a preprocessed dataset to disk and reload without
  re-running the catalogue query.

Further reading
---------------

- Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain imaging
  data structure for electroencephalography. *Scientific Data*, 6(1), 103.
  https://doi.org/10.1038/s41597-019-0104-8
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python.
  *Frontiers in Neuroscience*, 7, 267.
  https://doi.org/10.3389/fnins.2013.00267
- Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
  electroencephalographic (EEG) data acquisition and signal processing.
  *PeerJ Computer Science*, 10, e2256.
  https://doi.org/10.7717/peerj-cs.2256
