:og:description: What EEGDash Corpus is, why it exists, how it was built and checked, and where its numbers come from.

.. meta::
   :description: What EEGDash Corpus is, why it exists, how it was built and checked, and where its numbers come from.

.. include:: /_static/corpus_generated/numbers.inc

.. _corpus-introduction:

############
Introduction
############

What it is
==========

EEGDash Corpus is a training-ready collection of public EEG datasets, served through
EEGDash from OpenNeuro and NEMAR and converted once with the recipe of REVE, an EEG
foundation model: |recipe_sfreq| Hz, a |recipe_band| Hz band, |recipe_window|-second
samples in microvolts, left un-normalised, with each session's mean and standard
deviation and each channel's position stored beside the samples. The result is one
Zarr store that a trainer reads directly; Zarr is a chunked array format, so a trainer
reads the samples it needs, over the network or from a local copy, without unpacking
the store.

Why
===

The data behind EEG foundation models is fragmented across archives, file formats,
sampling rates, montages and licences. Each group collates, checks, licenses and
converts its own copy, and that work is not released in a form another model can train
on. The corpus does it once: a group can start from the data rather than from the
collation, with the licence of every dataset accounted for.

Starting point: REVE's corpus
=============================

**EEGDash Corpus v0** is the public part of REVE's pretraining corpus, rebuilt from the
archives. REVE's paper names |paper_named| datasets in its reference list of what it
pretrained on, and the release card of its public subset lists |card_entries| entries,
|card_only| of which the paper does not name. These pages track the union,
|universe| datasets:

- |v0_datasets| are converted and released in v0;
- |not_redistributable| cannot be redistributed: |blocked| are blocked by their licences
  (|blocked_nd| No Derivatives, |blocked_none| with no licence stated) and |gated| are
  gated by access agreements that forbid redistribution (|gated_names|);
- |excluded| are excluded for other reasons (|excluded_prose|).

Every dataset keeps its own licence, and the release is gated: access needs a Hugging
Face account that accepted its terms. Later versions will add public datasets beyond
REVE's, chosen by licence and usability.

In numbers
==========

What the release holds, against the datasets tracked; the v0 page measures the store
itself.

.. corpus-figure:: intro-kpis

How to get it
=============

Three ways, from the training-ready store to one raw dataset; :ref:`corpus-download` on
the v0 page gives the details.

**The converted store.** The Zarr store of v0 is a gated dataset repository on Hugging
Face: accept its terms with a Hugging Face account, log in with ``hf auth login``, then
download it whole (|v0_size|) or in parts with ``--include``, and open it with ``zarr``:

.. code-block:: bash

   hf download EEGDash/eegdash-corpus-v0 --repo-type dataset --local-dir eegdash-corpus-v0

.. code-block:: python

   import zarr  # zarr >= 3
   root = zarr.open_group("eegdash-corpus-v0/eegdash-corpus-v0.zarr", mode="r")

**Every raw dataset of the corpus.** Fetch the archives' own files through
:class:`~eegdash.corpus.EEGDashCorpus`, not the converted Zarr store. The list of
datasets ships with EEGDash, so no extra CSV download is needed:

.. code-block:: python

   from eegdash import EEGDashCorpus

   corpus = EEGDashCorpus("eegdash-corpus-v0", cache_dir="eegdash-cache")
   print(corpus.plan())
   corpus.download(n_jobs=16, n_datasets=4)

The object plans files from the catalogue, downloads a few datasets at a time,
verifies the cache by size or checksum, and reports its status. A custom CSV with a
``dataset_id`` column works too. The command-line interface currently takes a CSV
path or comma-separated ids rather than a corpus name; :ref:`corpus-download` gives
those commands and the optional CSV download.

**One raw dataset.** Any dataset of the tables, by its EEGDash id, as an
:class:`~eegdash.api.EEGDashDataset` whose recordings download when first read (here
AlexMI, a small motor-imagery dataset):

.. code-block:: python

   from eegdash import EEGDashDataset
   ds = EEGDashDataset(dataset="nm000138", cache_dir="eegdash-cache")

How it was built and checked
============================

**Fetch.** OpenNeuro and NEMAR datasets through EEGDash into a cache that mirrors the
archives.

**Convert.** The recipe above, per recording, with a report of what was dropped and
why. Each electrode takes its position from a bank of templates (REVE's public bank
plus the BioSemi-160 and BioSemi-256 caps); for electrodes the bank lacks, the
dataset's own ``electrodes.tsv``, aligned to the bank, places them. The conversion and
training code is published in `eegdash/eegdash-corpus
<https://github.com/eegdash/eegdash-corpus>`__ (BSD-3-Clause).

**Verify.** |check_loader_datasets| datasets compared against the authors' own loader
sample by sample; REVE-Base, retrained from scratch on data converted by this pipeline,
scores within |check_bench_gap| of the released weights on |check_bench_tasks| public
OpenEEGBench tasks.

Where the numbers come from
===========================

Every cell carries a provenance label: ``zarr`` (measured in the converted Zarr store),
``eegdash`` (EEGDash's catalogue records), ``card`` (REVE's release card), ``stated``
(REVE's paper), ``download`` (a downloaded copy). A bucket is one exact channel set of
one dataset in the Zarr store. Status vocabulary: Ingested and Converted (v0) · Blocked
(ND Licence) · Gated (Distribution Forbidden) · Excluded.

Licences
========

Every dataset tracked, by licence, coloured by status: the licence decides whether a
converted copy may be redistributed, and under which terms.

.. corpus-figure:: licences

Every dataset tracked
=====================

REVE's paper names |paper_named| datasets and its release card lists |card_entries|
entries, |card_only| of them not in the paper: |universe| datasets in all. |v0_datasets|
are converted and released in v0. |not_redistributable| cannot be redistributed:
|blocked| are blocked by their licences (|blocked_nd| No Derivatives, |blocked_none|
with no licence stated) and |gated| are gated by access agreements (|gated_names|).
|excluded| are excluded: |excluded_prose|. The table gives each dataset its status and,
for each one not in the corpus, the reason; filter by text or by status.

.. corpus-figure:: table-all

.. only:: not html

   The table is published as ``_static/corpus_generated/atlas.csv``.
