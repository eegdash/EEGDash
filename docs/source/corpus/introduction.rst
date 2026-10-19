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

EEGDash Corpus is an aggregated, training-ready collection of public EEG datasets for
pretraining foundation models. It is served in two forms: the raw datasets, each
reachable through EEGDash by its identifier, and a converted Zarr store, built by applying
one recipe once (|recipe_sfreq| Hz, a |recipe_band| Hz band, |recipe_window|-second
samples, microvolts left un-normalised, with each session's mean and standard
deviation and each channel's position stored beside the samples), that a trainer reads
directly. Zarr is the chunked array format the store is served in: a trainer reads the
samples it needs, over the network or from a local copy, without unpacking the store.

Why
===

The data behind EEG foundation models is fragmented across archives, file formats,
sampling rates, montages and licences. Each group collates, checks, licenses and
converts its own copy, and that work is not released in a form another model can train
on. The corpus does it once: a group can start from the data rather than from the
collation, with the licence of every dataset accounted for.

Starting point: REVE's corpus
=============================

The first versions cover the corpus of REVE, the largest published EEG foundation
model. REVE's paper names |paper_named| datasets in its reference list of what it
pretrained on, and the release card of its public subset lists |card_entries| entries,
|card_only| of which the paper does not name. These pages track the union,
|universe| datasets, and every count, chart and the roadmap are over all of them. Of
the |card_only| card-only entries, |card_only_prose| (the table's notes say why).

Versions
========

A version is a frozen table of datasets and the converted Zarr store built from it.
Versions are additive; each carries its own table, provenance and conversion report.

**v0** is REVE's public release rebuilt from the archives and converted:
|v0_datasets| datasets (|v0_hours| hours). **v1** extends it to every publicly usable
dataset named by REVE's paper: |v1_datasets| datasets, of the added ones
|v1_added_prose|. The |gated| gated datasets are out of every version: their
distribution is forbidden, so only weights trained on them can be released. The
|blocked| datasets whose licences block a converted copy (No Derivative terms, or no
licence) are listed with the reason. Later versions add public datasets beyond REVE's,
chosen by licence and usability.

How to get it today
===================

``EEGDashDataset(dataset="<id>", cache_dir=...)`` for any raw dataset of the table;
the converted Zarr stores of v0 and v1 are to be hosted through EEGDash.

How it was built and checked
============================

**Fetch.** OpenNeuro and NEMAR datasets through EEGDash into a cache that mirrors the
archives.

**Convert.** The recipe above, per recording, with a report of what was dropped and
why. Each electrode takes its position from a bank of templates (REVE's public bank
plus the BioSemi-160 and BioSemi-256 caps); for electrodes the bank lacks, the
dataset's own ``electrodes.tsv``, aligned to the bank, places them.

**Verify.** |check_loader_datasets| datasets compared against the authors' own loader
sample by sample; REVE-Base retrained on the Zarr store scores within |check_bench_gap| of
the released weights on |check_bench_tasks| public OpenEEGBench tasks.

Where the numbers come from
===========================

Every cell carries a provenance label: ``zarr`` (measured in the converted Zarr store),
``eegdash`` (EEGDash's catalogue records), ``card`` (REVE's release card), ``stated``
(REVE's paper), ``download`` (our copy). A bucket is one exact channel set of one
dataset in the Zarr store. Status vocabulary: Ingested and Converted (v0) · Ingested and
Converted (v1) · Ingested, but not Converted (v1) · To Be Ingested (v1) · Blocked (ND
Licence) · Gated (Distribution Forbidden) · Excluded.

The corpus in numbers
=====================

.. corpus-figure:: kpis

Licences
========

Every dataset tracked, by licence, coloured by status: the licence decides whether a
converted copy may be redistributed, and under which terms.

.. corpus-figure:: licences
