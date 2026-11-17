:html_theme.sidebar_primary.remove: true
:html_theme.sidebar_secondary.remove: true
:og:description: EEGDash Corpus: a training-ready collection of public EEG datasets for pretraining EEG foundation models. Release v0, its datasets, licences and download.

.. meta::
   :description: EEGDash Corpus: a training-ready collection of public EEG datasets for pretraining EEG foundation models. Release v0, its datasets, licences and download.

.. include:: /_static/corpus_generated/numbers.inc

.. _corpus:

##############
EEGDash Corpus
##############

EEGDash Corpus is a training-ready collection of public EEG datasets for pretraining
EEG foundation models, served through EEGDash and converted once with one recipe. Its
release, **EEGDash Corpus v0**, holds |v0_datasets| datasets (|v0_hours| hours,
|v0_subjects| subjects): of the |universe| datasets REVE's paper and release card
name, |v0_datasets| are converted and released, |not_redistributable| cannot be
redistributed (|blocked| blocked by their licences, |gated| gated by access
agreements) and |excluded| are excluded.

.. corpus-figure:: kpis

.. container:: corpus

   .. grid:: 1 1 3 3
      :gutter: 4

      .. grid-item-card:: Introduction
         :link: introduction
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         The corpus

         - What the corpus is and why it exists
         - REVE's corpus as the starting point
         - How it was built, checked and licensed
         - The |universe| datasets tracked: why |not_redistributable| cannot be redistributed and |excluded| are excluded

         .. rst-class:: corpus-card-find

         What you will find: the definitions every page uses, the provenance of each number, the licences, the table of every dataset tracked with the reason for each one not in the corpus.

      .. grid-item-card:: EEGDash Corpus v0
         :link: v0
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         The release

         - |v0_datasets| datasets, |v0_hours| hours, |v0_subjects| subjects
         - |v0_positions| electrode positions in |v0_buckets| channel sets
         - Hours, channels, subjects, rates, paradigms, recordings, ages and sex, tokens, montages
         - How to download and load it

         .. rst-class:: corpus-card-find

         What you will find: the converted Zarr store measured, the electrode map in 2-D and 3-D, the download, the table of its datasets.

      .. grid-item-card:: Pretraining REVE
         :link: pretraining_reve
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         The tutorial

         - Download the converted store
         - Install the code and REVE's trainer
         - Pretrain REVE with the paper's settings
         - One machine, or 16 GPUs on 4 machines

         .. rst-class:: corpus-card-find

         What you will find: three steps and the two scripts behind them, from an empty machine to a pretrained encoder, and the paper's run times and scores.

.. toctree::
   :hidden:

   Introduction <introduction>
   EEGDash Corpus v0 <v0>
   Pretraining REVE <pretraining_reve>
