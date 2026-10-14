:html_theme.sidebar_primary.remove: true
:html_theme.sidebar_secondary.remove: true
:og:description: EEGDash Corpus: an aggregated, training-ready collection of public EEG datasets for pretraining EEG foundation models, its versions, progress and licences.

.. meta::
   :description: EEGDash Corpus: an aggregated, training-ready collection of public EEG datasets for pretraining EEG foundation models, its versions, progress and licences.

.. include:: /_static/corpus_generated/numbers.inc

.. _corpus:

##############
EEGDash Corpus
##############

EEGDash Corpus is an aggregated, training-ready collection of public EEG datasets
for pretraining EEG foundation models, served through EEGDash as raw datasets and as
one converted Zarr store. Version 0 holds |v0_datasets| datasets (|v0_hours| hours,
converted); version 1 extends it to |v1_datasets| datasets.

.. corpus-figure:: kpis

.. container:: corpus

   .. grid:: 1 1 2 2
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

         .. rst-class:: corpus-card-find

         What you will find: the definitions every page uses, the provenance of each number, the licences.

      .. grid-item-card:: Progress and Roadmap
         :link: progress
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         Status

         - Where each of the |universe| datasets stands
         - What comes next, in order
         - From REVE's paper to the versions

         .. rst-class:: corpus-card-find

         What you will find: the progress strip, the roadmap, and the table of every dataset tracked.

      .. grid-item-card:: EEGDash Corpus v0
         :link: v0
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         Converted

         - |v0_datasets| datasets, |v0_hours| hours, |v0_subjects| subjects
         - |v0_positions| electrode positions in |v0_buckets| channel sets
         - Hours, channels, subjects, rates, paradigms

         .. rst-class:: corpus-card-find

         What you will find: the converted Zarr store measured, the electrode map in 2-D and 3-D, the table of v0.

      .. grid-item-card:: EEGDash Corpus v1
         :link: v1
         :link-type: doc
         :class-card: corpus-card

         .. rst-class:: corpus-card-label

         Next version

         - |v1_new| datasets added: |v1_added_short|
         - |v1_datasets| datasets, |v1_hours| hours measured so far
         - The same views as v0, the added datasets by status

         .. rst-class:: corpus-card-find

         What you will find: what v1 adds, its numbers and caveats, the table of v1.

.. toctree::
   :hidden:

   Introduction <introduction>
   Progress and Roadmap <progress>
   EEGDash Corpus v0 <v0>
   EEGDash Corpus v1 <v1>
