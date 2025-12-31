:html_theme.sidebar_secondary.remove: true

.. title:: EEGDash - Data-sharing interface for M/EEG and related (fNIRS, EMG)

.. raw:: html

    <style type="text/css">
    /* Visually hide H1 but keep for metadata */
    h1 {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
    </style>

.. container:: hf-hero

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: hf-hero-grid

      .. grid-item::
         :class: hf-hero-copy hf-reveal hf-delay-1

         .. rst-class:: hf-kicker

            EEG Dash

         .. rst-class:: hf-hero-title

            The open hub for M/EEG datasets and benchmarks.

         .. rst-class:: hf-hero-lede

            Data-sharing interface for EEG, MEG, iEEG, fNIRS, and EMG with standardized metadata,
            reproducible preprocessing, and model-ready benchmarks.

         .. container:: hf-hero-actions

            .. button-ref:: install/install
               :color: primary
               :class: sd-btn-lg hf-btn hf-btn-primary

               Get Started

            .. button-ref:: dataset_summary
               :color: secondary
               :class: sd-btn-lg hf-btn hf-btn-secondary

               Explore Datasets

         .. container:: hf-hero-meta

            .. container:: hf-pill

               Open source

            .. container:: hf-pill

               500+ datasets

            .. container:: hf-pill

               BIDS-first

         .. container:: hf-badges

            .. image:: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml/badge.svg
               :alt: Test Status
               :target: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml

            .. image:: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml/badge.svg
               :alt: Doc Status
               :target: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml

            .. image:: https://img.shields.io/pypi/v/eegdash?color=blue&style=flat-square
               :alt: PyPI
               :target: https://pypi.org/project/eegdash/

            .. image:: https://img.shields.io/github/stars/eegdash/eegdash?style=flat-square
               :alt: GitHub Stars
               :target: https://github.com/eegdash/EEGDash

      .. grid-item::
         :class: hf-hero-panel hf-reveal hf-delay-2

         .. container:: hf-hero-card

            .. rst-class:: hf-card-title

               Quickstart

            .. code-block:: bash

               pip install eegdash

            .. rst-class:: hf-card-note

               Pull standardized metadata, run EEGPrep pipelines, and publish model-ready features.

            .. container:: hf-card-actions

               .. button-ref:: user_guide
                  :color: secondary
                  :class: sd-btn-sm hf-btn hf-btn-ghost

                  Read the Guide

               .. button-ref:: generated/auto_examples/index
                  :color: secondary
                  :class: sd-btn-sm hf-btn hf-btn-ghost

                  Browse Examples

         .. container:: hf-hero-stats

            .. container:: hf-stat

               .. rst-class:: hf-stat-value

                  500+

               .. rst-class:: hf-stat-label

                  Datasets

            .. container:: hf-stat

               .. rst-class:: hf-stat-value

                  5

               .. rst-class:: hf-stat-label

                  Modalities

            .. container:: hf-stat

               .. rst-class:: hf-stat-value

                  BIDS

               .. rst-class:: hf-stat-label

                  Standard

.. raw:: html

   <h2 class="hf-section-title">Build, benchmark, and share at scale</h2>
   <p class="hf-section-subtitle">From ingestion to evaluation, EEG Dash keeps workflows reproducible and datasets easy to explore.</p>

.. grid:: 1 1 3 3
   :gutter: 4
   :class-container: hf-feature-grid

   .. grid-item-card:: Dataset Atlas
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-1

      :octicon:`database;1.6em;sd-text-primary`

      Browse curated metadata, licensing, and modality coverage across hundreds of studies.

   .. grid-item-card:: Pipeline Library
      :link: user_guide
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-2

      :octicon:`plug;1.6em;sd-text-primary`

      Run EEGPrep and MNE-compatible workflows with traceable preprocessing decisions.

   .. grid-item-card:: Benchmarks and HF Integration
      :link: generated/auto_examples/index
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-3

      :octicon:`rocket;1.6em;sd-text-primary`

      Publish reproducible baselines and export model-ready features for the Hugging Face ecosystem.

.. raw:: html

   <h2 class="hf-section-title">A research-ready workflow</h2>
   <p class="hf-section-subtitle">Keep ingestion, preprocessing, and benchmarking in one auditable pipeline.</p>

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: hf-step-grid

   .. grid-item::
      :class: hf-step hf-reveal hf-delay-1

      .. rst-class:: hf-step-title

         1. Ingest

      .. rst-class:: hf-step-text

         Validate metadata and harmonize datasets across labs and modalities.

   .. grid-item::
      :class: hf-step hf-reveal hf-delay-2

      .. rst-class:: hf-step-title

         2. Preprocess

      .. rst-class:: hf-step-text

         Apply standardized pipelines with transparent QC and reporting.

   .. grid-item::
      :class: hf-step hf-reveal hf-delay-3

      .. rst-class:: hf-step-title

         3. Benchmark

      .. rst-class:: hf-step-text

         Compare models across tasks, export features, and share results.

.. raw:: html

   <h2 class="hf-section-title hf-section-title-center">Institutions</h2>

.. container:: logos-container hf-logo-cloud

    .. container:: logo-item

        .. image:: _static/logos/ucsd_white.svg
            :alt: UCSD
            :class: can-zoom only-dark
            :width: 300px

        .. image:: _static/logos/ucsd_dark.svg
            :alt: UCSD
            :class: can-zoom only-light
            :width: 300px

    .. container:: logo-item

        .. image:: _static/logos/bgu_dark.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-dark
            :width: 300px

        .. image:: _static/logos/bgu_white.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-light
            :width: 300px

.. container:: hf-callout hf-reveal hf-delay-2

   .. rst-class:: hf-callout-title

      Build with the community

   .. rst-class:: hf-callout-text

      Share datasets, contribute pipelines, and help define open standards for M/EEG.

   .. container:: hf-callout-actions

      .. button-link:: https://github.com/eegdash/EEGDash
         :color: secondary
         :class: sd-btn-lg hf-btn hf-btn-secondary

         GitHub

      .. button-link:: https://discord.gg/8jd7nVKwsc
         :color: primary
         :class: sd-btn-lg hf-btn hf-btn-primary

         Join Discord

.. toctree::
   :hidden:

   Installing <install/install>
   User Guide <user_guide>
   API <api/api>
   Dataset Catalog <dataset_summary>
   Examples <generated/auto_examples/index>
