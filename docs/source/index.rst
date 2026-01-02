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

         .. rst-class:: hf-hero-title

            Search and use 500+ EEG/MEG datasets - BIDS-first.

         .. rst-class:: hf-hero-lede

            Discover standardized metadata, run reproducible pipelines, and export
            model-ready features in minutes.

         .. raw:: html

            <form class="hf-search" action="dataset_summary.html" method="get" role="search" aria-label="Dataset search">
              <label class="hf-sr-only" for="hf-search-input">Search datasets</label>
              <div class="hf-search-input-wrap">
                <span class="hf-search-icon" aria-hidden="true">&#128269;</span>
                <input
                  id="hf-search-input"
                  class="hf-search-input"
                  type="search"
                  name="q"
                  placeholder="Search datasets (e.g., sleep, P300, resting-state, TUEG)"
                  autocomplete="off"
                />
                <button class="hf-search-submit" type="submit">Search</button>
              </div>
              <div class="hf-search-suggest">
                <span class="hf-suggest-label">Suggested:</span>
                <a class="hf-suggest-link" data-query="ds004504" href="dataset_summary.html?q=ds004504">ds004504</a>
                <a class="hf-suggest-link" data-query="ds000117" href="dataset_summary.html?q=ds000117">ds000117</a>
                <a class="hf-suggest-link" data-query="nm000107" href="dataset_summary.html?q=nm000107">nm000107</a>
              </div>
            </form>

         .. container:: hf-hero-actions

            .. button-ref:: dataset_summary
               :color: primary
               :class: sd-btn-lg hf-btn hf-btn-primary

               Browse datasets

            .. button-ref:: install/install
               :color: secondary
               :class: sd-btn-lg hf-btn hf-btn-secondary

               Get started

      .. grid-item::
         :class: hf-hero-panel hf-reveal hf-delay-2

         .. container:: hf-hero-card hf-quickstart

            .. rst-class:: hf-card-title

               Quickstart

            .. tab-set::
               :class: hf-code-tabs

               .. tab-item:: Install

                  .. code-block:: bash

                     pip install eegdash

               .. tab-item:: First search

                  .. code-block:: python

                     from eegdash import EEGDash

                     eegdash = EEGDash()
                     records = eegdash.find(dataset="ds002718")
                     print(f"Found {len(records)} records.")

            .. rst-class:: hf-card-note

               Works with Python 3.10+. BIDS-first. Runs locally.

            .. container:: hf-card-actions

               .. button-ref:: user_guide
                  :color: primary
                  :class: sd-btn-sm hf-btn hf-btn-primary

                  Run your first search

               .. button-ref:: generated/api/index
                  :color: secondary
                  :class: sd-btn-sm hf-btn hf-btn-ghost

                  Read the Docs

.. raw:: html

   <h2 class="hf-section-title">At a glance</h2>
   <p class="hf-section-subtitle">Search-first discovery with reproducible pipelines and standardized metadata.</p>

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: hf-stat-grid

   .. grid-item-card:: Datasets
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-stat-card hf-reveal hf-delay-1

      .. rst-class:: hf-stat-value

         500+

      .. rst-class:: hf-stat-text

         Curated and standardized metadata ready to explore.

   .. grid-item-card:: Modalities
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-stat-card hf-reveal hf-delay-2

      .. rst-class:: hf-stat-value

         5

      .. rst-class:: hf-stat-text

         EEG, MEG, fNIRS, EMG, and iEEG coverage.

   .. grid-item-card:: BIDS-first
      :link: user_guide
      :link-type: doc
      :text-align: left
      :class-card: hf-stat-card hf-reveal hf-delay-3

      .. rst-class:: hf-stat-value

         BIDS

      .. rst-class:: hf-stat-text

         Interoperability and reproducibility baked in.

   .. grid-item-card:: Open source
      :link: https://github.com/eegdash/EEGDash
      :link-type: url
      :text-align: left
      :class-card: hf-stat-card hf-reveal hf-delay-3

      .. rst-class:: hf-stat-value

         GitHub

      .. rst-class:: hf-stat-text

         Community-driven datasets, pipelines, and benchmarks.

.. container:: hf-trust

   .. rst-class:: hf-trust-title

      Trusted by labs and open-source community

   .. container:: hf-trust-badges

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

   .. container:: logos-container hf-logo-cloud

      .. container:: logo-item

         .. image:: _static/logos/ucsd_white.svg
            :alt: UCSD
            :class: can-zoom only-dark
            :width: 260px

         .. image:: _static/logos/ucsd_dark.svg
            :alt: UCSD
            :class: can-zoom only-light
            :width: 260px

      .. container:: logo-item

         .. image:: _static/logos/bgu_dark.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-dark
            :width: 260px

         .. image:: _static/logos/bgu_white.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-light
            :width: 260px

   .. container:: hf-trust-actions

      .. button-link:: https://github.com/eegdash/EEGDash
         :color: secondary
         :class: sd-btn-sm hf-btn hf-btn-secondary

         GitHub

      .. button-link:: https://discord.gg/8jd7nVKwsc
         :color: primary
         :class: sd-btn-sm hf-btn hf-btn-primary

         Join Discord


.. container:: hf-callout hf-reveal hf-delay-2

   .. rst-class:: hf-callout-title

      Build with the community

   .. rst-class:: hf-callout-text

      Share datasets, contribute pipelines, and help define open standards for EEG and MEG.

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

   Datasets <dataset_summary>
   User Guide <user_guide>
   Install <install/install>
   Examples <generated/auto_examples/index>
   Docs <generated/api/index>
