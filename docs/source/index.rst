:html_theme.sidebar_secondary.remove: true
:og:description: EEGDash is a Python library for 700+ BIDS-first EEG, MEG, fNIRS, EMG, and iEEG datasets. Load, preprocess, and train PyTorch models with MNE-Python and braindecode in minutes.

.. title:: EEGDash: Python library for 700+ EEG/MEG datasets

.. meta::
   :description: EEGDash is a Python library for 700+ BIDS-first EEG, MEG, fNIRS, EMG, and iEEG datasets. Load, preprocess, and train PyTorch models with MNE-Python and braindecode in minutes.
   :keywords: EEGDash, EEG, MEG, dataset, Python library, BIDS, neuroscience, PyTorch, MNE-Python, braindecode, fNIRS, EMG, iEEG, OpenNeuro, NEMAR, machine learning, deep learning

.. container:: hf-hero

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: hf-hero-grid

      .. grid-item::
         :class: hf-hero-copy hf-reveal hf-delay-1

         .. raw:: html

            <h1 class="hf-hero-title">EEGDash: the Python library for 700+ BIDS-first EEG/MEG datasets.</h1>

         .. rst-class:: hf-hero-lede

            Install with ``pip install eegdash``, then load, preprocess, and train
            PyTorch models on open EEG/MEG data in minutes. Works hand-in-hand with
            MNE-Python and braindecode.

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
                  placeholder="Search datasets (e.g., visual, P300, resting-state)"
                  autocomplete="off"
                />
                <button class="hf-search-submit" type="submit">Search</button>
              </div>
              <div class="hf-search-suggest">
                <span class="hf-suggest-label">Suggested:</span>
                <a class="hf-suggest-link" data-query="ds004504" href="/api/dataset/eegdash.dataset.DS004504.html">ds004504</a>
                <a class="hf-suggest-link" data-query="ds000117" href="/api/dataset/eegdash.dataset.DS000117.html">ds000117</a>
                <a class="hf-suggest-link" data-query="nm000107" href="/api/dataset/eegdash.dataset.NM000107.html">nm000107</a>
              </div>
              <div class="hf-search-advanced">
                <button type="button" class="hf-suggest-link" data-open-eegdash-search
                        title="Open the search palette: filter by task, channels, sampling rate, license…">
                  Advanced search <kbd>Ctrl</kbd>+<kbd>K</kbd>
                </button>
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

               .. button-ref:: quickstart
                  :color: primary
                  :class: sd-btn-sm hf-btn hf-btn-primary

                  Run your first search

               .. button-ref:: api/api
                  :color: secondary
                  :class: sd-btn-sm hf-btn hf-btn-ghost

                  Read the Docs

.. raw:: html

   <h2 class="hf-section-title">At a glance</h2>
   <p class="hf-section-subtitle">Search-first discovery with reproducible pipelines and standardized metadata.</p>


.. container:: hf-badges

   .. Badges come from shield/badge services (GitHub Actions, img.shields.io,
      pepy.tech, codecov) whose SVGs all share the same natural height (20)
      but have different natural widths driven by their text content. Each
      ``:width:`` below matches the exact natural width of its SVG; every
      ``:height:`` is 20. Rendering at 1:1 scale avoids the aspect-ratio
      distortion that PSI flagged (e.g. Python versions badge: natural
      198x20), and every ``<img>`` ships with explicit width+height so the
      browser reserves accurate layout space — no CLS regression, SEO
      contract preserved.

   .. image:: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml/badge.svg
      :alt: Test Status
      :target: https://github.com/eegdash/EEGDash/actions/workflows/tests.yml
      :width: 104
      :height: 20

   .. image:: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml/badge.svg
      :alt: Doc Status
      :target: https://github.com/eegdash/EEGDash/actions/workflows/doc.yaml
      :width: 102
      :height: 20

   .. image:: https://img.shields.io/pypi/v/eegdash?color=blue&style=flat-square
      :alt: PyPI
      :target: https://pypi.org/project/eegdash/
      :width: 78
      :height: 20

   .. image:: https://img.shields.io/pypi/pyversions/eegdash?style=flat-square
      :alt: Python Versions
      :target: https://pypi.org/project/eegdash/
      :width: 198
      :height: 20

   .. image:: https://pepy.tech/badge/eegdash
      :alt: Downloads
      :target: https://pepy.tech/project/eegdash
      :width: 106
      :height: 20

   .. image:: https://codecov.io/gh/eegdash/EEGDash/branch/main/graph/badge.svg
      :alt: Code Coverage
      :target: https://codecov.io/gh/eegdash/EEGDash
      :width: 112
      :height: 20

   .. image:: https://img.shields.io/pypi/l/eegdash?style=flat-square
      :alt: License
      :target: https://github.com/eegdash/EEGDash/blob/main/LICENSE
      :width: 134
      :height: 20

   .. image:: https://img.shields.io/github/stars/eegdash/eegdash?style=flat-square
      :alt: GitHub Stars
      :target: https://github.com/eegdash/EEGDash
      :width: 60
      :height: 20


.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: hf-stat-grid

   .. grid-item-card:: Datasets
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-stat-card hf-reveal hf-delay-1

      .. rst-class:: hf-stat-value

         700+

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
      :link: quickstart
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

   .. rst-class:: hf-callout-support

      Support Institutions

   .. container:: logos-container hf-logo-cloud

      .. container:: logo-item

         .. image:: _static/logos/ucsd_white.svg
            :alt: UCSD
            :class: can-zoom only-dark
            :width: 260
            :height: 80

         .. image:: _static/logos/ucsd_dark.svg
            :alt: UCSD
            :class: can-zoom only-light
            :width: 260
            :height: 80

      .. container:: logo-item

         .. image:: _static/logos/bgu_dark.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-dark
            :width: 260
            :height: 80

         .. image:: _static/logos/bgu_white.svg
            :alt: Ben-Gurion University of the Negev (BGU)
            :class: can-zoom only-light
            :width: 260
            :height: 80

   .. rst-class:: hf-callout-funders

      Funders

   .. container:: hf-supporter-line

      .. image:: _static/logos/nsf_logo.png
         :alt: National Science Foundation (NSF)
         :class: hf-supporter-logo
         :width: 100
         :height: 100

      .. rst-class:: hf-supporter-text

         AWS Open Data Sponsorship Program


.. toctree::
   :hidden:

   Datasets <dataset_summary>
   Quick Start <quickstart>
   Install <install/install>
   Examples <generated/auto_examples/index>
   Concepts <concepts/index>
   Docs <api/api>
   References <references>
