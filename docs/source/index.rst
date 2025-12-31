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
                <a class="hf-suggest-link" data-query="sleep eeg" href="dataset_summary.html?q=sleep">Sleep EEG</a>
                <a class="hf-suggest-link" data-query="p300" href="dataset_summary.html?q=P300">P300</a>
                <a class="hf-suggest-link" data-query="resting-state" href="dataset_summary.html?q=resting-state">Resting-state</a>
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

            .. raw:: html

               <a class="hf-card-link" href="user_guide.html">Read docs</a>

.. raw:: html

   <section class="hf-trending">
     <div class="hf-trending-header">
       <h2 class="hf-section-title">Trending this week</h2>
       <p class="hf-section-subtitle">Curated picks from datasets, pipelines, and benchmarks.</p>
     </div>
     <div class="hf-trending-grid">
       <div class="hf-trending-card">
         <div class="hf-trending-title">Datasets</div>
         <div class="hf-trending-meta">Catalog preview</div>
         <a class="hf-trending-item" href="dataset_summary.html?q=sleep">
           <span class="hf-trending-name">Sleep EEG</span>
           <span class="hf-trending-desc">Sleep | EEG | BIDS</span>
         </a>
         <a class="hf-trending-item" href="dataset_summary.html?q=resting-state">
           <span class="hf-trending-name">Resting MEG</span>
           <span class="hf-trending-desc">Resting-state | MEG | BIDS</span>
         </a>
         <a class="hf-trending-item" href="dataset_summary.html?q=P300">
           <span class="hf-trending-name">ERP benchmarks</span>
           <span class="hf-trending-desc">P300 | EEG | Benchmarks</span>
         </a>
         <a class="hf-trending-link" href="dataset_summary.html">Browse all datasets</a>
       </div>
       <div class="hf-trending-card">
         <div class="hf-trending-title">Pipelines</div>
         <div class="hf-trending-meta">Reproducible workflows</div>
         <a class="hf-trending-item" href="user_guide.html">
           <span class="hf-trending-name">EEGPrep preprocessing</span>
           <span class="hf-trending-desc">Standardize inputs with one command.</span>
         </a>
         <a class="hf-trending-item" href="user_guide.html">
           <span class="hf-trending-name">BIDS validation</span>
           <span class="hf-trending-desc">Keep datasets compliant and shareable.</span>
         </a>
         <a class="hf-trending-item" href="generated/auto_examples/index.html">
           <span class="hf-trending-name">Feature export</span>
           <span class="hf-trending-desc">Generate model-ready features quickly.</span>
         </a>
         <a class="hf-trending-link" href="user_guide.html">Browse all pipelines</a>
       </div>
       <div class="hf-trending-card">
         <div class="hf-trending-title">Benchmarks</div>
         <div class="hf-trending-meta">Baseline comparisons</div>
         <a class="hf-trending-item" href="generated/auto_examples/index.html">
           <span class="hf-trending-name">ERP baselines</span>
           <span class="hf-trending-desc">Compare P300 classifiers.</span>
         </a>
         <a class="hf-trending-item" href="generated/auto_examples/index.html">
           <span class="hf-trending-name">Motor imagery</span>
           <span class="hf-trending-desc">BCI-ready benchmark tasks.</span>
         </a>
         <a class="hf-trending-item" href="generated/auto_examples/index.html">
           <span class="hf-trending-name">Sleep staging</span>
           <span class="hf-trending-desc">Reproducible sleep pipelines.</span>
         </a>
         <a class="hf-trending-link" href="generated/auto_examples/index.html">View benchmarks</a>
       </div>
     </div>
   </section>

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

.. raw:: html

   <h2 class="hf-section-title">What's inside EEGDash</h2>
   <p class="hf-section-subtitle">Everything you need to discover, prepare, and benchmark EEG and MEG data.</p>

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: hf-feature-grid

   .. grid-item-card:: Dataset discovery
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-1

      :octicon:`search;1.5em;sd-text-primary`

      Search metadata, modalities, tasks, and cohorts with unified filters.

   .. grid-item-card:: Reproducible preprocessing
      :link: user_guide
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-2

      :octicon:`plug;1.5em;sd-text-primary`

      One-command pipelines with EEGPrep, MNE, and BIDS alignment.

   .. grid-item-card:: Benchmarks and features
      :link: generated/auto_examples/index
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-3

      :octicon:`rocket;1.5em;sd-text-primary`

      Export model-ready features and compare baselines across datasets.

   .. grid-item-card:: BIDS-first interoperability
      :link: user_guide
      :link-type: doc
      :text-align: left
      :class-card: feature-card hf-reveal hf-delay-3

      :octicon:`checklist;1.5em;sd-text-primary`

      Keep metadata consistent and portable across teams and tools.

.. raw:: html

   <h2 class="hf-section-title">Workflow</h2>
   <p class="hf-section-subtitle">Discover -> Prepare -> Benchmark -> Export</p>

.. container:: hf-workflow

   .. container:: hf-workflow-step

      .. rst-class:: hf-workflow-title

         Discover

      .. rst-class:: hf-workflow-text

         Search datasets, metadata, and modalities in one place.

   .. container:: hf-workflow-step

      .. rst-class:: hf-workflow-title

         Prepare

      .. rst-class:: hf-workflow-text

         Run standardized preprocessing pipelines with EEGPrep.

   .. container:: hf-workflow-step

      .. rst-class:: hf-workflow-title

         Benchmark

      .. rst-class:: hf-workflow-text

         Compare baselines across tasks and cohorts.

   .. container:: hf-workflow-step

      .. rst-class:: hf-workflow-title

         Export

      .. rst-class:: hf-workflow-text

         Ship model-ready features and share results.

.. raw:: html

   <h2 class="hf-section-title">Featured datasets</h2>
   <p class="hf-section-subtitle">A quick look at what you can search today.</p>

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: hf-dataset-grid

   .. grid-item-card:: Sleep EEG
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      Sleep | EEG | BIDS

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">EEG</span>
           <span class="hf-tag">Sleep</span>
           <span class="hf-tag">BIDS</span>
         </div>

   .. grid-item-card:: Resting MEG
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      Resting-state | MEG | BIDS

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">MEG</span>
           <span class="hf-tag">Resting</span>
           <span class="hf-tag">BIDS</span>
         </div>

   .. grid-item-card:: ERP benchmarks
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      ERP | EEG | Benchmarks

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">ERP</span>
           <span class="hf-tag">EEG</span>
           <span class="hf-tag">Benchmark</span>
         </div>

   .. grid-item-card:: Motor imagery
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      Motor imagery | EEG | BCI

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">BCI</span>
           <span class="hf-tag">EEG</span>
           <span class="hf-tag">Motor</span>
         </div>

   .. grid-item-card:: Clinical EEG
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      Clinical | EEG | Cohorts

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">Clinical</span>
           <span class="hf-tag">EEG</span>
           <span class="hf-tag">Cohort</span>
         </div>

   .. grid-item-card:: fNIRS pipelines
      :link: dataset_summary
      :link-type: doc
      :text-align: left
      :class-card: hf-dataset-card

      fNIRS | BIDS | Pipelines

      .. raw:: html

         <div class="hf-card-tags">
           <span class="hf-tag">fNIRS</span>
           <span class="hf-tag">BIDS</span>
           <span class="hf-tag">Pipeline</span>
         </div>

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
   Pipelines <user_guide>
   Benchmarks <generated/auto_examples/index>
   Docs <install/install>
