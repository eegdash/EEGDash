:html_theme.sidebar_primary.remove: true
:html_theme.sidebar_secondary.remove: true

.. _api:

#############
API Reference
#############

The EEGDash API reference curates everything you need to integrate, extend,
and automate EEGDash—from core dataset helpers to feature extraction and rich
dataset metadata. The focus is interoperability, extensibility, and ease of use.

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

The API is organized into three main components:


.. grid:: 1
   :gutter: 4
   :class-container: sd-gap-4 sd-mb-4

   .. grid-item-card::
      :link: api_core
      :link-type: doc
      :text-align: center
      :class-card: api-grid-card
      :class-header: api-grid-card__header
      :class-body: api-grid-card__body
      :class-footer: api-grid-card__footer

      .. raw:: html

         <span class="fa-solid fa-microchip api-grid-card__icon" aria-hidden="true"></span>

      .. rst-class:: api-grid-card__title

      **Core API**
      ^^^

      Build, query, and manage EEGDash datasets and utilities.

      +++

      .. button-ref:: api_core
         :color: primary
         :class: api-grid-card__button
         :click-parent:

         → Explore Core API

   .. grid-item-card::
      :link: api_features
      :link-type: doc
      :text-align: center
      :class-card: api-grid-card
      :class-header: api-grid-card__header
      :class-body: api-grid-card__body
      :class-footer: api-grid-card__footer

      .. raw:: html

         <span class="fa-solid fa-wave-square api-grid-card__icon" aria-hidden="true"></span>

      .. rst-class:: api-grid-card__title

      **Feature engineering**
      ^^^

      Extract statistical, spectral, and machine-learning-ready features.

      +++

      .. button-ref:: api_features
         :color: primary
         :class: api-grid-card__button
         :click-parent:

         → Explore Feature Engineering

   .. grid-item-card::
      :link: dataset/api_dataset
      :link-type: doc
      :text-align: center
      :class-card: api-grid-card
      :class-header: api-grid-card__header
      :class-body: api-grid-card__body
      :class-footer: api-grid-card__footer

      .. raw:: html

         <span class="fa-solid fa-database api-grid-card__icon" aria-hidden="true"></span>

      .. rst-class:: api-grid-card__title

      **Dataset catalog**
      ^^^

      Browse dynamically generated dataset classes with rich metadata.

      +++

      .. button-ref:: dataset/api_dataset
         :color: primary
         :class: api-grid-card__button
         :click-parent:

         → Explore the Dataset API


********************
REST API Endpoints
********************

The EEGDash metadata server exposes a FastAPI REST interface for discovery and
querying. Base URL: `https://data.eegdash.org`_. Below is a concise map of the main
entrypoints and their purpose.

.. _https://data.eegdash.org: https://data.eegdash.org



Meta Endpoints
==============

- ``GET /``
  Returns API name, version, and available databases.
- ``GET /health``
  Returns API health and MongoDB connection status.
- ``GET /metrics``
  Prometheus metrics (if enabled).

Public Data Endpoints
=====================

- ``GET /api/{database}/records``
  Query records (files) with filter and pagination.
- ``GET /api/{database}/count``
  Count records matching a filter.
- ``GET /api/{database}/datasets/names``
  List unique dataset names from records.
- ``GET /api/{database}/metadata/{dataset}``
  Get metadata for a single dataset (from records).
- ``GET /api/{database}/datasets/summary``
  Get summary statistics and metadata for all datasets (with pagination, filtering).
  Query params: ``limit`` (1-1000), ``skip``, ``modality`` (eeg/meg/ieeg), ``source`` (openneuro/nemar/zenodo/etc.).
  Response includes aggregate totals for datasets, subjects, files, and size.
- ``GET /api/{database}/datasets/summary/{dataset_id}``
  Get detailed summary for a specific dataset.
  ``dataset_id`` may be the dataset ID or dataset name.
- ``GET /api/{database}/datasets/{dataset_id}``
  Get a specific dataset document by ID.
- ``GET /api/{database}/datasets``
  List dataset documents (with filtering and pagination).

Admin Endpoints (require Bearer token)
======================================

- ``POST /admin/{database}/records``
  Insert a single record (file document).
- ``POST /admin/{database}/records/bulk``
  Insert multiple records (max 1000 per request).
- ``POST /admin/{database}/datasets``
  Insert or update a single dataset document (upsert by ``dataset_id``).
- ``POST /admin/{database}/datasets/bulk``
  Insert or update multiple dataset documents (max 500 per request).
- ``PATCH /admin/{database}/records``
  Update records matching a filter (only ``$set`` allowed).
- ``GET /admin/security/blocked``
  List blocked IPs and offense counts.
- ``POST /admin/security/unblock``
  Unblock a specific IP.
    

******************
Related Guides
******************

- :doc:`Tutorial gallery <../generated/auto_examples/index>`
- :doc:`Dataset summary <../dataset_summary>`
- :doc:`Installation guide <../install/install>`

.. toctree::
   :hidden:

   api_core
   api_features
   dataset/api_dataset
   ../developer_notes
