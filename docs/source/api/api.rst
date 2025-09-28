:html_theme.sidebar_primary.remove: true
:html_theme.sidebar_secondary.remove: true

.. _api:

API Reference
=============

The EEGDash API reference curates everything you need to integrate, extend,
and automate EEGDash—from core dataset helpers to feature extraction and rich
dataset metadata.

.. container:: api-hero-buttons

   .. button-link:: api_core
      :color: primary
      :ref-type: doc
      :shadow:

      Get Started

   .. button-link:: dataset/api_dataset
      :color: secondary
      :ref-type: doc
      :shadow:

      Dataset Catalog

.. grid:: 1
   :gutter: 2
   :class-container: sd-gap-4 sd-mb-4

   .. grid-item::

      .. card:: Core API
         :link: api_core
         :link-type: doc
         :shadow: sm
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 api-card
         :class-title: api-card__title

         .. container:: api-card__content

            .. container:: api-card__icon

               .. raw:: html

                  <span class="fa-solid fa-microchip" aria-hidden="true"></span>
                  <span class="sd-sr-only">Core API documentation</span>

            .. container:: api-card__text

               Build, query, and manage EEGDash datasets and utilities.

               .. rst-class:: api-card__cta

               :doc:`→ Explore Core API <api_core>`

   .. grid-item::

      .. card:: Feature Engineering
         :link: api_features
         :link-type: doc
         :shadow: sm
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 api-card
         :class-title: api-card__title

         .. container:: api-card__content

            .. container:: api-card__icon

               .. raw:: html

                  <span class="fa-solid fa-wave-square" aria-hidden="true"></span>
                  <span class="sd-sr-only">Feature engineering documentation</span>

            .. container:: api-card__text

               Extract statistical, spectral, and machine-learning ready features.

               .. rst-class:: api-card__cta

               :doc:`→ Explore Feature Engineering <api_features>`

   .. grid-item::

      .. card:: Dataset Catalog
         :link: dataset/api_dataset
         :link-type: doc
         :shadow: sm
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 api-card
         :class-title: api-card__title

         .. container:: api-card__content

            .. container:: api-card__icon

               .. raw:: html

                  <span class="fa-solid fa-database" aria-hidden="true"></span>
                  <span class="sd-sr-only">Dataset catalog documentation</span>

            .. container:: api-card__text

               Browse dynamically generated dataset classes with rich metadata.

               .. rst-class:: api-card__cta

               :doc:`→ Explore the Dataset Catalog <dataset/api_dataset>`

Related guides
--------------

- :doc:`Tutorial gallery <../generated/auto_examples/index>`
- :doc:`Dataset summary <../dataset_summary>`
- :doc:`Installation guide <../install/install>`

.. toctree::
   :hidden:

   api_core
   api_features
   dataset/api_dataset
