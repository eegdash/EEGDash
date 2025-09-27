:html_theme.sidebar_primary.remove: true

.. _api:

API Reference
=============

.. grid:: 1
   :gutter: 2
   :class-container: sd-mb-4

   .. grid-item::

      .. card:: Core API
         :link: api_core
         :link-type: doc
         :shadow: md
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 sd-text-center api-card
         :class-title: sd-bg-primary sd-text-white sd-font-weight-bold

         .. raw:: html

            <span class="fa-solid fa-microchip fa-3x sd-text-primary" aria-hidden="true"></span>
            <span class="sd-sr-only">Core API documentation</span>

         Build, query, and manage EEGDash datasets and utilities.

   .. grid-item::

      .. card:: Feature Engineering
         :link: api_features
         :link-type: doc
         :shadow: md
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 sd-text-center api-card
         :class-title: sd-bg-primary sd-text-white sd-font-weight-bold

         .. raw:: html

            <span class="fa-solid fa-wave-square fa-3x sd-text-primary" aria-hidden="true"></span>
            <span class="sd-sr-only">Feature engineering documentation</span>

         Extract statistical, spectral, and machine-learning ready features.

   .. grid-item::

      .. card:: Dataset Catalog
         :link: dataset/api_dataset
         :link-type: doc
         :shadow: md
         :width: 100%
         :class-card: sd-border-0 sd-rounded-3 sd-text-center api-card
         :class-title: sd-bg-primary sd-text-white sd-font-weight-bold

         .. raw:: html

            <span class="fa-solid fa-database fa-3x sd-text-primary" aria-hidden="true"></span>
            <span class="sd-sr-only">Dataset catalog documentation</span>

         Browse dynamically generated dataset classes with rich metadata.

.. toctree::
   :hidden:

   api_core
   api_features
   dataset/api_dataset
