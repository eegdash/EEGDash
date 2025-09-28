:html_theme.sidebar_primary.remove: true
:html_theme.sidebar_secondary.remove: true

.. _api:

#############
API Reference
#############

The EEGDash API reference curates everything you need to integrate, extend,
and automate EEGDash—from core dataset helpers to feature extraction and rich
dataset metadata.

The focus of EEGDash is interopinteroperability, extensibility, and ease of use.

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

         → Explore the Dataset Catalog
    

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
