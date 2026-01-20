:hide_sidebar: true
:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _data_summary:

.. raw:: html

   <script>document.documentElement.classList.add('dataset-summary-page');</script>

.. rst-class:: dataset-summary-article

Datasets Catalog
================

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications.


.. grid:: 1 2 2 4
    :gutter: 3
    :margin: 4 4 0 0

    .. grid-item-card::  Datasets
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`database;2em;sd-text-primary`
        
        **|datasets_total|**

    .. grid-item-card::  Subjects
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`people;2em;sd-text-secondary`
        
        **|subjects_total|**

    .. grid-item-card::  Modalities
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`pulse;2em;sd-text-success`
        
        **|modalities_total|**

    .. grid-item-card::  Sources
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`globe;2em;sd-text-warning`
        
        **|sources_total|**


.. raw:: html

   <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>

.. tab-set::

   .. tab-item:: Dataset Table

      .. include:: dataset_summary/table.rst

   .. tab-item:: Participant Distribution

      .. include:: dataset_summary/kde.rst

   .. tab-item:: Dataset Flow

      .. include:: dataset_summary/sankey.rst

   .. tab-item:: Dataset Treemap

      .. include:: dataset_summary/treemap.rst

   .. tab-item:: Scatter of Sample Size vs. Recording Duration

      .. include:: dataset_summary/bubble.rst

The archive is currently still in :bdg-danger:`beta testing` mode, so be kind. 
