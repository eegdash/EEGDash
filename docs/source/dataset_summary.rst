:hide_sidebar: true
:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true
:og:description: Browse 700+ BIDS-first EEG/MEG datasets in the EEGDash Python catalog. Filter by modality, task, subjects, and licence; load any row with a single line of Python.

.. meta::
   :description: Browse 700+ BIDS-first EEG/MEG datasets in the EEGDash Python catalog. Filter by modality, task, subjects, and licence; load any row with a single line of Python.

.. _data_summary:

.. only:: html

   .. raw:: html

      <script>
        document.documentElement.classList.add('dataset-summary-page');
        // Augment sphinx-design tabs with ARIA semantics. sphinx-design renders
        // a <input type="radio"> + <label> pattern where the input is visually
        // hidden but still tabindex=0, and neither element gets ARIA roles.
        // Screen readers announce the radios as unlabelled. Wire up role="tab",
        // aria-label, aria-selected, and the containing tablist role so they're
        // discoverable as a real tab panel.
        window.addEventListener('DOMContentLoaded', function () {
          document.querySelectorAll('.sd-tab-set').forEach(function (set) {
            set.setAttribute('role', 'tablist');
            var inputs = set.querySelectorAll('input[type="radio"]');
            inputs.forEach(function (input) {
              var label = set.querySelector('label[for="' + input.id + '"]');
              if (!label) return;
              var name = label.textContent.trim();
              input.setAttribute('role', 'tab');
              input.setAttribute('aria-label', name);
              input.setAttribute('aria-selected', input.checked ? 'true' : 'false');
              input.addEventListener('change', function () {
                inputs.forEach(function (i) {
                  i.setAttribute('aria-selected', i.checked ? 'true' : 'false');
                });
              });
            });
          });
        });
      </script>

.. rst-class:: dataset-summary-article

Datasets Catalog
================

EEG-DaSh is a data-sharing archive for MEEG (EEG, MEG) recordings contributed by collaborating labs. It preserves publicly funded research data and exposes it in a form that machine learning and deep learning workflows can use directly.


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

    .. grid-item-card::  Duration (hours)
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`clock;2em;sd-text-info`

        **|duration_hours|**

    .. grid-item-card::  Modalities
        :class-card: dataset-counter-card
        :text-align: center

        :octicon:`pulse;2em;sd-text-success`

        **|modalities_total|**


.. only:: html

   .. raw:: html

      <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>

   .. tab-set::

      .. tab-item:: Dataset Table

         .. dataset-figure:: table

      .. tab-item:: Participant Distribution

         .. dataset-figure:: kde

      .. tab-item:: Dataset Flow

         .. dataset-figure:: sankey

      .. tab-item:: Dataset Treemap

         .. dataset-figure:: treemap

      .. tab-item:: Clinical Breakdown

         .. dataset-figure:: clinical

      .. tab-item:: Dataset Growth

         .. dataset-figure:: growth

      .. tab-item:: Dataset Map

         .. dataset-figure:: bubble

      .. tab-item:: Subject Distribution

         .. dataset-figure:: moabb

.. only:: not html

   Browse the catalog interactively at
   `eegdash.org/dataset_summary.html <https://eegdash.org/dataset_summary.html>`__.
   The same data is available programmatically in Python::

       from eegdash import EEGDashDataset

       # List every dataset
       records = EEGDashDataset.list_datasets()

       # Filter by task, modality, subject count, …
       rest_datasets = EEGDashDataset.list_datasets(task="rest")

   Or via the HTTP API at ``https://data.eegdash.org``
   (see the ``/docs`` Swagger UI and the
   `api catalog </.well-known/api-catalog>`__).

The archive is still in :bdg-danger:`beta testing` mode, so be kind.
