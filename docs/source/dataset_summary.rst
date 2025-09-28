:hide_sidebar: true
:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _data_summary:

.. raw:: html

   <script>document.documentElement.classList.add('dataset-summary-page');</script>

.. rst-class:: dataset-summary-article

Datasets 
=========

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications.

The archive is currently still in :bdg-danger:`beta testing` mode, so be kind. 

.. raw:: html

  <figure class="eegdash-figure" style="margin: 0 0 1.25rem 0;">

.. raw:: html
  :file: ../build/dataset_bubble.html

.. raw:: html

  <figcaption class="eegdash-caption">
    Figure: Dataset landscape. Each bubble represents a dataset: x-axis shows the number of records,
    y-axis the number of subjects, bubble area encodes on-disk size, and color indicates sampling frequency band.
    Hover for details and use the legend to highlight groups.
  </figcaption>
  </figure>


.. raw:: html

  <figure class="eegdash-figure" style="margin: 1.0rem 0 0 0;">


MEEG Datasets Table
===================

The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, 
involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. 

In addition, EEG-DaSh will incorporate a subset of the data converted from `NEMAR <https://nemar.org/>`__, which includes 330 MEEG BIDS-formatted datasets, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

.. raw:: html
  :file: ../build/dataset_summary_table.html

.. raw:: html

  <figcaption class="eegdash-caption">
    Table: Sortable catalogue of EEG‑DaSh datasets. Use the “Filters” button to open column filters;
    click a column header to jump directly to a filter pane. The Total row is pinned at the bottom.
    * means that we use the median value across multiple recordings in the dataset, and empty cells
    when the metainformation is not extracted yet.
  </figcaption>
  </figure>

Pathology, modality, and dataset type now surface as consistent color-coded tags so you can scan the table at a glance and reuse the same visual language as the model catalog.
