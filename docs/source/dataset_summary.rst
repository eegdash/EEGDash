.. meta::
   :hide_sidebar: true

:html_theme.sidebar_secondary.remove:
:html_theme.sidebar_primary.remove:

.. _data_summary:
.. automodule:: eegdash.dataset

.. currentmodule:: eegdash.dataset

To leverage recent and ongoing advancements in large-scale computational methods and to ensure the preservation of scientific data generated from publicly funded research, the EEG-DaSh data archive will create a data-sharing resource for MEEG (EEG, MEG) data contributed by collaborators for machine learning (ML) and deep learning (DL) applications.

The archive is currently still in <span style="color: red;">beta testing</span> mode, so be kind. 

EEG Dash Datasets
==================

The data in EEG-DaSh originates from a collaboration involving 25 laboratories, encompassing 27,053 participants. This extensive collection includes MEEG data, which is a combination of EEG and MEG signals. The data is sourced from various studies conducted by these labs, involving both healthy subjects and clinical populations with conditions such as ADHD, depression, schizophrenia, dementia, autism, and psychosis. Additionally, data spans different mental states like sleep, meditation, and cognitive tasks. In addition, EEG-DaSh will incorporate a subset of the data converted from NEMAR, which includes 330 MEEG BIDS-formatted datasets, further expanding the archive with well-curated, standardized neuroelectromagnetic data.

Columns definitions for the table below:
   - **dataset**: Name of the dataset.
   - **n_records**: Number of EEG records in the dataset.
   - **n_subjects**: Number of subjects in the dataset.
   - **n_tasks**: Number of experimental tasks in the dataset.
   - **nchans_set**: Set of EEG channel counts used in the dataset.
   - **sampling_freqs**: Set of sampling frequencies used in the dataset.
   - **duration_hours_total**: Total duration of all recordings in hours.


Datasets
======================

.. csv-table::
   :file: ../build/dataset_summary.csv
   :header-rows: 1
   :class: sortable



.. raw:: html
   <style>
     /* Make this page full-width and remove side padding */
     :root {
       --pst-page-max-width: 100%;
       --pst-content-max-width: 100%;
     }
     .bd-main .bd-content .bd-article-container {
       max-width: 100%;
       padding-left: 0;
       padding-right: 0;
     }
     /* Ensure the DataTable uses the full width */
     table.sortable { width: 100% !important; }
   </style>

   <link href="https://cdn.datatables.net/v/bm/jq-3.7.0/dt-2.3.2/af-2.7.0/b-3.2.4/b-html5-3.2.4/cr-2.1.1/fh-4.0.3/r-3.0.5/datatables.min.css"
         rel="stylesheet"
         integrity="sha384-aemAM3yl2c0KAZZkR1b1AwMH2u3r1NHOppsl5i6Ny1L5pfqn7SDH52qdaa1TbyN9"
         crossorigin="anonymous">

   <script src="https://cdn.datatables.net/v/bm/jq-3.7.0/dt-2.3.2/af-2.7.0/b-3.2.4/b-html5-3.2.4/cr-2.1.1/fh-4.0.3/r-3.0.5/datatables.min.js"
           integrity="sha384-CKcCNsP1rMRsJFtrN6zMWK+KIK/FjYiV/d8uOp0LZtbEVzbidk105YcuVncAhBR8"
           crossorigin="anonymous"></script>

   <script>
     document.addEventListener('DOMContentLoaded', function () {
       const tables = document.querySelectorAll('table.sortable');
       tables.forEach(function (tbl) {
         // Use the jQuery plugin that ships in the bundle
         $(tbl).DataTable({
           paging: false,
           searching: false,
           info: false,
           ordering: true,
           responsive: true,
           fixedHeader: true,
           // Avoid re-initialization if this script runs more than once
           retrieve: true,
           scrollX: true
         });
       });
     });
   </script>

